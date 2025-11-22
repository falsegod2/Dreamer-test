import io
import os
import cv2
import math
import time
import copy
import json
import random
import pathlib
import collections
import numpy as np
import torch
from torch import distributions as torchd
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from datetime import datetime
from bisect import insort
from collections import defaultdict

# 尝试导入 wandb，如果未安装则忽略
try:
    import wandb
except ImportError:
    wandb = None

# =============================================================================
# 1. 通用工具类与函数 (General Utilities)
# =============================================================================

class Once:
    """只执行一次的触发器"""
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

class Until:
    """直到某个步数前一直返回 True"""
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until

class Every:
    """每隔一定步数触发一次"""
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count

def set_seed_everywhere(seed):
    """设置所有随机数生成器的种子，确保可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"已设置全局随机种子: {seed}")

def enable_deterministic_run():
    """开启 PyTorch 确定性模式（会降低性能，但用于调试很重要）"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print("已开启确定性运行模式 (Deterministic Mode)。")

def args_type(default):
    """参数类型转换辅助函数，用于解析配置文件或命令行参数"""
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def recursively_collect_optim_state_dict(obj, path="", optimizers_state_dicts=None, visited=None):
    """递归收集模型中所有优化器的状态（用于保存 Checkpoint）"""
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr})
    
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(attr, new_path, optimizers_state_dicts, visited)
            )
    return optimizers_state_dicts

def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    """递归加载优化器状态"""
    for path, state_dict in optimizers_state_dicts.items():
        try:
            keys = path.split(".")
            obj_now = obj
            for key in keys:
                obj_now = getattr(obj_now, key)
            obj_now.load_state_dict(state_dict)
        except AttributeError:
            print(f"警告: 无法加载优化器状态 {path}")

# =============================================================================
# 2. 分布与数学工具 (Distributions & Math)
# =============================================================================

class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """自定义的 OneHot Categorical 分布，支持 unimix_ratio 平滑"""
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("Seed support not implemented")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample

class ThresholdBuffer:
    """使用 Welford 在线算法计算均值和方差，用于动态阈值"""
    def __init__(self):
        self.n = 0       
        self.mean = 0    
        self.M2 = 0      

    def count(self):
        return self.n
    
    def fmean(self):
        return self.mean if self.n > 0 else 0
    
    def std_dev(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 0
    
    def add(self, number):
        self.n += 1
        delta = number - self.mean
        self.mean += delta / self.n
        delta2 = number - self.mean
        self.M2 += delta * delta2
    
    def get_threshold(self):
        if self.n > 0:
            return self.fmean() + self.std_dev()
        else:
            return 1.0

def convert(value, precision=32):
    """将 numpy 数组转换为指定精度的类型"""
    if value is None:
        return None
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(f"不支持的数据类型: {value.dtype}")
    return value.astype(dtype)

# =============================================================================
# 3. 图像处理工具 (Image Processing)
# =============================================================================

MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

def resized_if_need(image, target_size=(256, 160)):
    if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
    return image
    
def normalize_numpy(image, mean, std):
    for c in range(image.shape[2]):
        image[:, :, c] = (image[:, :, c] - mean[c]) / std[c]
    return image

def MCResize(image, label, target_size=(256, 160)):
    """调整图像大小以适应模型输入"""
    image = resized_if_need(image, target_size)
    label = resized_if_need(label, target_size)
    return image, label

def MyToTensor(image, label):
    """转换为 Tensor 并调整维度顺序 (H, W, C) -> (C, H, W)"""
    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(label).permute(2, 0, 1)

def MNormalize(image, label):
    """归一化图像"""
    return normalize_numpy(image / 255.0, MC_IMAGE_MEAN, MC_IMAGE_STD), label

# =============================================================================
# 4. 数据与 Replay Buffer 工具 (Data & Cache)
# =============================================================================

def selective_deepcopy(episode):
    """
    深拷贝 Episode 数据，防止修改原始 Replay Buffer。
    只对易变的图像和热力图数据进行 Deepcopy 以节省开销。
    """
    episode_copy = episode.copy()
    if "zoomed_image" in episode_copy:
        episode_copy["zoomed_image"] = copy.deepcopy(episode_copy["zoomed_image"])
    if "heatmap_on_zoomed" in episode_copy:
        episode_copy["heatmap_on_zoomed"] = copy.deepcopy(episode_copy["heatmap_on_zoomed"])
    return episode_copy

def replace_none_with_zeros(episode):
    """将数据中的 None 占位符替换为全零数组"""
    if "image" in episode and len(episode["image"]) > 0:
        reference_shape = episode["image"][0].shape
    else:
        # 如果没有 image，无法推断形状，这里抛出异常或返回
        raise ValueError("Episode 数据中缺少图像数据，无法推断形状。")

    if "zoomed_image" in episode:
        episode["zoomed_image"] = [
            np.zeros(reference_shape) if img is None else img for img in episode["zoomed_image"]
        ]
    
    heatmap_shape = None
    if "heatmap" in episode and len(episode["heatmap"]) > 0:
        heatmap_shape = episode["heatmap"][0].shape
    
    if "heatmap_on_zoomed" in episode and heatmap_shape:
        episode["heatmap_on_zoomed"] = [
            np.zeros(heatmap_shape) if hm is None else hm for hm in episode["heatmap_on_zoomed"]
        ]
    
    return episode

def count_steps(folder_path):
    """
    统计文件夹中所有 .npz 文件代表的总步数。
    文件名格式通常为: {episode_id}-{length}.npz
    """
    path = pathlib.Path(folder_path).expanduser()
    if not path.exists():
        return 0
    
    total_steps = 0
    for filename in path.glob("*.npz"):
        try:
            # 解析文件名末尾的长度 (例如 'uuid-100.npz' -> 100)
            # 减1是因为通常 trajectory length = steps + 1 (初始状态)
            length = int(filename.stem.split("-")[-1]) - 1
            total_steps += max(0, length)
        except (ValueError, IndexError):
            print(f"警告: 无法解析文件长度 {filename}")
            continue
    
    print(f"已统计 {folder_path} 中的已有步数: {total_steps}")
    return total_steps

def load_episodes(directory, limit=None, reverse=True):
    """从磁盘加载 Episode 数据到内存"""
    print(f"正在加载回合数据: {directory} (上限: {limit} 步)")
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total_steps = 0
    
    filenames = sorted(directory.glob("*.npz"))
    if reverse:
        filenames = reversed(filenames)

    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                # 将 numpy.npz 对象转换为字典
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"无法加载文件 {filename}: {e}")
            continue
            
        # 使用文件名（无后缀）作为键
        episodes[str(filename.stem)] = episode
        
        # 计算步数 (长度 - 1)
        steps = len(episode["reward"]) - 1
        total_steps += steps
        
        if limit and total_steps >= limit:
            break
            
    return episodes

def add_to_cache(cache, env_id, transition):
    """将单步 Transition 添加到内存缓存 (Cache) 中"""
    if env_id not in cache:
        cache[env_id] = dict()
        for key, val in transition.items():
            cache[env_id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[env_id]:
                # 如果是新出现的 key，先用 0 填充之前的历史，保持长度一致
                # (通常为了处理某些 info 字段在中间突然出现的情况)
                current_len = len(next(iter(cache[env_id].values())))
                cache[env_id][key] = [convert(0 * val)] * (current_len - 1)
                cache[env_id][key].append(convert(val))
            else:
                cache[env_id][key].append(convert(val))

def calculate_accumulated_reward(rewards, intrinsics, gamma):
    """计算累积折扣奖励 (用于长短期想象中的价值估计)"""
    if len(rewards) == 0:
        return 0
    
    rewards = np.array(rewards)
    intrinsics = np.array(intrinsics)
    # 生成 gamma 的幂次序列: [1, gamma, gamma^2, ...]
    gammas = np.power(gamma, np.arange(len(rewards)))
    
    discounted_sum = np.sum((rewards + intrinsics) * gammas)
    gamma_sum = np.sum(gammas)
    
    # 返回归一化后的累积奖励
    return discounted_sum / (gamma_sum + 1e-8)

def save_episodes(directory, episodes):
    """将内存中的 Episodes 保存到磁盘"""
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        # 文件名格式: {id}-{length}.npz
        filepath = directory / f"{filename}-{length}.npz"
        
        with io.BytesIO() as f1:
            episode_copy = selective_deepcopy(episode)
            episode_copy = replace_none_with_zeros(episode_copy)
            np.savez_compressed(f1, **episode_copy)
            f1.seek(0)
            with filepath.open("wb") as f2:
                f2.write(f1.read())
            del episode_copy
    return True

def erase_over_episodes(cache, dataset_size):
    """如果 Cache 超过大小限制，删除最旧的数据"""
    step_in_dataset = 0
    # 按键排序（通常包含时间戳），从新到旧遍历
    sorted_items = reversed(sorted(cache.items(), key=lambda x: x[0]))
    
    for key, ep in sorted_items:
        steps = len(ep["reward"]) - 1
        if not dataset_size or (step_in_dataset + steps) <= dataset_size:
            step_in_dataset += steps
        else:
            del cache[key]
            
    return step_in_dataset

def from_generator(generator, batch_size):
    """
    将生成器产生的数据打包成 Batch。
    这是连接 sample_episodes 和训练循环的关键桥梁。
    """
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(generator))
            except StopIteration:
                return # 数据耗尽
        
        # 将 batch 列表转换为字典形式: {key: stack(values)}
        data = {}
        keys = batch[0].keys()
        for key in keys:
            data[key] = np.stack([item[key] for item in batch], axis=0)
            
        yield data

def sample_episodes(episodes, length, seed=0):
    """
    数据采样器生成器。
    从 episodes 中采样长度为 length 的序列片段 (chunks)。
    """
    np_random = np.random.RandomState(seed)
    
    while True:
        # 1. 筛选有效 episodes 并计算采样概率
        valid_episodes = []
        lengths = []
        
        for ep in episodes.values():
            # 检查是否有数据
            if len(ep) == 0: continue
            ep_len = len(next(iter(ep.values())))
            if ep_len >= 2: # 至少要有两帧才能构成 transition
                valid_episodes.append(ep)
                lengths.append(ep_len)
        
        if not valid_episodes:
            # 如果没有数据，稍作等待，避免死循环占用 CPU
            time.sleep(1)
            continue
            
        probs = np.array(lengths) / np.sum(lengths)
        
        # 2. 构建 Batch Chunk
        current_len = 0
        ret = None # 最终返回的片段字典
        
        while current_len < length:
            # 按概率采样一个 episode
            episode = np_random.choice(valid_episodes, p=probs)
            total = len(next(iter(episode.values())))
            
            # 处理数据副本
            episode_copy = selective_deepcopy(episode)
            episode_copy = replace_none_with_zeros(episode_copy)
            
            if ret is None:
                # 第一次采样：随机选一个起点
                # 保证不选到最后一步，因为需要 next_state
                index = int(np_random.randint(0, total - 1))
                end_index = min(index + length, total)
                
                ret = {
                    k: v[index : end_index].copy()
                    for k, v in episode_copy.items()
                    if "log_" not in k
                }
                # 标记序列起点
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 后续采样：拼接到 ret 后面
                # 从新 episode 的头部开始接
                needed = length - current_len
                end_index = min(needed, total)
                
                ret = {
                    k: np.append(ret[k], v[0 : end_index].copy(), axis=0)
                    for k, v in episode_copy.items()
                    if "log_" not in k
                }
                # 标记拼接点为 is_first
                if "is_first" in ret:
                    ret["is_first"][current_len] = True
                    
            current_len = len(next(iter(ret.values())))
            del episode_copy
            
        yield ret

def make_dataset(episodes, config):
    """创建数据加载流水线"""
    print("创建数据集生成器...")
    generator = sample_episodes(episodes, config.batch_length, seed=config.seed)
    dataset = from_generator(generator, config.batch_size)
    return dataset

# =============================================================================
# 5. 环境构建与配置 (Environment Construction)
# =============================================================================

from omegaconf import OmegaConf
from minedojo.sim import MineDojoSim
from minedojo.tasks import MetaTaskBase, _meta_task_make, _parse_inventory_dict, ALL_TASKS_SPECS
import envs.wrappers as wrappers
# 在函数外部导入自定义 Wrapper，避免循环引用或缩进问题
from wrappers_own import (
    MinedojoScreenshotWrapper, 
    MinedojoRewardWrapper,
    MinedojoSuccessWrapper, 
    MinedojoTerminalWrapper, 
    ConcentrationWrapper,
    MinedojoLSImagineWrapper, 
    MinedojoSemifastResetWrapper, 
    ClipWrapper
)
from reward_own import MinedojoClipReward, MinedojoConcentrationReward

# 加载任务配置
try:
    CUSTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("envs/tasks/task_specs.yaml"))
except Exception:
    CUSTOM_TASK_SPECS = {}

def get_specs(task, **kwargs):
    """解析任务 ID 和配置参数"""
    if task in CUSTOM_TASK_SPECS:
        yaml_specs = CUSTOM_TASK_SPECS[task].copy()
        task_id = yaml_specs.pop("task_id", task)
        assert "sim" in yaml_specs, "task_specs.yaml must define sim attribute"
    else:
        yaml_specs = dict()
        task_id = task

    sim_specs = yaml_specs.pop("sim_specs", dict())
    
    # 默认任务配置
    task_specs = dict(
        clip=False,
        fake_clip=False,
        fake_dreamer=False,
        subgoals=False,
    )
    task_specs.update(**yaml_specs)
    task_specs.update(**kwargs)
    
    assert not (task_specs["clip"] and task_specs["fake_clip"]), "不能同时开启 Clip 和 Fake Clip"

    return task_id, task_specs, sim_specs

def _get_minedojo_specs(task_id, task_specs, sim_specs):
    """获取 MineDojo 底层模拟器的配置"""
    if task_id in ALL_TASKS_SPECS:
        minedojo_specs = ALL_TASKS_SPECS[task_id]
        if OmegaConf.is_config(minedojo_specs):
            minedojo_specs = OmegaConf.to_container(minedojo_specs)
        minedojo_specs.pop("prompt", None)
        meta_task_cls = minedojo_specs.pop("__cls__")
    else:
        minedojo_specs = dict()
        meta_task_cls = task_id

    # 基础配置
    minedojo_specs.update(dict(
        image_size=(160, 256), 
        fast_reset=False, 
        event_level_control=False,
        use_voxel=False,
        use_lidar=False
    ))

    # 如果任务涉及 Blocks (方块)，激活 Voxel 观测
    terminal_conds = task_specs.get("terminal_specs", {})
    success_conds = task_specs.get("success_specs", {})
    
    has_blocks = False
    for specs in [terminal_conds, success_conds]:
        for key in ["all", "any"]:
            if key in specs and any(x == "blocks" for x in specs[key]):
                has_blocks = True
                break
    
    if has_blocks:
        minedojo_specs["use_voxel"] = True
        minedojo_specs["voxel_size"] = dict(xmin=-3, ymin=-1, zmin=-3, xmax=3, ymax=1, zmax=3)

    minedojo_specs.update(**sim_specs)

    if "initial_inventory" in minedojo_specs:
        minedojo_specs["initial_inventory"] = _parse_inventory_dict(minedojo_specs["initial_inventory"])

    return meta_task_cls, minedojo_specs

def _add_wrappers(
    env: MetaTaskBase, 
    task_id: str, 
    LS_Imagine_specs: dict = None,
    screenshot_specs: dict = None,
    reward_specs: dict = None,
    success_specs: dict = None,
    terminal_specs: dict = None,
    clip_specs: dict = None,
    concentration_specs: dict = None,
    fast_reset: int = None,
    log_dir: str = None,
    freeze_equipped: bool = True,
    **kwargs
):
    """为环境添加各种 Wrapper"""
    
    if terminal_specs is None:
        terminal_specs = dict(max_steps=500, on_death=True)
    if success_specs and terminal_specs:
        success_specs["max_steps"] = terminal_specs["max_steps"]
        
    # 基础功能 Wrapper
    if screenshot_specs:
        env = MinedojoScreenshotWrapper(env, log_dir=log_dir, **screenshot_specs)
    if reward_specs:
        env = MinedojoRewardWrapper(env, **reward_specs)
    if success_specs:
        env = MinedojoSuccessWrapper(env, **success_specs)

    env = MinedojoTerminalWrapper(env, **terminal_specs)
    """
    基础功能包装
    执行完这一块，env 对象的嵌套结构是 
    Terminal(Success(Reward(Screenshot(BaseEnv))))
    调用env的相同函数“依次”是有严格顺序的：
        调用：从外到内，一层层深入。
        返回（和修改）：从内到外，一层层冒出。
    """
    # 奖励塑造 (Reward Shaping) Wrapper
    if clip_specs is not None:
        #clip_reward是一个clip功能包，用于调用基于clip的各种工具
        clip_reward = MinedojoClipReward()
        env = ClipWrapper(env, clip_reward, **clip_specs)

    if concentration_specs is not None:
        unet_ckpt = concentration_specs.get("unet_checkpoint_dir", "envs/tasks/base/unet_checkpoint")
        sigma_w = concentration_specs.get("gaussian_sigma_weight", 0.5)
        concentration_reward = MinedojoConcentrationReward(
            unet_checkpoint_dir=unet_ckpt, 
            output_dir=log_dir, 
            gaussian_sigma_weight=sigma_w
        )
        env = ConcentrationWrapper(env, concentration_reward, **concentration_specs)

    # LS-Imagine 核心逻辑 Wrapper
    env = MinedojoLSImagineWrapper(env, **(LS_Imagine_specs or {}))

    # 快速重置 Wrapper (用于加速训练)
    if fast_reset is not None:
        wrapped = env
        while hasattr(wrapped, "env"):
            if isinstance(wrapped.env, MineDojoSim):
                wrapped.env = MinedojoSemifastResetWrapper(
                    wrapped.env,
                    reset_freq=fast_reset,
                    random_teleport_range=200
                )
                break
            wrapped = wrapped.env

    return env

def make_env(config, mode, id):
    """
    环境创建工厂函数。
    被 expr.py 调用。
    """
    suite, task = config.task.split("_", 1)
    print(f"正在创建环境: suite={suite}, task={task}, mode={mode}, id={id}")

    log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # 内部辅助 make 函数
    def _make(task_name, **kwargs):
        task_id, task_specs, sim_specs = get_specs(task_name, **kwargs)
        meta_task_cls, minedojo_specs = _get_minedojo_specs(task_id, task_specs, sim_specs)
        
        # 创建基础环境
        env = _meta_task_make(meta_task_cls, **minedojo_specs)
        # 添加 Wrappers
        env = _add_wrappers(env, task_id, log_dir=log_dir, **task_specs)
        return env

    # 构建环境
    kwargs = dict(
        log_dir=log_dir,
        target_item=config.target_item
    )
    env = _make(task, **kwargs)
    
    # 添加通用 OneHot 动作空间 Wrapper
    env = wrappers.OneHotAction(env)
    """
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)
    """
    return env

# =============================================================================
# 6. 日志与统计 (Logging & Stats)
# =============================================================================

class ScoreStorage:
    """用于跟踪 'Zoomed' 图像评分的缓冲区"""
    def __init__(self, max_steps=1000):
        self.data = defaultdict(list)
        self.max_steps = max_steps
    
    def add(self, env_id, step, score_on_zoomed):
        insort(self.data[env_id], (score_on_zoomed, step))
    
    def get_and_remove_less_than(self, env_id, current_step, score):
        if env_id not in self.data:
            return []
        
        removable_indices = []
        steps = []
        # 找出分数低于当前分数或步数过旧的记录
        for i, (score_on_zoomed, step) in enumerate(self.data[env_id]):
            if score_on_zoomed < score or (current_step - step >= self.max_steps):
                removable_indices.append(i)
                steps.append(step)
        
        # 删除
        for index in reversed(removable_indices):
            del self.data[env_id][index]
        
        return steps
    
    def remove_all(self, env_id):
        if env_id in self.data:
            del self.data[env_id]
    
    def count_data_pairs(self, env_id):
        return len(self.data[env_id])

class Logger:
    """统一日志记录器 (Tensorboard + WandB + Console)"""
    def __init__(self, config, logdir, step):
        print(f"日志系统启动，目标目录: {logdir}，起始步数: {step}")
        self._wandb = config.use_wandb and (wandb is not None)
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

        if self._wandb:
            if config.wandb_key is None:
                raise ValueError("配置中 wandb_key 为空")
            wandb.login(key=config.wandb_key)
            wandb.init(project="LS-Imagine", dir=str(logdir), config=config.__dict__, name=str(logdir), save_code=True)
            wandb.config.update({"initial_step": step})
        else:
            self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        
        # 控制台输出
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        
        # 文件记录
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

        # WandB 记录
        if self._wandb:
            for name, value in scalars:
                wandb.log({name: value}, step=step)
            for name, value in self._images.items():
                wandb.log({name: [wandb.Image(value)]}, step=step)
            for name, value in self._videos.items():
                # WandB 视频格式处理
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                # (B, T, H, W, C) -> (T, C, H, W*B)
                B, T, H, W, C = value.shape
                value = value.transpose(1, 4, 2, 0, 3).reshape((T, C, H, B * W))
                wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)
        
        # TensorBoard 记录
        else:
            for name, value in scalars:
                prefix = "scalars/" if "/" not in name else ""
                self._writer.add_scalar(prefix + name, value, step)
            for name, value in self._images.items():
                self._writer.add_image(name, value, step)
            for name, value in self._videos.items():
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                B, T, H, W, C = value.shape
                value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
                self._writer.add_video(name, value, step, fps=16)
            self._writer.flush()

        # 清空缓存
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / (duration + 1e-8)

    def finish(self):
        if self._wandb:
            wandb.finish()

# =============================================================================
# 7. 模拟循环 (Simulation Loop)
# =============================================================================

def simulate(
    agent,
    envs, 
    cache, 
    directory, 
    logger, 
    step_calculator,
    max_steps,
    gamma,
    is_eval=False,
    limit=None, 
    steps=0, 
    episodes=0,
    state=None,
    is_training=False,
):
    """
    核心模拟循环：控制 Agent 与 Environment 的交互。
    负责：Step 环境、重置环境、收集数据到 Cache、计算 Score、保存 Episode。
    执行数据收集的核心函数

    让 agent 和 envs 交互，把数据存入 cache，并保存到 directory
    
    在 expr.py 中被调用了两次
       Prefill（预填充）：用一个“随机智能体”来收集一些初始数据 
       Train（训练）：在主循环中，用我们正在学习的 LS_Imagine 智能体来收集数据
    """
    print(f"开始模拟... (目录: {directory}, 目标步数: {steps}, 目标回合: {episodes})")
    
    # 初始化或解包状态
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        # reward/info 初始化
        information = [{}] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, _, information = state
        
    while (steps and step < steps) or (episodes and episode < episodes):
        
        # --- 1. 环境重置 (Reset) ---
        if done.any():
            # 找出需要重置的环境索引
            indices = [index for index, d in enumerate(done) if d]
            # 如果是评估模式，或者该环境确实已经结束 (real_done)
            indices = [index for index in indices if information[index].get("real_done", True)]
            
            # 执行 Reset
            if indices:
                results = [envs[i].reset() for i in indices]
                # 解包结果 (假设 env.reset 返回 (obs, info) 或直接 obs)
                # 这里假设 env 是自定义 Wrapper，reset 可能返回函数需调用
                results = [r() if callable(r) else r for r in results] 

                for index, result in zip(indices, results):
                    # 处理初始观测
                    t = result.copy() if isinstance(result, dict) else result
                    t = {k: convert(v) for k, v in t.items()}
                    
                    # 初始化 Transition 字段
                    t["reward"] = 0.0
                    t["discount"] = 1.0
                    
                    # 将初始状态写入 Cache
                    add_to_cache(cache, envs[index].id, t)

                    # 如果初始状态就有 Zoomed Score，记录下来
                    current_step = 0
                    if t.get("is_zoomed", False):
                        step_calculator.add(envs[index].id, current_step, t["score_on_zoomed"])

                    obs[index] = result

        # --- 2. Agent 决策 (Action) ---
        # 堆叠观测数据 batch
        obs_batch = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        
        # 调用 Agent 获取动作
        action, agent_state = agent(obs_batch, done, agent_state)

        # 格式化 Action (转回 numpy, 处理 tensor)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        
        # --- 3. 环境步进 (Step) ---
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() if callable(r) else r for r in results]
        
        # 解包 step 结果: obs, reward, done, info
        obs_next, reward_next, done_next, info_next = zip(*[p[:4] for p in results])

        obs = list(obs_next)
        reward = list(reward_next)
        done = np.stack(done_next)
        information = list(info_next)
        
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= (1 - done.astype(int)) # 如果 done，重置 length

        # --- 4. 数据缓存 (Cache) ---
        for i in range(len(results)):
            a, result, env = action[i], results[i], envs[i]
            o, r, d, info = result
            
            # 准备 Transition 数据
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            transition["success"] = info.get("success", False)
            transition["first_success_step"] = info.get("first_success_step", max_steps)
            
            # 存入 Cache
            add_to_cache(cache, env.id, transition)

            # LS-Imagine 逻辑：计算 Jumping Steps 和 Accumulated Reward
            ep_len = len(cache[env.id]["reward"])
            current_step_idx = ep_len - 1
            
            if transition.get("is_zoomed", False) and not d:
                step_calculator.add(env.id, current_step_idx, transition["score_on_zoomed"])

            # 处理之前的 Zoomed 状态
            # 这里的逻辑大概是：如果当前状态的分数比之前的某个 Zoomed 状态高，
            # 那么之前的那个状态可以“跳跃”到当前状态。
            valid_prev_steps = step_calculator.get_and_remove_less_than(env.id, current_step_idx, transition.get("score", 0))
            
            if valid_prev_steps:
                for prev_step in valid_prev_steps:
                    # 计算跳跃步数
                    cache[env.id]["jumping_steps"][prev_step] = current_step_idx - prev_step
                    # 计算跳跃区间的累积奖励
                    rewards_slice = cache[env.id]["reward"][prev_step+1 : current_step_idx]
                    intrinsics_slice = cache[env.id]["intrinsic"][prev_step+1 : current_step_idx]
                    
                    acc_reward = calculate_accumulated_reward(rewards_slice, intrinsics_slice, gamma)
                    cache[env.id]["accumulated_reward"][prev_step] = acc_reward
                    cache[env.id]["is_calculated"][prev_step] = True

            # 如果数据对处理完了，标记 real_done
            if step_calculator.count_data_pairs(env.id) == 0:
                information[i]['real_done'] = True

        # --- 5. 回合结束处理 (Episode End) ---
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            for i in indices:
                # 只有在真正结束时才保存 (除非是 Eval 模式)
                if (not is_eval) and (not information[i].get("real_done", False)):
                    continue

                # 保存 Episode 到磁盘
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})

                # 清理临时存储
                step_calculator.remove_all(envs[i].id)
                
                # 计算统计数据
                ep_len = len(cache[envs[i].id]["reward"]) - 1
                ep_rew = float(np.array(cache[envs[i].id]["reward"])[:max_steps+1].sum())
                ep_suc = 1 if any(np.array(cache[envs[i].id]["success"])[:max_steps+1]) else 0
                first_suc = min(cache[envs[i].id]["first_success_step"][-1], max_steps)
                
                # 记录标量日志 (log_xxx)
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(key, float(np.array(cache[envs[i].id][key]).sum()))
                        cache[envs[i].id].pop(key)

                # 训练模式日志
                if not is_eval:
                    # 控制数据集大小，删除旧数据
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", ep_rew)
                    logger.scalar("train_length", ep_len)
                    logger.scalar("train_episodes", len(cache))
                    logger.scalar("train_success", ep_suc)
                    logger.scalar("train_first_success_step", first_suc)
                    logger.write(step=logger.step)
                
                # 评估模式日志 (聚合计算)
                else:
                    # 使用 local 变量存储评估统计 (注意：simulate 函数重入时会重置，这里假设单次调用)
                    if "eval_stats" not in locals():
                        eval_stats = defaultdict(list)
                        eval_done_flag = False
                    
                    eval_stats["return"].append(ep_rew)
                    eval_stats["length"].append(ep_len)
                    eval_stats["success"].append(ep_suc)
                    eval_stats["first_success"].append(first_suc)

                    # 记录视频 (只记第一个)
                    video = cache[envs[i].id].get("image", None)
                    if video is not None and len(eval_stats["return"]) == 1:
                         logger.video("eval_policy", np.array(video)[None])

                    # 如果收集够了 episode 数量
                    current_eval_eps = len(eval_stats["return"])
                    if current_eval_eps >= episodes and not eval_done_flag:
                        logger.scalar("eval_return", np.mean(eval_stats["return"]))
                        logger.scalar("eval_length", np.mean(eval_stats["length"]))
                        logger.scalar("eval_episodes", current_eval_eps)
                        logger.scalar("eval_success", np.mean(eval_stats["success"]))
                        logger.scalar("eval_first_success_step", np.mean(eval_stats["first_success"]))
                        logger.write(step=logger.step)
                        eval_done_flag = True

    # 评估模式下清理 Cache，节省内存
    if is_eval:
        while len(cache) > 1:
            cache.popitem(last=False)

    return (step - steps, episode - episodes, done, length, obs, agent_state, reward, information)