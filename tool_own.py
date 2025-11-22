import numpy as np
import time
import copy
import cv2
import torch
from torchvision.transforms import Normalize
import math
import os
from minedojo.sim import MineDojoSim
import envs.wrappers as wrappers
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
import random
import json
import collections
import pathlib
from bisect import insort
from collections import defaultdict
import io
from torch import distributions as torchd
from torch.nn import functional as F

class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until

class Every:
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
    
class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample

def make_env(config, mode, id):
    """创建环境的辅助函数。"""
    suite, task = config.task.split("_", 1)
    print(f"创建环境: suite={suite}, task={task}")

    """
    log_dir: 一个用于环境日志的特定目录
    kwargs: 传递环境日志
    """
    log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    kwargs=dict(
            log_dir=log_dir,
            target_item=config.target_item
        )
    
    env = make(task, **kwargs)
    env = wrappers.OneHotAction(env)


    from omegaconf import OmegaConf
    from envs.tasks.minedojo import make_minedojo
    CUTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("envs/tasks/task_specs.yaml"))

    def get_specs(task, **kwargs):
        # Get task data and task id
        if task in CUTOM_TASK_SPECS:
            yaml_specs = CUTOM_TASK_SPECS[task].copy()
            task_id = yaml_specs.pop("task_id", task)
            assert "sim" in yaml_specs, "task_specs.yaml must define sim attribute"
        else:
            yaml_specs = dict()
            task_id = task

        # Get minedojo specs
        sim_specs = yaml_specs.pop("sim_specs", dict())

        # Get our task specs
        task_specs = dict(
            clip=False,
            fake_clip=False,
            fake_dreamer=False,
            subgoals=False,
        )
        task_specs.update(**yaml_specs)
        task_specs.update(**kwargs)
        assert not (task_specs["clip"] and task_specs["fake_clip"]), "Can only use one reward shaper"

        return task_id, task_specs, sim_specs
    
    from minedojo.tasks import MetaTaskBase, _meta_task_make, _parse_inventory_dict, ALL_TASKS_SPECS
    def _get_minedojo_specs(task_id, task_specs, sim_specs):
        if task_id in ALL_TASKS_SPECS:
            minedojo_specs = ALL_TASKS_SPECS[task_id]
            if OmegaConf.is_config(minedojo_specs):
                minedojo_specs = OmegaConf.to_container(minedojo_specs)
            minedojo_specs.pop("prompt", None)
            meta_task_cls = minedojo_specs.pop("__cls__")
        else:
            minedojo_specs = dict()
            meta_task_cls = task_id

        minedojo_specs.update(dict(
            image_size=(160, 256), 
            fast_reset=False, 
            event_level_control=False,
            use_voxel=False,
            use_lidar=False
        ))

        # If using blocks condition, activate voxels
        if ("terminal_specs" in task_specs and \
                ("all" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["all"]) or \
                "any" in task_specs["terminal_specs"] and any(x == "blocks" for x in task_specs["terminal_specs"]["any"]))) or \
            ("success_specs" in task_specs and \
                ("all" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["all"]) or \
                "any" in task_specs["success_specs"] and any(x == "blocks" for x in task_specs["success_specs"]["any"]))):
            minedojo_specs["use_voxel"] = True
            minedojo_specs["voxel_size"] = dict(xmin=-3, ymin=-1, zmin=-3, xmax=3, ymax=1, zmax=3)

        minedojo_specs.update(**sim_specs)

        if "initial_inventory" in minedojo_specs:
            minedojo_specs["initial_inventory"] = _parse_inventory_dict(minedojo_specs["initial_inventory"])

        return meta_task_cls, minedojo_specs
    
    from typing import Dict
    from wrappers_own import MinedojoScreenshotWrapper,MinedojoRewardWrapper
    from wrappers_own import MinedojoSuccessWrapper,MinedojoTerminalWrapper,ConcentrationWrapper
    from wrappers_own import MinedojoLSImagineWrapper,MinedojoSemifastResetWrapper
    from wrappers_own import ClipWrapper
    from reward_own import MinedojoClipReward,MinedojoConcentrationReward

    def _add_wrappers(
        env: MetaTaskBase, 
        task_id: str, 
        LS_Imagine_specs: Dict = None,
        screenshot_specs: Dict = None,
        reward_specs: Dict = None,
        success_specs: Dict = None,
        terminal_specs: Dict = None,
        clip_specs: Dict = None,
        concentration_specs: Dict = None,
        fast_reset: int = None,
        log_dir: str = None,
        freeze_equipped: bool = True,
        **kwargs
        ):

        if terminal_specs is None:
            terminal_specs = dict(max_steps=500, on_death=True)
        if success_specs and terminal_specs:
            success_specs["max_steps"] = terminal_specs["max_steps"]
        if screenshot_specs:
            env = MinedojoScreenshotWrapper(env, log_dir=log_dir, **screenshot_specs)
        if reward_specs:
            env = MinedojoRewardWrapper(env, **reward_specs)
        if success_specs:
            env = MinedojoSuccessWrapper(env, **success_specs)

        env = MinedojoTerminalWrapper(env, **terminal_specs)
        """
        核心功能包装
        执行完这一块，env 对象的嵌套结构是 
        Terminal(Success(Reward(Screenshot(BaseEnv))))
        调用env的相同函数“依次”是有严格顺序的：
            调用：从外到内，一层层深入。
            返回（和修改）：从内到外，一层层冒出。
        """
        # Add reward shaping wrapper
        # 奖励塑造包装
        if clip_specs is not None:
            #clip_reward是一个clip功能包，用于调用基于clip的各种工具
            clip_reward = MinedojoClipReward()
            env = ClipWrapper(env, clip_reward, **clip_specs)

        if concentration_specs is not None:
            unet_checkpoint_dir = concentration_specs["unet_checkpoint_dir"] if "unet_checkpoint_dir" in concentration_specs else "envs/tasks/base/unet_checkpoint"
            gaussian_sigma_weight = concentration_specs["gaussian_sigma_weight"] if "gaussian_sigma_weight" in concentration_specs else 0.5
            concentration_reward = MinedojoConcentrationReward(unet_checkpoint_dir=unet_checkpoint_dir, output_dir=log_dir, gaussian_sigma_weight=gaussian_sigma_weight)
            env = ConcentrationWrapper(env, concentration_reward, **concentration_specs)

        env = MinedojoLSImagineWrapper(env, **LS_Imagine_specs)

        # If we don't care about start position, use fast reset to speed training and prevent memory leaks
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

    def make(task: str, **kwargs):
        task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict
        meta_task_cls, minedojo_specs = _get_minedojo_specs(task_id, task_specs, sim_specs)
        
        env = _meta_task_make(meta_task_cls, **minedojo_specs)
        env = _add_wrappers(env, task_id, **task_specs)

        return env
    

# --- 1. 工具函数 ---
def selective_deepcopy(episode):
    """
    防止破坏回放缓冲区（Replay Buffer）中的原始数据
    copy.deepcopy 会创建一个全新的对象，并把原始对象的所有内容复制过去
    """
    episode_copy = episode.copy()

    if "zoomed_image" in episode_copy:
        episode_copy["zoomed_image"] = copy.deepcopy(episode_copy["zoomed_image"])
    
    if "heatmap_on_zoomed" in episode_copy:
        episode_copy["heatmap_on_zoomed"] = copy.deepcopy(episode_copy["heatmap_on_zoomed"])
    
    return episode_copy

def replace_none_with_zeros(episode):
    """
    将在 episode 列表中作为占位符（placeholder）的 None 值，
    替换为具有正确形状和数据类型的全零 NumPy 数组（np.zeros）
    """
    if "image" in episode and episode["image"]:
        reference_shape = episode["image"][0].shape
    else:
        raise ValueError("Image data is missing or malformed in episode.")
    
    if "zoomed_image" in episode:
        episode["zoomed_image"] = [
            np.zeros(reference_shape) if img is None else img for img in episode["zoomed_image"]
        ]
    
    if "heatmap" in episode and episode["heatmap"]:
        heatmap_shape = episode["heatmap"][0].shape
    else:
        heatmap_shape = None
    
    if "heatmap_on_zoomed" in episode and heatmap_shape:
        episode["heatmap_on_zoomed"] = [
            np.zeros(heatmap_shape) if hm is None else hm for hm in episode["heatmap_on_zoomed"]
        ]
    
    return episode

def args_type(default):
    """
    根据从 configs.yaml 文件中加载的值的类型，来自动创建一个正确的类型转换函数
    """
    #用户传入的字符串
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

    #处理那些不是从命令行传入的字符串、而是直接从 YAML 文件加载的默认值
    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def set_seed_everywhere(seed):
    """设置所有库（torch, numpy, random）的随机种子。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"设置全局随机种子为: {seed}")
    pass

def enable_deterministic_run():
    """开启 PyTorch 的确定性算法，确保可复现性。"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print("开启确定性运行模式。")
    pass

def count_steps(folder_path):
    """统计一个文件夹中所有 .npz 文件的总步数。"""
    print(f"统计 {folder_path} 中的已有步数...")
    # 假设返回一个整数
    return 0 

def load_episodes(directory, limit=None, reverse=True):
    """从目录加载 .npz 回合数据到内存中。"""
    print(f"从 {directory} 加载回合数据，上限: {limit}...")

    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            """
            注意，这里的total加的是episode的长度，因此limit限制的是所有episodes加起来的内部步数
            因此将limit设成10是完全不合理的，相当于只取上一次的更新的episode，也就是replay buffer里只有一个序列
            """
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes

def make_dataset(episodes, config):
    """将内存中的回合数据转换为一个可迭代的数据集生成器。"""
    print("创建数据集生成器...")
    # 假设返回一个生成器
    return iter([])



def recursively_collect_optim_state_dict(
        obj, path="", optimizers_state_dicts=None, visited=None
):
    """递归收集 agent 内部所有优化器的状态。"""
    print("收集优化器状态...")
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    """递归加载优化器状态。"""
    print("加载优化器状态...")
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
    pass

class Logger:
    """用于记录标量、视频和日志到控制台及文件的类。"""
    def __init__(self, config, logdir, step):
        print(f"日志系统启动，目标目录: {logdir}，起始步数: {step}")
        self._wandb = config.use_wandb
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

        if self._wandb:
            if config.wandb_key is None:
                raise ValueError("wandb_key is None")
            wandb.login(key=config.wandb_key)
            wandb.init(project="LS-Imgine", dir=str(logdir), config = config.__dict__, name = str(logdir), save_code=True)
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
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

        if self._wandb:
            for name, value in scalars:
                wandb.log({name: value}, step=step)

            for name, value in self._images.items():
                wandb.log({name: [wandb.Image(value)]}, step=step)

            for name, value in self._videos.items():
                name = name if isinstance(name, str) else name.decode("utf-8")
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                B, T, H, W, C = value.shape
                value = value.transpose(1, 4, 2, 0, 3).reshape((T, C, H, B * W))
                wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)

            self._scalars = {}
            self._images = {}
            self._videos = {}

        else:
            for name, value in scalars:
                if "/" not in name:
                    self._writer.add_scalar("scalars/" + name, value, step)
                else:
                    self._writer.add_scalar(name, value, step)
            for name, value in self._images.items():
                self._writer.add_image(name, value, step)
            for name, value in self._videos.items():
                name = name if isinstance(name, str) else name.decode("utf-8")
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                B, T, H, W, C = value.shape
                value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
                self._writer.add_video(name, value, step, 16)

            self._writer.flush()
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
        return steps / duration

    def offline_scalar(self, name, value, step):
        if self._wandb:
            wandb.log({name: value}, step=step)
        else:
            self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((T, C, H, B * W))
        if self._wandb:
            wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)
        else:
            self._writer.add_video(name, value, step, 16)

    def finish(self):
        if self._wandb:
            wandb.finish()

class ScoreStorage:
    """LS-Imagine 专用的工具，用于存储和计算“跳跃”步骤。"""
    def __init__(self, max_steps=1000):
        print(f"ScoreStorage 初始化，最大步数: {max_steps}")
        self.data = defaultdict(list)
        self.max_steps = max_steps
    
    def add(self, env_id, step, score_on_zoomed):
        insort(self.data[env_id], (score_on_zoomed, step))
    
    def get_and_remove_less_than(self, env_id, current_step, score):
        if env_id not in self.data:
            return []
        
        removable_indices = []
        steps = []
        for i, (score_on_zoomed, step) in enumerate(self.data[env_id]):
            if score_on_zoomed < score or (current_step - step >= self.max_steps):
                removable_indices.append(i)
                steps.append(step)
        
        for index in reversed(removable_indices):
            del self.data[env_id][index]
        
        return steps
    
    def remove_all(self, env_id):
        if env_id in self.data:
            del self.data[env_id]
    
    def count_data_pairs(self, env_id):
        return len(self.data[env_id])
    


def sample_episodes(episodes, length, seed=0):
    """
    从所有存储的完整 "episodes" (回合) 中，高效地采样出固定长度length的短序列batch,里面用 "chunks"填充，用于训练
        batch的长度length指的是一个batch里需要填充多少个episodes

        数据连续性： 世界模型（RSSM）需要连续的数据序列（batch_length）来学习时序动态。
        数据稀疏性： 完整的 episode 可能非常长（几千步），但有趣或重要的事件（如找到钻石）可能只占一小部分。
    """
    def _get_episode_probabilities(episodes):
        """
        计算采样概率。
        遍历所有可用的 episode，计算它们的长度，并返回一个加权概率分布。
        一个 episode 被选中的概率与其长度（包含的步数）成正比。
        确保了模型更频繁地从包含更多数据的长 episode 中学习。
        """
        valid_episodes = []
        episode_lengths = []
        
        for episode in episodes.values():
            try:
                # 检查 episode 是否有数据
                # next(iter(episode.values())) 会获取第一个键的值（例如 'image' 数组）
                #只需要获得第一个键的值的维度，因为同一个episode中每个键的值的数量是一样的，代表着该episode的长度
                length = len(next(iter(episode.values())))
                
                # 确保 episode 至少有1个初始状态和1个步进（即至少2个数据点）
                if length >= 2:
                    valid_episodes.append(episode)
                    episode_lengths.append(length)
                    
            except StopIteration:
                # 捕获空的 episode (例如 {})，并忽略它
                continue
                
        lengths_array = np.array(episode_lengths)
        total_steps = np.sum(lengths_array)
        
        if total_steps == 0:
            # 如果episodes的长度为0，也就是环境步数为0 (例如刚开始训练)，返回 None
            return None, None
            
        # 概率 = 该episode的长度 / 所有episodes的总长度
        probabilities = lengths_array / total_steps
        return valid_episodes, probabilities
    
    """
    一个从 'episodes' 集合中采样 'length' 长度序列的生成器。
    
    它会持续不断地生成（yield）一个又一个的训练序列（chunks）。
    
    参数:
    episodes (dict): 包含所有 episode 数据的字典 (例如 train_eps)。
    length (int): 目标序列长度 (即 config.batch_length)。
    seed (int): 随机种子。
    """
    
    # 初始化此生成器专用的随机数生成器
    np_random = np.random.RandomState(seed)

    # 这是一个生成器，会无限循环地产生数据
    while True:
        
        # 1. 获取最新的 episode 概率
        # 这必须在每次循环时都做，因为 'episodes' 字典（回放缓冲区）
        # 在训练过程中会不断被添加新的数据。
        valid_episodes, probabilities = _get_episode_probabilities(episodes)
        
        if valid_episodes is None:
            # 如果当前没有有效的 episode (例如，所有 episode 都太短)
            # 暂停1秒，等待更多数据被采集
            print("警告: 缓冲区中没有长度 >= 2 的有效 episodes。")
            time.sleep(1)
            continue

        # ---
        # 2. 构建一个大小为 'length' 的批次 (batch)
        # ---
        current_batch_size = 0
        batch_to_yield = None # 这是我们将要构建和返回的批次

        # 循环直到我们的批次被填满，其中length = config.batch_length = 32
        while current_batch_size < length:
            
            # --- 2a. 加权采样一个 episode ---
            episode_data = np_random.choice(
                valid_episodes,  # 从有效 episode 列表中选
                p=probabilities  # 使用加权概率
            )
            
            # --- 2b. 清理数据 (用于后续切片) ---
            # 必须操作副本，否则会修改回放缓冲区中的原始数据
            episode_copy = selective_deepcopy(episode_data)
            episode_copy = replace_none_with_zeros(episode_copy)
            
            total_steps_in_episode = len(next(iter(episode_copy.values())))

            # --- 2c. 从 episode 中切片数据 ---
            if not batch_to_yield:
                # --- 情况A: 这是批次的第一个 chunk ---
                
                # 从 episode 中随机选择一个起始点
                # (不能选最后一步，因为至少需要一个完整的 transition)
                start_index = int(np_random.randint(0, total_steps_in_episode - 1))
                end_index = min(start_index + length, total_steps_in_episode)
                
                # 执行切片
                batch_to_yield = {
                    k: v[start_index : end_index].copy()
                    for k, v in episode_copy.items()
                    if "log_" not in k # 忽略日志数据
                }
                
                # **关键**: 将此 chunk 的第一个数据点标记为 'is_first'
                # 这会重置 RSSM 的隐藏状态
                #is_first=True 保证了模型在处理一个新序列（或被拼接上的序列）的开头时，
                # 会清空（重置）它的循环记忆（hidden state）
                if "is_first" in batch_to_yield:
                    batch_to_yield["is_first"][0] = True
                
            else:
                # --- 情况B: 我们在追加数据以填满批次 ---
                # (因为上一个 chunk 不够长)
                
                steps_needed = length - current_batch_size
                
                # 从新 episode 的*开头*开始取
                start_index = 0
                end_index = min(start_index + steps_needed, total_steps_in_episode)
                
                # 将新 chunk 追加到 'batch_to_yield'
                batch_to_yield = {
                    k: np.append(
                        batch_to_yield[k], # 已有的数据
                        v[start_index : end_index].copy(), # 新数据
                        axis=0
                    )
                    for k, v in episode_copy.items()
                    if "log_" not in k
                }
                
                # **关键**: 标记这个新 chunk 的起始点为 'is_first'
                if "is_first" in batch_to_yield:
                    batch_to_yield["is_first"][current_batch_size] = True

            # --- 2d. 更新批次大小并清理 ---
            current_batch_size = len(next(iter(batch_to_yield.values())))
            del episode_copy # 释放副本内存
        
        # 3. 产生（yield）这个填满的批次
        # 'while True' 循环将立即开始构建下一个批次
        yield batch_to_yield


#MCResize,MNormalize,MyToTensor
"""
RandomGenerator的辅助函数
流水线专门用于将 MineDojo 环境的原始观测图像转换为 Multimodal U-Net 模型所需要的输入格式
"""
MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
MC_NORMALIZER = Normalize(mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)

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

#确保输入图像符合 U-Net 的输入尺寸要求（即 224x224）
def MCResize(image, label, target_size=(256, 160)):
    image = resized_if_need(image, target_size)
    label = resized_if_need(label, target_size)
    
    return image, label

#将 NumPy 数组转换为 PyTorch Tensor，并调整通道顺序
def MyToTensor(image, label):
    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(label).permute(2, 0, 1)

#将像素值标准化，以便模型更容易收敛或推理
def MNormalize(image, label):
    return normalize_numpy(image / 255.0, MC_IMAGE_MEAN, MC_IMAGE_STD), label

"""
concentration_reward的辅助函数
"""
class ThresholdBuffer:
    """
    Welford's Online Algorithm（Welford 在线算法）
    """
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



"""
模拟与收集replay buffer的辅助函数
"""
def convert(value, precision=32):
    '''
    convert numpy array to specified precision
    '''
    
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
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))

def calculate_accumulated_reward(rewards, intrinsics, gamma):
    if len(rewards) == 0:
        return 0
    
    rewards = np.array(rewards)
    intrinsics = np.array(intrinsics)
    gammas = np.power(gamma, np.arange(len(rewards)))
    discounted_rewards = rewards * gammas
    discounted_intrinsics = intrinsics * gammas
    total_reward = np.sum(discounted_rewards + discounted_intrinsics)
    gamma_sum = np.sum(gammas)
    
    return total_reward / gamma_sum 

def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            episode_copy = selective_deepcopy(episode)
            episode_copy = replace_none_with_zeros(episode_copy)
            np.savez_compressed(f1, **episode_copy)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
            del episode_copy
    return True

def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset

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
    执行数据收集的核心函数
    让 agent 和 envs 交互，把数据存入 cache，并保存到 directory
    
    在 expr.py 中被调用了两次
       Prefill（预填充）：用一个“随机智能体”来收集一些初始数据 
       Train（训练）：在主循环中，用我们正在学习的 LS_Imagine 智能体来收集数据
    """
    print(f"开始在 {directory} 中模拟...")
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
        information = [{}] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward, information = state
        
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            indices = [index for index in indices if information[index].get("real_done", True)]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results] 

            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()} # convert: convert numpy array to specified precision
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)

                current_step = 0
                if t["is_zoomed"] == True:
                    step_calculator.add(envs[index].id, current_step, t["score_on_zoomed"])

                # replace obs with done by initial state
                obs[index] = result

        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)

        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done, info = zip(*[p[:4] for p in results])

        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        information = list(info)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done

        # add to cache
        for tmp_index in range(len(results)):
            a, result, env = action[tmp_index], results[tmp_index], envs[tmp_index]

            o, r, d, info = result
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
            add_to_cache(cache, env.id, transition)

            length = len(cache[env.id]["reward"]) 
            current_step = length - 1
            if transition["is_zoomed"] == True and not d:
                step_calculator.add(env.id, current_step, transition["score_on_zoomed"])

            tmp_list = step_calculator.get_and_remove_less_than(env.id, current_step, transition["score"])
            if len(tmp_list) > 0:
                for ss in tmp_list:
                    cache[env.id]["jumping_steps"][ss] = current_step - ss
                    cache[env.id]["accumulated_reward"][ss] = calculate_accumulated_reward(cache[env.id]["reward"][ss+1:current_step], cache[env.id]["intrinsic"][ss+1:current_step], gamma)
                    cache[env.id]["is_calculated"][ss] = True

            if step_calculator.count_data_pairs(env.id) == 0:
                information[tmp_index]['real_done'] = True

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                if (not is_eval) and (not information[i].get("real_done", False)):
                    continue

                save_episodes(directory, {envs[i].id: cache[envs[i].id]})

                step_calculator.remove_all(envs[i].id)
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"])[0:max_steps+1].sum())
                suc = 1 if any(np.array(cache[envs[i].id]["success"])[:max_steps+1]) else 0
                first_success_step = min(cache[envs[i].id]["first_success_step"][-1], max_steps)
                video = cache[envs[i].id]["image"]
                
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.scalar(f"train_success", suc)
                    logger.scalar(f"train_first_success_step", first_success_step)
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_success = []
                        eval_first_success_step = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)
                    eval_success.append(suc)
                    eval_first_success_step.append(first_success_step)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    success_rate = sum(eval_success) / len(eval_success)
                    if len(eval_first_success_step) > 0:
                        first_success_step = sum(eval_first_success_step) / len(eval_first_success_step)
                    else:
                        first_success_step = max_steps

                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.scalar(f"eval_success", success_rate)
                        logger.scalar(f"eval_first_success_step", first_success_step)
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward, information)