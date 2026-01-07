import torch
from torch import nn
import numpy as np
import argparse
import ruamel.yaml as yaml
import pathlib
import sys
from datetime import datetime
import os
import time

from parallel import Parallel, Damy
from torch import distributions as torchd
import functools

# 自定义模块
import tool_own
import model_own
import reward_own
import exploration as expl

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()

class LS_Imagine(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, env):
        super(LS_Imagine, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tool_own.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tool_own.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tool_own.Once()
        self._should_reset = tool_own.Every(config.reset_every)
        self._should_expl = tool_own.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        #CORE1
        self.env = env

        self._wm = model_own.WorldModel(obs_space, act_space, self._step, config, self.env)
        self._task_behavior = model_own.ImagBehavior(config, self._wm)



        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        # start = (post, post_zoomed)

        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        '''
        intrinsic = lambda f, s, a: self._wm.heads["intrinsic"](
            self._wm.dynamics.get_feat(s)
        ).mode() 
        '''
        is_end = lambda s: self._wm.heads["end"](
            self._wm.dynamics.get_feat(s)
        ).mean

        metrics.update(self._task_behavior._train(post, reward, is_end)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def main(config):
    """
    0.设置云盘保存时限 (由参数控制)
    """
    last_backup_time = time.time() # 初始化时间
    
    if config.enable_cloud_backup:
        print(f"云盘备份已开启。策略: 每 {config.cloud_backup_interval/3600:.2f} 小时全量覆盖一次。")
    else:
        print("云盘备份已关闭 (本地模式)。")
    
    """
    1. 全局设置与日志初始化 (Global Setup & Logging)
    """
    tool_own.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tool_own.enable_deterministic_run()

    """--- 配置路径结构 ---"""
    '''
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir = logdir / config.task / f'seed_{config.seed}'
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
    '''
    """--- 配置路径结构 ---"""
    # [修改开始] 支持指定恢复路径
    if hasattr(config, 'resume_path') and config.resume_path:
        # 如果指定了 resume_path，直接使用该路径，不再生成新时间戳
        logdir = pathlib.Path(config.resume_path).expanduser()
        print(f" 检测到恢复模式，将直接使用现有目录: {logdir}")
    else:
        # 否则，按照原逻辑生成新时间戳
        logdir = pathlib.Path(config.logdir).expanduser()
        logdir = logdir / config.task / f'seed_{config.seed}'
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = logdir / timestamp
    # [修改结束]
    
    config.logdir = logdir

    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    
    """ --- 归一化步数配置 (Step Normalization) ---"""
    #使用了 //=（向下取整除法赋值），
    #目的是将配置文件中的总帧数除以动作重复次数，从而转换为智能体在总帧数内需要决策的次数
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat # 多少帧/次评估
    config.log_every //= config.action_repeat 
    config.time_limit //= config.action_repeat

    """--- 创建目录与Logger ---"""
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    
    step = tool_own.count_steps(config.traindir)
    logger = tool_own.Logger(config, logdir, config.action_repeat * step)
    print(f"日志将保存在: {logdir}")

    """
    2. 环境与任务配置 (Environment & Task Specs)
    """    

    """ --- 加载离线数据路径 (如果有) ---"""
    directory = config.traindir
    train_eps = tool_own.load_episodes(directory, limit=config.dataset_size)
    #train_eps 是驻留在内存中的巨大字典，包含了图像、动作、奖励等所有数据

    directory = config.evaldir
    eval_eps = tool_own.load_episodes(directory, limit=1)
    """
    情景一：离线训练模式 (Offline Training)
        如果指定了 offline_traindir，并且 load_episodes 的 limit=10：
        现象：Agent 只会加载磁盘中最新的 1 个 Episode（假设该 Episode 长度大于 10 步）。
        能否运行？：能运行。
        因为 1 个 Episode（通常 1000 步）大于 batch_length (32 步)，make_dataset 和 sample_episodes 可以正常从这 1 个 Episode 中采样切片，不会报错。
        训练后果：严重的过拟合 (Severe Overfitting)。
        世界模型和 Agent 会反复在这同一个 Episode 上训练成千上万次。
        模型会迅速“背诵”下这 1000 步的所有像素变化，但在任何稍微不同的场景下（泛化能力）都会彻底失效。
        LS-Imagine 需要学习长短期跳转，单个 Episode 无法提供足够的多样性来学习这种复杂的动态。
    情景二：在线训练模式 - 恢复训练 (Online Resume)
        如果你是在一个已经有大量 .npz 文件的目录下继续训练（Resume），并且 load_episodes 的 limit=10：
        现象：“部分失忆” (Partial Amnesia)。
        程序启动时，只“回忆”起上次运行的最后 1 个 Episode。之前几天跑出来的几千个 Episodes 都会被忽略，没有加载进内存。
        后续过程：
        程序进入 simulate 环节。注意，此时 simulate 使用的是配置文件中未修改的 dataset_size (1,000,000)。
        Agent 会基于这仅有的 1 个 Episode 开始新的交互。
        随着交互进行，新的数据会不断填入 Buffer。因为 dataset_size 很大，Buffer 会正常增长，直到存满 1,000,000 步。
        训练后果：冷启动问题。
        虽然最终 Buffer 会恢复正常，但训练初期由于丢失了历史经验，Agent 的表现可能会突然大幅下降，需要重新花费大量时间去探索和收集数据，才能恢复到之前的性能水平。
    情景三：在线训练模式 - 全新开始 (Fresh Start)
        如果你在一个空目录下开始新训练：
        现象：没有任何影响。
        原因：目录下没有文件，load_episodes 什么也读不到（无论 limit 是 10 还是 100 万）。程序会进入 prefill 阶段（随机探索），此时使用的是配置文件中的 dataset_size，一切正常运行。
    """

    """ --- 获取任务详细规格 (MineDojo Specs) --- """
    suite, task = config.task.split("_", 1)
    task_kwargs = dict(target_item=config.target_item)
    task_id, task_specs, sim_specs = tool_own.get_specs(task, **task_kwargs)

    """ --- 将任务参数注入Config --- """
    config.episode_max_steps = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['max_steps'] = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['gaussian_reward_weight'] = config.gaussian_reward_weight
    task_specs['concentration_specs']['gaussian_sigma_weight'] = config.gaussian_sigma_weight

    """ 确定目标物体 (用于Clip Score) """
    if 'all' in task_specs['success_specs']:
        task_specs['clip_specs']['target_object'] = task_specs['success_specs']['all']['item']['type']
    else:
        task_specs['clip_specs']['target_object'] = task_specs['success_specs']['any']['item']['type']

    """ --- 实例化环境 (Parallel or Dummy) ---"""
    make = lambda mode, id: tool_own.make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    # --- 动作空间配置 ---
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    #step_calculator = tool_own.ScoreStorage(max_steps=config.episode_max_steps)
    state = None
    """
    3. 预填充 Replay Buffer (Prefill Phase)
    """
    if not config.resume_path:
        prefill = max(0, config.prefill - tool_own.count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")

        # --- 定义随机策略 (Random Actor) ---
        if hasattr(acts, "discrete"):
            random_actor = tool_own.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ), 1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        # --- 执行随机探索 ---
        state = tool_own.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            #step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=prefill,
            is_training=False,
        )

        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")


    """
    4. Agent 初始化 (Agent Initialization)
    """
    print("Simulate agent.")
    train_dataset = tool_own.make_dataset(train_eps, config)
    eval_dataset = tool_own.make_dataset(eval_eps, config)
    
    # 初始化 LS-Imagine 模型
    agent = LS_Imagine(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        train_envs,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)

    # --- 加载 Checkpoint (Resume) ---
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tool_own.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    """
    5. 主训练循环 (Main Training Loop)
    """
    # 确保评估在最后一步也会执行一次
    while agent._step < config.steps + config.eval_every:
        logger.write()

        # --- Phase A: 评估 (Evaluation) ---
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            
            tool_own.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                #step_calculator,
                config.episode_max_steps,
                config.discount,
                is_eval=True,
                episodes=config.eval_episode_num,
                is_training=False,
            )
            
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        # --- Phase B: 训练 (Training) ---
        print("Start training.")
        state = tool_own.simulate(
            agent,  # LS_Imagine call
            train_envs,
            train_eps,
            config.traindir,
            logger,
            #step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            is_training=True,
        )

        # --- Phase C: 保存 Checkpoint (Saving) ---
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tool_own.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")  

        """
        按时间间隔执行云端备份 (仅当开关开启时)
        """
        if config.enable_cloud_backup:
            current_time = time.time()
            # 使用 config.cloud_backup_interval 代替原来的 BACKUP_INTERVAL
            if current_time - last_backup_time >= config.cloud_backup_interval:
                print(f"已过去 {(current_time - last_backup_time)/3600:.2f} 小时，开始执行云端备份...")
                
                # 调用 tool_own.py 里的备份函数
                try:
                    tool_own.save_colab(config, logdir)
                    last_backup_time = current_time
                except Exception as e:
                    print(f"警告：云端备份失败，但这不会影响训练主进程。错误信息: {e}")
            else:
                remaining_time = config.cloud_backup_interval - (current_time - last_backup_time)
                print(f"距离下次云端备份还有: {remaining_time/60:.1f} 分钟") # 觉得吵可以注释掉
    
    """
    6. 清理与结束 (Cleanup)
    """
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    logger.finish()

if __name__ == "__main__":
    """
    配置的优先级顺序：
    1.（最低）configs.yaml 里的 defaults 块：基础设置。
    2.（中等）configs.yaml 里的指定块(比如minedojo)：用 --configs 指定，用来覆盖 defaults。
    3.（最高）命令行的单独参数：用来覆盖前面所有的设置。
    """

    #（1）找出用户指定的“配置方案”（中等优先级）/即minedojo
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    """
    args：--configs 指定的方案，也就是minedojo 。
    remaining：包含命令行中除了--configs外所有单独参数（任务以及保存路径），这些参数稍后会再次处理。
    """

    #（2）加载并合并“基础”和“中等”配置
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent/ "configs.yaml").read_text())
    """
    加载 configs.yaml 文件到一个字典 configs 中
    pathlib.Path(sys.argv[0]).parent:获取当前正在运行的脚本（expr.py）所在的目录
    / "configs.yaml"：将路径指向同目录下的 configs.yaml 文件
    """
    #用于深度合并基础base与中等update两个字典
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name]) 
    #defaults = {}：字典用来存放“基础”和“中等”配置合并后的配置

    #（3）创建“最终”解析器（准备最高优先级）
    parser = argparse.ArgumentParser()
    """
    代码有了一份完整的配置字典，需要创建一个新的、全功能的解析器，
    来处理步骤 1 中被搁置的那些参数（remaining）
    """
    for key, value in sorted(defaults.items(), key = lambda x: x[0]):
            arg_type = tool_own.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    #(4)应用“最高优先级”配置并运行
    main(parser.parse_args(remaining))

