from gym import Wrapper
import os
from PIL import Image
import random
import cv2
from datetime import datetime
from typing import Dict
import numpy as np
import torch
import gym.spaces as spaces
import copy
from collections import OrderedDict
from minedojo.sim.wrappers.fast_reset import FastResetWrapper

def name_match(target_name, obs_name):
    return target_name.replace(" ", "_") == obs_name.replace(" ", "_")

class MinedojoScreenshotWrapper(Wrapper):
    """
    专门处理 MineDojo 环境中的截图
    """
    
    def __init__(self, env, log_dir, reset_flag=False, step_flag=False, save_freq=1, save_dir='', HUD=False, **kwargs):
        super().__init__(env)
        self.wrapper_name = "ScreenshotWrapper"
        self.reset_flag = reset_flag
        self.step_flag = step_flag
        self.ff = False
        self.save_freq = save_freq
        self.freq = 0
        self.episode = 0
        self.steps = 0
        self.HUD = HUD
        self.mask = cv2.imread('envs/tasks/base/HUD_mask.png', cv2.IMREAD_GRAYSCALE)
        if save_dir:
            self.log_dir = save_dir
        else:
            self.log_dir = os.path.join(log_dir, "screenshot")

    def reset(self):
        obs = super().reset()

        self.ff = True
        self.freq = 0
        self.episode += 1
        self.steps = 0

        if self.reset_flag and self.ff:
            self.screenshot(obs, "reset", self.episode, self.steps)
            self.ff = False
        
        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.freq += 1
        self.steps += 1

        if self.step_flag and random.random() < (1.0 / self.save_freq):
            self.screenshot(obs, "step", self.episode, self.steps)

        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        return obs, reward, done, info
    

    def screenshot(self, obs, type, episode_num, step_num):
        full_path = os.path.join(self.log_dir, f"episode_{episode_num}", "image")

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        img = self._get_curr_frame(obs)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] 
        filename = f"{step_num}.png"
        filepath = os.path.join(full_path, filename)
        
        img.save(filepath)

    def remove_HUD(self, obs):
        img = self._get_curr_frame(obs)
        result = cv2.inpaint(img, self.mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        result = result.transpose((2, 0, 1))

        return result
    
    #静态方法（有 @staticmethod）是一个与“汽车”这个概念相关，但不需要一辆“具体汽车”就能执行的工具函数,它没有 self 参数
    @staticmethod
    def _get_curr_frame(obs):
        """
        从 MineDojo 环境的观测（obs）字典中提取并格式化图像
        """
        curr_frame = obs["rgb"].copy()
        curr_frame = curr_frame.transpose((1, 2, 0))
        return curr_frame # shape: (160, 256, 3)
    
    @staticmethod
    def get_resolution():
        """
        直接返回一个元组（tuple）， (160, 256)，代表这个包装器处理的图像的标准分辨率
        """
        return (160, 256)
    
class MinedojoRewardWrapper(Wrapper):
    """
    监控智能体在环境中的“物品栏”（Inventory），并在智能体获取到指定物品时，给予它一个额外的、自定义的奖励。
    """

    def __init__(self, env, item_rewards: Dict[str, Dict[str, int]] = dict()):
        super().__init__(env)
        self.wrapper_name = "RewardWrapper"
        self.item_rewards = item_rewards
        """
        item_rewards是一个字典，定义了要跟踪的物品和奖励规则。
        例如：
        reward_specs:   
            item_rewards: 
            log:
                reward: 1
        """
        self.last_inventory = None
        self.reward_count = None 

    def reset(self):
        """
        在每个新 episode (回合) 开始时，重置所有内部状态
            last_inventory: 物品栏内的所有我们关心的物品数量
            reward_count: 累计的物品栏奖励
        """
        self.last_inventory = {item: 0 for item in self.item_rewards}
        self.reward_count = {item: 0 for item in self.item_rewards}

        return super().reset()

    def step(self, action):
        # 1. 先让原始环境执行一步
        obs, reward, done, info = super().step(action)
        # 2. 遍历所有我们关心的物品
        for item in self.item_rewards:
            #获取当前物品数量
            curr_inv = self._get_item_count(obs, item)
            #计算与上一步遍历的差值
            item_diff = curr_inv - self.last_inventory[item]
            #检查是否有奖励数量限制（在我的方法中没有限制）
            if "quantity" in self.item_rewards[item]:
                item_diff = min(item_diff, self.item_rewards[item]["quantity"] - self.reward_count[item])

            #获取这个物品对应的奖励值
            item_reward = self.item_rewards[item]["reward"] if "reward" in self.item_rewards[item] else 1
            #把新获得的物品奖励，添加到原始环境的奖励上
            reward += item_diff * item_reward

            #更新状态
            self.reward_count[item] += item_diff
            self.last_inventory[item] = curr_inv

        return obs, reward, done, info
    
    @staticmethod
    def _get_item_count(obs, item):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) if name_match(item, name))
    
class MinedojoTerminalWrapper(Wrapper):
    def __init__(self, env, max_steps: int = 500, on_death=True, all: Dict = dict(), any: Dict = dict(), stagger_max_steps=False):
        super().__init__(env)
        self.wrapper_name = "TerminalWrapper"
        self.max_steps = max_steps
        self.on_death = on_death
        self.all_conditions = all
        self.any_conditions = any
        self.t = 0
        self.curr_max_steps = self.max_steps
        self.stagger_max_steps = stagger_max_steps

    def reset(self):
        #将步数计数器 self.t 归零
        self.t = 0
        if self.stagger_max_steps:
            self.curr_max_steps = np.random.randint((self.max_steps*3)//4, self.max_steps+1)
        else:
            self.curr_max_steps = self.max_steps

        return super().reset()
    
    def step(self, action):
        # 1. 先让原始环境执行一步
        obs, reward, done, info = super().step(action)
        self.t += 1

        #用来区分“真正的结束”（如死亡、成功）和“人为的结束”（如超时）
        info['real_done'] = False

        #超时检查
        if self.t >= self.curr_max_steps:
            done = True
            if self.t > self.curr_max_steps:
                info["success"] = False

        #死亡检查
        if self.on_death:
            if self._check_condition("death", {}, obs):
                done = True
                info['real_done'] = True


        if len(self.all_conditions) > 0:
            if all(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.all_conditions.items()
            ):
                done = True
                info['real_done'] = True

        if len(self.any_conditions) > 0:
            if any(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.any_conditions.items()
            ):
                done = True
                info['real_done'] = True

        return obs, reward, done, info

    def _check_condition(self, condition_type, condition_info, obs):
        if condition_type == "item":
            return self._check_item_condition(condition_info, obs)
        elif condition_type == "blocks":
            return self._check_blocks_condition(condition_info, obs)
        elif condition_type == "death":
            return self._check_death_condition(condition_info, obs)
        else:
            raise NotImplementedError("{} terminal condition not implemented".format(condition_type))
        
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False

    @staticmethod
    def _check_death_condition(condition_info, obs):
        return obs["life_stats"]["life"].item() == 0
    
class MinedojoSuccessWrapper(Wrapper):
    def __init__(self, env, terminal: bool = True, reward: int = 0, all: Dict = dict(), any: Dict = dict(), max_steps: int = 500):
        super().__init__(env)
        self.wrapper_name = "SuccessWrapper"
        self.terminal = terminal
        self.all_conditions = all
        self.any_conditions = any
        self.success_reward = reward
        self._first_success = True
        self._max_steps = max_steps
        self.steps = 0
        self.first_success_step = 0

    def reset(self):
        self._first_success = True
        self.steps = 0
        self.first_success_step = self._max_steps
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["success"] = info.pop("success", False)

        if len(self.all_conditions) > 0:
            info["success"] = info["success"] or all(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.all_conditions.items()
            )

        if len(self.any_conditions) > 0:
            info["success"] = info["success"] or any(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.any_conditions.items()
            )

        if self.terminal:
            done = done or info["success"]
        else:
            done = False

        if info["success"] and self._first_success:
            self._first_success = False
            reward += self.success_reward
            self.first_success_step = min(self._max_steps, self.steps)

        info["first_success_step"] = self.first_success_step
        info["success"] = info["success"] or (not self._first_success)

        self.steps += 1

        return obs, reward, done, info

    def _check_condition(self, condition_type, condition_info, obs):
        if condition_type == "item":
            return self._check_item_condition(condition_info, obs)
        elif condition_type == "blocks":
            return self._check_blocks_condition(condition_info, obs)
        else:
            raise NotImplementedError("{} terminal condition not implemented".format(condition_type))
    
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False



class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, dense_reward=.01, smoothing=1, target_object='log', **kwargs):
        super().__init__(env)
        self.clip = clip # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.dense_reward = dense_reward
        self.smoothing = smoothing

        # --- 任务流 ---
        self.task_prompts = prompts
        self.task_buffer = None
        self.task_clip_state = (None, None) # (past_frames, text_feats)
        self.task_last_score = 0

        # --- 探索流 ---
        self.expl_prompts = [f"Explore the widest possible area to find {target_object}"]
        self.expl_buffer = None
        self.expl_clip_state = (None, None) # (past_frames, text_feats)
        self.expl_last_score = 0
        
    def reset(self, **kwargs):
        # 缓存文本特征text_feats ，重置帧缓冲区past_frames 
        self.task_clip_state = None, self.task_clip_state[1]
        self.expl_clip_state = None, self.expl_clip_state[1]

        self.task_buffer = None
        self.expl_buffer = None
        self.task_last_score = 0
        self.expl_last_score = 0

        obs = self.env.reset(**kwargs)
        obs['intrinsic'] = 0.0
        obs['score'] = 0.0

        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # --- 1. 处理任务奖励流 (注入到 obs) ---
        if len(self.task_prompts) > 0:
            logits, self.task_clip_state = self.clip.get_logits(obs, self.task_prompts, self.task_clip_state)
            logits = logits.detach().cpu()

            self.task_buffer = self._insert_buffer(self.task_buffer, logits[:1])
            score = self._get_score()

            #agent只会因为“比上一步做得更好”而获得内在奖励
            if score > self.task_last_score:
                obs['intrinsic'] = self.dense_reward * score
                self.task_last_score = score
            else:
                obs['intrinsic'] = 0.0

            obs['score'] = self.dense_reward * score

        else:
            obs['intrinsic'] = 0.0
            obs['score'] = 0.0

        # --- 2. 处理探索奖励流 (注入到 info) ---
        if len(self.expl_prompts) > 0:
            logits, self.expl_clip_state = self.clip.get_logits(obs, self.expl_prompts, self.expl_clip_state)
            logits = logits.detach().cpu()

            self.expl_buffer = self._insert_buffer(self.expl_buffer, logits[:1])
            expl_score = self._get_expl_score()

            if expl_score > self.expl_last_score:
                info['expl_intrinsic'] = self.dense_reward * expl_score
                self.expl_last_score = expl_score
            else:
                info['expl_intrinsic'] = 0.0

        else:
            info['expl_intrinsic'] = 0.0

        info["clip_score"] = obs['intrinsic']
        info["clip_last_score"] = self.task_last_score
        info["clip_dense_reward"] = self.dense_reward    

        return obs, reward, done, info 

    """
    _get_score 和 _get_expl_score 计算的是buffer内分数的平均值，这使得奖励信号更平滑，不易突变。
    并使用了一个硬编码的 sigmoid 函数 (1 / (1 + th.exp(1.2 * (21.8 - score))))。
    这会将 logits 的平均值（原始值可能在 20-22 附近）“压扁”到一个 0 到 1 之间的概率值。
        21.8是及格线
        1.2是敏感度
    """
    def _get_score(self):
        score = torch.mean(self.task_buffer)
        return (1 / (1 + torch.exp(1.2 * (21.8 - score)))).item()
    
    def _get_expl_score(self):
        score = torch.mean(self.expl_buffer)
        return (1 / (1 + torch.exp(1.2 * (21.8 - score)))).item()


    def _insert_buffer(self, task_buffer, logits):
        """
        将 logits 插入固定大小的 FIFO 缓冲区。
        不使用原始的 logits，而是维护一个大小为 self.smoothing 的滑动窗口（FIFO 缓冲区）。
        """
        if task_buffer is None:
            task_buffer = logits.unsqueeze(0)
        elif task_buffer.shape[0] < self.smoothing:
            task_buffer = torch.cat([task_buffer, logits.unsqueeze(0)], dim=0)
        else:
            task_buffer = torch.cat([task_buffer[1:], logits.unsqueeze(0)], dim=0)
        return task_buffer


class ConcentrationWrapper(Wrapper):
    def __init__(self, env, concentration, prompts=None, dense_reward=0.01, mineclip_dense_reward=0.01, max_steps=1000, gaussian_reward_weight=1.0, **kwargs):
        super().__init__(env)
        self.concentration = concentration # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.gaussian_reward_weight = gaussian_reward_weight

        self.episode = 0
        self.steps = 0
        self.last_score = 0

        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.episode += 1
        self.steps = 0

        self.last_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0
        obs = self.env.reset(**kwargs)

        score, zoom_in_prob, check_threshold = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
        zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
        if is_check:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
        else:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = 0.0, 0.0, 0.0, False, False

        obs['is_zoomed'] = is_zoomed
        obs['jump'] = jump
        obs['jumping_steps'] = self.max_steps
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = False
        obs['reward_on_zoomed'] = 0.0
        obs['intrinsic_on_zoomed'] = 0.0
        obs['score_on_zoomed'] = 0.0
        obs['zoomed_image'] = zoomed_image

        if score > self.last_score:
            obs['intrinsic'] += self.dense_reward * score * self.gaussian_reward_weight
            self.last_score = score

        obs['score'] += self.dense_reward * score

        if is_zoomed:
            if gaussian_on_zoomed > self.last_score and gaussian_on_zoomed > self.last_zoom_in_gaussian_score:
                obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed * self.gaussian_reward_weight
                self.last_zoom_in_gaussian_score = gaussian_on_zoomed

            obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed

            if mineclip_on_zoomed > self.last_zoom_in_mineclip_score:
                obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                self.last_zoom_in_mineclip_score = mineclip_on_zoomed

            obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
           
        obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
        if is_zoomed:
            obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
        else:
            obs['heatmap_on_zoomed'] = obs['heatmap']
        
        return obs
    
    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            score, zoom_in_prob, check_threshold = self.concentration.get_reward(obs, self.prompt, self.episode, self.steps)
            zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
            if is_check:
                mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
            else:
                mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = 0.0, 0.0, 0.0, False, False
            
            obs['is_zoomed'] = is_zoomed
            obs['jump'] = jump
            obs['jumping_steps'] = self.max_steps
            obs['accumulated_reward'] = 0.0
            obs['is_calculated'] = False
            obs['reward_on_zoomed'] = reward
            obs['intrinsic_on_zoomed'] = 0.0
            obs['score_on_zoomed'] = 0.0
            obs['zoomed_image'] = zoomed_image

            if score > self.last_score:
                obs['intrinsic'] += self.dense_reward * score * self.gaussian_reward_weight
                self.last_score = score

            obs['score'] += self.dense_reward * score

            if is_zoomed:
                if gaussian_on_zoomed > self.last_score and gaussian_on_zoomed > self.last_zoom_in_gaussian_score:
                    obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed * self.gaussian_reward_weight
                    self.last_zoom_in_gaussian_score = gaussian_on_zoomed

                obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed
                
                if mineclip_on_zoomed > info["clip_last_score"] and mineclip_on_zoomed > self.last_zoom_in_mineclip_score:
                    self.mineclip_dense_reward = info["clip_dense_reward"]
                    obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                    self.last_zoom_in_mineclip_score = mineclip_on_zoomed

                obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed

            obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
            if is_zoomed:
                obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
            else:
                obs['heatmap_on_zoomed'] = obs['heatmap']
                
        return obs, reward, done, info
    

BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.])),
    "turn_down": dict(camera=np.array([10.0, 0.])),
    "turn_left": dict(camera=np.array([0., -10.0])),
    "turn_right": dict(camera=np.array([0., 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    'camera': np.array([0., 0.]), 
    'smelt': 'none', 
    'craft': 'none', 
    'craft_with_table': 'none', 
    'forward': np.array(0), 
    'back': np.array(0), 
    'left': np.array(0), 
    'right': np.array(0), 
    'jump': np.array(0), 
    'sneak': np.array(0), 
    'sprint': np.array(0), 
    'use': np.array(0), 
    'attack': np.array(0), 
    'drop': 0, 
    'swap_slot': OrderedDict([('source_slot', 0), ('target_slot', 0)]), 
    'pickItem': 0, 
    'hotbar.1': 0, 
    'hotbar.2': 0, 
    'hotbar.3': 0, 
    'hotbar.4': 0, 
    'hotbar.5': 0, 
    'hotbar.6': 0, 
    'hotbar.7': 0, 
    'hotbar.8': 0, 
    'hotbar.9': 0,
}

class MinedojoLSImagineWrapper(Wrapper):
    def __init__(self, env, repeat=1, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70)):
        super().__init__(env)
        self.wrapper_name = "LSImagineWrapper"

        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())

        self.observation_space = spaces.Dict(
            {
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                'heatmap': spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
                'jump': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_calculated': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'reward_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'intrinsic': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'intrinsic_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'score': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'score_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'jumping_steps': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'accumulated_reward': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True
        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        obs = self.env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def step(self, action):
        action = copy.deepcopy(self._action_values[action])
        action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", info["real_done"]))
        obs = self._obs(obs)

        assert "pov" not in obs, list(obs.keys())

        return obs, reward, done, info

    def _obs(self, obs):
        image = obs['rgb'] # 3 * H * W
        image = image.transpose(1, 2, 0).astype(np.uint8) # H * W * 3
        image = cv2.resize(image, (64, 64)) # 64 * 64 * 3

        if 'zoomed_image' in obs:
            zoomed_image = obs['zoomed_image'] # H * W * 3
            zoomed_image = zoomed_image.astype(np.uint8) # H * W * 3
            zoomed_image = cv2.resize(zoomed_image, (64, 64)) # 64 * 64 * 3
        else:
            zoomed_image = np.zeros_like(image)

        heatmap = cv2.resize(obs['heatmap'] if 'heatmap' in obs else np.zeros((64, 64, 1)), (64, 64))
        heatmap_on_zoomed = cv2.resize(obs['heatmap_on_zoomed'] if 'heatmap_on_zoomed' in obs else np.zeros((64, 64, 1)), (64, 64))
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_on_zoomed = np.clip(heatmap_on_zoomed * 255, 0, 255).astype(np.uint8)

        obs = {
            'image': image,
            'heatmap': heatmap,
            'jump': obs['jump'] if 'jump' in obs else False,
            'is_zoomed': obs['is_zoomed'] if 'is_zoomed' in obs else False,
            'is_calculated': obs['is_calculated'] if 'is_calculated' in obs else False,
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
            'reward_on_zoomed': obs['reward_on_zoomed'] if 'reward_on_zoomed' in obs else 0.0,
            'intrinsic': obs['intrinsic'] if 'intrinsic' in obs else 0.0,
            'intrinsic_on_zoomed': obs['intrinsic_on_zoomed'] if 'intrinsic_on_zoomed' in obs else 0.0,
            'score': obs['score'] if 'score' in obs else 0.0,
            'score_on_zoomed': obs['score_on_zoomed'] if 'score_on_zoomed' in obs else 0.0,
            'jumping_steps': obs['jumping_steps'] if 'jumping_steps' in obs else 1000.0,
            'accumulated_reward': obs['accumulated_reward'] if 'accumulated_reward' in obs else 1000.0,
        }

        if obs["is_zoomed"]:
            obs["zoomed_image"] = zoomed_image
            obs["heatmap_on_zoomed"] = heatmap_on_zoomed
        else:
            obs["zoomed_image"] = None
            obs["heatmap_on_zoomed"] = None

        
        for key, value in obs.items():
            if key in self.observation_space:
                space = self.observation_space[key]
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                assert (key, value, value.dtype, value.shape, space)
        return obs

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action


    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions


# Fast reset wrapper saves time but doesn't replace blocks
# Occasionally doing a hard reset should prevent state shift
class MinedojoSemifastResetWrapper(FastResetWrapper):

    def __init__(self, *args, reset_freq=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.reset_count = 0

    def reset(self):
        if self.reset_count < self.reset_freq:
            self.reset_count += 1
            return super().reset()
        else:
            self.reset_count = 0
            return self.env.reset()
