from gym import Wrapper
import os
from PIL import Image
import random
import cv2
from datetime import datetime
from typing import Dict
import numpy as np


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


class ClipWrapper(Wrapper):
    