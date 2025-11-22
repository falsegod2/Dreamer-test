from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod


class RewardWrapper(Wrapper, ABC):
    def __init__(self, env, item_rewards: Dict[str, Dict[str, int]] = dict()):
        super().__init__(env)
        self.wrapper_name = "RewardWrapper"
        self.item_rewards = item_rewards
        self.last_inventory = None
        self.reward_count = None 

    def reset(self):
        self.last_inventory = {item: 0 for item in self.item_rewards}
        self.reward_count = {item: 0 for item in self.item_rewards}

        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        for item in self.item_rewards:
            curr_inv = self._get_item_count(obs, item)
            item_diff = curr_inv - self.last_inventory[item]
            if "quantity" in self.item_rewards[item]:
                item_diff = min(item_diff, self.item_rewards[item]["quantity"] - self.reward_count[item])
            item_reward = self.item_rewards[item]["reward"] if "reward" in self.item_rewards[item] else 1
            reward += item_diff * item_reward
            self.reward_count[item] += item_diff
            self.last_inventory[item] = curr_inv

        return obs, reward, done, info

    """
    @abstractstaticmethod 的完整含义是：
    “我（RewardWrapper）在此立下一个合同：所有想使用我的子类，
    必须提供一个名为 _get_Ditem_count 的静态工具函数，这个函数必须接受 obs 和 item 两个参数。”

    @abstractstaticmethod 这个“单行”装饰器是 Python 2 和早期 Python 3 的写法，现在已被弃用
    # 现代 Python 的写法：
    from abc import ABC, abstractmethod
        @staticmethod
        @abstractmethod
    """
    @abstractstaticmethod
    def _get_item_count(obs, item):
        raise NotImplementedError()
