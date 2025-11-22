import torch
from omegaconf import OmegaConf
from mineclip import MineCLIP
from typing import List, Tuple, Dict, Optional


class MinedojoClipReward():
    """
    接收环境观察 (obs) 和文本提示 (prompts)，并计算一个代表“视频与文本匹配程度”的奖励值
    """
    def __init__(self, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
        #kwargs.pop若在kwargs中找不到第一个键“arch”，则会返回第二个参数“vit_base_p16_fz.v2.t2”，即默认值
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.resolution = self.get_resolution()
        self.device = kwargs.pop("device", "cuda")
        self.model = None
        
        self._load_mineclip(ckpt, kwargs)

    @staticmethod
    def _get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        return torch.from_numpy(curr_frame)

    @staticmethod
    def get_resolution():
        return (160, 256)
    
    def _load_mineclip(self, ckpt, config):
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = torch.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()
        print("MineCLIP 模型加载并设置到 eval() 模式。") 

    def get_reward(
            self, 
            obs: Dict,
            prompt: str,
            neg_prompts: List[str],
            state: Tuple[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
            """计算单个正面提示的奖励。"""
            
            all_prompts = [prompt] + neg_prompts
            logits, new_state = self.get_logits(obs, all_prompts, state)
            
            # logits 现在对应 [prompt, neg1, neg2, ...]
            # _get_reward_from_logits 自动处理这个
            reward = self._get_reward_from_logits(logits)

            return reward, new_state
    
    def get_rewards(
            self, 
            obs: Dict,
            prompts: List[str],
            neg_prompts: List[str],
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[List[float], Tuple[torch.Tensor, torch.Tensor]]:
            """计算多个正面提示的奖励 (优化版)。"""
            
            all_prompts = prompts + neg_prompts
            logits, new_state = self.get_logits(obs, all_prompts, state)
            
            rewards = []
            num_pos = len(prompts)
            
            # 将负面 logits 只切片一次
            neg_logits = logits[num_pos:] 

            for i in range(num_pos):
                # 正面 logit
                pos_logit = logits[i:i+1]
                # 组合 [pos_i, neg1, neg2, ...]
                combined_logits = torch.cat((pos_logit, neg_logits))
                
                reward = self._get_reward_from_logits(combined_logits)
                rewards.append(reward)
                
            return rewards, new_state
    
    #核心代码
    def get_logits(
            self,
            obs: Dict,
            prompts: List[str],
            state: Tuple[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            计算视频观察和文本提示列表之间的 logits。
            """
            if self.model is None:
                raise RuntimeError("必须在使用 .get_logits() 之前调用 .load_model()")

            # 1. 解包状态
            past_frames, text_feats = state if state is not None else (None, None)

            # 2. 从 obs 提取帧 (委托给子类)
            curr_frame = self._get_curr_frame(obs)

            with torch.no_grad():
                # 3. 缓存或计算文本特征
                if text_feats is None:
                    text_feats = self._get_text_feats(prompts)

                # 4. 计算图像/视频特征 (和以前一样),先拼接新帧，到第六步再丢弃旧帧
                image_feats_seq = self._get_image_feats(curr_frame, past_frames)
                video_feats = self._get_video_feats(image_feats_seq)
                
                # 5. 计算 Logits
                logits = self.model.forward_reward_head(
                    video_feats.to(self.device), 
                    text_tokens=text_feats.to(self.device)
                )[0][0] # P

            # 6. 准备并返回新的状态
            # (image_feats_seq[0, 1:] 丢弃最旧的帧, 保留 15 帧)
            new_past_frames = image_feats_seq[0, 1:].cpu()
            new_state = (new_past_frames, text_feats.cpu())
            """
            text_feats:
                是一个缓存。prompts 通常在整个 episode 中是不变的。
                在第一步（state 为 None），get_logits 会调用 _get_text_feats 来计算文本特征，并将其存储在 state 中。
                在之后的所有步骤中，它都会重用这个缓存的 text_feats。
            past_frames:
                这是最关键的部分。它是一个 (1, 15, 512) 维度的张量，代表过去15帧的图像特征。
                它充当一个“滑动窗口缓冲区”。
            """
            
            return logits, new_state
    
    # -----------------------------------------------
    # 内部辅助方法 
    # -----------------------------------------------
    def _get_reward_from_logits(self, logits: torch.Tensor) -> float:
        """从 [pos, neg1, neg2, ...] 的 logits 计算奖励。"""
        probs = torch.softmax(logits, 0)
        # 奖励 = P(positive) - P(random_guess)
        reward = max(probs[0].item() - (1.0 / logits.shape[0]), 0)
        return reward
    
    def _get_image_feats(
        self,
        curr_frame: torch.Tensor,
        past_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将 1 个当前帧和 15 个过去帧合并为 16 帧特征序列。
        先拼接新帧，后减去最久的帧，达到滑动窗口的目的
        """
        
        # 确保当前帧有 (1, 1, 3, H, W) 的形状
        batched_frame = curr_frame.clone()
        while len(batched_frame.shape) < 5:
            batched_frame = batched_frame.unsqueeze(0)
        
        curr_frame_feats = self.model.forward_image_features(batched_frame.to(self.device))
        # -> (1, 1, 512)

        # 准备过去的帧
        if past_frames is None:
            # 如果是 episode 开始，创建 15 帧的零占位符
            past_frames = torch.zeros((1, 15, curr_frame_feats.shape[-1]), device=self.device)
        else:
            past_frames = past_frames.to(self.device)
            # 确保 (1, 15, 512)
            while len(past_frames.shape) < 3:
                past_frames = past_frames.unsqueeze(0)

        # 拼接：(1, 15, 512) + (1, 1, 512) -> (1, 16, 512)
        return torch.cat((past_frames, curr_frame_feats), dim=1)
    
    def _get_video_feats(self, image_feats: torch.Tensor) -> torch.Tensor:
        """将 16 帧序列 (1, 16, 512) 压缩为 (1, 512) 的视频特征。"""
        return self.model.forward_video_features(image_feats.to(self.device))

    def _get_text_feats(self, prompts: List[str]) -> torch.Tensor:
        """将文本字符串列表编码为 (P, 512) 的特征。"""
        text_feats = self.model.encode_text(prompts)
        return text_feats
    
