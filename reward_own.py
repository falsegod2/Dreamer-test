import torch
from omegaconf import OmegaConf
from mineclip import MineCLIP
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

import os
import math
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import torch.utils.checkpoint as checkpoint

from omegaconf import OmegaConf
from torch import nn
from einops import rearrange
from datetime import datetime
from scipy.stats import kurtosis
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.transforms import Normalize

import tool_own
import model_own

class MinedojoClipReward():
    """
    接收环境观察 (obs) 和文本提示 (prompts)，并计算一个代表“视频与文本匹配程度”的奖励值
    只用到了get_logits函数
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

    def _load_mineclip(self, ckpt, config):
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = torch.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()
        print("MineCLIP 模型加载并设置到 eval() 模式。") 
    '''
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
    '''
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
    @staticmethod
    def get_resolution():
        return (160, 256)

    @staticmethod
    def _get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        return torch.from_numpy(curr_frame)

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
    
class RandomGenerator(object):
    """
    
    """
    def __init__(self, output_height=224, output_width=224):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image, label):
        image, label = tool_own.MCResize(image, label, target_size=(self.output_width, self.output_height))
        image, label = tool_own.MNormalize(image, label)
        image, label = tool_own.MyToTensor(image, label)

        return image, label

class Config:
    #unet配置
    class AUG:
        AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
        COLOR_JITTER = 0.4
        CUTMIX = 1.0
        CUTMIX_MINMAX = None
        MIXUP = 0.8
        MIXUP_MODE = "batch"
        MIXUP_PROB = 1.0
        MIXUP_SWITCH_PROB = 0.5
        RECOUNT = 1
        REMODE = "pixel"
        REPROB = 0.25

    class DATA:
        BATCH_SIZE = 72
        CACHE_MODE = "part"
        DATASET = "imagenet"
        DATA_PATH = ""
        IMG_SIZE = 224
        INTERPOLATION = "bicubic"
        NUM_WORKERS = 8
        PIN_MEMORY = True
        ZIP_MODE = False

    class MODEL:
        DROP_PATH_RATE = 0.2
        DROP_RATE = 0.0
        HEADS = 8
        LABEL_SMOOTHING = 0.1
        NAME = "swin_tiny_patch4_window7_224"
        NUM_CLASSES = 1000
        PRETRAIN_CKPT = "./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
        RESUME = ""
        SWIN = type('SWIN', (), {
            'APE': False,
            'DECODER_DEPTHS': [2, 2, 2, 1],
            'DEPTHS': [2, 2, 2, 2],
            'EMBED_DIM': 96,
            'FINAL_UPSAMPLE': "expand_first",
            'IN_CHANS': 3,
            'MLP_RATIO': 4.0,
            'NUM_HEADS': [3, 6, 12, 24],
            'PATCH_NORM': True,
            'PATCH_SIZE': 4,
            'QKV_BIAS': True,
            'QK_SCALE': None,
            'WINDOW_SIZE': 7
        })
        TEXT_FEATURE_DIM = 512
        TYPE = "swin"

    class TRAIN:
        ACCUMULATION_STEPS = 0
        AUTO_RESUME = True
        BASE_LR = 0.0005
        CLIP_GRAD = 5.0
        EPOCHS = 300
        LR_SCHEDULER = type('LR_SCHEDULER', (), {
            'DECAY_EPOCHS': 30,
            'DECAY_RATE': 0.1,
            'NAME': "cosine"
        })
        MIN_LR = 5e-06
        OPTIMIZER = type('OPTIMIZER', (), {
            'BETAS': (0.9, 0.999),
            'EPS': 1e-08,
            'MOMENTUM': 0.9,
            'NAME': "adamw"
        })
        START_EPOCH = 0
        USE_CHECKPOINT = False
        WARMUP_EPOCHS = 20
        WARMUP_LR = 5e-07
        WEIGHT_DECAY = 0.05

    EVAL_MODE = True
    LOCAL_RANK = 0
    OUTPUT = ""
    PRINT_FREQ = 10
    SAVE_FREQ = 1
    SEED = 0
    TAG = "default"
    TEST = type('TEST', (), {
        'CROP': True
    })
    THROUGHPUT_MODE = False

def save_image(img, index, name='curr_frame', output_dir='output_tmp'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imsave(os.path.join(output_dir, f"{index}_{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), img)

def save_mask(out_np, index, name='mask', output_dir='output_tmp'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, mask in enumerate(out_np):
        plt.imsave(os.path.join(output_dir, f"{index}_{name}_{i}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), mask, cmap='jet', vmin=0, vmax=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def resized_if_need(image, target_size=(256, 160)):
    if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
    return image

class MinedojoConcentrationReward():
    def __init__(self, ckpt="weights/mineclip_attn.pth", unet_checkpoint_dir="envs/tasks/base/unet_checkpoint", output_dir="output_tmp", gaussian_sigma_weight=0.5, **kwargs) -> None:
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.text_feature = None
        self.prompts = None
        self.unet_checkpoint_dir = unet_checkpoint_dir
        self.output_dir = output_dir

        self.resolution = self.get_resolution() # (160, 256)
        self.u_net_resolution = 224
        self.device = kwargs.pop("device", "cuda")
        self.model = None
        self.unet = None
        self.gaussian = self._generate_gaussian_distribution(height=self.resolution[0], width=self.resolution[1], peak=1.0, sigma_x=self.resolution[1]*gaussian_sigma_weight, sigma_y=self.resolution[0]*gaussian_sigma_weight)
        self.gaussian_mean = np.mean(self.gaussian)
        self.mask = None
        self.mask_on_zoomed_image = None
        self.preprocess = RandomGenerator(output_height=self.u_net_resolution, output_width=self.u_net_resolution)
        self.best_value_on_mask = 0

        self.have_center = False

        self.ker_size = 0.15
        self.strides = 9
        self.stride = 1
        self.zoom_in_frames = []
        self.video_feats = None
        self.zoom_in_score = None

        self.index = 0

        self.unet_cfg = Config()
        self._load_mineclip(ckpt, kwargs)
        self._load_unet()

        self.gaussian_score = 0
        self.zoom_in_prob = 0
        self.num_above_threshold = 0

        self.blur_x = 51
        self.blur_y = 79 

        self.check_threshold_buffer = tool_own.ThresholdBuffer()
        self.gaussian_buffer = tool_own.ThresholdBuffer()

        self.check_threshold = 1

        self.curr_frame = None
        self.zoomed_frame = None

        
    @staticmethod
    def get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        curr_frame = curr_frame.transpose((1, 2, 0))
        return curr_frame # shape: (160, 256, 3)
    
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

    def _load_unet(self):
        self.unet = model_own.MCUnet(self.unet_cfg, img_size=self.u_net_resolution, num_classes=1).cuda()
        snapshot = os.path.join(self.unet_checkpoint_dir, 'swin_unet_checkpoint.pth')
        msg = self.unet.load_state_dict(torch.load(snapshot))
        print("self trained swin unet",msg)
        self.unet.eval()

    def _get_text_feats(
        self,
        prompts: str
    ) -> torch.Tensor:

        if self.prompts is not None and self.text_feature is not None and self.prompts == prompts:
            return self.text_feature # shape: [P, 512]
        
        else:
            self.prompts = prompts
            self.text_feature = self.model.encode_text(prompts)
            assert len(self.text_feature.shape) == 2 and self.text_feature.shape[0] == len(prompts), "Found shape {}".format(self.text_feature.shape)
            return self.text_feature # shape: [P, 512]

    def _generate_mask(self,
                       obs: Dict,
                       prompts: List[str]):
        self.index += 1
        self.curr_frame = self.get_curr_frame(obs) # shape: [160, 256, 3]
        random_lable = np.random.rand(self.u_net_resolution, self.u_net_resolution, 1)
        img, _ = self.preprocess(self.curr_frame, random_lable)
        img = img.unsqueeze(0) # shape: [1, 3, 224, 224]

        with torch.no_grad():
            texts_feats = self._get_text_feats(prompts).cuda().float()
            img = img.cuda().float().expand(texts_feats.shape[0], -1, -1, -1)
            out = self.unet(img, texts_feats) # out.shape: [P, 1, 224, 224]
            out_np = out.squeeze(1).cpu().detach().numpy() # out_np.shape: [P, 224, 224]
            out_np = out_np.transpose((1, 2, 0))
            out_np = resized_if_need(out_np, target_size=(self.resolution[1], self.resolution[0]))
            out_np = out_np.transpose((2, 0, 1)).squeeze(0)
            out_np = cv2.GaussianBlur(out_np, (self.blur_x, self.blur_y), 0)
            out_np = out_np[np.newaxis, :]

        return out_np

    def _generate_gaussian_distribution(self, height=160, width=256, peak=1.0, sigma_x=128.0, sigma_y=80.0):
        if sigma_x == 0 or sigma_y == 0:
            return np.full((height, width), peak)

        x = np.linspace(-width // 2, width // 2, width)
        y = np.linspace(-height // 2, height // 2, height)
        x, y = np.meshgrid(x, y)
        gaussian = peak * np.exp(-((x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))

        return gaussian

    def get_reward(
            self,
            obs: Dict,
            prompts: List[str],
            episode_num: int,
            step_num: int
    ):
        masks = self._generate_mask(obs, prompts)
        self.mask = np.max(masks, axis=0) * 255.0 # shape: [160, 256]

        score = 0
        for mask in masks:
            score += (np.mean(mask * self.gaussian)/self.gaussian_mean)

        self.gaussian_score = score
        self.gaussian_buffer.add(self.gaussian_score)
        heatmap_normalized = self.mask / 255.0

        kurtosis_value = kurtosis(heatmap_normalized.flatten())
        normalized_kurtosis = sigmoid(kurtosis_value)

        self.zoom_in_prob = normalized_kurtosis * (np.max(heatmap_normalized) - np.mean(heatmap_normalized))
        self.check_threshold_buffer.add(self.zoom_in_prob)
        self.check_threshold = self.check_threshold_buffer.get_threshold()   
        
        return score, self.zoom_in_prob, self.check_threshold
        
    def get_heatmap(self, is_zoomed=False):
        if is_zoomed:
            return np.expand_dims(self.mask_on_zoomed_image, axis=-1) # [H, W, 1]
        else:
            return np.expand_dims(self.mask, axis=-1) / 255.0 # [H, W, 1]

    def generate_zoom_in_frame(self, ):
        if self.check_threshold >= self.zoom_in_prob:
            return self.curr_frame, False
        
        image_tensor = torch.from_numpy(self.curr_frame).unsqueeze(0).float().to(self.device) # shape: [1, 160, 256, 3]
        B, H, W, C = image_tensor.shape
        heatmap_normalized = self.mask / 255.0
        threshold_value = (np.max(heatmap_normalized) + np.min(heatmap_normalized)) / 2.0 + np.std(heatmap_normalized)
        _, binary_image = cv2.threshold(heatmap_normalized, threshold_value, 1, cv2.THRESH_BINARY)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 10))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return self.curr_frame, False

        max_mean_value = 0
        max_area_ratio = 0
        centroid_x = 0
        centroid_y = 0
        self.have_center = False

        for contour in contours:
            mask = np.zeros_like(heatmap_normalized)
            cv2.drawContours(mask, [contour], -1, (1), thickness=cv2.FILLED)
            mean_val = np.mean(heatmap_normalized[mask == 1])
            M = cv2.moments(contour)
            if self.gaussian[int(M['m01'] / M['m00']), int(M['m10'] / M['m00'])] >= self.gaussian_mean:
                self.have_center = True
            if mean_val > max_mean_value:
                max_mean_value = mean_val
                max_area_ratio = np.sum(mask) / mask.size
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])

        self.num_above_threshold = max_area_ratio / (self.gaussian[centroid_y, centroid_x])
        proportion = torch.tensor([self.num_above_threshold], device=self.device)
        proportion = torch.clamp(proportion, 0.01, 1.0)
        sqrt_proportion = torch.sqrt(proportion) # shape: [1]
        mask = torch.from_numpy(self.mask / 255.0).unsqueeze(0).float().to(self.device) # shape: [1, 160, 256]

        window_width = (sqrt_proportion * W).int()
        window_height = (sqrt_proportion * H).int()
        cur_window_height, cur_window_width = window_height[0], window_width[0]
        kernel = torch.ones((1, 1, cur_window_height, cur_window_width), device=self.device) 
        kernel_size = cur_window_height * cur_window_width
        
        mask = mask.unsqueeze(1) # shape: [1, 1, 160, 256]
        conv_result = F.conv2d(mask, kernel, stride=self.stride)

        best_value, best_idx = torch.max(conv_result.view(-1), 0)
        self.best_value_on_mask = (best_value / kernel_size).item()
        best_y, best_x = divmod(best_idx.item(), conv_result.shape[-1])

        left_top = torch.tensor([best_x, best_y], device=self.device) * self.stride
        right_bottom = left_top + torch.tensor([cur_window_width, cur_window_height], device=self.device)
        window = image_tensor[:, int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0]), :] 
        window = window.permute(0, 3, 1, 2) # shape: [1, 3, window_height, window_width]
        zoomed_image = F.interpolate(window, size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # shape: [1, 160, 256, 3]
        self.zoomed_frame = zoomed_image.squeeze(0).detach().cpu().numpy().astype(np.uint8) # shape: [160, 256, 3]

        return self.zoomed_frame, True
    
    def compute_reward_on_zoomed_image(self):
        random_lable = np.random.rand(self.u_net_resolution, self.u_net_resolution, 1)
        img, _ = self.preprocess(self.zoomed_frame, random_lable)
        img = img.unsqueeze(0)

        with torch.no_grad():
            texts_feats = self._get_text_feats(self.prompts).cuda().float()
            img = img.cuda().float().expand(texts_feats.shape[0], -1, -1, -1)

            out = self.unet(img, texts_feats) # out.shape: [P, 1, 224, 224]
            out_np = out.squeeze(1).cpu().detach().numpy() # out_np.shape: [P, 224, 224]
            out_np = out_np.transpose((1, 2, 0))
            out_np = resized_if_need(out_np, target_size=(self.resolution[1], self.resolution[0]))
            out_np = out_np.transpose((2, 0, 1)).squeeze(0)
            out_np = cv2.GaussianBlur(out_np, (self.blur_x, self.blur_y), 0)
            out_np = out_np[np.newaxis, :]

        self.mask_on_zoomed_image = np.max(out_np, axis=0)

        zoomed_gaussian = 0
        for mask in out_np:
            zoomed_gaussian += (np.mean(mask * self.gaussian)/self.gaussian_mean)

        kurtosis_value = kurtosis(self.mask_on_zoomed_image.flatten())
        normalized_kurtosis = sigmoid(kurtosis_value)
        zoom_in_prob_on_zoomed_image = normalized_kurtosis * (np.max(self.mask_on_zoomed_image) - np.mean(self.mask_on_zoomed_image))
        zoomed_reward = self.best_value_on_mask

        if zoomed_gaussian < self.gaussian_score + 2.0 * self.gaussian_buffer.std_dev():
            is_zoomed = False
        else:
            is_zoomed = True

        jump = is_zoomed and self.have_center
        
        return zoomed_reward, zoomed_gaussian, zoom_in_prob_on_zoomed_image, is_zoomed, jump
    
    def save_img_and_mask(self):
        print(self.index)
        save_image(self.curr_frame, self.index)
        save_mask(np.expand_dims(self.mask / 255.0, axis=0), self.index)
        save_image(self.zoomed_frame, self.index, name='zoomed')
        save_mask(np.expand_dims(self.mask_on_zoomed_image, axis=0), self.index, name='zoomed_mask')