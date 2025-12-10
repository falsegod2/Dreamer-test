# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import math
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from affordance_map.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys, MultimodalSwinTransformerSys

import networks
import tools

logger = logging.getLogger(__name__)
to_np = lambda x: x.detach().cpu().numpy()

def probability_to_bool(input_tensor, jump_prob=1.0):
    """根据概率将张量转换为布尔掩码"""
    random_tensor = torch.rand_like(input_tensor)
    random_mask = random_tensor < jump_prob # bool
    
    bool_tensor = input_tensor > 0.5 # bool
    bool_tensor = bool_tensor & random_mask
    
    return bool_tensor

class RewardEMA:
    """
    运行时的奖励均值和标准差估计 (Exponential Moving Average)
    用于标准化奖励信号，使训练更稳定
    """

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # 原地更新 ema_vals
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class MCUnet(nn.Module):
    """多模态 U-Net，用于处理图像和文本特征"""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MCUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = MultimodalSwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                text_feature_dim=config.MODEL.TEXT_FEATURE_DIM,
                                heads=config.MODEL.HEADS)

    def forward(self, x, p):
        # 如果是灰度图，转为 3 通道
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x, p)
        return logits

    def load_from(self, config):
        """加载预训练权重，包含针对性的键名处理"""
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class WorldModel(nn.Module):
    """
    世界模型 (World Model) 主类。
    
    功能：
    1. 编码 (Encoder): Image -> Embedding
    2. 动力学 (Dynamics): 学习 p(s_t+1 | s_t, a_t)。
       - 支持标准 Dreamer (单流)
       - 支持 Iso-Dream++ (双流: S可控 + Z不可控)
    3. 预测头 (Heads): 预测图像重建、奖励、结束标志等。
    4. LS-Imagine 支持: 处理 Zoomed 图片以学习长视距跳跃。
    """
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config

        # --- 1. 编码器 (Encoder) ---
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim

        # --- 2. 动力学核心 (RSSM Core) ---
        if config.use_iso_dream:
            print(f"🔥 WorldModel: 启用 Iso-Dream 双流架构 (Z-Stoch={config.dyn_stoch_z}, Z-Deter={config.dyn_deter_z})")
            self._init_iso_dream_dynamics(config, act_space)
        else:
            self._init_standard_dynamics(config, act_space)

        # --- 3. 预测头 (Heads) ---
        # 所有的预测头都基于 feat_size (S+Z 或 S) 进行预测
        self.heads = nn.ModuleDict()
        self._init_heads(config, shapes)

        # --- 4. 优化器 ---
        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt, use_amp=self._use_amp,
        )
        
        # --- 5. 打印网络结构摘要 (增强可读性) ---
        self._print_summary()
    
    def _init_iso_dream_dynamics(self, config, act_space):
        """初始化 Iso-Dream 双流架构 (S: 可控, Z: 不可控)"""
        # S 分支 (可控, 接收动作)
        self.dynamics_s = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act,
            config.norm, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_min_std, config.unimix_ratio, config.initial,
            config.num_actions, self.embed_size, config.device,
            action_free=False 
        )
        # Z 分支 (不可控, 不接收动作)
        self.dynamics_z = networks.RSSM(
            config.dyn_stoch_z, config.dyn_deter_z, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act,
            config.norm, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_min_std, config.unimix_ratio, config.initial,
            config.num_actions, self.embed_size, config.device,
            action_free=True 
        )
        
        # 计算特征总维度 (S + Z)
        self.feat_size = self._get_rssm_feat_size(config, is_iso=True)

        # 逆动力学头 (Inverse Dynamics): 仅用于 S 分支
        # 作用: 强迫 S 分支学习与动作相关的信息 (s_t, s_t+1 -> a_t)
        if config.dyn_discrete:
            feat_s_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_s_dim = config.dyn_stoch + config.dyn_deter
            
        self.heads["inverse"] = networks.MLP(
            feat_s_dim * 2, # 输入: [Feature_t, Feature_t+1]
            (config.num_actions,), 
            config.reward_head["layers"], config.units, config.act, config.norm,
            dist="onehot" if hasattr(act_space, "n") else "normal",
            outscale=1.0, device=config.device, name="Inverse"
        )
        self.dynamics = None # 禁用单流变量，防止误用

    def _init_standard_dynamics(self, config, act_space):
        """初始化标准 Dreamer 单流架构"""
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_rec_depth, config.dyn_discrete, config.act,
            config.norm, config.dyn_mean_act, config.dyn_std_act,
            config.dyn_min_std, config.unimix_ratio, config.initial,
            config.num_actions, self.embed_size, config.device,
            action_free=False
        )
        self.feat_size = self._get_rssm_feat_size(config, is_iso=False)

    def _get_rssm_feat_size(self, config, is_iso=False):
        """辅助函数: 计算 RSSM 输出特征的维度"""
        def get_dim(stoch, deter):
            if config.dyn_discrete:
                return stoch * config.dyn_discrete + deter
            else:
                return stoch + deter
        
        feat_s = get_dim(config.dyn_stoch, config.dyn_deter)
        if is_iso:
            feat_z = get_dim(config.dyn_stoch_z, config.dyn_deter_z)
            return feat_s + feat_z
        return feat_s

    def _init_heads(self, config, shapes):
        """初始化所有的预测头 (Decoder, Reward, End, LS-Imagine Heads)"""
        # 1. 图像重建 (Decoder)
        self.heads["decoder"] = networks.MultiDecoder(self.feat_size, shapes, **config.decoder)
        
        # 2. 基础 RL 头
        self.heads["reward"] = self._make_mlp(config.reward_head, name="Reward")
        self.heads["end"] = self._make_mlp(config.end_head, dist="binary", name="End")
        
        # 3. LS-Imagine 特有头
        self.heads["jump"] = self._make_mlp(config.jump_head, dist="binary", name="Jump")
        self.heads["intrinsic"] = self._make_mlp(config.intrinsic_head, name="Intrinsic")
        
        # Jumping Steps & Accumulated Reward 需要双倍特征输入 (Jump from A to B)
        self.heads["jumping_steps"] = self._make_mlp(
            config.jumping_steps_head, inp_dim=self.feat_size * 2, name="JumpingSteps"
        )
        self.heads["accumulated_reward"] = self._make_mlp(
            config.accumulated_reward_head, inp_dim=self.feat_size * 2, name="AccumulatedReward"
        )

        # 损失函数权重缩放
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            end=config.end_head["loss_scale"],
            jump=config.jump_head["loss_scale"],
            intrinsic=config.intrinsic_head["loss_scale"],
            jumping_steps=config.jumping_steps_head["loss_scale"],
            accumulated_reward=config.accumulated_reward_head["loss_scale"],
        )

    def _make_mlp(self, cfg, inp_dim=None, dist=None, name="MLP"):
        """创建 MLP 头的辅助函数"""
        if inp_dim is None: inp_dim = self.feat_size
        if dist is None: dist = cfg.get("dist", "mse") # Default
        
        shape = (255,) if dist == "symlog_disc" else ()
        return networks.MLP(
            inp_dim, shape, cfg["layers"], self._config.units, 
            self._config.act, self._config.norm, 
            dist=dist, outscale=cfg["outscale"], 
            device=self._config.device, name=name
        )
    
    def _print_summary(self):
        """在命令行输出模型结构摘要"""
        print("\n" + "="*50)
        print(f"🏗️  World Model Structure ({'Iso-Dream++' if self._config.use_iso_dream else 'Standard'})")
        print(f"   - Feature Size: {self.feat_size}")
        print(f"   - Params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
        print(f"   - Heads: {list(self.heads.keys())}")
        print("="*50 + "\n")
    
    # =========================================================================
    # 核心训练循环 (Refactored)
    # =========================================================================
    def _train(self, data_origin):
        """
        训练入口。
        逻辑流：
        1. 预处理数据 (Normal & Zoomed)。
        2. 短视距学习 (Short-term Stream): 学习一步步的动态 (p(s_t+1|s_t))。
        3. 长视距学习 (Long-term Stream): 学习跳跃动态 (LS-Imagine 特有)。
        4. 计算所有 Head 的 Loss。
        5. 梯度更新。
        """
        # 1. 预处理数据
        data = self.preprocess(data_origin, zoomed=False)
        data_zoomed = self.preprocess(data_origin, zoomed=True)

        zoomed_num = torch.sum(data["is_zoomed"]).item()
        calculated_num = torch.sum(data_zoomed["is_calculated"]).item()

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # --- Step A: 图像编码 ---
                embed = self.encoder(data)
                
                # --- Step B: 短视距动力学学习 (Short-term Stream) ---
                # 返回:
                # post: 后验状态 (Posterior) - 包含睁眼看世界的信息
                # feat: 拼接好的特征向量 - 用于输入给 Heads
                # dyn_losses: 动力学损失 (KL Div, Inverse Loss 等)
                post, feat, dyn_losses_img = self._step_short_term(data, embed)

                # --- Step C: 通用预测头损失 (Decoder, Reward...) ---
                head_losses_img = self._compute_head_losses(feat, data, is_zoomed_branch=False)

                # --- Step D: 长视距/放大动力学学习 (LS-Imagine Zoomed Stream) ---
                total_loss = sum(dyn_losses_img.values()) + sum(head_losses_img.values())
                
                metrics = {} # 用于收集日志
                post_zoomed = None

                if zoomed_num > 0:
                    # 如果这批数据里有 Zoomed 图片，开启 LS-Imagine 分支
                    embed_zoomed = self.encoder(data_zoomed)
                    
                    # LS-Imagine 核心: 从 Normal Stream 中挑选状态，并在 Zoomed Stream 中推演
                    post_zoomed, feat_zoomed, dyn_losses_jmp = self._step_long_term_zoomed(
                        data, data_zoomed, embed_zoomed, post
                    )
                    
                    # 计算 Zoomed 分支的预测头损失 (如 jumping_steps)
                    head_losses_jmp = self._compute_head_losses(
                        feat_zoomed, data_zoomed, is_zoomed_branch=True, 
                        feat_base=self.get_feat(post) # 需要基础特征来计算 jump 差值
                    )
                    
                    # 合并损失 (加权)
                    loss_jmp = sum(dyn_losses_jmp.values()) + sum(head_losses_jmp.values())
                    total_loss += loss_jmp * self._config.long_term_branch_weight
                    
                    # 记录 Zoomed Metrics
                    metrics.update({f"zoomed_{k}": to_np(v.mean()) for k, v in dyn_losses_jmp.items()})
                    metrics.update({f"zoomed_{k}_loss": to_np(v.mean()) for k, v in head_losses_jmp.items()})

                # --- Step E: 反向传播 ---
                metrics.update(self._model_opt(total_loss, self.parameters()))

        # --- Step F: 整理日志 ---
        metrics.update({f"{k}_loss": to_np(v.mean()) for k, v in head_losses_img.items()})
        metrics.update({k: to_np(v.mean()) for k, v in dyn_losses_img.items()})
        
        # 传递 context 给 Exploration 模块使用
        context = dict(embed=embed, feat=feat, post=post)
    
        return post, post_zoomed, context, metrics

    # =========================================================================
    # 内部逻辑实现
    # =========================================================================

    def _step_short_term(self, data, embed):
        """处理标准的短视距序列数据"""
        metrics = {}
        
        if self._config.use_iso_dream:
            # --- Iso-Dream 双流逻辑 ---
            # 1. Observe (S & Z)
            post_s, prior_s = self.dynamics_s.observe(embed, data["action"], data["is_first"])
            post_z, prior_z = self.dynamics_z.observe(embed, None, data["is_first"]) # Z 无动作
            
            # 2. KL Loss (Rep & Dyn)
            loss_s, _, _, _ = self.dynamics_s.kl_loss(post_s, prior_s, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            loss_z, _, _, _ = self.dynamics_z.kl_loss(post_z, prior_z, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            
            # 3. Inverse Dynamics Loss (仅 S 分支)
            # 作用: 迫使 S 分支包含动作信息，实现解耦
            feat_s = self.dynamics_s.get_feat(post_s)
            inv_input = torch.cat([feat_s[:, :-1], feat_s[:, 1:]], dim=-1)
            target_action = data["action"][:, :-1, :-1] # 移除 mask 维度
            pred_action_dist = self.heads["inverse"](inv_input)
            inv_loss = -pred_action_dist.log_prob(target_action).mean()
            
            # 汇总
            post = {**post_s, **{f"z_{k}": v for k, v in post_z.items()}} # 合并字典
            feat = torch.cat([feat_s, self.dynamics_z.get_feat(post_z)], dim=-1) # 拼接特征
            losses = {"kl": loss_s + loss_z, "inv": inv_loss * self._config.iso_inv_scale}
            
        else:
            # --- 标准单流逻辑 ---
            post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
            loss, _, _, _ = self.dynamics.kl_loss(post, prior, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            feat = self.dynamics.get_feat(post)
            losses = {"kl": loss}
            
        return post, feat, losses

    def _step_long_term_zoomed(self, data, data_zoomed, embed_zoomed, post_full):
        """
        处理 LS-Imagine 的 Zoomed 数据。
        逻辑：找出哪些 step 是 "is_zoomed" 的，提取其对应的隐状态，然后用 Zoomed 图片进行单步推演。
        """
        # 1. 筛选数据 (Filter)
        # 只有被标记为 is_zoomed 的数据才参与计算
        mask = data["is_zoomed"].squeeze(-1).bool()
        
        # 筛选 Embedding 和 Action
        embed_z_sel = embed_zoomed[mask].unsqueeze(1) # [Batch_Zoom, 1, Emb]
        action_z_sel = data_zoomed["action"][mask].unsqueeze(1)
        first_z_sel = data_zoomed["is_first"][mask].unsqueeze(1)
        
        # 筛选初始状态 (S & Z)
        # 注意：这里需要把 post_full (Time, Batch) 转置并筛选，再转回去给 observe_zoomed 用
        def filter_state(state_dict, prefix=""):
            # 过滤出符合 prefix 的键，并根据 mask 筛选
            filtered = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    # v: [Batch, Time, ...] -> [Batch_Zoom] -> [Batch_Zoom, 1, ...]
                    clean_k = k[len(prefix):]
                    filtered[clean_k] = v[mask].unsqueeze(1)
            return filtered

        if self._config.use_iso_dream:
            # --- Iso-Dream Zoomed ---
            post_s_init = filter_state(post_full, prefix="") # 不带前缀的是 S
            post_z_init = filter_state(post_full, prefix="z_")
            
            # 推演 (Observe Zoomed)
            # 注意: observe_zoomed 是基于 post_s_init 作为上下文，用 embed_z_sel 进行更新
            # 这是为了模拟: "如果我现在看清楚了远方(Zoomed)，我的状态会变成什么样?"
            post_sz, prior_sz = self.dynamics_s.observe_zoomed(
                embed_z_sel, action_z_sel, first_z_sel, post_s_init, post_s_init
            )
            # Z 分支: 传入 Dummy Action (复用 action_z_sel)
            post_zz, prior_zz = self.dynamics_z.observe_zoomed(
                embed_z_sel, action_z_sel, first_z_sel, post_z_init, post_z_init
            )
            
            # KL Loss
            loss_sz, _, _, _ = self.dynamics_s.kl_loss(post_sz, prior_sz, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            loss_zz, _, _, _ = self.dynamics_z.kl_loss(post_zz, prior_zz, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            
            # 结果打包
            post_final = {**post_sz, **{f"z_{k}": v for k, v in post_zz.items()}}
            feat_final = torch.cat([self.dynamics_s.get_feat(post_sz), self.dynamics_z.get_feat(post_zz)], dim=-1)
            losses = {"kl": loss_sz + loss_zz}
            
        else:
            # --- Standard Zoomed ---
            post_init = filter_state(post_full)
            post_z, prior_z = self.dynamics.observe_zoomed(
                embed_z_sel, action_z_sel, first_z_sel, post_init, post_init
            )
            loss, _, _, _ = self.dynamics.kl_loss(post_z, prior_z, self._config.kl_free, self._config.dyn_scale, self._config.rep_scale)
            
            post_final = post_z
            feat_final = self.dynamics.get_feat(post_z)
            losses = {"kl": loss}
            
        return post_final, feat_final, losses

    def _compute_head_losses(self, feat, data, is_zoomed_branch=False, feat_base=None):
        """计算各个预测头的 Loss"""
        losses = {}
        
        # 如果是 Zoomed 分支，我们需要筛选 data
        if is_zoomed_branch:
            mask = data["is_zoomed"].squeeze(-1).bool()
            # 筛选出需要计算 loss 的数据，并增加时间维度 [Batch_Zoom, 1, ...]
            target_data = {k: v[mask].unsqueeze(1) for k, v in data.items() if isinstance(v, torch.Tensor)}
            # is_calculated 用于掩码 jumping_steps 等只有部分计算过的 Loss
            calc_mask = data["is_calculated"][mask].unsqueeze(1)
        else:
            target_data = data
            calc_mask = None

        for name, head in self.heads.items():
            # 跳过特殊头或不需要梯度的头
            if name == "inverse": continue 
            
            # 1. 准备输入特征
            # 对于 'jumping_steps' 和 'accumulated_reward'，需要拼接 [Before_Jump, After_Jump]
            if name in ["jumping_steps", "accumulated_reward"]:
                if not is_zoomed_branch: continue # 短视距分支不计算这个
                
                # feat_base 是跳转前的特征 (从 Short-term 筛选而来)
                # feat_base 也需要根据 mask 筛选
                mask = data["is_zoomed"].squeeze(-1).bool()
                feat_b = feat_base[mask].unsqueeze(1)
                
                inp = torch.cat([feat_b, feat], dim=-1)
            else:
                inp = feat

            # 2. 梯度阻断 (根据配置决定是否训练 Head)
            if name not in self._config.grad_heads:
                inp = inp.detach()
            
            # 3. 预测 & 计算 Loss
            pred = head(inp)
            if isinstance(pred, dict): # Decoder 输出是字典
                for k, v in pred.items():
                    loss = -v.log_prob(target_data[k])
                    losses[f"{name}_{k}"] = loss
            else:
                loss = -pred.log_prob(target_data[name])
                
                # 特殊处理: 只有计算过的才计入 Loss
                if name in ["jumping_steps", "accumulated_reward"] and calc_mask is not None:
                    loss *= calc_mask
                
                losses[name] = loss

        # 4. 应用 Loss Scale
        scaled_losses = {k: v * self._scales.get(k.split('_')[0], 1.0) for k, v in losses.items()}
        return scaled_losses
        
    # =========================================================================
    # 辅助函数
    # =========================================================================
    def get_dist(self, state):
        """
        [新增] 通用分布获取方法。
        用于计算熵 (Entropy)。在 Iso-Dream 模式下，我们主要关注可控分支 (S) 的熵。
        """
        if self._config.use_iso_dream:
            # 1. 拆分出 S 分支的状态
            state_s = {k: v for k, v in state.items() if not k.startswith("z_")}
            # 2. 返回 S 分支的分布
            return self.dynamics_s.get_dist(state_s)
        else:
            # 原版逻辑
            return self.dynamics.get_dist(state)

    def get_feat(self, state):
        """
        通用特征提取方法，自动适配 Iso-Dream 双流架构。
        """
        if self._config.use_iso_dream:
            # 1. 拆分混合状态字典
            state_s = {k: v for k, v in state.items() if not k.startswith("z_")}
            state_z = {k[2:]: v for k, v in state.items() if k.startswith("z_")}
            
            # 2. 分别提取特征
            feat_s = self.dynamics_s.get_feat(state_s)
            feat_z = self.dynamics_z.get_feat(state_z)
            
            # 3. 拼接
            return torch.cat([feat_s, feat_z], dim=-1)
        else:
            # 原版逻辑
            return self.dynamics.get_feat(state)
        
    def preprocess(self, obs, zoomed=False):
        """预处理：归一化图像，处理动作维度 (LS-Imagine 13维动作)"""
        obs = obs.copy()
        
        # 1. 图像归一化 [0, 255] -> [0, 1]
        if zoomed:
            obs["image"] = torch.Tensor(obs["zoomed_image"]) / 255.0
            obs["heatmap"] = torch.Tensor(obs["heatmap_on_zoomed"]).unsqueeze(-1) / 255.0
            # 使用 Zoomed 特有的奖励和状态
            obs["reward"] = obs["reward_on_zoomed"]
            obs['intrinsic'] = obs['intrinsic_on_zoomed']
            # 清零 Reset 标志 (Zoomed 总是连续的)
            obs['is_first'] = np.zeros_like(obs['is_first'])
        else:
            obs["image"] = torch.Tensor(obs["image"]) / 255.0
            obs["heatmap"] = torch.Tensor(obs["heatmap"]).unsqueeze(-1) / 255.0

        # 2. 动作维度处理 (12 -> 13)
        # LS-Imagine 增加第 13 维作为 "Jump Flag"
        if "action" in obs:
            act = obs["action"]
            if not zoomed:
                # 普通动作：第 13 维补 0
                padding = np.zeros((*act.shape[:-1], 1), dtype=act.dtype)
                obs["action"] = np.concatenate((act, padding), axis=-1)
            else:
                # Zoom 动作：全 0，第 13 维设 1 (标记为 Jump)
                # 注意：RSSM 的 Zoomed Observe 其实不太关心这个动作，主要是给 Inverse Head 用
                new_act = np.zeros((*act.shape[:-1], act.shape[-1] + 1), dtype=act.dtype)
                new_act[..., -1] = 1
                obs["action"] = new_act

        # 3. 清理不需要的键
        for k in ["zoomed_image", "heatmap_on_zoomed", "reward_on_zoomed", "intrinsic_on_zoomed"]:
            obs.pop(k, None)

        # 4. 转 Tensor 并移至 GPU
        tensor_keys = ["is_zoomed", "jump", "is_calculated", "is_terminal"]
        for k in tensor_keys:
            if k in obs:
                obs[k] = torch.Tensor(obs[k]).unsqueeze(-1)
        
        # 这里的 keys 映射需要根据实际数据微调
        obs["end"] = obs["is_terminal"] 
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)

        return {k: torch.Tensor(v).to(self._config.device) if isinstance(v, (np.ndarray, list)) else v 
                for k, v in obs.items()}
    
    def video_pred(self, data):
        """视频预测 (用于 Tensorboard 可视化)"""
        data = self.preprocess(data, zoomed=False)
        embed = self.encoder(data)
        
        # 1. 观测前 5 帧
        obs_slice = slice(0, 5)
        if self._config.use_iso_dream:
            post_s, _ = self.dynamics_s.observe(embed[:6, obs_slice], data["action"][:6, obs_slice], data["is_first"][:6, obs_slice])
            post_z, _ = self.dynamics_z.observe(embed[:6, obs_slice], None, data["is_first"][:6, obs_slice])
            
            # 重建
            feat = torch.cat([self.dynamics_s.get_feat(post_s), self.dynamics_z.get_feat(post_z)], dim=-1)
            recon = self.heads["decoder"](feat)["image"].mode()[:6]
            
            # 2. 想象后续帧 (Open-loop)
            init_s = {k: v[:, -1] for k, v in post_s.items()}
            init_z = {k: v[:, -1] for k, v in post_z.items()}
            
            prior_s = self.dynamics_s.imagine_with_action(data["action"][:6, 5:], init_s)
            # Z 分支传入 dummy action 仅为了长度对齐
            prior_z = self.dynamics_z.imagine_with_action(data["action"][:6, 5:], init_z)
            
            feat_prior = torch.cat([self.dynamics_s.get_feat(prior_s), self.dynamics_z.get_feat(prior_z)], dim=-1)
            openl = self.heads["decoder"](feat_prior)["image"].mode()
        else:
            # 标准逻辑 (略，类似)
            post, _ = self.dynamics.observe(embed[:6, obs_slice], data["action"][:6, obs_slice], data["is_first"][:6, obs_slice])
            recon = self.heads["decoder"](self.dynamics.get_feat(post))["image"].mode()[:6]
            init = {k: v[:, -1] for k, v in post.items()}
            prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
            openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()

        # 拼接: 真值 | 重建 | 预测 (误差)
        model_pred = torch.cat([recon, openl], 1)
        truth = data["image"][:6] + 0.5
        model_pred = model_pred + 0.5
        error = (model_pred - truth + 1.0) / 2.0
        
        return torch.cat([truth, model_pred, error], 2)


class ImagBehavior(nn.Module):
    """
    想象策略模块 (Imagination Behavior).
    
    核心职责:
    1. 在 Latent Space 中进行想象 (Rollout)。
    2. LS-Imagine 核心: 动态决定是"迈小步"(Step) 还是 "跳大步"(Jump)。
    3. 训练 Actor (策略) 最大化长期价值，训练 Critic (价值) 预测未来回报。
    """

    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        
        # --- 1. 配置参数 ---
        self.jump_prob = config.jump_prob
        # 预计算 Gamma 累积表 (用于长视距跳跃的折扣计算)
        # gamma_sum[k] = \sum_{i=0}^{k} gamma^i
        self.gamma_sum = [(1 - self._config.discount ** (i + 1)) / (1 - self._config.discount) 
                          for i in range(self._config.episode_max_steps)]
        self.gamma_sum = torch.tensor(self.gamma_sum, dtype=torch.float32, device=config.device)

        # --- 2. 确定输入特征维度 ---
        # 自动获取 S+Z 的总维度
        if hasattr(world_model, "feat_size"):
            feat_size = world_model.feat_size
        else:
            # Fallback: 手动计算 (防守性代码)
            feat_s = config.dyn_stoch + config.dyn_deter
            if config.dyn_discrete: feat_s += (config.dyn_stoch * (config.dyn_discrete - 1))
            
            feat_z = 0
            if config.use_iso_dream:
                feat_z = config.dyn_stoch_z + config.dyn_deter_z
                if config.dyn_discrete: feat_z += (config.dyn_stoch_z * (config.dyn_discrete - 1))
            
            feat_size = feat_s + feat_z

        # --- 3. 初始化网络 ---
        # Actor: s_t -> a_t
        self.actor = networks.MLP(
            feat_size, (config.num_actions,), config.actor["layers"], config.units,
            config.act, config.norm, config.actor["dist"], config.actor["std"],
            config.actor["min_std"], config.actor["max_std"], absmax=1.0,
            temp=config.actor["temp"], unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"], name="Actor",
        )
        
        # Value (Critic): s_t -> V(s_t)
        self.value = networks.MLP(
            feat_size, (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"], config.units, config.act, config.norm,
            config.critic["dist"], outscale=config.critic["outscale"],
            device=config.device, name="Value",
        )
        
        # Slow Value (Target Network): 用于稳定训练
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        # --- 4. 优化器 ---
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor", self.actor.parameters(), config.actor["lr"],
            config.actor["eps"], config.actor["grad_clip"], **kw
        )
        self._value_opt = tools.Optimizer(
            "value", self.value.parameters(), config.critic["lr"],
            config.critic["eps"], config.critic["grad_clip"], **kw
        )
        
        if self._config.reward_EMA:
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,              # 短视距数据的初始状态
        start_zoomed,       # 长视距(Zoomed)数据的初始状态
        objective,          # 奖励函数预测器
        intrinsic_objective,# 内在奖励预测器 (如 CLIP reward)
        jumping_steps_predictor,      # 预测跳跃步数
        accumulated_reward_predictor, # 预测跳跃期间的累积奖励
        jump_indicator,     # 预测是否应该跳跃
        is_end,             # 预测是否结束
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                
                # === Phase 1: 准备初始状态 (Data Augmentation) ===
                # 将正常的 start 状态和 Zoomed 的 start 状态拼接，增加 Batch Size
                # [B_normal, ...] + [B_zoomed, ...] -> [B_total, ...]
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                start = {k: flatten(v) for k, v in start.items()} 
                
                if start_zoomed is not None and self.jump_prob > 0.0:
                    start_zoomed = {k: flatten(v) for k, v in start_zoomed.items()}
                    for k, v in start.items():
                        start[k] = torch.cat((v, start_zoomed[k]), dim=0)

                # === Phase 2: 执行想象 (Imagination Rollout) ===
                # 这是 LS-Imagine 的核心：在想象中动态决定是 Step 还是 Jump
                imag_feat, imag_state, imag_action, jump_record = self._rollout_imagination(
                    start, self.actor, jump_indicator, is_end
                )

                # === Phase 3: 计算回报 (Calculate Rewards) ===
                # 预测每一步的即时奖励
                reward = objective(imag_feat, imag_state, imag_action)
                intrinsic_reward = intrinsic_objective(imag_feat, imag_state, imag_action)
                total_reward = reward + intrinsic_reward

                # === Phase 4: 计算价值目标 (Compute Targets) ===
                # 使用 Lambda-Target 计算长期价值
                # 注意：这里会用到 jumping_steps 来调整折扣因子 gamma
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, total_reward, jump_record, 
                    jumping_steps_predictor, accumulated_reward_predictor, is_end
                )

                # === Phase 5: 计算 Actor Loss ===
                actor_ent = self.actor(imag_feat).entropy()
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, base, jump_record
                )
                
                # 加上熵正则化 (鼓励探索)
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)

        # === Phase 6: 更新 Critic (Value) ===
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # Critic 预测当前状态价值
                value = self.value(imag_feat[:-1].detach())
                target = torch.stack(target, dim=1) # 这里的 target 是从上面计算出来的 Lambda-Return
                
                # Value Loss: 预测值 vs 目标值 (回归)
                value_loss = -value.log_prob(target.detach())
                
                # Slow Target 正则化
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(imag_feat[:-1].detach())
                    value_loss -= value.log_prob(slow_target.mode().detach())
                
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        # === Phase 7: 更新权重与日志 ===
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        # 打印训练看板 (每 100 次更新打印一次)
        self._print_train_info(metrics, jump_record)

        return imag_feat, imag_state, imag_action, weights, metrics

    def _rollout_imagination(self, start, policy, jump_indicator, is_end):
        """
        [核心逻辑] LS-Imagine 的混合想象循环。
        模拟未来 H 步，并在每一步动态决定是 "Short-term Step" 还是 "Long-term Jump"。
        """
        state_num = start['deter'].shape[0]
        
        # 初始化轨迹容器
        imag_state = {k: v.unsqueeze(0) for k, v in start.items()} # [1, N, ...]
        jump_record_list = [] # 记录每一步是否发生了跳跃
        imag_action_list = []
        
        # 1. 初始动作 (用于占位，实际还没执行)
        # LS-Imagine 的特征提取
        feat = self._world_model.get_feat(start).detach()
        action_example = policy(feat).sample()
        
        # --- 循环推演 (Horizon - 1) ---
        for i in range(self._config.imag_horizon - 1):
            # 取出当前时刻的状态 (Last state)
            current_state = {k: v[-1] for k, v in imag_state.items()}
            
            # A. 决策: 是否跳跃?
            # 计算跳跃概率 (Jump Prob) * 未结束概率 (Not End)
            should_jump_prob = jump_indicator(current_state) * (1.0 - is_end(current_state))
            # 转换为布尔掩码 (Mask): True 表示这一个样本要跳跃
            jump_mask = probability_to_bool(should_jump_prob, self.jump_prob).squeeze() # [N]
            
            jump_record_list.append(jump_mask)

            # B. 分流处理
            # 分支 1: 需要跳跃的状态 (Jumpy)
            # 先执行一步 _jumpy (模拟长视距跳跃)，得到 state_after_jump
            # 再基于 state_after_jump 执行一步 _imagine (模拟跳跃后的第一步调整)
            jump_nodes = {k: v[jump_mask] for k, v in current_state.items()}
            
            # [Jump] 
            _, state_after_jump_1, _ = self._jumpy(jump_nodes, policy, horizon=1)
            # [Step after Jump]
            _, state_after_jump_2, _ = self._imagine(state_after_jump_1, policy, horizon=1)

            # 分支 2: 普通推演的状态 (Short-term)
            # 对所有状态执行一步普通推演
            _, state_next_all, action_next_all = self._imagine(current_state, policy, horizon=1)
            
            # C. 状态合并 (Merge)
            # 将 "Jump" 分支的结果覆盖回 "All" 分支中对应的位置
            # 只有那些 mask 为 True 的位置会被长视距结果替代
            for k in state_next_all:
                state_next_all[k][jump_mask] = state_after_jump_2[k]
            
            # D. 记录轨迹
            imag_action_list.append(action_next_all)
            for k in imag_state:
                imag_state[k] = torch.cat((imag_state[k], state_next_all[k].unsqueeze(0)), dim=0)

        # --- 结尾处理 ---
        # 补全最后一步的特征和动作
        last_state = {k: v[-1] for k, v in imag_state.items()}
        last_feat = self._world_model.get_feat(last_state).detach()
        last_action = policy(last_feat).sample()
        imag_action_list.append(last_action)
        
        # 整理 Tensor 形状
        # Actions: [Horizon, N, Act_Dim]
        imag_action = torch.stack(imag_action_list, dim=0)
        
        # Jump Record: [Horizon, N] (最后一步补 0)
        jump_record = torch.stack(jump_record_list, dim=0)
        last_jump_pad = torch.zeros((1, state_num), device=self._config.device)
        jump_record = torch.cat((jump_record, last_jump_pad), dim=0)

        # Features: [Horizon, N, Feat_Dim]
        imag_feat = self._world_model.get_feat(imag_state)

        return imag_feat, imag_state, imag_action, jump_record

    def _imagine(self, start, policy, horizon):
        """
        短视距想象 (Short-term Imagination)。
        Standard Dreamer dynamics.
        """
        # 适配 Iso-Dream: 区分 S 和 Z 分支
        use_iso = self._config.use_iso_dream
        dyn_s = self._world_model.dynamics_s if use_iso else self._world_model.dynamics
        dyn_z = self._world_model.dynamics_z if use_iso else None

        def step(prev, _):
            state, _, _ = prev
            
            # 1. 提取特征 (S+Z)
            if use_iso:
                feat = self._world_model.get_feat(state)
                # 拆分状态供分别推演
                state_s = {k: v for k, v in state.items() if not k.startswith("z_")}
                state_z = {k[2:]: v for k, v in state.items() if k.startswith("z_")}
            else:
                feat = dyn_s.get_feat(state)
            
            # 2. 策略选择动作
            action = policy(feat.detach()).sample()
            
            # 3. 动力学推演
            # 构造 LS-Imagine 特有的 13维动作 (最后一位是 Jump Flag=0)
            zeros_tensor = torch.zeros(action.shape[0], 1).to(action.device)
            env_action = torch.cat((action, zeros_tensor), dim=-1)

            if use_iso:
                # S分支: 随动作变化
                succ_s = dyn_s.img_step(state_s, env_action)
                # Z分支: 自由演化 (Dummy Action)
                succ_z = dyn_z.img_step(state_z, env_action) 
                
                # 合并
                succ = {**succ_s, **{f"z_{k}": v for k, v in succ_z.items()}}
            else:
                succ = dyn_s.img_step(state, env_action)

            return succ, feat, action

        # 使用 static_scan 批量执行 horizon 步
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        
        # 整理输出格式
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if horizon == 1:
            states = {k: v.squeeze(0) for k, v in succ.items()}
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)

        return feats, states, actions

    def _jumpy(self, start, policy, horizon):
        """
        长视距想象 (Long-term / Jumpy Imagination)。
        核心区别: Action 的 Jump Flag 被置为 1。
        """
        use_iso = self._config.use_iso_dream
        dyn_s = self._world_model.dynamics_s if use_iso else self._world_model.dynamics
        dyn_z = self._world_model.dynamics_z if use_iso else None

        def step(prev, _):
            state, _, _ = prev
            
            # 1. 提取特征
            if use_iso:
                feat = self._world_model.get_feat(state)
                state_s = {k: v for k, v in state.items() if not k.startswith("z_")}
                state_z = {k[2:]: v for k, v in state.items() if k.startswith("z_")}
            else:
                feat = dyn_s.get_feat(state)
            
            # 2. 策略选择动作 (仅用于生成特征，实际动作被 Jump Flag 覆盖)
            action = policy(feat.detach()).sample()

            # 3. 构造 JUMP 动作
            # [Action, 1] -> 告诉 WorldModel 这是一个跳跃步
            jump_action = torch.zeros(action.shape[0], action.shape[1] + 1).to(action.device)
            jump_action[:, -1] = 1 # Set Jump Flag to 1
            
            # 4. 动力学推演 (注意：这里使用的是 jump_action)
            if use_iso:
                succ_s = dyn_s.img_step(state_s, jump_action)
                succ_z = dyn_z.img_step(state_z, jump_action)
                succ = {**succ_s, **{f"z_{k}": v for k, v in succ_z.items()}}
            else:
                succ = dyn_s.img_step(state, jump_action)
            
            return succ, feat, action
        
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if horizon == 1:
            states = {k: v.squeeze(0) for k, v in succ.items()}
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, jump_record, jumping_steps_predictor, accumulated_reward_predictor, is_end):
        """
        计算 Lambda-Target (TD-lambda)。
        这是强化学习的核心：估计"在这个状态下，未来能拿多少分"。
        
        LS-Imagine 的特殊之处：
        普通的 TD-Error 是 r_t + gamma * V(s_{t+1})。
        但如果是跳跃步，时间跨度不是 1，而是 k 步。
        公式变为: r_{accumulated} + gamma^k * V(s_{t+k})
        """
        # 1. 预测跳跃步数 (k) 和 累积奖励 (r_acc)
        # 输入: [Feat_t, Feat_t+1]
        fc = torch.cat((imag_feat[:-1], imag_feat[1:]), dim=-1) 
        
        pred_steps = jumping_steps_predictor(fc).mode() # [L-1, N, 1]
        pred_acc_reward = accumulated_reward_predictor(fc).mode()
        
        # 对齐 shape (最后一步补0)
        pred_steps = torch.cat([pred_steps, torch.zeros_like(pred_steps[0]).unsqueeze(0)], dim=0)
        pred_acc_reward = torch.cat([pred_acc_reward, torch.zeros_like(pred_acc_reward[0]).unsqueeze(0)], dim=0)
        
        # 2. 计算修正后的 累积奖励
        # 因为 accumulated_reward 是归一化的，这里乘回 gamma_sum 还原
        # gamma_sum index: (steps - 2) 是为了处理索引偏移
        gamma_idx = (pred_steps - 2).clamp(0, self._config.episode_max_steps - 1).long()
        # 注意: 如果 pred_steps < 2，index 会是 0。
        
        # 这里使用 gather 或者简单的索引来获取对应的 gamma_sum
        # 简化写法 (假设 gamma_sum 在 GPU):
        correction = self.gamma_sum[gamma_idx.squeeze(-1)].unsqueeze(-1)
        pred_acc_reward *= correction
        
        # 3. 混合 Short/Long 逻辑
        # 如果是 Jump 步 (jump_record=1): 使用 pred_steps 和 pred_acc_reward
        # 如果是 Step 步 (jump_record=0): 使用 1 和 reward (单步)
        
        # 真正的跳跃步数: Jump时用预测值，不Jump时为1
        real_steps = (pred_steps - 1) * jump_record + 1 
        # 真正的奖励: Jump时用累积值，不Jump时为0 (后续 lambda_return 会处理基础 reward)
        # 注意：这里的逻辑比较 tricky，原代码是在 lambda_return 内部处理混合
        # 这里我们准备好 raw data 传进去
        
        value = self.value(imag_feat).mode()
        end = is_end(imag_state)
        
        # 4. 调用工具函数计算 Lambda Return
        # 这是一个递归计算过程 (Backprop through time)
        targets = tools.lambda_return_for_ls_imagine(
            reward[1:], 
            value[:-1], 
            gamma=self._config.discount, 
            end=end[:-1],
            jumping_steps=real_steps[:-1], 
            accumulated_reward=pred_acc_reward[:-1] * jump_record[:-1], # 只有 Jump 时才有这一项
            bootstrap=value[-1], 
            lambda_=self._config.discount_lambda, 
            axis=0
        )
        
        # 5. 计算权重 (Discount weight)
        # 越远的未来，权重越低
        # 这里的 discount 需要考虑跳跃步数：gamma ^ real_steps
        step_discount = self._config.discount ** real_steps
        discount_factor = step_discount * (1.0 - end)
        
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount_factor[:1]), discount_factor[:-1]], 0), 0
        ).detach()

        return targets, weights, value[:-1]

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base, jump_record):
        """计算 Actor 的 Policy Gradient Loss"""
        metrics = {}
        policy = self.actor(imag_feat.detach())
        
        # 计算 Advantage (优势函数)
        # Adv = Q(s,a) - V(s)
        # Target ≈ Q(s,a), Base ≈ V(s)
        target = torch.stack(target, dim=1)
        
        # Reward EMA (标准化 Advantage，使训练更稳定)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            advantage = normed_target - normed_base
            
            metrics["r_ema_05"] = self.ema_vals[0]
            metrics["r_ema_95"] = self.ema_vals[1]
        else:
            advantage = target - base

        # Policy Gradient:
        # Loss = - log_prob(a|s) * Advantage
        log_prob = policy.log_prob(imag_action)[:-1].unsqueeze(-1)
        
        # 混合梯度 (Dynamics Backprop vs REINFORCE)
        # LS-Imagine 论文通常使用 Reinforce 或 Both
        if self._config.imag_gradient == "dynamics":
            actor_target = advantage
        elif self._config.imag_gradient == "reinforce":
            actor_target = log_prob * advantage.detach()
        elif self._config.imag_gradient == "both":
            mix = self._config.imag_gradient_mix
            actor_target = mix * advantage + (1 - mix) * (log_prob * advantage.detach())
        
        # Masking: 跳跃步不更新 Actor (因为跳跃步的动作是虚构的/Jump Flag)
        # 只有正常的 Step 步才更新策略
        valid_mask = (1.0 - jump_record[:-1])
        
        actor_loss = -weights[:-1] * valid_mask * actor_target
        
        return actor_loss, metrics

    def _update_slow_target(self):
        """软更新 Target Network"""
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def _print_train_info(self, metrics, jump_record):
        """
        [新增] 命令行训练监控。
        打印：跳跃频率、平均值、EMA统计等。
        """
        if self._updates % 100 != 0: return # 减少刷屏频率

        jump_rate = torch.mean(jump_record.float()).item()
        val_mean = metrics.get("value_mean", 0.0)
        
        print(f"\n[ImagBehavior] Step {self._updates}")
        print(f"  Jump Rate: {jump_rate:.2%} (Avg jumps per rollout)")
        print(f"  Value Mean: {val_mean:.3f}")
        if "r_ema_95" in metrics:
            print(f"  Reward EMA: Low={metrics['r_ema_05']:.3f}, High={metrics['r_ema_95']:.3f}")