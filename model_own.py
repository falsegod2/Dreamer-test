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

#from mineclip import MineCLIP


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


class MCUnetConfig:
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



class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config, env):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()} # 'image': (64, 64, 3)
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        
        #CORE1
        # --- MCUnet 加载部分 ---
        # 使用本地定义的 MCUnetConfig 避开 yacs 依赖
        self.env = env

        prompt = ["Cut a tree"]
        self.prompt = env._get_text_feats(prompt)

        self.mc_unet = MCUnet(MCUnetConfig, img_size=224, num_classes=1).cuda()
        snapshot = os.path.join("affordance_map/finetune_unet/finetune_checkpoints/harvest_log_in_plains", 'swin_unet_checkpoint.pth')
        msg = self.mc_unet.load_state_dict(torch.load(snapshot))
        print("self trained swin unet",msg)
        print(f"Successfully loaded MCUnet from {snapshot}")
        self.mc_unet.eval()

        #self.clip = MineCLIP(**config).to(self.device)


        # 2. 定义双分支动力学
        # 信号分支：任务相关，受控
        self.dynamics_s = self.dynamics # 复用原有的 dynamics 配置
        # 干扰分支：任务无关，不受控
        self.dynamics_d = copy.deepcopy(self.dynamics)       
        #CORE1

        self.heads = nn.ModuleDict()

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        #维度变为 feat_size * 2
        #CORE1
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size * 2, shapes, **config.decoder # 接收拼接后的 feat_s 和 feat_d
        )

        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        self.heads["end"] = networks.MLP(
            feat_size,
            (),
            config.end_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.end_head["outscale"],
            device=config.device,
            name="End",
        )

        '''
        self.heads["intrinsic"] = networks.MLP(
            feat_size,
            (255,) if config.intrinsic_head["dist"] == "symlog_disc" else (),
            config.intrinsic_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.intrinsic_head["dist"],
            outscale=config.intrinsic_head["outscale"],
            device=config.device,
            name="Intrinsic",
        )
        '''
        for name in config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            end=config.end_head["loss_scale"],
            #intrinsic=config.intrinsic_head["loss_scale"],
        )

    def _train(self, data_origin):
        
        data = self.preprocess(data_origin)

        '''
        zoomed_num = torch.sum(data["is_zoomed"]).item()
        calculated_num = torch.sum(data_zoomed["is_calculated"]).item()
        '''

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                
                #CORE1
                # --- 步骤 1: 生成空间掩码 (Affordance Mask) ---
                # data['p'] 是任务目标文本嵌入
                # 注意：MCUnet 输出是 [B*T, 1, 224, 224]
                # out_mask 形状: (B*T, 1, 224, 224)

                aff_mask_flat = self.mc_unet(data['image'],self.prompt) 
                # 还原维度以配合 Encoder 处理 (B, T, 1, H, W)
                aff_mask = aff_mask_flat.reshape(
                    data['image'].shape[0], data['image'].shape[1], 1, 
                    aff_mask_flat.shape[-2], aff_mask_flat.shape[-1]
                )
                
                # --- 步骤 2: 空间特征分拣 (Encoder 侧介入) ---
                # 提取“受控且任务相关”特征 (Signal)
                embed_s = self.encoder(data, mask=aff_mask)
                # 提取“任务无关干扰”特征 (Distractor)
                embed_d = self.encoder(data, mask=(1.0 - aff_mask))

                # --- 步骤 3: 双流动力学观察 (RSSM 侧介入) ---
                post_s, prior_s = self.dynamics_s.observe(embed_s, data["action"], data["is_first"])
                # 干扰分支传入零动作，使其学习背景动态
                post_d, prior_d = self.dynamics_d.observe(embed_d, torch.zeros_like(data["action"]), data["is_first"])

                # KL 损失 (加权控制信息带宽)
                kl_free = self._config.kl_free # 1.0
                dyn_scale = self._config.dyn_scale # 0.5
                rep_scale = self._config.rep_scale # 0.1

                kl_loss_s, kl_val_s, dyn_loss_s, rep_loss_s = self.dynamics_s.kl_loss(post_s, prior_s, kl_free, dyn_scale, rep_scale)
                kl_loss_d, kl_val_d, dyn_loss_d, rep_loss_d = self.dynamics_d.kl_loss(post_d, prior_d, kl_free, dyn_scale, rep_scale)


                # --- 步骤 4: 提取解耦特征 ---
                feat_s = self.dynamics_s.get_feat(post_s)
                feat_d = self.dynamics_d.get_feat(post_d)         
                #CORE1
                '''
                embed = self.encoder(data)
                #embed_zoomed = self.encoder(data_zoomed)
                # process original data
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                kl_free = self._config.kl_free # 1.0
                dyn_scale = self._config.dyn_scale # 0.5
                rep_scale = self._config.rep_scale # 0.1

                kl_loss_img, kl_value_img, dyn_loss_img, rep_loss_img = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                assert kl_loss_img.shape == embed.shape[:2], kl_loss_img.shape
                '''
                #CORE1
                # --- 步骤 5: 各个预测头 (Heads) 的分流逻辑 ---
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    
                    # A. 任务相关 Head: 只看信号分支 (Reward & End)
                    # 科学性：梯度只回传给信号分支和 Mask，倒逼模型把有用信息分拣出来
                    if name in ["reward", "end"]:
                        feat = feat_s
                    
                    # B. 重构 Head: 组合两分支特征以还原全景图像
                    # 科学性：保证世界模型对环境有全量感知
                    elif name == "decoder":
                        feat = torch.cat([feat_s, feat_d], dim=-1)
                    
                    # C. 其他辅助任务 Head (如有)
                    else:
                        feat = feat_s

                    # 梯度控制与前向
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                #CORE1
                """
                preds = {}
                for name, head in self.heads.items():
                    # When processing original data, "jumping_steps" and "accumulated_reward" are not used
                    '''
                    if name == "jumping_steps" or name == "accumulated_reward":
                        continue
                    '''
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                """
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    #assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                    
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                '''
                kl_loss = kl_loss_img
                kl_value = kl_value_img
                dyn_loss = dyn_loss_img
                rep_loss = rep_loss_img
                '''
                #CORE1
                model_loss = sum(scaled.values()) + kl_loss_s + kl_loss_d
                #model_loss = sum(scaled.values()) + kl_loss

            #CORE1
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
            #metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        # 指标上报
        metrics.update({f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()})
        #CORE1
        # --- 信号分支 (Signal) 核心指标 ---
        metrics.update({
            "kl_s": to_np(torch.mean(kl_val_s)),
            "dyn_loss_s": to_np(torch.mean(dyn_loss_s)),
            "rep_loss_s": to_np(torch.mean(rep_loss_s)),
            "post_ent_s": to_np(torch.mean(self.dynamics_s.get_dist(post_s).entropy())),
            "prior_ent_s": to_np(torch.mean(self.dynamics_s.get_dist(prior_s).entropy())),
        })

        # --- 干扰分支 (Distractor) 核心指标 ---
        # 监控干扰分支是为了确保它确实在捕获“无关噪声”
        metrics.update({
            "kl_d": to_np(torch.mean(kl_val_d)),
            "dyn_loss_d": to_np(torch.mean(dyn_loss_d)),
            "rep_loss_d": to_np(torch.mean(rep_loss_d)),
            "post_ent_d": to_np(torch.mean(self.dynamics_d.get_dist(post_d).entropy())),
            "prior_ent_d": to_np(torch.mean(self.dynamics_d.get_dist(prior_d).entropy())),
        }) 
        #CORE1

        metrics["model_loss"] = to_np(torch.mean(model_loss))       
        '''
        if zoomed_num > 0:
            metrics.update({f"zoomed_{name}_loss": to_np(torch.mean(loss)) for name, loss in losses_zoomed.items()})
        '''
        #metrics["kl_free"] = kl_free
        #metrics["dyn_scale"] = dyn_scale
        #metrics["rep_scale"] = rep_scale
        '''
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        '''
        '''
        if zoomed_num > 0:
            metrics["dyn_loss_img"] = to_np(torch.mean(dyn_loss_img))
            metrics["dyn_loss_jmp"] = to_np(torch.mean(dyn_loss_jmp))
            metrics["rep_loss_img"] = to_np(torch.mean(rep_loss_img))
            metrics["rep_loss_jmp"] = to_np(torch.mean(rep_loss_jmp))
        '''
        
        """
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            '''
            if zoomed_num > 0:
                metrics["prior_zoomed_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior_zoomed).entropy())
                )
                metrics["post_zoomed_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post_zoomed).entropy())
                )    
            '''
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )

        post = {k: v.detach() for k, v in post.items()}
        '''
        if zoomed_num > 0:
            post_zoomed = {k: v.detach() for k, v in post_zoomed.items()}
            return post, post_zoomed, context, metrics
        else:
        '''
        """
        # context 仅返回信号分支，供后续策略学习(ImagBehavior)使用
        context = dict(
            embed=embed_s,
            feat=feat_s,
            kl=kl_val_s,
            postent=self.dynamics_s.get_dist(post_s).entropy(),
        )       
        return post_s, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        
        #if not zoomed:
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        obs["heatmap"] = torch.Tensor(obs["heatmap"]).unsqueeze(-1) / 255.0
        if "action" in obs:
            '''
            original_action = obs["action"]
            zeros_array = np.zeros((original_action.shape[0], original_action.shape[1], 1), dtype=original_action.dtype)
            new_action = np.concatenate((original_action, zeros_array), axis=-1)
            obs["action"] = new_action  # [16, 64, 13]
            '''
            obs["action"] = obs["action"] # 保持原始维度 (如 12 维)

        '''
        else:
            obs["is_zoomed"] = np.zeros_like(obs["is_zoomed"])
            obs["jump"] = np.zeros_like(obs["jump"])
            obs["image"] = torch.Tensor(obs["zoomed_image"]) / 255.0
            obs["heatmap"] = torch.Tensor(obs["heatmap_on_zoomed"]).unsqueeze(-1) / 255.0
            obs["reward"] = obs["reward_on_zoomed"]
            obs['intrinsic'] = obs['intrinsic_on_zoomed']
        
            # for states after zooming, clear the action and add it to 13th dimension, and set the last dimension to 1
            if "action" in obs:
                new_action = np.zeros((obs["action"].shape[0], obs["action"].shape[1], obs["action"].shape[2] + 1), dtype=obs["action"].dtype) # [64, 16, 13]
                new_action[:, :, -1] = 1
                obs["action"] = new_action
        '''
        '''
        obs.pop("zoomed_image", None)
        obs.pop("heatmap_on_zoomed", None)
        obs.pop("reward_on_zoomed", None)
        obs.pop("intrinsic_on_zoomed", None)
        '''
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train end_head
        assert "is_terminal" in obs

        '''
        if zoomed:
            obs['is_first'] = np.zeros_like(obs['is_first'])
        '''
        '''
        obs["is_zoomed"] = torch.Tensor(obs["is_zoomed"]).unsqueeze(-1)
        obs["jump"] = torch.Tensor(obs["jump"]).unsqueeze(-1)
        obs["is_calculated"] = torch.Tensor(obs["is_calculated"]).unsqueeze(-1)
        '''
        obs["end"] = torch.Tensor(obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
    
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]

        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.jump_prob = config.jump_prob
        self.gamma_sum = [(1 - self._config.discount ** (i + 1)) / (1 - self._config.discount) for i in range(self._config.episode_max_steps)]
        self.gamma_sum = torch.tensor(self.gamma_sum, dtype=torch.float32, device=config.device)

        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        #start_zoomed,
        objective,
        #intrinsic_objective,
        #jumping_steps_predictor,
        #accumulated_reward_predictor,
        #jump_indicator,
        is_end,
    ):

        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # add post-jump state to start
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                start = {k: flatten(v) for k, v in start.items()} # [512, xx, xx]
                '''
                if start_zoomed is not None and self.jump_prob > 0.0:
                    start_zoomed = {k: flatten(v) for k, v in start_zoomed.items()} # [n, xx, xx]
                    for k, v in start.items():
                        start[k] = torch.cat((v, start_zoomed[k]), dim=0) # shape: [N = 512 + n, xx, xx]
                '''
                state_num = start['deter'].shape[0] # 512 or N

                imag_state = {} # [L, N, xx, xx]
                for k, v in start.items():
                    imag_state[k] = v.unsqueeze(0) # [1, N, xx, xx]

                action_example = self.actor(self._world_model.dynamics.get_feat(start).detach()).sample()
                action_dimension = action_example.shape[-1]

                #jump_record = torch.empty((0, state_num), device=self._config.device) # [0, N]
                imag_action = torch.empty((0, state_num, action_dimension), device=self._config.device) # [0, N, xx]

                for _ in range (self._config.imag_horizon - 1):
                    checking_state = {} # [N, xx, xx]
                    for key, tensor in imag_state.items():
                        checking_state[key] = tensor[-1, :, ...]

                    # check if checking_state can jump
                    '''
                    jump_tensor = jump_indicator(checking_state) # [N, 1]
                    end_factor = is_end(checking_state) # [N, 1]
                    indices = probability_to_bool(jump_tensor * (1.0 - end_factor), self.jump_prob).squeeze() # [N]
                    '''
                    '''
                    jump_record = torch.cat((jump_record, indices.unsqueeze(0)), dim=0) # [L, N]
                    '''
                    '''
                    jump_state = {} # [X, xx, xx]
                    for key, tensor in checking_state.items():
                        jump_state[key] = tensor[indices]
                    
                    # for states identified as requiring a jumpy transition, execute long-term imagination.
                    _, state_after_jumping, _ = self._jumpy(
                        jump_state, self.actor, 1
                    )
                    '''
                    '''
                    _, state_after_jumping, _ = self._imagine(
                        state_after_jumping, self.actor, 1
                    ) # [X, xx, xx]
                    '''
                    # For other states, execute short-term imagination.
                    _, state_after_imagination, ac = self._imagine(
                        checking_state, self.actor, 1
                    ) # [N, xx, xx]

                    # save action for this step to imag_action
                    imag_action = torch.cat((imag_action, ac.unsqueeze(0)), dim=0)

                    '''
                    for key in state_after_imagination:
                        state_after_imagination[key][indices] = state_after_jumping[key]
                    '''
                    for key in imag_state:
                        imag_state[key] = torch.cat((imag_state[key], state_after_imagination[key].unsqueeze(0)), dim=0)

                '''
                last_jump_record = torch.zeros((1, state_num), device=self._config.device) # [1, N]
                jump_record = torch.cat((jump_record, last_jump_record), dim=0) # [L, N]
                jump_num = torch.sum(jump_record)
                '''
                imag_feat = self._world_model.dynamics.get_feat(imag_state) # [L, N, xx, xx]
                inp = imag_feat[-1].detach()
                last_imag_action = self.actor(inp).sample() # [N, xx, xx]
                imag_action = torch.cat((imag_action, last_imag_action.unsqueeze(0)), dim=0) # [L, N, xx]

                #  Data augmentation (using the state after long-term transition as the starting point for a new imagination sequence)
                new_state = {}
                '''
                new_jump_tensor = jump_indicator(imag_state) # [L, N, 1]
                new_end_factor = is_end(imag_state) # [L, N, 1]
                zoom_indices = new_jump_tensor * (1.0 - new_end_factor)
                '''
                '''
                max_values, _ = zoom_indices.max(dim=0, keepdim=True)
                max_mask = (zoom_indices == max_values)
                zoom_indices *= max_mask.float()
                zoom_indices = probability_to_bool(zoom_indices, self.jump_prob).squeeze() # [L, N]
                '''
                '''
                new_num = torch.sum(zoom_indices) # Y

                for key, tensor in imag_state.items():
                    new_state[key] = tensor[zoom_indices] # [Y, xx, xx]

                _, new_state_after_jump, _ = self._jumpy(
                    new_state, self.actor, 1
                ) 

                _, new_state_after_jump, _ = self._imagine(
                    new_state_after_jump, self.actor, 1
                ) 

                new_feat, new_state_sequence, new_action = self._imagine(
                    new_state_after_jump, self.actor, self._config.imag_horizon
                ) # [L, N, xx, xx]
                '''
                '''
                new_jump_record = torch.zeros((self._config.imag_horizon, new_num), device=self._config.device) # [L, Y]

                for key, tensor in imag_state.items():
                    imag_state[key] = torch.cat((tensor, new_state_sequence[key]), dim=1) # [L, N+Y, xx, xx]
                '''
                '''
                imag_feat = torch.cat((imag_feat, new_feat), dim=1) # [L, N+Y, xx]
                imag_action = torch.cat((imag_action, new_action), dim=1) # [L, N+Y, 12]
                jump_record = torch.cat((jump_record, new_jump_record), dim=1) # [L, N+Y]
                '''
                imagination_num_tensor = torch.tensor(state_num, dtype=torch.float32, device=imag_feat.device)
                
                reward = objective(imag_feat, imag_state, imag_action)
                '''
                intrinsic_reward = intrinsic_objective(imag_feat, imag_state, imag_action)
                reward += intrinsic_reward
                '''

                actor_ent = self.actor(imag_feat).entropy() 
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                '''
                jump_sequence_record = torch.any(jump_record, dim=0, keepdim=True).repeat(jump_record.size(0), 1) # [L, N]
                jump_record_tensor = jump_record.unsqueeze(-1) # [L, N, 1]
                jump_sequence_record = jump_sequence_record.unsqueeze(-1) # [L, N, 1]
                '''
                target_TD_lambda_Return, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, is_end
                )

                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target_TD_lambda_Return,
                    weights,
                    base,
                )

                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target_TD_lambda_Return = torch.stack(target_TD_lambda_Return, dim=1)

                value_loss = -value.log_prob(target_TD_lambda_Return.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target_TD_lambda_Return, "target_TD_lambda_Return"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(imagination_num_tensor, "imagination_num"))

        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    #可视化“想象”过程
    def save_state_sequence(self, imag_feat, freq=0.1):
        """
        为 Dreamer 框架简化的想象序列可视化函数。
        移除了对 jump_record 的依赖。
        """
        if random.random() > freq:
            return

        output_dir = os.path.join(self._config.logdir, "check_obs")
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # imag_feat 形状通常为 [L, N, Feat_Size] (L为时间步，N为Batch大小)
        num_steps, batch_size, _ = imag_feat.shape
        
        # 1. 随机选择 Batch 中的一个序列进行可视化
        random_index = np.random.randint(batch_size)
        feats = imag_feat[:, random_index, :] 

        def tensor_to_image(tensor):
            array = tensor.detach().cpu().numpy().astype(np.uint8)
            return Image.fromarray(array)

        def apply_colormap_to_heatmap(heatmap_tensor, cmap='jet', vmin=0, vmax=1):
            heatmap = heatmap_tensor.detach().cpu().numpy().squeeze()
            normed_heatmap = (heatmap - vmin) / (vmax - vmin) 
            normed_heatmap = np.clip(normed_heatmap, 0, 1) 
            colormap = plt.get_cmap(cmap)
            colored_heatmap = colormap(normed_heatmap) 
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_heatmap)

        def blend_images(image, heatmap, alpha=0.5):
            image = image.convert("RGB")
            heatmap = heatmap.convert("RGB")
            return Image.blend(image, heatmap, alpha)

        # 2. 解码图像序列
        image_list = []
        for feat in feats:
            # 增加维度以匹配 Decoder 输入 [1, 1, Feat_Size]
            feat = feat.unsqueeze(0).unsqueeze(0)

            # 解码原始图像
            image_tensor = self._world_model.heads["decoder"](feat)["image"].mode()
            image_tensor = image_tensor.squeeze(0).squeeze(0)
            image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0) * 255.0
            image = tensor_to_image(image_tensor)

            # 解码热力图 (如果有)
            heatmap_tensor = self._world_model.heads["decoder"](feat)["heatmap"].mode()
            heatmap_tensor = heatmap_tensor.squeeze(0).squeeze(0)
            heatmap_tensor = torch.clamp(heatmap_tensor, min=0.0, max=1.0)
            heatmap = apply_colormap_to_heatmap(heatmap_tensor, cmap='jet', vmin=0, vmax=1)

            # 混合图像
            blended_image = blend_images(image, heatmap, alpha=0.3)
            image_list.append((image, heatmap, blended_image))

        # 3. 拼接并保存图像 (统一使用蓝色边框)
        border_width = 2
        total_width = sum(img.width for img, _, _ in image_list) + border_width * len(image_list) * 2
        max_height = max(img.height for img, _, _ in image_list) + border_width * 2
        composite_image = Image.new('RGB', (total_width, max_height * 3), color='white')

        x_offset = 0
        border_color = 'blue' # 统一边框颜色

        for i, (orig_img, heatmap, blended_img) in enumerate(image_list):
            for row_idx, img in enumerate([orig_img, heatmap, blended_img]):
                y_offset = max_height * row_idx
                # 绘制带边框的图像
                bordered_img = Image.new('RGB', (img.width + border_width * 2, img.height + border_width * 2), border_color)
                bordered_img.paste(img, (border_width, border_width))
                composite_image.paste(bordered_img, (x_offset, y_offset))
            x_offset += orig_img.width + border_width * 2

        composite_image.save(os.path.join(output_dir, f"{current_time}.png"))
        

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            
            '''
            # When interacting with the wm, the action needs to be expanded to 13 dimensions.
            zeros_tensor = torch.zeros(action.shape[0], 1).to(action.device)
            new_action = torch.cat((action, zeros_tensor), dim=-1)
            '''
            
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )

        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        if horizon == 1:
            for k, v in succ.items():
                succ[k] = v.squeeze(0)
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)
            return feats, succ, actions
        else:
            return feats, states, actions

    '''
    def _jumpy(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()

            new_action = torch.zeros(action.shape[0], action.shape[1] + 1).to(action.device)
            new_action[:, -1] = 1
            succ = dynamics.img_step(state, new_action)
            return succ, feat, action
        
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        if horizon == 1:
            for k, v in succ.items():
                succ[k] = v.squeeze(0)
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)
            return feats, succ, actions
        else:
            return feats, states, actions
    '''
            
    def _compute_target(self, imag_feat, imag_state, reward, is_end):
        fc = torch.cat((imag_feat[:-1], imag_feat[1:]), dim=-1) # [L - 1, N, 2xx]
        '''
        jumping_steps = jumping_steps_predictor(fc, None, None) # [L - 1, N, 1]
        jumping_steps = torch.cat([jumping_steps, torch.zeros_like(jumping_steps[0]).unsqueeze(0)], dim=0) # [L, N, 1]
        accumulated_reward = accumulated_reward_predictor(fc, None, None) # [L - 1, N, 1]
        accumulated_reward = torch.cat([accumulated_reward, torch.zeros_like(accumulated_reward[0]).unsqueeze(0)], dim=0) # [L, N, 1]
        accumulated_reward *= self.gamma_sum[(jumping_steps - 2).clamp(0, self._config.episode_max_steps - 1)]
        '''
        end = is_end(imag_state) # [L, N, 1]
        discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        '''
        jumping_steps = (jumping_steps - 1) * jump_record + 1
        accumulated_reward *= jump_record
        '''
        discount = discount * (1.0 - end)
        
        #Critic 的训练目标（实际回报目标）
        target_TD_lambda_Return = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[:-1],
            #end[:-1],
            #jumping_steps[:-1],
            #accumulated_reward[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        
        self.save_state_sequence(imag_feat)
        
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        return target_TD_lambda_Return, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target_TD_lambda_Return,
        weights,
        base,
        #jump_record,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target_TD_lambda_Return = torch.stack(target_TD_lambda_Return, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target_TD_lambda_Return, self.ema_vals)
            normed_target = (target_TD_lambda_Return - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target_TD_lambda_Return - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target_TD_lambda_Return - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target_TD_lambda_Return + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        
        #jump_mask = 1.0 - jump_record
        #actor_loss = -weights[:-1] * jump_mask[:-1] * actor_target
        actor_loss = -weights[:-1] * actor_target

        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1