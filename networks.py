import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools

# --- [新增] 逆动力学预测头 ---
class InverseHead(nn.Module):
    def __init__(self, inp_dim, out_dim, layers, units, act=nn.SiLU):
        super(InverseHead, self).__init__()
        self._layers = nn.Sequential()
        current_dim = inp_dim
        for i in range(layers):
            self._layers.add_module(f"linear_{i}", nn.Linear(current_dim, units))
            self._layers.add_module(f"act_{i}", act())
            current_dim = units
        self._layers.add_module("out", nn.Linear(current_dim, out_dim))

    def forward(self, feat_prev, feat_curr):
        # 拼接前后两帧的特征 (Batch, T, Feat*2)
        x = torch.cat([feat_prev, feat_curr], dim=-1)
        return self._layers(x)

class RSSM(nn.Module):
    """
    循环状态空间模型 (Recurrent State Space Model).
    
    它是 Dreamer 和 LS-Imagine 的核心“大脑”，包含两个关键过程：
    1. 想象 (Imagination / Prior): p(z_t | h_t)
       - 俗称“闭眼预测”。根据历史记忆预测下一刻会发生什么。
    2. 观察 (Observation / Posterior): q(z_t | h_t, x_t)
       - 俗称“睁眼修正”。看到真实画面后，修正自己的预测。
       
    支持:
    - 离散/连续 Latent Space。
    - Iso-Dream++ 双流架构 (S-Branch 可控 / Z-Branch 不可控)。
    """

    def __init__(
        self,
        stoch=30,       # z_t: 随机状态维度 (代表当前时刻的随机性/模糊性)
        deter=200,      # h_t: 确定性状态维度 (GRU隐藏层，代表长期记忆)
        hidden=200,     # MLP 中间层维度
        rec_depth=1,    # GRU 层数
        discrete=False, # 是否使用离散分布 (DreamerV3/MinDojo 推荐 True)
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None, # 动作空间维度
        embed=None,       # CNN 编码后的图像特征维度
        device=None,
        action_free=False,  # [Iso-Dream] 是否为不可控分支 (Z分支)
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._discrete = discrete
        self._action_free = action_free
        self._num_actions = num_actions + 1 if num_actions else 0 # +1 for empty/pad action
        self._embed = embed
        self._device = device
        self._unimix_ratio = unimix_ratio
        self._initial = initial

        # 激活函数配置
        self._act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._norm = norm

        # --- 构建网络层 ---
        self._build_components()
        
        # --- 打印结构摘要 ---
        self._print_summary()

    def _build_components(self):
        """构建 RSSM 的四个核心组件"""
        
        # 1. 想象输入层 (Img In): 处理 [z_{t-1}, a_{t-1}] -> GRU 输入
        if self._action_free:
            # Z分支 (不可控): 仅依赖上一时刻状态，不看动作
            inp_dim = self._stoch * (self._discrete or 1)
        else:
            # S分支 (可控): 依赖上一时刻状态 + 动作
            inp_dim = self._stoch * (self._discrete or 1) + self._num_actions
            
        self._img_in_layers = self._make_mlp_layers(inp_dim, self._hidden)

        # 2. 循环记忆单元 (Cell): 更新长期记忆 h_t
        self._cell = GRUCell(self._hidden, self._deter, norm=self._norm)

        # 3. 想象输出头 (Img Out / Prior): h_t -> z_t (Prior)
        # "根据记忆猜测现在的情况"
        self._img_out_layers = self._make_mlp_layers(self._deter, self._hidden)
        self._imgs_stat_layer = self._make_dist_head(self._hidden)

        # 4. 观察输出头 (Obs Out / Posterior): [h_t, embed_t] -> z_t (Posterior)
        # "结合记忆和真实看到的画面，确定现在的情况"
        self._obs_out_layers = self._make_mlp_layers(self._deter + self._embed, self._hidden)
        self._obs_stat_layer = self._make_dist_head(self._hidden)

        # 初始状态参数
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def _make_mlp_layers(self, in_dim, out_dim):
        """辅助函数：构建 MLP 层 (Linear + Norm + Act)"""
        layers = []
        layers.append(nn.Linear(in_dim, out_dim, bias=False))
        if self._norm:
            layers.append(nn.LayerNorm(out_dim, eps=1e-03))
        layers.append(self._act())
        model = nn.Sequential(*layers)
        model.apply(tools.weight_init)
        return model

    def _make_dist_head(self, in_dim):
        """辅助函数：构建分布参数预测头"""
        if self._discrete:
            # 输出 logits: [stoch, classes]
            out_dim = self._stoch * self._discrete
        else:
            # 输出 mean, std: 2 * stoch
            out_dim = 2 * self._stoch
        
        layer = nn.Linear(in_dim, out_dim)
        layer.apply(tools.uniform_weight_init(1.0))
        return layer

    def _print_summary(self):
        """在命令行输出 RSSM 结构信息"""
        type_str = "Action-Free (Z-Branch)" if self._action_free else "Action-Conditioned (S-Branch)"
        print(f"  [RSSM] Initialized {type_str}")
        print(f"         Latent: Stoch={self._stoch}, Deter={self._deter}, Discrete={self._discrete}")

    # =========================================================================
    # 核心初始化逻辑
    # =========================================================================
    def initial(self, batch_size):
        """生成初始的隐状态 (通常是0或者学习到的参数)"""
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
            
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            # 如果是 learned，使用 Parameter W 来填充 deter
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    # =========================================================================
    # 核心功能 1: 序列观察 (用于训练 World Model)
    # =========================================================================
    def observe(self, embed, action, is_first, state=None):
        """
        处理整个时间序列数据。
        Input: 
            embed: (Batch, Time, Embed_Dim) - 真实图像特征
            action: (Batch, Time, Act_Dim)
        Output:
            post: 后验状态序列 (用于重建图像，计算 Loss)
            prior: 先验状态序列 (用于计算 KL Loss)
        """
        # (Batch, Time, ...) -> (Time, Batch, ...) 为了循环处理
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        # [Iso-Dream 修正] 如果是 Z 分支，外部可能传入 None action
        if action is None:
            # 创建占位符动作，确保 static_scan 格式正确
            batch, time_steps = embed.shape[:2]
            action = torch.zeros((batch, time_steps, 1), device=embed.device)

        if state is None:
            state = self.initial(action.shape[0])
            
        # 转换维度
        embed, action, is_first = swap(embed), swap(action), swap(is_first)

        # === 核心循环 ===
        # 这一步会依次调用 self.obs_step 处理每一个时间步
        post, prior = tools.static_scan(
            lambda prev, *args: self.obs_step(prev[0], *args),
            (action, embed, is_first),
            (state, state),
        )

        # (Time, Batch) -> (Batch, Time) 转回来
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    # =========================================================================
    # 核心功能 2: Zoomed 观察 (用于 LS-Imagine 长视距训练)
    # =========================================================================
    def observe_zoomed(self, embed_zoomed, action_zoomed, is_first_zoomed, rely_post, rely_prior):
        """
        LS-Imagine 特有功能。
        基于 Short-term 分支的状态 (rely_post)，使用 Zoomed 图像进行推演。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        embed_zoomed = swap(embed_zoomed)
        action_zoomed = swap(action_zoomed)
        is_first_zoomed = swap(is_first_zoomed)
        
        rely_post = {k: swap(v) for k, v in rely_post.items()}
        rely_prior = {k: swap(v) for k, v in rely_prior.items()}

        # static_scan_zoomed 是特制的工具函数，支持传入 context (rely_state)
        post_zoomed, prior_zoomed = tools.static_scan_zoomed(
            lambda rely_state, prev_act, emb, first: self.obs_step(
                rely_state, prev_act, emb, first
            ),
            (action_zoomed, embed_zoomed, is_first_zoomed), 
            (rely_post, rely_prior),
        )

        post_zoomed = {k: swap(v) for k, v in post_zoomed.items()}
        prior_zoomed = {k: swap(v) for k, v in prior_zoomed.items()}

        return post_zoomed, prior_zoomed

    # =========================================================================
    # 核心功能 3: 纯想象 (用于训练 Actor)
    # =========================================================================
    def imagine_with_action(self, action, state):
        """
        在 Latent Space 中闭眼推演。只使用 Action，不使用图像。
        用于 Actor-Critic 训练。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        assert isinstance(state, dict), state

        # 循环调用 img_step (Prior Step)
        prior = tools.static_scan(self.img_step, [action], state)
        
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    # =========================================================================
    # 原子操作 A: 观察步 (Posterior Step)
    # =========================================================================
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        单步观察更新：
        1. 拿到上一步的状态 (prev_state) 和动作 (prev_action)。
        2. 先闭眼预测当前状态 (Prior)。
        3. 再看一眼真实图像特征 (embed)，修正得到当前状态 (Posterior)。
        """
        # 1. 检查是否需要重置状态 (Episode 开始)
        # 如果 is_first=1，说明这是新回合，忽略上一步的状态，重置为 initial
        if prev_action is not None and prev_action.shape[-1] != self._num_actions:
             # Action 维度对齐 (Pad 0)
             shape = list(prev_action.shape[:-1]) + [self._num_actions - prev_action.shape[-1]]
             prev_action = torch.cat([prev_action, torch.zeros(shape, device=prev_action.device)], -1)

        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= (1.0 - is_first) # Mask action
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                # 混合：如果是 first，用 init_state；否则用 prev_state
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # 2. 闭眼预测 (Prior)
        prior = self.img_step(prev_state, prev_action, sample)

        # 3. 睁眼修正 (Posterior)
        # 输入: [Prior 记忆 deter, 真实图像特征 embed]
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        
        # 得到分布参数 (logits 或 mean/std)
        stats = self._suff_stats_layer("obs", x)
        
        # 采样得到后验随机状态 z_t
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
            
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # =========================================================================
    # 原子操作 B: 想象步 (Prior Step)
    # =========================================================================
    def img_step(self, prev_state, prev_action, sample=True):
        """
        单步想象推演：
        h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
        z_t ~ Prior(h_t)
        """
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)

        # 1. 准备输入
        if self._action_free:
            # Z 分支: 忽略动作
            x = prev_stoch
        else:
            # S 分支: 状态 + 动作
            if prev_action is None:
                 raise ValueError("Action-Conditioned RSSM requires action input")
            x = torch.cat([prev_stoch, prev_action], -1)

        # 2. 特征提取 & GRU 更新
        x = self._img_in_layers(x)
        
        prev_deter = prev_state["deter"]
        x, deter = self._cell(x, [prev_deter])
        deter = deter[0] # 新的确定性状态 h_t

        # 3. 预测先验分布 (Prior Distribution)
        x = self._img_out_layers(x)
        stats = self._suff_stats_layer("ims", x)

        # 4. 采样先验随机状态 z_t
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
            
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    # =========================================================================
    # 工具函数
    # =========================================================================
    def get_feat(self, state):
        """提取特征供 Actor/Critic 使用: Concat[z_t, h_t]"""
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        """获取 PyTorch 分布对象"""
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist
    
    def get_stoch(self, deter):
        """仅根据 deter (h) 获取 stoch (z) 的模式"""
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        """根据 hidden 计算分布参数 (logits 或 mean/std)"""
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            # 连续分布逻辑 (略微复杂，涉及std处理)
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            # ... 激活函数处理 ...
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """
        计算 KL 散度 Loss。
        Goal: 让 Prior (想象) 尽可能接近 Posterior (现实)。
        """
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        # sg = Stop Gradient
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # 1. Representation Loss: 让后验去接近先验 (防止后验作弊，只看图不看记忆)
        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        
        # 2. Dynamics Loss: 让先验去接近后验 (训练想象能力)
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )

        # Free Bits: 容忍微小的误差，防止后验坍缩
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            if self._dist == "normal_std_fixed":
                mean = torch.sigmoid(mean)
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
