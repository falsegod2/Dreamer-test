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
    def __init__(
        self,
        stoch=30,       # z_t: 随机状态的维度 (Stochastic) -> 代表对当前时刻的模糊理解
        deter=200,      # h_t: 确定性状态的维度 (Deterministic) -> GRU的隐藏层，代表长期记忆
        hidden=200,     # 神经网络中间层的单元数
        rec_depth=1,    # GRU的层数
        discrete=False, # 是否使用离散分布 (DreamerV3 通常用离散分布，更稳定)
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        # ... 其他参数主要用于配置激活函数、初始化等 ...
        num_actions=None, # 动作空间大小
        embed=None,       # 图像编码后的特征维度
        device=None,
        action_free=False,  # <--- [新增] 标记是否为 Action-Free 分支
    ):
        super(RSSM, self).__init__()
        # ... 保存参数 ...
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._action_free = action_free  # <--- [记录]
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions + 1 # +1 是因为有时会填充一个空动作
        self._embed = embed
        self._device = device

        inp_layers = []
        """
        --- 1. 想象输入层 (Prior Network) ---
        作用：处理 [上一步随机状态 z + 动作 a] -> 准备输入给 GRU
        也就是：根据“现状”和“动作”，推测“变化”
        """
        # --- [修正] 维度计算逻辑 ---
        if self._action_free:
            # Z分支 (不可控): 不接收动作，所以没有 num_actions，也没有 +1
            if self._discrete:
                inp_dim = self._stoch * self._discrete
            else:
                inp_dim = self._stoch
        else:
            # S分支 (可控): 接收动作
            # 必须加上 LS-Imagine 特有的 "+1" (因为 preprocess 里拼接了 extra dim)
            if self._discrete:
                inp_dim = self._stoch * self._discrete + (num_actions or 0) + 1
            else:
                inp_dim = self._stoch + (num_actions or 0) + 1
        # -------------------------
        
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
    
        #定义全连接层+归一化+激活函数
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)

        """
        # --- 2. 核心记忆单元 (GRU Cell) ---
        # 作用：更新长期记忆 h_t。 h_t = GRU(h_{t-1}, 变化输入)
        """
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        """
        # --- 3. 想象输出层 (Prior Output) ---
        # 作用：基于 GRU 的记忆 h_t，预测下一步的随机状态 z_t (先验/闭眼猜)
        """
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        """
        # --- 4. 观察输出层 (Posterior Output) ---
        # 作用：基于 [记忆 h_t + 真实图像特征 embed]，修正随机状态 z_t (后验/睁眼看)
        # 输入维度 = 记忆(deter) + 眼睛看到的(embed)
        """
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        # ... 归一化和激活 ...
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        """
        # --- 5. 分布映射层 (Heads) ---
        # 作用：把上面神经网络输出的 hidden features 映射成概率分布的参数 (logits 或 mean/std)
        # 分别用于生成“想象的 z” (imgs_stat) 和 “观察到的 z” (obs_stat)
        """
        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        # ... 连续分布的处理 ...
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    # =========================================================================
    # 初始化函数
    # =========================================================================
    def initial(self, batch_size):
        # ... 初始化 h_0 ...
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        # ... 初始化 z_0 ...
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
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
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)
        
    # =========================================================================
    # 核心功能1：观察序列 (Training 阶段用)
    # =========================================================================
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        # === [关键修复] 保证 action 永远不是 None ===
        if action is None:
            # 如果是 Z 分支，造一个假的 action 占位符
            # 形状参考 embed: (Batch, Time, ...) -> Action: (Batch, Time, 1)
            # 这里的 1 只是占位，RSSM.img_step 会根据 self._action_free=True 忽略它
            batch, time_steps = embed.shape[:2]
            action = torch.zeros((batch, time_steps, 1), device=embed.device)
        # ==========================================

        if state is None:
            state = self.initial(action.shape[0])
            
        embed, action, is_first = swap(embed), swap(action), swap(is_first)

        post, prior = tools.static_scan(
            # 注意这里使用 *args 接收，你在上一步改对了，这里保持
            lambda prev, *args: self.obs_step(prev[0], *args),
            (action, embed, is_first),
            (state, state),
        )

        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    # =========================================================================
    # 核心功能2：LS-Imagine 特有的 Zoom 观察
    # =========================================================================
    def observe_zoomed(self, embed_zoomed, action_zoomed, is_first_zoomed, rely_post, rely_prior):
        """
        这是论文特有的。处理那些经过 Zoom-In (放大) 处理的“跳跃”数据。
        逻辑和 observe 基本一样，但是它依赖于主分支的状态 (rely_post, rely_prior) 作为上下文。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed_zoomed, action_zoomed, is_first_zoomed = swap(embed_zoomed), swap(action_zoomed), swap(is_first_zoomed)

        rely_post = {k: swap(v) for k, v in rely_post.items()}
        rely_prior = {k: swap(v) for k, v in rely_prior.items()}

        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post_zoomed, prior_zoomed = tools.static_scan_zoomed(
            lambda rely_state, prev_act, embed_zoomed, is_first_zoomed: self.obs_step(
                rely_state, prev_act, embed_zoomed, is_first_zoomed
            ),
            (action_zoomed, embed_zoomed, is_first_zoomed), 
            (rely_post, rely_prior),
        )

        post_zoomed = {k: swap(v) for k, v in post_zoomed.items()}
        prior_zoomed = {k: swap(v) for k, v in prior_zoomed.items()}

        return post_zoomed, prior_zoomed

    # =========================================================================
    # 核心功能3：纯想象 (Actor Training 阶段用)
    # =========================================================================
    def imagine_with_action(self, action, state):
        """
        输入：当前状态 state，和一串计划要做的动作 action
        输出：想象出的未来状态 prior
        注意：这里没有 embed 输入，因为是“闭眼”想未来。
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        # 循环调用 img_step (只预测，不修正)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior



    # =========================================================================
    # 原子操作 A：单步观察 (Posterior)
    # =========================================================================
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        这一步发生了什么：
        1. 此时时刻是 t。
        2. 我有 t-1 时刻的状态 (prev_state) 和动作 (prev_action)。
        3. 我看到了 t 时刻的真实图片特征 (embed)。
        """
        # 1. 检查是否是 Episode 的第一步
        # 如果 is_first=True，说明上一步的状态是上一个回合的，不能用。
        # 需要重置为 initial 状态 (全0)
        if prev_action is not None and prev_action.shape[-1] != self._num_actions:
            shape = prev_action.shape
            new_shape = list(shape[:-1]) + [1]
            zero_tensor = torch.zeros(*new_shape).to(prev_action.device)
            prev_action = torch.cat((prev_action, zero_tensor), dim=-1)


        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
            # prev_action.requires_grad_()
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # 2. 先做“闭眼预测” (Prior)
        # 算出：基于历史，我认为现在应该是什么样？
        prior = self.img_step(prev_state, prev_action)
        # 3. 融合“预测”和“真实所见”
        # 把 [预测的记忆 deter] 和 [真实看到的 embed] 拼起来
        x = torch.cat([prior["deter"], embed], -1)
        # 4. 通过神经网络，计算“睁眼修正”后的分布 (Posterior)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x) # 得到分布参数(logits)

        # 5. 从分布中采样得到随机状态 z (stoch)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        # post 包含：修正后的 z (stoch), 记忆 h (deter), 分布参数
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior
    
    # =========================================================================
    # 原子操作 B：单步想象 (Prior)
    # =========================================================================
    def img_step(self, prev_state, prev_action, sample=True):
        """
        Input:
            prev_state: 上一时刻的潜在状态 (包含 stoch, deter)
            prev_action: 上一时刻采取的动作 (Action-Free 分支时应为 None 或被忽略)
        Output:
            prior: 对当前时刻状态的先验预测 (不包含后验信息)
        """
        # ==========================================================
        # [关键修复] 必须先从 prev_state 中提取 stoch
        # 无论后面分支如何走，这个变量都必须存在
        # ==========================================================
        prev_stoch = prev_state["stoch"]
        
        # 1. 如果是离散状态，展平特征
        if self._discrete:
            # 确保 shape 逻辑正确
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        
        # 2. 根据分支类型决定输入内容
        if self._action_free:
            # === Z 分支 (不可控) ===
            # 强制忽略动作，只看上一时刻的状态
            x = prev_stoch
        else:
            # === S 分支 (可控) ===
            # 必须包含动作
            if prev_action is None:
                raise ValueError("Error: Action-Conditioned RSSM requires a valid action input!")
            
            # 拼接: [State, Action]
            x = torch.cat([prev_stoch, prev_action], -1)

        # 3. 线性层特征提取
        x = self._img_in_layers(x)
        
        # 4. GRU 循环层推演
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter]) 
        deter = deter[0] 

        # 5. 输出先验分布参数
        x = self._img_out_layers(x)
        stats = self._suff_stats_layer("ims", x)
        
        # 6. 采样得到随机状态 z_t
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
            
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior


    # =========================================================================
    # 辅助功能
    # =========================================================================
    def get_feat(self, state):
        """
        把 h (deter) 和 z (stoch) 拼起来。
        这是为了给下游的 Actor/Critic 网络提供完整的信息。
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        """
        把神经网络输出的 logits 转换成 PyTorch 的分布对象。
        方便计算 entropy (熵) 或 log_prob (对数概率)。
        """
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
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
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
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    # =========================================================================
    # 损失函数
    # =========================================================================
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """
        计算 KL 散度 Loss。
        目的：让“闭眼预测 (prior)”尽可能接近“睁眼观察 (post)”。
        这样以后即使闭着眼 (Imagine)，也能对世界有准确的认知。
        """
        
        kld = torchd.kl.kl_divergence

        # sg (stop gradient): 停止梯度传播
        # Dreamer 算法通常使用这种技巧来分别优化 Representation 和 Dynamics
        # Representation Loss: 让后验去接近先验（让观察更符合逻辑）
        # Dynamics Loss: 让先验去接近后验（让预测更符合现实）
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # 计算两个分布之间的距离
        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )

        # Free bits: 如果误差小于 free (例如 1.0)，就不惩罚。防止模型坍塌。
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        """
        # 加权求和
        # 这个loss值能够同时更新生成post的Encoder
          以及生成prior的Dynamics
        """
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
