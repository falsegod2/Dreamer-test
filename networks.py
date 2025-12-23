import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


import torch
from torch import nn
from torch import distributions as torchd
import tools
import numpy as np

class RSSM(nn.Module):
    def __init__(
        self, stoch=30, deter=200, hidden=200, rec_depth=1, discrete=False,
        act="SiLU", norm=True, mean_act="none", std_act="softplus", min_std=0.1,
        unimix_ratio=0.01, initial="learned", num_actions=None, embed=None, device=None,
    ):
        super(RSSM, self).__init__()
        # 1. 维度拆分：s(受控), z(非受控)
        self._stoch_s = stoch // 2
        self._stoch_z = stoch - self._stoch_s
        self._deter_s = deter // 2
        self._deter_z = deter - self._deter_s
        
        self._stoch, self._deter = stoch, deter
        self._hidden, self._min_std = hidden, min_std
        self._rec_depth, self._discrete = rec_depth, discrete
        act_fn = getattr(torch.nn, act)
        self._mean_act, self._std_act = mean_act, std_act
        self._unimix_ratio, self._initial = unimix_ratio, initial
        self._num_actions = num_actions + 1
        self._embed, self._device = embed, device

        # 2. 受控分支网络 (输入包含动作)
        stoch_size_s = self._stoch_s * (self._discrete if self._discrete else 1)
        self._img_in_s = self._make_layer(stoch_size_s + self._num_actions, norm, act_fn)
        self._cell_s = GRUCell(self._hidden, self._deter_s, norm=norm)
        self._img_out_s = self._make_layer(self._deter_s, norm, act_fn)
        self._obs_out_s = self._make_layer(self._deter_s + self._embed, norm, act_fn)

        # 3. 非受控分支网络 (动作无关)
        stoch_size_z = self._stoch_z * (self._discrete if self._discrete else 1)
        self._img_in_z = self._make_layer(stoch_size_z, norm, act_fn)
        self._cell_z = GRUCell(self._hidden, self._deter_z, norm=norm)
        self._img_out_z = self._make_layer(self._deter_z, norm, act_fn)
        self._obs_out_z = self._make_layer(self._deter_z + self._embed, norm, act_fn)

        # 4. 映射层与逆动力学
        self._stat_s_img = self._make_stat_layer(self._stoch_s)
        self._stat_s_obs = self._make_stat_layer(self._stoch_s)
        self._stat_z_img = self._make_stat_layer(self._stoch_z)
        self._stat_z_obs = self._make_stat_layer(self._stoch_z)
        self._inverse_dynamics = nn.Sequential(
            nn.Linear(self._deter_s * 2, self._hidden), act_fn(),
            nn.Linear(self._hidden, num_actions)
        )

        if self._initial == "learned":
            self.W = torch.nn.Parameter(torch.zeros((1, self._deter), device=torch.device(self._device)), requires_grad=True)

    def _make_layer(self, inp_dim, norm, act_fn):
        layers = [nn.Linear(inp_dim, self._hidden, bias=False)]
        if norm: layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        layers.append(act_fn()); net = nn.Sequential(*layers)
        net.apply(tools.weight_init); return net

    def _make_stat_layer(self, dim):
        l = nn.Linear(self._hidden, dim * (self._discrete if self._discrete else 2))
        l.apply(tools.uniform_weight_init(1.0)); return l

    def initial(self, batch_size):
        # 1. 初始化确定性状态 (deter)
        deter_s = torch.zeros(batch_size, self._deter_s).to(self._device)
        deter_z = torch.zeros(batch_size, self._deter_z).to(self._device)
        
        if self._initial == "learned":
            W_s, W_z = torch.split(torch.tanh(self.W), [self._deter_s, self._deter_z], -1)
            deter_s = W_s.repeat(batch_size, 1)
            deter_z = W_z.repeat(batch_size, 1)

        # 2. 构造初始状态字典
        state = {"deter_s": deter_s, "deter_z": deter_z}
        
        # --- 修复点：获取受控分支(s)的初始随机状态及分布参数 ---
        stoch_s, stats_s = self._get_stoch_and_stats_init(deter_s, "s")
        state["stoch_s"] = stoch_s
        state.update({f"s_{k}": v for k, v in stats_s.items()}) # 添加 s_logit 或 s_mean/std
        
        # --- 修复点：获取非受控分支(z)的初始随机状态及分布参数 ---
        stoch_z, stats_z = self._get_stoch_and_stats_init(deter_z, "z")
        state["stoch_z"] = stoch_z
        state.update({f"z_{k}": v for k, v in stats_z.items()}) # 添加 z_logit 或 z_mean/std
        
        return state

    # --- 辅助函数：统一获取初始随机态和参数 ---
    def _get_stoch_and_stats_init(self, deter, branch):
        net = self._img_out_s if branch == "s" else self._img_out_z
        stat_layer = self._stat_s_img if branch == "s" else self._stat_z_img
        stoch_dim = self._stoch_s if branch == "s" else self._stoch_z
        
        x = net(deter)
        stats = self._suff_stats_layer(stat_layer, x, stoch_dim)
        stoch = self.get_dist(stats).mode()
        return stoch, stats

    # --- 必须保留的核心接口：处理正常序列 ---
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(prev_state[0], prev_act, embed, is_first),
            (action, embed, is_first), (state, state),
        )
        return {k: swap(v) for k, v in post.items()}, {k: swap(v) for k, v in prior.items()}

    # --- 必须保留的核心接口：处理缩放跳跃序列 ---
    def observe_zoomed(self, embed_z, action_z, is_f_z, rely_post, rely_prior):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed_z, action_z, is_f_z = swap(embed_z), swap(action_z), swap(is_f_z)
        rely_p = {k: swap(v) for k, v in rely_post.items()}
        rely_pr = {k: swap(v) for k, v in rely_prior.items()}
        # 注意：这里调用的是重构后的 obs_step，它会自动处理解耦状态
        post_z, prior_z = tools.static_scan_zoomed(
            lambda r_s, p_a, e_z, i_f_z: self.obs_step(r_s, p_a, e_z, i_f_z),
            (action_z, embed_z, is_f_z), (rely_p, rely_pr),
        )
        return {k: swap(v) for k, v in post_z.items()}, {k: swap(v) for k, v in prior_z.items()}

    # --- 必须保留的核心接口：闭眼想象 ---
    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)[0]
        return {k: swap(v) for k, v in prior.items()}

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(self._device)
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_s = self.initial(len(is_first))
            for k, v in prev_state.items():
                is_f_r = torch.reshape(is_first, is_first.shape + (1,) * (len(v.shape) - len(is_first.shape)))
                prev_state[k] = v * (1.0 - is_f_r) + init_s[k] * is_f_r

        prior = self.img_step(prev_state, prev_action)
        # 受控后验
        stats_s = self._suff_stats_layer(self._stat_s_obs, self._obs_out_s(torch.cat([prior["deter_s"], embed], -1)), self._stoch_s)
        stoch_s = self.get_dist(stats_s).sample() if sample else self.get_dist(stats_s).mode()
        # 非受控后验
        stats_z = self._suff_stats_layer(self._stat_z_obs, self._obs_out_z(torch.cat([prior["deter_z"], embed], -1)), self._stoch_z)
        stoch_z = self.get_dist(stats_z).sample() if sample else self.get_dist(stats_z).mode()

        post = {"stoch_s": stoch_s, "deter_s": prior["deter_s"], "stoch_z": stoch_z, "deter_z": prior["deter_z"]}
        # 合并分布参数以便后续计算 KL (添加前缀防止冲突)
        post.update({f"s_{k}": v for k, v in stats_s.items()})
        post.update({f"z_{k}": v for k, v in stats_z.items()})
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # 1. 受控分支演化 (带动作)
        prev_s = prev_state["stoch_s"]
        if self._discrete: prev_s = prev_s.reshape(list(prev_s.shape[:-2]) + [-1])
        x_s, deter_s = self._cell_s(self._img_in_s(torch.cat([prev_s, prev_action], -1)), [prev_state["deter_s"]])
        stats_s = self._suff_stats_layer(self._stat_s_img, self._img_out_s(x_s), self._stoch_s)
        stoch_s = self.get_dist(stats_s).sample() if sample else self.get_dist(stats_s).mode()

        # 2. 非受控分支演化 (无动作)
        prev_z = prev_state["stoch_z"]
        if self._discrete: prev_z = prev_z.reshape(list(prev_z.shape[:-2]) + [-1])
        x_z, deter_z = self._cell_z(self._img_in_z(prev_z), [prev_state["deter_z"]])
        stats_z = self._suff_stats_layer(self._stat_z_img, self._img_out_z(x_z), self._stoch_z)
        stoch_z = self.get_dist(stats_z).sample() if sample else self.get_dist(stats_z).mode()

        prior = {"stoch_s": stoch_s, "deter_s": deter_s[0], "stoch_z": stoch_z, "deter_z": deter_z[0]}
        prior.update({f"s_{k}": v for k, v in stats_s.items()})
        prior.update({f"z_{k}": v for k, v in stats_z.items()})
        return prior

    def _suff_stats_layer(self, layer, x, dim):
        x = layer(x)
        if self._discrete: return {"logit": x.reshape(list(x.shape[:-1]) + [dim, self._discrete])}
        mean, std = torch.split(x, [dim] * 2, -1)
        mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
        std = {"softplus": lambda: torch.softplus(std), "abs": lambda: torch.abs(std + 1)}[self._std_act]()
        return {"mean": mean, "std": std + self._min_std}

    def get_dist(self, stats):
        if self._discrete:
            return torchd.independent.Independent(tools.OneHotDist(stats["logit"], unimix_ratio=self._unimix_ratio), 1)
        return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(stats["mean"], stats["std"]), 1))

    def get_feat(self, state):
        s_stoch = state["stoch_s"]
        z_stoch = state["stoch_z"]
        if self._discrete:
            # 修改点：将 -1 替换为明确的维度乘积 [stoch_dim * discrete_num]
            # 这样即使 Batch 维度为 0，PyTorch 也能通过明确的末尾维度正常处理
            s_stoch = s_stoch.reshape(list(s_stoch.shape[:-2]) + [self._stoch_s * self._discrete])
            z_stoch = z_stoch.reshape(list(z_stoch.shape[:-2]) + [self._stoch_z * self._discrete])
        return torch.cat([s_stoch, state["deter_s"], z_stoch, state["deter_z"]], -1)

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        # 适配双分支的 KL 损失计算
        def get_branch_dist(state, prefix):
            # 从合并后的 state 字典中提取对应分支的参数
            branch_stats = {k[2:]: v for k, v in state.items() if k.startswith(f"{prefix}_")}
            return self.get_dist(branch_stats)

        kld = torchd.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # 受控分支与非受控分支分别计算 KL
        kl_s = kld(get_branch_dist(post, "s"), get_branch_dist(sg(prior), "s"))
        kl_z = kld(get_branch_dist(post, "z"), get_branch_dist(sg(prior), "z"))
        
        # 按照 DreamerV3 逻辑计算 rep_loss 和 dyn_loss
        rep_loss = torch.clip(kl_s + kl_z, min=free)
        
        dyn_kl_s = kld(get_branch_dist(sg(post), "s"), get_branch_dist(prior, "s"))
        dyn_kl_z = kld(get_branch_dist(sg(post), "z"), get_branch_dist(prior, "z"))
        dyn_loss = torch.clip(dyn_kl_s + dyn_kl_z, min=free)

        return dyn_scale * dyn_loss + rep_scale * rep_loss, (kl_s + kl_z), dyn_loss, rep_loss

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
