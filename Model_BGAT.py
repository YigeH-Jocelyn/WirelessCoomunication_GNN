import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLAMP_MIN = -200.0
CLAMP_MAX =  200.0
########################################
# 1) BGATAttention: The GAT Layer Component
########################################
class BGATAttention(nn.Module):
    def __init__(self, user_dim, ant_dim, edge_dim, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # self.norm_user = nn.LayerNorm(hidden_dim)
        # self.norm_ant = nn.LayerNorm(hidden_dim)
        # self.norm_edge = nn.LayerNorm(edge_dim)
        self.leaky = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(0.2)

        # self.user_projs = nn.ModuleList([nn.Linear(user_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        # self.ant_projs = nn.ModuleList([nn.Linear(ant_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        # self.edge_projs = nn.ModuleList([nn.Linear(edge_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.user_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(user_dim, self.head_dim, bias=False),
                nn.LayerNorm(self.head_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_heads)])
        self.ant_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ant_dim, self.head_dim, bias=False),
                nn.LayerNorm(self.head_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_heads)])
        self.edge_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_dim, self.head_dim, bias=False),
                nn.LayerNorm(self.head_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_heads)])
        self.attn_vecs = nn.ParameterList([nn.Parameter(torch.Tensor(self.head_dim)) for _ in range(num_heads)])

        self.residual_proj = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.residual_proj_ant = nn.Sequential(
            nn.Linear(ant_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        for seq in (self.user_projs + self.ant_projs + self.edge_projs):
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        a=0.2,
                        mode='fan_in',
                        nonlinearity='leaky_relu'
                    )
                    # layer.weight.data.mul_(0.5)
        for module in [self.residual_proj, self.residual_proj_ant]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.2,
                        mode='fan_in',
                        nonlinearity='leaky_relu'
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        for attn_vec in self.attn_vecs:
            nn.init.kaiming_normal_(attn_vec.view(1, -1), a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        # for seq in (self.user_projs + self.ant_projs + self.edge_projs):
        #     for layer in seq:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.xavier_uniform_(layer.weight)
        # for module in (self.residual_proj, self.residual_proj_ant):
        #     for layer in module:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.xavier_uniform_(layer.weight)
        #             if layer.bias is not None:
        #                 nn.init.zeros_(layer.bias)
        # for attn_vec in self.attn_vecs:
        #     nn.init.xavier_uniform_(attn_vec.view(1, -1))

    def forward(self, user_feats, ant_feats, edge_feats):
        head_outputs_user = []
        head_outputs_ant = []
        for k in range(self.num_heads):
            U = self.user_projs[k](user_feats) # shape: (Batch, M, head_dim)
            U = U.unsqueeze(2) # (Batch, M, 1, head_dim)
            A = self.ant_projs[k](ant_feats) # shape: (B, N, head_dim)
            A = A.unsqueeze(1) # (Batch, 1, N, head_dim)
            E = self.edge_projs[k](edge_feats) # shape: (Batch, M, N, head_dim)

            intermediate = self.leaky(U + A + E)
            attn_score = torch.einsum('bmnd,d->bmn', intermediate, self.attn_vecs[k]) * self.scale # (Batch, M, N, head_dim)
            attn_score = attn_score - attn_score.max(dim=-1, keepdim=True)[0]

            alpha = F.softmax(attn_score, dim=-1) # (Batch, M, N)

            user_out_k = (A * alpha.unsqueeze(-1)).sum(dim=2) # (Batch, M, head_dim)
            ant_out_k = (U * alpha.unsqueeze(-1)).sum(dim=1) # (Batch, N, head_dim)

            head_outputs_user.append(user_out_k)
            head_outputs_ant.append(ant_out_k)

        user_out = torch.cat(head_outputs_user, dim=-1) + self.residual_proj(user_feats) # (B, M, num_heads*head_dim)
        ant_out = torch.cat(head_outputs_ant, dim=-1) + self.residual_proj_ant(ant_feats) # (B, N, num_heads*head_dim)

        user_out = F.relu(user_out)
        ant_out = F.relu(ant_out)

        return user_out, ant_out

########################################
# 2) BGATBlock: GAT, MLP, and Readout (for both δ and power)
########################################
class BGATBlock(nn.Module):
    def __init__(self, user_dim, ant_dim, edge_dim, hidden_dim, num_heads,
                 waveguide_bound, delta_min, H, Pmax, N, M):
        super().__init__()
        self.attention = BGATAttention(user_dim, ant_dim, edge_dim, hidden_dim, num_heads)
        self.layer_norm_user = nn.LayerNorm(hidden_dim)
        self.layer_norm_ant = nn.LayerNorm(hidden_dim)
        self.layer_norm_delta = nn.LayerNorm(N)
        self.layer_norm_power = nn.LayerNorm(N)

        self.mlp_ant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2, bias=True),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.1)
        )
        self.raw_readout_delta = nn.Sequential(
            nn.Linear(N, 2 * N),
            nn.ReLU(),
            nn.Linear(2 * N, N)
        )
        self.raw_readout_power = nn.Sequential(
            nn.Linear(N, 2 * N),
            nn.ReLU(),
            nn.Linear(2 * N, N)
        )

        for module in self.mlp_ant:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # module.weight.data.mul_(0.5)
                nn.init.constant_(module.bias, 0.1)

        for module in self.raw_readout_delta:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # module.weight.data.mul_(0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.raw_readout_power:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # module.weight.data.mul_(0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # for module in self.mlp_user:
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         nn.init.zeros_(module.bias)
        # for module in self.mlp_ant:
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         nn.init.zeros_(module.bias)
        #
        # for module in self.raw_readout_delta:
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)
        # for module in self.raw_readout_power:
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)

        self.dropout = nn.Dropout(0.1)
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H
        self.Pmax = Pmax
        self.N = N
        self.M = M

    def forward(self, user_feats_init, ant_feats, edge_feats):
        # 1) Apply GAT with **initial user features**
        u_attn, a_attn = self.attention(user_feats_init, ant_feats, edge_feats)
        a_attn = self.layer_norm_ant(a_attn)

        # 2) Process via MLPs
        ant_out_2d = self.dropout(self.mlp_ant(a_attn))
        ant_out_2d = torch.clamp(ant_out_2d, CLAMP_MIN, CLAMP_MAX)

        raw_delta = self.raw_readout_delta(ant_out_2d[..., 0])
        raw_power = self.raw_readout_power(ant_out_2d[..., 1])
        raw_delta = torch.clamp(raw_delta, min=0.0, max=1e2)
        raw_power = torch.nan_to_num(raw_power, nan=0.0, posinf=1, neginf=0)

        # 3) First Readout NN (Delta)
        Bmax_val = 2 * self.waveguide_bound - (self.N - 1) * self.delta_min
        delta_aux = F.relu(raw_delta)
        sum_delta = delta_aux.sum(dim=1, keepdim=True)
        scale_factor_delta = Bmax_val / torch.maximum(
            torch.tensor(Bmax_val, device=sum_delta.device),
            sum_delta
        )
        scaled_delta = scale_factor_delta * delta_aux
        scaled_delta = torch.clamp(scaled_delta, 0.0, 1e2)

        # 4) Compute New Positions
        x = torch.zeros_like(scaled_delta)
        x[:, 0] = scaled_delta[:, 0] - self.waveguide_bound
        for n in range(1, self.N):
            x[:, n] = x[:, n - 1] + scaled_delta[:, n] + self.delta_min - self.waveguide_bound
        positions = torch.stack([x, torch.zeros_like(x), torch.full_like(x, self.H)], dim=-1)

        # 5) Second Readout NN (Power)
        eps = 1e-3
        power_aux = F.relu(raw_power)
        power_aux = torch.nan_to_num(power_aux, nan=0.0, posinf=1, neginf=0)
        power_aux = torch.clamp(power_aux, 0.0, 1)

        sum_power = power_aux.sum(dim=1, keepdim=True)
        sum_power = torch.nan_to_num(sum_power, nan=eps, posinf=eps, neginf=-eps)
        denom = torch.clamp(sum_power, min=eps)

        pmax_tensor = torch.tensor(self.Pmax, device=sum_power.device, dtype=power_aux.dtype).detach()
        pmax_tensor = torch.nan_to_num(pmax_tensor, nan=1.0, posinf=1, neginf=0.0)

        scale_factor_power = pmax_tensor / denom
        scale_factor_power = torch.nan_to_num(scale_factor_power, nan=1.0, posinf=1.0, neginf=0)
        scale_factor_power = scale_factor_power.detach()

        scaled_power = scale_factor_power * power_aux
        scaled_power = torch.nan_to_num(scaled_power, nan=0.0, posinf=1, neginf=0)
        scaled_power = torch.clamp(scaled_power, 0.0, 1)

        # scale_factor_power = self.Pmax / torch.maximum(
        #     torch.tensor(self.Pmax, device=sum_power.device),
        #     sum_power
        # )
        # scaled_power = scale_factor_power * power_aux
        # scaled_power = torch.clamp(scaled_power, 0.0, 1e3)
        # scaled_power = torch.nan_to_num(scaled_power, nan=0.0, posinf=1e3, neginf=-1e3)

        ant_feats_out = torch.stack([scaled_delta, scaled_power], dim=-1)

        user_xy = user_feats_init[..., :2]
        edge_feats_new = torch.norm(
            user_xy.unsqueeze(2) - positions[..., :2].unsqueeze(1),
            dim=-1, keepdim=True
        )

        return ant_feats_out, edge_feats_new, positions


########################################
# 3) BGATModel: Stacking Multiple Blocks
########################################
class BGATModel(nn.Module):
    def __init__(self, D_blocks, user_dim, ant_dim, hidden_dim, num_heads,
                 waveguide_bound, delta_min, H, Pmax, N, M, L):
        super().__init__()
        self.blocks = nn.ModuleList([
            BGATBlock(user_dim, ant_dim, edge_dim=1, hidden_dim=hidden_dim, num_heads=num_heads,
                      waveguide_bound=waveguide_bound, delta_min=delta_min, H=H, Pmax=Pmax,N=N, M=M)
            for _ in range(D_blocks)
        ])
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H
        self.Pmax = Pmax
        self.N = N
        self.M = M
        self.L = L

    def forward(self, user_feats, delta_init, power_init):
        user_feats_init = user_feats[..., :2]
        ant_feats = torch.stack([delta_init, power_init], dim=-1)

        x0 = torch.zeros_like(delta_init)
        x0[:, 0] = delta_init[:, 0] - self.waveguide_bound
        for n in range(1, self.N):
            x0[:, n] = x0[:, n - 1] + delta_init[:, n] + self.delta_min - self.waveguide_bound

        init_positions = torch.stack([
            x0, torch.zeros_like(x0), torch.full_like(x0, self.H)
        ], dim=-1)

        user_xy = user_feats_init[..., :2]
        edge_feats = torch.norm(
            user_xy.unsqueeze(2) - init_positions[..., :2].unsqueeze(1),
            dim=-1, keepdim=True
        )
        intermediate_outputs = []
        final_positions = init_positions
        for block in self.blocks:
            ant_feats, edge_feats, final_positions = block(user_feats_init, ant_feats, edge_feats)
            curr_power = ant_feats[..., 1]
            intermediate_outputs.append((curr_power, final_positions))

        # scaled_power = ant_feats[..., 1]
        # scaled_delta = ant_feats[..., 0]

        final_power, final_positions = intermediate_outputs[-1]

        # return scaled_power, scaled_delta, final_positions
        return final_power, final_positions, intermediate_outputs
