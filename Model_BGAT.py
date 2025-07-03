import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLAMP_MIN = -200.0
CLAMP_MAX =  200.0
########################################
# 1) BGATAttention: The GAT Layer 
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
# 2) BGATBlock: GAT, MLP, and Readout
########################################
class BGATBlock(nn.Module):
    def __init__(self, user_dim, ant_dim, edge_dim, hidden_dim, num_heads,
                 waveguide_bound, delta_min, H, Pmax, N, M):
        super().__init__()
        self.attention = BGATAttention(user_dim, ant_dim, edge_dim, hidden_dim, num_heads)

        self.mlp_user = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.ReLU()
        )

        self.mlp_ant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Only 1D output for delta
            nn.ReLU()
        )

        self.raw_readout_delta = nn.Sequential(
            nn.Linear(N, 2 * N),
            nn.ReLU(),
            nn.Linear(2 * N, N)
        )

        self.dropout = nn.Dropout(0.1)

        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H
        self.N = N

    def forward(self, user_feats_init, ant_feats, edge_feats):
        # GAT attention
        u_attn, a_attn = self.attention(user_feats_init, ant_feats, edge_feats)

        user_out = self.dropout(self.mlp_user(u_attn))
        ant_out = self.dropout(self.mlp_ant(a_attn)).squeeze(-1)  # shape: (B, N)

        raw_delta = self.raw_readout_delta(ant_out)  # (B, N)
        delta_aux = F.relu(raw_delta)

        Bmax_val = 2 * self.waveguide_bound - (self.N - 1) * self.delta_min
        sum_delta = torch.clamp(delta_aux.sum(dim=1, keepdim=True), 1e-6, 1e6)
        scale_factor = Bmax_val / torch.maximum(torch.tensor(Bmax_val, device=delta_aux.device), sum_delta)
        scaled_delta = torch.clamp(scale_factor * delta_aux, 0.0, 1000)

        # Compute positions
        x = torch.zeros_like(scaled_delta)
        x[:, 0] = scaled_delta[:, 0] - self.waveguide_bound
        for n in range(1, self.N):
            x[:, n] = x[:, n - 1] + scaled_delta[:, n] + self.delta_min - self.waveguide_bound

        positions = torch.stack([
            x, torch.zeros_like(x), torch.full_like(x, self.H)
        ], dim=-1)

        # Update edge features
        user_xy = user_feats_init[..., :2]
        edge_feats_new = torch.norm(
            user_xy.unsqueeze(2) - positions[..., :2].unsqueeze(1),
            dim=-1, keepdim=True
        )

        # Only return delta and updated positions
        return scaled_delta.unsqueeze(-1), edge_feats_new, positions


########################################
# 3) BGATModel: Stacking Multiple Blocks
########################################

class BGATModel(nn.Module):
    def __init__(self, D_blocks, user_dim, ant_dim, hidden_dim, num_heads,
                 waveguide_bound, delta_min, H, Pmax, N, M, L):
        super().__init__()
        self.blocks = nn.ModuleList([
            BGATBlock(user_dim, ant_dim, edge_dim=1, hidden_dim=hidden_dim, num_heads=num_heads,
                      waveguide_bound=waveguide_bound, delta_min=delta_min, H=H, Pmax=Pmax, N=N, M=M)
            for _ in range(D_blocks)
        ])
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H
        self.Pmax = Pmax
        self.N = N
        self.M = M
        self.L = L

        # Learnable scalar split factor δ in logit space
        self.logit_delta = nn.Parameter(torch.tensor(0.0))  # sigmoid(logit) in (0,1)

    def forward(self, user_feats, delta_init):
        user_feats_init = user_feats[..., :2]

        # [B, N, 1]
        ant_feats = delta_init.unsqueeze(-1)
        B, M, _ = user_feats_init.shape
        B, N, _ = ant_feats.shape

        # Compute initial positions
        x0 = torch.zeros_like(delta_init)
        x0[:, 0] = delta_init[:, 0] - self.waveguide_bound
        for n in range(1, N):
            x0[:, n] = x0[:, n - 1] + delta_init[:, n] + self.delta_min - self.waveguide_bound

        init_positions = torch.stack([
            x0, torch.zeros_like(x0), torch.full_like(x0, self.H)
        ], dim=-1)

        # Edge features (user-antenna distances)
        user_xy = user_feats_init[..., :2]
        edge_feats = torch.norm(
            user_xy.unsqueeze(2) - init_positions[..., :2].unsqueeze(1),
            dim=-1, keepdim=True
        )

        final_positions = init_positions
        intermediate_outputs = []  # << add this line
        delta_scalar = torch.sigmoid(self.logit_delta)

        for block in self.blocks:
            ant_feats, edge_feats, final_positions = block(user_feats_init, ant_feats, edge_feats)
            # delta_this_block = ant_feats[..., 0]  # shape: [B, N]
            # intermediate_outputs.append((delta_this_block, final_positions))
            intermediate_outputs.append((delta_scalar, final_positions))

        return delta_scalar, final_positions, intermediate_outputs


