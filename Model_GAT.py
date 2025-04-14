import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##############################################################################
# 1) GAT Layer (user-based)
##############################################################################
class GATLayer(nn.Module):
    def __init__(self, user_dim, out_dim, num_heads,hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_src = nn.ModuleList([
            nn.Sequential(
                nn.Linear(user_dim, self.head_dim, bias=False),
                nn.LayerNorm(self.head_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_heads)])
        self.W_tgt = nn.ModuleList([
            nn.Sequential(
                nn.Linear(user_dim, self.head_dim, bias=False),
                nn.LayerNorm(self.head_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_heads)])

        self.attn_vecs = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.head_dim)) for _ in range(num_heads)
        ])

        self.residual_proj = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # for seq in (self.W_src+self.W_tgt):
        #     for layer in seq:
        #         if isinstance(layer, nn.Linear):
        #             nn.init.xavier_uniform_(layer.weight)
        for seq in (self.W_src+self.W_tgt):
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        a=0.2,
                        mode='fan_in',
                        nonlinearity='leaky_relu'
                    )
        for attn_vec in self.attn_vecs:
            nn.init.kaiming_normal_(attn_vec.view(1, -1), a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        for layer in self.residual_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # for a in self.attn_vecs:
        #     nn.init.xavier_uniform_(a.view(1, -1))

        # for layer in self.residual_proj:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)

        self.leaky = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, user_feats):
        head_outputs = []
        for k in range(self.num_heads):
            U_src = self.W_src[k](user_feats)
            U_tgt = self.W_tgt[k](user_feats)
            U_i = U_src.unsqueeze(2)
            U_j = U_tgt.unsqueeze(1)

            attn_input = self.leaky(U_i + U_j)
            e = torch.einsum('bijh,h->bij', attn_input, self.attn_vecs[k])* self.scale
            e = e - e.max(dim=-1, keepdim=True)[0]

            alpha = F.softmax(e, dim=2)

            aggregated = torch.einsum('bij,bjd->bid', alpha, U_tgt)
            head_outputs.append(aggregated)

        out = torch.cat(head_outputs, dim=-1) + self.residual_proj(user_feats)

        out = F.relu(out)

        return out


##############################################################################
# 2) Independent GAT Baseline Model (replicating Appendix B of the paper)
##############################################################################
class GATModel(nn.Module):
    def __init__(self,
                 in_dim,  # user location
                 hidden_dim,  # hidden dimension per head
                 num_layers,  # number of GAT layers
                 num_heads,  # number of attention heads
                 N,  # Number of antennas
                 waveguide_bound,  # D: half-range for antenna deployment
                 delta_min,  # Δ: minimum distance
                 Pmax,  # Total power budget
                 H  # Antenna height
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_antennas = N
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.Pmax = Pmax
        self.H = H
        self.B_max = 2 * self.waveguide_bound - (self.num_antennas - 1) * self.delta_min
        self.layer_norm_ant = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.gat_layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_layers):
            head_dim = hidden_dim // num_heads
            layer = GATLayer(
                user_dim=current_dim,
                out_dim=head_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim
            )
            self.gat_layers.append(layer)
            current_dim = hidden_dim

        self.pool = lambda x: x.max(dim=1)[0]

        self.mlp = nn.Sequential(
            nn.Linear(current_dim, current_dim, bias=True),
            nn.ReLU(),
            nn.Linear(current_dim, 2 * self.num_antennas, bias=True),
            nn.ReLU(),
        )

        self.raw_readout_delta = nn.Sequential(
            nn.Linear(self.num_antennas, 2 * self.num_antennas),
            nn.ReLU(),
            nn.Linear(2 * self.num_antennas, self.num_antennas)
        )
        self.raw_readout_power = nn.Sequential(
            nn.Linear(self.num_antennas, 2 * self.num_antennas),
            nn.ReLU(),
            nn.Linear(2 * self.num_antennas, self.num_antennas)
        )

        for module in self.mlp:
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

    def forward(self, user_feats):
        x = user_feats
        for layer in self.gat_layers:
            x = layer(x)

        # x = self.layer_norm_ant(x)
        pooled = self.pool(x)
        mlp_out = self.dropout(self.mlp(pooled))

        p_pred, d_pred = torch.split(mlp_out, self.num_antennas, dim=-1)

        raw_delta = self.raw_readout_delta(d_pred)
        raw_power = self.raw_readout_power(p_pred)
        raw_delta = torch.clamp(raw_delta, min=0.0, max=1e2)
        raw_power = torch.nan_to_num(raw_power, nan=0.0, posinf=1, neginf=0)

        # ----- antenna intervals -----
        delta = F.relu(raw_delta)
        sum_delta = delta.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        # B_max_tensor = torch.tensor(self.B_max, device=delta.device, dtype=delta.dtype)
        scale_factor_delta = self.B_max / torch.maximum(
            torch.tensor(self.B_max, device=sum_delta.device),
            sum_delta
        )
        delta_scaled = delta * scale_factor_delta
        delta_scaled = torch.clamp(delta_scaled, 0.0, 1e2)

        # ----- power -----
        # eps = 1e-3
        power = F.relu(raw_power)
        sum_power = power.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        # P_max_tensor = torch.tensor(self.Pmax, device=power.device, dtype=power.dtype)
        scale_factor_power = self.Pmax / torch.maximum(
            torch.tensor(self.Pmax, device=sum_power.device),
            sum_power
        )
        power_scaled = power * scale_factor_power

        return power_scaled, delta_scaled

    def delta_to_position(self, delta_pred):
        x_coords = torch.zeros_like(delta_pred)
        x_coords[:, 0] = delta_pred[:, 0] - self.waveguide_bound
        for n in range(1, self.num_antennas):
            x_coords[:, n] = x_coords[:, n - 1] + delta_pred[:, n] + self.delta_min
        pos_3d = torch.stack([
            x_coords,
            torch.zeros_like(x_coords),
            torch.full_like(x_coords, self.H)
        ], dim=-1)
        return pos_3d
