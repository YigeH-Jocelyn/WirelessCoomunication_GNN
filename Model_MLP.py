import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, M, N,Pmax_linear,delta_min,H,waveguide_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * M, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.readout_power = nn.Sequential(
            nn.Linear(64, 2 * N), nn.ReLU(),
            nn.Linear(2 * N, N)
        )
        self.readout_delta = nn.Sequential(
            nn.Linear(64, 2 * N), nn.ReLU(),
            nn.Linear(2 * N, N)
        )
        self.Pmax_linear = Pmax_linear
        self.delta_min = delta_min
        self.waveguide_bound = waveguide_bound
        self.N = N
        self.H = H

    def delta_to_position(self, delta):
        B, N_ant = delta.shape
        delta_aux = F.relu(delta)
        sum_delta = delta_aux.sum(dim=1, keepdim=True) + 1e-6
        Bmax = 2 * self.waveguide_bound - (N_ant - 1) * self.delta_min
        scaled_delta = Bmax * delta_aux / torch.clamp(sum_delta, min=1e-6)

        x = torch.zeros_like(scaled_delta)
        x[:, 0] = scaled_delta[:, 0] - self.waveguide_bound
        for n in range(1, N_ant):
            x[:, n] = x[:, n - 1] + scaled_delta[:, n] + self.delta_min
        return torch.stack([x, torch.zeros_like(x), torch.full_like(x, self.H)], dim=-1)

    def forward(self, users):
        B = users.shape[0]
        x = users.view(B, -1)
        feat = self.net(x)
        p_raw = self.readout_power(feat)
        d_raw = self.readout_delta(feat)
        eps = 1e-6
        p = F.relu(p_raw) + eps
        p = self.Pmax_linear * p / p.sum(dim=1, keepdim=True).clamp_min(eps)
        Bmax = 2 * self.waveguide_bound - (self.N - 1) * self.delta_min
        d = F.relu(d_raw) + eps
        d = Bmax * d / d.sum(dim=1, keepdim=True).clamp_min(eps)
        return p, d

