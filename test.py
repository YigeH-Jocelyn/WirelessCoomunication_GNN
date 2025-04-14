import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gc
torch.cuda.empty_cache()
gc.collect()

# --------------------------- Hyperparameters ---------------------------
M = 2                                       # Number of users
N = 4                                       # Number of antennas
L = 100.0                                   # User area range
D = 50.0                                    # Half-range for antenna deployment
H = 5.0                                     # Antenna height (m)
Pmax_linear = 10**(30 / 10 - 3)             # Total power budget (1 W ~ 30 dBm)
sigma2 = 10**(-90 / 10 - 3)                 # Noise power
fc = 6e9                                    # Carrier frequency (Hz)
c = 3e8                                     # Speed of light (m/s)
Pc = 0.5                                    # Constant Power Consumption
n_neff = 1.4                                # Effective Refractive Index
lambda_fs = c / fc                          # Free-space wavelength
lambda_R = c / (fc * n_neff)                # Waveguide wavelength
delta_min = lambda_fs / 2                   # Minimum antenna spacing
eta = (c ** 2) / ((4 * math.pi * fc) ** 2)  # Path loss constant
d_blocks = 5                                # Number of Blocks
h_dim = 32                                  # Hidden Layers
o_dim = 2                                   # User dimension

# Training Parameters
NUM_SAMPLES_TRAIN = 100000
NUM_SAMPLES_TEST = 1000
EPOCHS = 1000
BATCH_SIZE = 2048
LR = 5e-5
CLIP_GRAD = 5

# Set seeds
SEED = 3407
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# --------------------------- Data Generation ---------------------------
def generate_users_2d(M, L):
    xy = np.random.uniform(-L, L, size=(M, 2))
    z = np.zeros((M, 1))
    return np.concatenate([xy, z], axis=1)

def init_intervals(N, D, delta_min):
    Bmax = 2 * D - (N - 1) * delta_min
    return np.ones(N) * (Bmax / (N-1))

def intervals_to_positions(delta, D, delta_min, H):
    L = D
    positions = np.zeros((len(delta), 3))
    x = delta[0] - L
    positions[0] = [x, 0, H]
    for i in range(1, len(delta)):
        x = positions[i-1][0] + delta[i] + delta_min - L
        positions[i] = [x, 0, H]
    return positions

class PinchingDataset(Dataset):
    def __init__(self, num_samples):
        self.data = []
        for _ in range(num_samples):
            users = generate_users_2d(M, L)
            delta = init_intervals(N, D, delta_min)
            p = np.ones(N) * (Pmax_linear / N)
            ants = intervals_to_positions(delta, D, delta_min, H)
            self.data.append({
                'users': users,
                'delta': delta,
                'power': p,
                'antennas': ants
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.tensor(sample['users'], dtype=torch.float32).to(device),
            torch.tensor(sample['delta'], dtype=torch.float32).to(device),
            torch.tensor(sample['power'], dtype=torch.float32).to(device),
            torch.tensor(sample['antennas'], dtype=torch.float32).to(device)
        )

# ----------------------- EE Calculation --------------------------
def data_rate(users, antennas, power, feed_point, eta, sigma2):
    diff = users.unsqueeze(2) - antennas.unsqueeze(1)
    d_ua = torch.norm(diff, dim=-1).clamp_min(1e-6)
    phase_fs = -2 * math.pi * d_ua / lambda_fs
    feed_dist = torch.norm(feed_point - antennas, dim=-1)
    phase_wg = 2.0 * math.pi * feed_dist / lambda_R
    phase_total = phase_fs - phase_wg.unsqueeze(1)
    amplitude = (eta ** 0.5) / d_ua.clamp_min(1e-6)

    p_sqrt = torch.sqrt(power.clamp_min(0)).unsqueeze(1)
    phase_complex = torch.exp(1j * phase_total)
    combined = p_sqrt * amplitude * phase_complex

    combined_sum = combined.sum(dim=2)
    snr = torch.abs(combined_sum) ** 2 / sigma2
    rates = torch.log2(1.0 + snr.clamp_min(1e-9))
    return rates

def ee_loss(users, antennas, power, eta, sigma2):
    feed_point = torch.tensor([[-D, 0, H]], dtype=torch.float32, device=antennas.device)
    feed_point = feed_point.expand(antennas.size(0), 1, 3)
    rate = data_rate(users, antennas, power, feed_point, eta, sigma2).sum(dim=1)
    power_total = power.sum(dim=1) + Pc
    return -torch.mean(rate / power_total)

# ------------------------ BGAT Model --------------------------------
class BGATAttention(nn.Module):
    def __init__(self, user_dim, ant_dim, edge_dim, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.user_projs = nn.ModuleList([nn.Linear(user_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.ant_projs  = nn.ModuleList([nn.Linear(ant_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.edge_projs = nn.ModuleList([nn.Linear(edge_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.attn_vecs  = nn.ParameterList([nn.Parameter(torch.Tensor(self.head_dim)) for _ in range(num_heads)])
        self.residual_proj = nn.Linear(user_dim, hidden_dim, bias=False)

        for proj in self.user_projs + self.ant_projs + self.edge_projs:
            nn.init.kaiming_normal_(proj.weight, mode='fan_in', nonlinearity='relu')
        for a in self.attn_vecs:
            nn.init.kaiming_normal_(a.view(-1, 1), mode='fan_in', nonlinearity='relu')

        self.leaky = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_feats, ant_feats, edge_feats):
        head_outputs_user = []
        head_outputs_ant = []
        for k in range(self.num_heads):
            U = self.user_projs[k](user_feats).unsqueeze(2)
            A = self.ant_projs[k](ant_feats).unsqueeze(1)
            E = self.edge_projs[k](edge_feats)
            attn_score = torch.sum(self.attn_vecs[k] * self.leaky(U + A + E), dim=-1) * self.scale
            alpha = F.softmax(attn_score, dim=-1)
            alpha = F.dropout(alpha, p=0.3, training=self.training)
            user_out = ((A + E) * alpha.unsqueeze(-1)).sum(dim=2)
            ant_out  = ((U + E) * alpha.unsqueeze(-1)).sum(dim=1)
            head_outputs_user.append(user_out)
            head_outputs_ant.append(ant_out)
        user_out = torch.cat(head_outputs_user, dim=-1)+ self.residual_proj(user_feats)
        ant_out = torch.cat(head_outputs_ant, dim=-1)

        return user_out, ant_out

class BGATBlock(nn.Module):
    def __init__(self, user_dim, ant_dim, edge_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = BGATAttention(user_dim, ant_dim, edge_dim, hidden_dim, num_heads)
        self.layer_norm_user = nn.LayerNorm(hidden_dim)
        self.layer_norm_ant = nn.LayerNorm(hidden_dim)
        self.mlp_user = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.mlp_ant = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.dropout = nn.Dropout(0.2)

        for m in self.mlp_user:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.mlp_ant:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_feats, ant_feats, edge_feats):
        u_attn, a_attn = self.attention(user_feats, ant_feats, edge_feats)
        u_attn = self.layer_norm_user(u_attn)
        a_attn = self.layer_norm_ant(a_attn)
        user_out = self.dropout(self.mlp_user(u_attn))
        ant_out = self.dropout(self.mlp_ant(a_attn))
        return user_out, ant_out

class BGATModel(nn.Module):
    def __init__(self, D_blocks=d_blocks, user_dim=2, ant_dim=2, hidden_dim=h_dim, out_dim=o_dim, num_heads=8,
                 waveguide_bound=D, delta_min=delta_min, H=H, Pmax=Pmax_linear):
        super().__init__()
        self.blocks = nn.ModuleList([
            BGATBlock(user_dim, ant_dim, edge_dim=1, hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(D_blocks)
        ])

        self.antenna_projection = nn.Linear(out_dim, 2)
        nn.init.kaiming_normal_(self.antenna_projection.weight, mode='fan_in', nonlinearity='relu')
        if self.antenna_projection.bias is not None:
            nn.init.zeros_(self.antenna_projection.bias)

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
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H
        self.Pmax = Pmax

    def forward(self, users, delta_init, power_init):
        B, M, _ = users.shape
        N_ant = delta_init.shape[1]

        ant_feats = torch.stack([power_init, delta_init], dim=-1)
        user_feats = users[..., :2]

        for block in self.blocks:
            ants_pos = self.delta_to_position(ant_feats[..., 1])
            edge_feats = torch.norm(
                user_feats.unsqueeze(2) - ants_pos[..., :2].unsqueeze(1),
                dim=-1, keepdim=True
            )
            user_feats, ant_feats = block(user_feats, ant_feats, edge_feats)

        raw_delta = self.raw_readout_delta(ant_feats[..., 1])
        delta_aux = torch.clamp(F.relu(raw_delta), min=1e-3)
        sum_delta = delta_aux.sum(dim=1, keepdim=True)
        Bmax_val = 2 * self.waveguide_bound - (N_ant - 1) * self.delta_min
        scale_factor_delta = Bmax_val / sum_delta
        scaled_delta = scale_factor_delta * delta_aux

        x = torch.zeros_like(scaled_delta)
        x[:, 0] = scaled_delta[:, 0] - self.waveguide_bound
        for n in range(1, N_ant):
            x[:, n] = x[:, n - 1] + scaled_delta[:, n] + self.delta_min - self.waveguide_bound
        final_positions = torch.stack(
            [x, torch.zeros_like(x), torch.full_like(x, self.H)], dim=-1
        )

        raw_power = self.raw_readout_power(ant_feats[..., 0])
        power_aux = torch.clamp(F.relu(raw_power), min=1e-3)
        sum_power = power_aux.sum(dim=1, keepdim=True) + 1e-6
        scale_factor_power = self.Pmax / torch.maximum(
            torch.tensor(self.Pmax, device=power_aux.device),
            sum_power
        )
        scaled_power = scale_factor_power * power_aux

        return scaled_power, scaled_delta, final_positions

    def delta_to_position(self, delta):
        B, N_ant = delta.shape
        delta_aux = F.relu(delta)
        sum_delta = delta_aux.sum(dim=1, keepdim=True) + 1e-6
        Bmax = 2 * self.waveguide_bound - (N_ant - 1) * self.delta_min
        scaled_delta = Bmax * delta_aux / torch.clamp(sum_delta, min=1e-6)

        x = torch.zeros_like(scaled_delta)
        x[:, 0] = scaled_delta[:, 0] - self.waveguide_bound
        for n in range(1, N_ant):
            x[:, n] = x[:, n-1] + scaled_delta[:, n] + self.delta_min

        return torch.stack([x, torch.zeros_like(x), torch.full_like(x, self.H)], dim=-1)

# -------------------------- GAT Model (Baseline) ---------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        Wh = self.W(h)
        B, M, _ = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(B, M, M, -1)
        Wh_j = Wh.unsqueeze(1).expand(B, M, M, -1)
        e = self.leakyrelu(torch.matmul(torch.cat([Wh_i, Wh_j], dim=-1), self.a))
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class GATModel(nn.Module):
    def __init__(self, num_users, num_antennas,H=H, waveguide_bound=D, delta_min=delta_min, in_dim=3, hidden_dim=16, num_layers=2):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATLayer(hidden_dim, hidden_dim))
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * num_antennas)
        )
        self.num_antennas = num_antennas
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min
        self.H = H

    def forward(self, users):
        h = users
        for layer in self.gat_layers:
            h = layer(h)
        pooled, _ = torch.max(h, dim=1)
        readout = self.mlp(pooled)
        raw_delta, raw_power = torch.split(readout, self.num_antennas, dim=-1)
        eps = 1e-6
        raw_delta = F.relu(raw_delta) + eps
        raw_power = F.relu(raw_power) + eps
        Bmax = 2 * self.waveguide_bound - (self.num_antennas - 1) * self.delta_min
        delta = Bmax * raw_delta / (raw_delta.sum(dim=1, keepdim=True) + eps)
        power = Pmax_linear * raw_power / (raw_power.sum(dim=1, keepdim=True) + eps)
        return power, delta

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

# ------------------------------ MLP Model ----------------------------------
class MLPModel(nn.Module):
    def __init__(self, M, N, waveguide_bound=D, delta_min=delta_min):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * M, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        # Two separate readouts for power and delta (matching the paper’s idea)
        self.readout_power = nn.Sequential(
            nn.Linear(64, 2 * N), nn.ReLU(),
            nn.Linear(2 * N, N)
        )
        self.readout_delta = nn.Sequential(
            nn.Linear(64, 2 * N), nn.ReLU(),
            nn.Linear(2 * N, N)
        )
        self.waveguide_bound = waveguide_bound
        self.delta_min = delta_min

    def forward(self, users):
        B = users.shape[0]
        x = users.view(B, -1)
        feat = self.net(x)
        p_raw = self.readout_power(feat)  # shape: [B, N]
        d_raw = self.readout_delta(feat)    # shape: [B, N]
        eps = 1e-6
        p = F.relu(p_raw) + eps
        p = Pmax_linear * p / p.sum(dim=1, keepdim=True).clamp_min(eps)
        Bmax = 2 * D - (N - 1) * delta_min
        d = F.relu(d_raw) + eps
        d = Bmax * d / d.sum(dim=1, keepdim=True).clamp_min(eps)
        return p, d

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
        return torch.stack([x, torch.zeros_like(x), torch.full_like(x, H)], dim=-1)


# ------------------------- Training & Evaluation ---------------------------
def train_model(model, train_loader, test_loader, model_type, lr=LR, epochs=EPOCHS, patience=20, tolerance=1e-2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    best_ee = -np.inf

    epoch_list, train_loss_list,train_ee_list, val_loss_list, val_ee_list = [], [], [], [], []
    epochs_no_improve, last_train_loss, last_val_ee = 0, None, None

    for epoch in range(epochs):
        model.train()
        train_loss, train_ee = 0, 0
        for users, delta, power, ants in train_loader:
            optimizer.zero_grad()
            if model_type == 'MLP':
                p_pred, d_pred = model(users)
                ants_pred = model.delta_to_position(d_pred)
            elif model_type == 'BGAT':
                p_pred, d_pred, ants_pred = model(users, delta, power)
            elif model_type == 'GAT':
                p_pred, d_pred = model(users)
                ants_pred = model.delta_to_position(d_pred)
            loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            train_loss += loss.item() * users.size(0)
            train_ee += -loss.item() * users.size(0)

        model.eval()
        val_loss, val_ee = 0, 0
        with torch.no_grad():
            for users, delta, power, ants in test_loader:
                if model_type == 'MLP':
                    p_pred, d_pred = model(users)
                    ants_pred = model.delta_to_position(d_pred)
                elif model_type == 'BGAT':
                    p_pred, d_pred, ants_pred = model(users, delta, power)
                elif model_type == 'GAT':
                    p_pred, d_pred = model(users)
                    ants_pred = model.delta_to_position(d_pred)
                loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
                val_loss += loss.item() * users.size(0)
                val_ee += -loss.item() * users.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_ee = train_ee / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_ee = val_ee / len(test_loader.dataset)

        scheduler.step()

        epoch_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        val_ee_list.append(avg_val_ee)
        train_ee_list.append(avg_train_ee)
        val_loss_list.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train EE: {avg_train_ee:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val EE: {avg_val_ee:.4f}')

        if last_train_loss is not None and last_val_ee is not None:
            diff_loss, diff_ee = abs(avg_train_loss - last_train_loss), abs(avg_val_ee - last_val_ee)
            epochs_no_improve = epochs_no_improve + 1 if diff_loss < tolerance and diff_ee < tolerance else 0
        else:
            epochs_no_improve = 0
        if avg_val_ee > best_ee:
            best_ee, epochs_no_improve = avg_val_ee, 0

        last_train_loss, last_val_ee = avg_train_loss, avg_val_ee

        if epochs_no_improve >= patience:
            break

    print(f'Best Validation EE: {best_ee:.4f}')

    return best_ee, {
        'epoch': epoch_list,
        'train_loss': train_loss_list,
        'train_ee': train_ee_list,
        'val_loss': val_loss_list,
        'val_ee': val_ee_list
    }

# --------------------------- Main Execution --------------------------------
if __name__ == "__main__":
    train_dataset = PinchingDataset(NUM_SAMPLES_TRAIN)
    test_dataset = PinchingDataset(NUM_SAMPLES_TEST)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    metrics_dict = {}
    models = {
        "BGAT": BGATModel(D_blocks=d_blocks),
        "GAT": GATModel(num_users=M, num_antennas=N, in_dim=3, hidden_dim=16, num_layers=2),
        "MLP": MLPModel(M, N)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name} Model...")
        best_ee, metrics = train_model(model, train_loader, test_loader, model_type=model_name)
        metrics_dict[model_name] = metrics

    output_path = r"C:\Users\jocel\Desktop\Education\UToronto\2025 Winter\ECE2500Y\results\results.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for model_name, mtr in metrics_dict.items():
            df = pd.DataFrame(mtr)
            df.to_excel(writer, sheet_name=model_name, index=False)

# --------------------------- Plotting Metrics ---------------------------
    def plot_metric(metric_name, ylabel, title):
        plt.figure(figsize=(8, 6))
        for model_name, mtr in metrics_dict.items():
            plt.plot(mtr['epoch'], mtr[metric_name], marker='o', linestyle='-', label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot Training Loss vs. Epoch
    plot_metric("train_loss", "Training Loss", "Training Loss vs. Epoch")
    # Plot Validation Loss vs. Epoch
    plot_metric("val_loss", "Validation Loss", "Validation Loss vs. Epoch")
    # Plot Training EE vs. Epoch
    plot_metric("train_ee", "Training EE", "Training Energy Efficiency vs. Epoch")
    # Plot Validation EE vs. Epoch
    plot_metric("val_ee", "Validation EE", "Validation Energy Efficiency vs. Epoch")


