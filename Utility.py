import math
import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------- Hyperparameters ---------------------------
M = 1                                       # Number of users
N = 4                                       # Number of antennas
L = 100.0                                   # User area range
D = 50.0                                    # Half-range for antenna deployment
H = 5.0                                     # Antenna height (m)
Pmax_linear = 10**(30 / 10 - 3)             # Total power budget
sigma2 = 10**(-90 / 10 - 3)                 # Noise power
fc = 6e9                                    # Carrier frequency (Hz)
c = 3e8                                     # Speed of light (m/s)
Pc = 0.5                                    # Constant Power Consumption
n_neff = 1.4                                # Effective Refractive Index
lambda_fs = c / fc                          # Free-space wavelength
lambda_R = c / (fc * n_neff)                # Waveguide wavelength
delta_min = lambda_fs / 2                   # Minimum antenna spacing
eta = (c ** 2) / ((4 * math.pi * fc) ** 2)  # Path loss constant
d_blocks = 2                                # Number of Blocks
h_dim = 32                                  # Hidden Layers
o_dim = 2                                   # User dimension
NUM_SAMPLES_TRAIN = 100000                  # Number of Samples in each Training set
NUM_SAMPLES_TEST = 1000                     # Number of Samples in each Test set

device = torch.device("cpu")

def generate_users_2d(M, L):
    xy = np.random.uniform(-L, L, size=(M, 2)) # for M user locations, uniformly sample from [-L,L]
    z = np.zeros((M, 1)) # 0 is set for z
    return np.concatenate([xy, z], axis=1) # location is in 3D as [x, y, 0]

def init_intervals(N, D, delta_min):
    Bmax = 2 * D - (N - 1) * delta_min # this is used to calculate the maximum available span for antennas
    return np.ones(N) * (Bmax / (N-1)) # used to evenly divided the total length to N inter-antenna intervals

def intervals_to_positions(delta, D, delta_min, H):
    positions = np.zeros((len(delta)+1, 3))
    x = -D # ensure the length intervals of rest antennas don't exceed the range by shifting the first antenna's position by -D
    positions[0] = [x, 0, H]
    for i in range(1, len(delta)):
        x = positions[i-1][0] + delta[i] + delta_min
        positions[i] = [x, 0, H]
    return positions

class PinchingDataset(Dataset):
    def __init__(self, num_samples, num_users=M):
        self.data = []
        self.num_users = num_users
        for _ in range(num_samples):
            users = generate_users_2d(self.num_users, L)
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

def data_rate(users, antennas, power, feed_point, eta, sigma2):
    diff = users.unsqueeze(2) - antennas.unsqueeze(1)  # [batch, M, N, 3]
    diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)

    d_ua = torch.norm(diff, dim=-1).clamp_min(0.1)  # avoid division by 0
    d_ua = torch.nan_to_num(d_ua, nan=0.1, posinf=1e6, neginf=0.1)

    phase_fs = 2.0 * math.pi * d_ua / lambda_fs  # Free-space phase shift

    diff_feed = feed_point - antennas
    diff_feed = torch.nan_to_num(diff_feed, nan=0.0, posinf=1e6, neginf=-1e6)

    feed_dist = torch.norm(diff_feed, dim=-1)
    feed_dist = torch.nan_to_num(feed_dist, nan=0.0, posinf=1e6, neginf=0.0)

    phase_wg = 2.0 * math.pi * feed_dist / lambda_R
    amplitude = (eta ** 0.5) / d_ua.clamp_min(1e-6)
    p_sqrt = torch.sqrt(power.clamp_min(0) + 1e-8).unsqueeze(1)
    hm = amplitude * torch.exp(-1j * phase_fs)
    sm = p_sqrt * torch.exp(-1j * phase_wg.unsqueeze(1))

    hm_real = torch.nan_to_num(hm.real, nan=0.0, posinf=1e6, neginf=-1e6)
    hm_imag = torch.nan_to_num(hm.imag, nan=0.0, posinf=1e6, neginf=-1e6)
    hm = torch.complex(hm_real, hm_imag)

    sm_real = torch.nan_to_num(sm.real, nan=0.0, posinf=1e6, neginf=-1e6)
    sm_imag = torch.nan_to_num(sm.imag, nan=0.0, posinf=1e6, neginf=-1e6)
    sm = torch.complex(sm_real, sm_imag)

    combined = hm * sm
    combined_sum = combined.sum(dim=2)

    noise_real = torch.randn_like(combined_sum.real) * math.sqrt(sigma2 / 2)
    noise_imag = torch.randn_like(combined_sum.imag) * math.sqrt(sigma2 / 2)
    noise = noise_real + 1j * noise_imag

    received_signal = combined_sum + noise
    rec_real = torch.nan_to_num(received_signal.real, nan=0.0, posinf=1e6, neginf=-1e6)
    rec_imag = torch.nan_to_num(received_signal.imag, nan=0.0, posinf=1e6, neginf=-1e6)
    received_signal = torch.complex(rec_real, rec_imag)

    abs_received = torch.abs(received_signal)
    abs_received = torch.clamp(abs_received, 0.0, 1e3)

    snr = (abs_received ** 2) / sigma2
    snr = torch.nan_to_num(snr, nan=1e-9, posinf=1e6, neginf=1e-9)
    snr = snr.clamp(1e-9, 1e6)

    snr_safe = torch.nan_to_num(snr, nan=1e-9, posinf=1e6, neginf=1e-9)
    snr_safe = snr_safe.clamp(1e-9, 1e6)

    rates = torch.log1p(snr_safe)
    rates = torch.nan_to_num(rates, nan=0.0, posinf=1e6, neginf=-1e6)
    return rates

def dynamic_ee_loss(users, positions, delta_scalar, eta, sigma2, Pin=1.0, Pc=0.5, xi=1.0):
    # delta_scalar is δ=cos(κℓ)
    B, _, _ = users.shape  # users shape: (B, 1, 3)
    B, N, _ = positions.shape

    delta = delta_scalar.clamp(0.01, 0.99)

    beta = 2 * math.pi * n_neff / lambda_fs # waveguide phase shift
    alpha = 2 * math.pi / lambda_fs # free-space phase shift

    u = users[:, 0, :]  # (B, 3) extract the single user’s coordinates

    Dn = torch.norm(u.unsqueeze(1) - positions, dim=-1).clamp(min=1e-3)  # (B, N)
    x_n = positions[:, :, 0]  # (B, N)

    n_indices = torch.arange(N, device=users.device).float().view(1, N)
    delta_pow = delta.view(-1, 1) ** n_indices  # (B, N)

    phase = -1j * (beta * x_n + alpha * Dn)  # (B, N)
    h = (delta_pow / Dn) * torch.exp(phase)  # (B, N)
    h_sum = h.sum(dim=1)  # (B,)

    snr = (xi ** 2) * (1 - delta ** 2) * Pin * torch.abs(h_sum) ** 2 / sigma2  # (B,)
    rate = torch.log1p(snr.clamp_min(1e-9))  # (B,)
    ee = rate / (Pin + Pc)  # (B,)
    return -torch.mean(ee)

def average_power(power):
    # Compute total power per sample.
    power_total = power.sum(dim=1) + Pc
    # Average over the batch to get a single scalar.
    return torch.mean(power_total)



