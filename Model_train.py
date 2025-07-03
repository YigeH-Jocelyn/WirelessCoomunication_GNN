import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import gc
torch.cuda.empty_cache()
gc.collect()

from Utility import PinchingDataset, dynamic_ee_loss, average_power
from Utility import M, N, D, H, L, Pmax_linear, sigma2, delta_min,eta, d_blocks, h_dim, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST
from Model_BGAT import BGATModel
from Model_GAT import GATModel
from Model_MLP import MLPModel

# Training Parameters
EPOCHS = 5000
BATCH_SIZE = 2048
LR = 5e-3
CLIP_GRAD = 0.5

# Set seeds
SEED = 3407
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

def train_model_dynamic(model, train_loader, test_loader, lr=LR, epochs=EPOCHS,
                        patience=50, tolerance=1e-2, clip_grad=CLIP_GRAD):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    best_ee = -float("inf")
    epoch_list, train_loss_list, train_ee_list, val_loss_list, val_ee_list = [], [], [], [], []
    power_list = []
    epochs_no_improve, last_train_loss, last_val_ee = 0, None, None
    delta_vals = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_ee = 0, 0
        delta_vals.append(torch.sigmoid(model.logit_delta).item())

        for users, delta, _, _ in train_loader:  # no power, no antenna needed
            users, delta = users.to(device), delta.to(device)
            optimizer.zero_grad()

            delta_scalar, final_positions, intermediate_outputs = model(users, delta)

            loss = 0.0
            for _, pos in intermediate_outputs:
                loss += dynamic_ee_loss(users, pos, delta_scalar, eta, sigma2)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * users.size(0)
            train_ee += -loss.item() * users.size(0)

        model.eval()
        val_loss, val_ee = 0, 0
        with torch.no_grad():
            for users, delta, _, _ in test_loader:
                users, delta = users.to(device), delta.to(device)
                delta_scalar, final_positions, intermediate_outputs = model(users, delta)

                loss = 0.0
                for _, pos in intermediate_outputs:
                    loss += dynamic_ee_loss(users, pos, delta_scalar, eta, sigma2)

                val_loss += loss.item() * users.size(0)
                val_ee += -loss.item() * users.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_ee = train_ee / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_ee = val_ee / len(test_loader.dataset)

        epoch_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        train_ee_list.append(avg_train_ee)
        val_loss_list.append(avg_val_loss)
        val_ee_list.append(avg_val_ee)

        print(f'Epoch {epoch+1}/{epochs} | Train EE: {avg_train_ee:.4f} | Val EE: {avg_val_ee:.4f} | '
              f'Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | δ: {delta_scalar.item():.4f}')

        if last_train_loss is not None and last_val_ee is not None:
            diff_loss = abs(avg_train_loss - last_train_loss)
            diff_ee = abs(avg_val_ee - last_val_ee)
            if diff_loss < tolerance and diff_ee < tolerance:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
        else:
            epochs_no_improve = 0

        if avg_val_ee > best_ee:
            best_ee = avg_val_ee
            epochs_no_improve = 0

        last_train_loss, last_val_ee = avg_train_loss, avg_val_ee

        if epochs_no_improve >= patience:
            break

    print(f'Best Validation EE: {best_ee:.4f}')
    return best_ee, {
        'epoch': epoch_list,
        'train_loss': train_loss_list,
        'train_ee': train_ee_list,
        'val_loss': val_loss_list,
        'val_ee': val_ee_list,
        'delta': delta_vals
    }

if __name__ == "__main__":
    train_dataset = PinchingDataset(NUM_SAMPLES_TRAIN)
    test_dataset = PinchingDataset(NUM_SAMPLES_TEST)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    metrics_dict = {}
    models = {
        # "BGAT": BGATModel(D_blocks=d_blocks, user_dim=2, ant_dim=2, hidden_dim=h_dim,num_heads=8,
        #                   waveguide_bound=D, delta_min=delta_min, H=H, Pmax=Pmax_linear,N=N,M=M,L=L),
        "BGAT": BGATModel(D_blocks=d_blocks, user_dim=2, ant_dim=1, hidden_dim=h_dim, num_heads=8,
                          waveguide_bound=D, delta_min=delta_min, H=H, Pmax=Pmax_linear, N=N, M=M, L=L),
        # "GAT": GATModel(in_dim=3, hidden_dim=h_dim,num_layers=2, num_heads=8, delta_min=delta_min, H=H,
        #                 waveguide_bound=D, N=N, Pmax=Pmax_linear),
        # "MLP": MLPModel(M=M, N=N, Pmax_linear=Pmax_linear,delta_min=delta_min,waveguide_bound=D,H=H)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name} Model...")
        # best_ee, metrics = train_model(model, train_loader, test_loader, model_type=model_name)
        if model_name == "BGAT":
            best_ee, metrics = train_model_dynamic(model, train_loader, test_loader)
        else:
            best_ee, metrics = train_model_dynamic(model, train_loader, test_loader, model_type=model_name)

        metrics_dict[model_name] = metrics

    output_path = r"C:\Users\jocel\Desktop\Education\UToronto\2025 Winter\ECE2500Y\results\Results.xlsx"
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

