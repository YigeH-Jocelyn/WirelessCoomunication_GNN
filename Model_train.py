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

from Utility import PinchingDataset, ee_loss, average_power
from Utility import M, N, D, H, L, Pmax_linear, sigma2, delta_min,eta, d_blocks, h_dim, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST
from Model_BGAT import BGATModel
from Model_GAT import GATModel
from Model_MLP import MLPModel

# Training Parameters
EPOCHS = 1000
BATCH_SIZE = 2048
LR = 5e-5
CLIP_GRAD = 0.5

# Set seeds
SEED = 3407
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

def train_model(model, train_loader, test_loader, model_type, lr=LR, epochs=EPOCHS, patience=50, tolerance=1e-1):
    model.to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=100, T_mult=1, eta_min=1e-5
    # )
    # num_training_steps = EPOCHS * len(train_loader)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)

    best_ee = -np.inf
    epoch_list, train_loss_list,train_ee_list, val_loss_list, val_ee_list = [], [], [], [], []
    power_list = []
    epochs_no_improve, last_train_loss, last_val_ee = 0, None, None

    for epoch in range(epochs):
        model.train()
        train_loss, train_ee = 0, 0
        train_power_sum = 0
        for users, delta, power, ants in train_loader:
            users, delta, power, ants = users.to(device), delta.to(device), power.to(device), ants.to(device)
            optimizer.zero_grad()
            if model_type == 'MLP':
                p_pred, d_pred = model(users)
                ants_pred = model.delta_to_position(d_pred)
                loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
            elif model_type == 'BGAT':
                p_pred, d_pred, ants_pred = model(users, delta, power)
                # loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
                loss = 0.0
                for i, (p, pos) in enumerate(ants_pred, start=0):
                    weight = (i + 1) / len(ants_pred)
                    # weight = 1
                    loss += weight * ee_loss(users, pos, p, eta, sigma2)

            elif model_type == 'GAT':
                p_pred, d_pred = model(users)
                ants_pred = model.delta_to_position(d_pred)
                loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
            loss.backward()
            # total_grad_norm = 0.0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm(2)
            #         total_grad_norm += grad_norm.item() ** 2
            #         print(f"Gradient norm for {name}: {grad_norm.item():.4f}")
            # total_grad_norm = total_grad_norm ** 0.5
            # print(f"Total gradient norm: {total_grad_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD)
            torch.autograd.set_detect_anomaly(True)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * users.size(0)
            train_ee += -loss.item() * users.size(0)

            batch_avg_power = average_power(power)
            train_power_sum += batch_avg_power.item() * users.size(0)

        model.eval()
        val_loss, val_ee = 0, 0
        val_power_sum = 0
        with torch.no_grad():
            for users, delta, power, ants in test_loader:
                if model_type == 'MLP':
                    p_pred, d_pred = model(users)
                    ants_pred = model.delta_to_position(d_pred)
                    loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
                elif model_type == 'BGAT':
                    p_pred, d_pred, ants_pred = model(users, delta, power)
                    # loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
                    loss = 0.0
                    for i, (p, pos) in enumerate(ants_pred):
                        weight = (i + 1) / len(ants_pred)
                        # weight = 1
                        loss += weight * ee_loss(users, pos, p, eta, sigma2)
                elif model_type == 'GAT':
                    p_pred, d_pred = model(users)
                    ants_pred = model.delta_to_position(d_pred)
                    loss = ee_loss(users, ants_pred, p_pred, eta, sigma2)
                val_loss += loss.item() * users.size(0)
                val_ee += -loss.item() * users.size(0)

                batch_avg_power = average_power(power)
                val_power_sum += batch_avg_power.item() * users.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_ee = train_ee / len(train_loader.dataset)
        avg_train_power = train_power_sum / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_ee = val_ee / len(test_loader.dataset)
        avg_val_power = val_power_sum / len(test_loader.dataset)

        epoch_list.append(epoch + 1)
        train_loss_list.append(avg_train_loss)
        val_ee_list.append(avg_val_ee)
        train_ee_list.append(avg_train_ee)
        val_loss_list.append(avg_val_loss)
        power_list.append((avg_train_power, avg_val_power))

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train EE: {avg_train_ee:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val EE: {avg_val_ee:.4f} | '
              f'Avg Val Power: {avg_val_power:.4f}')

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

if __name__ == "__main__":
    train_dataset = PinchingDataset(NUM_SAMPLES_TRAIN)
    test_dataset = PinchingDataset(NUM_SAMPLES_TEST)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    metrics_dict = {}
    models = {
        "BGAT": BGATModel(D_blocks=d_blocks, user_dim=2, ant_dim=2, hidden_dim=h_dim,num_heads=8,
                          waveguide_bound=D, delta_min=delta_min, H=H, Pmax=Pmax_linear,N=N,M=M,L=L),
        "GAT": GATModel(in_dim=3, hidden_dim=h_dim,num_layers=2, num_heads=8, delta_min=delta_min, H=H,
                        waveguide_bound=D, N=N, Pmax=Pmax_linear),
        "MLP": MLPModel(M=M, N=N, Pmax_linear=Pmax_linear,delta_min=delta_min,waveguide_bound=D,H=H)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name} Model...")
        best_ee, metrics = train_model(model, train_loader, test_loader, model_type=model_name)
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

