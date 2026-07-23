# Input: Heat, Change in heat also.

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.table as mtable

# ==========================================
# 1. CONFIGURATION
# ==========================================
SUBSET_PATH = r"D:\Relationship (correlation) between the Signals\FREI datasets"
VAL_PATH = r"D:\Relationship (correlation) between the Signals\Validation - Data"

WINDOW_SIZE = 1500  
STEP_SIZE = 3       
SAMPLES_PER_EPOCH = 500 
NUM_STATIC_VAL_WINDOWS = 400 

# Restored network capacity to handle sharp waves
HIDDEN_SIZE = 128 
NUM_LAYERS = 4      
DROPOUT = 0.4       
LEARNING_RATE = 0.001 
EPOCHS = 500 
BATCH_SIZE = 8
PATIENCE = 20 

base_path = r"D:\Relationship (correlation) between the Signals\Results2\LSTM14_Predictions"

# ==========================================
# 2. DATA LOADING & DATASETS
# ==========================================
def load_full_datasets(folder_path):
    data_list, file_names, scaler_dict = [], [], {} 
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for file in files:
        try:
            df = pd.read_csv(file, sep='\t')
            if df.shape[1] >= 3 and len(df) > WINDOW_SIZE:
                data_list.append(df)
                fname = os.path.basename(file).replace('.txt', '')
                file_names.append(fname)
                
                heat = (df.iloc[:, 0].values * -1).reshape(-1, 1)
                pressure = df.iloc[:, 2].values.reshape(-1, 1)
                
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                scaler_x.fit(heat)
                scaler_y.fit(pressure)
                
                scaler_dict[fname] = {'x': scaler_x, 'y': scaler_y}
        except Exception as e:
            pass
    return data_list, file_names, scaler_dict

def filter_by_phi(data_list, file_names, target_phi):
    filtered_data, filtered_names = [], []
    for df, fname in zip(data_list, file_names):
        if f"Phi_{target_phi}" in fname:
            filtered_data.append(df)
            filtered_names.append(fname)
    return filtered_data, filtered_names

class OpenLoopDataset(Dataset):
    def __init__(self, data_list, file_names, scaler_dict, window_size=1500, step_size=4, samples_per_epoch=500):
        self.data = data_list
        self.file_names = file_names
        self.scaler_dict = scaler_dict
        self.W = window_size
        self.step = step_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self): return self.samples_per_epoch

    def __getitem__(self, idx):
        file_idx = random.randint(0, len(self.data) - 1)
        file_df = self.data[file_idx]
        global_scalers = self.scaler_dict[self.file_names[file_idx]] 
        
        start_idx = random.randint(0, len(file_df) - self.W)
        reduced_window = file_df.iloc[start_idx : start_idx + self.W][::self.step]
        
        time_full = np.linspace(0.0, 1.0, len(file_df))
        reduced_time = time_full[start_idx : start_idx + self.W][::self.step].reshape(-1, 1)
        
        heat = (reduced_window.iloc[:, 0].values * -1).reshape(-1, 1)
        pressure = reduced_window.iloc[:, 2].values.reshape(-1, 1)

        h_scaled = global_scalers['x'].transform(heat)
        p_scaled = global_scalers['y'].transform(pressure)

        # FEATURE ENGINEERING: Calculate Delta Heat (Acceleration)
        h_diff = np.diff(h_scaled, axis=0, prepend=h_scaled[0:1])

        # Input is now 3 features: [Heat, Delta_Heat, Time]
        X_combined = np.hstack((h_scaled, h_diff, reduced_time))
        y_target = p_scaled 

        return torch.tensor(X_combined, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

class StaticValidationDataset(Dataset):
    def __init__(self, data_list, file_names, scaler_dict, window_size=1500, step_size=4, num_samples=100):
        self.static_windows = []
        for _ in range(num_samples):
            file_idx = random.randint(0, len(data_list) - 1)
            file_df = data_list[file_idx]
            global_scalers = scaler_dict[file_names[file_idx]]
            
            start_idx = random.randint(0, len(file_df) - window_size)
            reduced_window = file_df.iloc[start_idx : start_idx + window_size][::step_size]
            
            time_full = np.linspace(0.0, 1.0, len(file_df))
            reduced_time = time_full[start_idx : start_idx + window_size][::step_size].reshape(-1, 1)
            
            heat = (reduced_window.iloc[:, 0].values * -1).reshape(-1, 1)
            pressure = reduced_window.iloc[:, 2].values.reshape(-1, 1)

            h_scaled = global_scalers['x'].transform(heat)
            p_scaled = global_scalers['y'].transform(pressure)

            # FEATURE ENGINEERING: Calculate Delta Heat
            h_diff = np.diff(h_scaled, axis=0, prepend=h_scaled[0:1])

            X_combined = np.hstack((h_scaled, h_diff, reduced_time))
            y_target = p_scaled
            
            self.static_windows.append((torch.tensor(X_combined, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)))

    def __len__(self): return len(self.static_windows)
    def __getitem__(self, idx): return self.static_windows[idx]

# ==========================================
# 3. MODEL DEFINITION & CUSTOM LOSS
# ==========================================
class StandardLSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class PeakWeightedLoss(nn.Module):
    def __init__(self, peak_weight=2.0):
        super().__init__()
        self.peak_weight = peak_weight

    def forward(self, pred, target):
        base_error = (pred - target) ** 2
        # Dynamic multiplier: Errors on large spikes are penalized up to 3x more heavily
        dynamic_multiplier = 1.0 + (torch.abs(target) * self.peak_weight)
        weighted_error = base_error * dynamic_multiplier
        return torch.mean(weighted_error)

# ==========================================
# 4. EXECUTION & TRAINING
# ==========================================
if __name__ == "__main__":
    master_train_data, master_train_names, master_train_scalers = load_full_datasets(SUBSET_PATH)
    master_val_data, master_val_names, master_val_scalers = load_full_datasets(VAL_PATH)

    for target_phi in ["1p0", "1p2"]:
        print(f"\n========================================================")
        print(f"   STARTING ADVANCED TRAINING FOR PHI = {target_phi}  ")
        print(f"========================================================")
        
        save_folder = os.path.join(base_path, f"Advanced_LSTM_Phi_{target_phi}")
        os.makedirs(save_folder, exist_ok=True)
        CHECKPOINT_PATH = os.path.join(save_folder, f"best_model_{target_phi}.pth")

        train_data_list, train_file_names = filter_by_phi(master_train_data, master_train_names, target_phi)
        val_data_list, val_file_names = filter_by_phi(master_val_data, master_val_names, target_phi)
        if len(train_data_list) == 0: continue

        train_dataset = OpenLoopDataset(train_data_list, train_file_names, master_train_scalers, WINDOW_SIZE, STEP_SIZE, SAMPLES_PER_EPOCH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = StaticValidationDataset(val_data_list, val_file_names, master_val_scalers, WINDOW_SIZE, STEP_SIZE, NUM_STATIC_VAL_WINDOWS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Input Size is now 3: [Heat, Delta Heat, Time]
        model = StandardLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1, dropout=DROPOUT)
        
        criterion = PeakWeightedLoss(peak_weight=2.0)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss = float('inf')
        counter = 0
        history_train_loss = []
        history_val_loss = []

        for epoch in range(EPOCHS):
            model.train()
            t_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            model.eval()
            with torch.no_grad():
                v_loss = sum(criterion(model(vx), vy).item() for vx, vy in val_loader) / len(val_loader)
            
            avg_train = t_loss/len(train_loader)
            history_train_loss.append(avg_train)
            history_val_loss.append(v_loss)
            
            print(f"Phi {target_phi} | Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_train:.5f} | Val: {v_loss:.5f}")
            
            scheduler.step(v_loss)

            if v_loss < best_loss:
                best_loss, counter = v_loss, 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
            else:
                counter += 1
                if counter >= PATIENCE: 
                    print("Early Stopping.")
                    break

        # --- PLOT 1: LOSS CURVE ---
        plt.figure(figsize=(8, 5))
        plt.plot(history_train_loss, color='blue', label='Train Loss')
        plt.plot(history_val_loss, color='red', linestyle='--', label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        # Small legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small', borderaxespad=0.)
        plt.savefig(os.path.join(save_folder, f"1_Loss_Curve.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 2: UNIFORM WINDOWS & TABLE METRICS ---
        print(f"\n--- GENERATING PLOTS ---")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()
        
        global_results_table = []
        
        with torch.no_grad():
            eval_fname = val_file_names[0]
            file_df = val_data_list[0]
            global_scalers = master_val_scalers[eval_fname]

            uniform_starts = np.linspace(0, len(file_df) - WINDOW_SIZE, num=6, dtype=int)
            
            for i, start_idx in enumerate(uniform_starts):
                reduced_window = file_df.iloc[start_idx : start_idx + WINDOW_SIZE][::STEP_SIZE]
                reduced_time = np.linspace(0.0, 1.0, len(file_df))[start_idx : start_idx + WINDOW_SIZE][::STEP_SIZE].reshape(-1, 1)
                
                heat = (reduced_window.iloc[:, 0].values * -1).reshape(-1, 1)
                pressure = reduced_window.iloc[:, 2].values.reshape(-1, 1)

                h_scaled = global_scalers['x'].transform(heat)
                p_scaled = global_scalers['y'].transform(pressure)

                h_diff = np.diff(h_scaled, axis=0, prepend=h_scaled[0:1])

                X_combined = np.hstack((h_scaled, h_diff, reduced_time))
                val_X_input = torch.tensor(X_combined, dtype=torch.float32).unsqueeze(0)
                
                pred_scaled = model(val_X_input)
                
                pred_original = global_scalers['y'].inverse_transform(pred_scaled.squeeze().numpy().reshape(-1, 1))
                real_original = pressure

                w_rmse = np.sqrt(mean_squared_error(real_original, pred_original))
                w_r2 = r2_score(real_original, pred_original)
                w_pcc, _ = pearsonr(real_original.flatten(), pred_original.flatten())
                
                global_results_table.append([f"W{i+1}", f"{w_rmse:.4f}", f"{w_r2:.4f}", f"{w_pcc:.4f}"])
                
                plt.figure(figsize=(10, 4))
                plt.plot(real_original, color='blue', label='Real')
                plt.plot(pred_original, color='red', linestyle='--', label='Generated')
                plt.title(f"Window {i+1} / 6 - {eval_fname}\nRMSE: {w_rmse:.4f} | R2: {w_r2:.4f} | PCC: {w_pcc:.4f}")
                plt.xlabel('Time Steps')
                plt.ylabel('Pressure')
                plt.grid(True, alpha=0.3)
                # Small legend outside the plot
                plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', borderaxespad=0.)
                plt.savefig(os.path.join(save_folder, f"2_Window_{i+1}.png"), dpi=150, bbox_inches='tight')
                plt.close()

        # --- PLOT 3: PERFORMANCE METRICS TABLE ---
        avg_rmse = np.mean([float(row[1]) for row in global_results_table])
        avg_r2 = np.mean([float(row[2]) for row in global_results_table])
        avg_pcc = np.mean([float(row[3]) for row in global_results_table])
        global_results_table.append(["Average", f"{avg_rmse:.4f}", f"{avg_r2:.4f}", f"{avg_pcc:.4f}"])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        table = mtable.table(ax, cellText=global_results_table, 
                             colLabels=["Window", "RMSE", "R2", "PCC"], 
                             loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        plt.title("Performance Metrics", pad=20)
        plt.savefig(os.path.join(save_folder, f"3_Metrics_Table.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # --- PLOT 4 & 5: MACRO TEST (FULL DATASET) ---
        with torch.no_grad():
            pressure_full = file_df.iloc[:, 2].values.reshape(-1, 1)
            time_full = np.linspace(0.0, 1.0, len(file_df)).reshape(-1, 1)
            
            heat_dec = (file_df.iloc[:, 0].values * -1).reshape(-1, 1)[::STEP_SIZE]
            pressure_dec = file_df.iloc[:, 2].values.reshape(-1, 1)[::STEP_SIZE]
            time_dec = time_full[::STEP_SIZE]
            
            h_scaled = global_scalers['x'].transform(heat_dec)
            p_scaled = global_scalers['y'].transform(pressure_dec)
            h_diff_full = np.diff(h_scaled, axis=0, prepend=h_scaled[0:1])
            
            X_combined = np.hstack((h_scaled, h_diff_full, time_dec))
            val_X_input = torch.tensor(X_combined, dtype=torch.float32).unsqueeze(0)
            
            pred_scaled = model(val_X_input)
            pred_original = global_scalers['y'].inverse_transform(pred_scaled.squeeze().numpy().reshape(-1, 1))
            real_dec_original = pressure_dec
            
            x_gen = np.arange(0, len(pred_original)) * STEP_SIZE
            
            # Plot 4: Downsampled
            plt.figure(figsize=(12, 5))
            plt.plot(real_dec_original, color='blue', label='Real')
            plt.plot(pred_original, color='red', linestyle='--', label='Generated')
            plt.title(f"Full Dataset (Downsampled) - {eval_fname}")
            plt.xlabel('Time Steps')
            plt.ylabel('Pressure')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', borderaxespad=0.)
            plt.savefig(os.path.join(save_folder, f"4_Full_Dataset_Downsampled.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 5: Raw
            plt.figure(figsize=(12, 5))
            plt.plot(pressure_full, color='blue', alpha=0.5, label='Real (Raw)')
            plt.plot(x_gen, pred_original, color='red', label='Generated')
            plt.title(f"Full Dataset (Raw) - {eval_fname}")
            plt.xlabel('Time Steps')
            plt.ylabel('Pressure')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', borderaxespad=0.)
            plt.savefig(os.path.join(save_folder, f"5_Full_Dataset_Raw.png"), dpi=150, bbox_inches='tight')
            plt.close()
