import os
import glob
import re
from typing import Tuple, Dict, List
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- New Imports for Automated Analysis ---
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from sklearn.linear_model import LinearRegression

from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# --- 1. DATA HANDLING (No changes) ---
class FlameDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 500, downsample_step: int = 10):
        self.sequence_length = sequence_length
        self.downsample_step = downsample_step
        self.final_seq_len = sequence_length // downsample_step
        self.samples = []
        self._prepare_samples(data_dir)
        if not self.samples:
            print(f"Warning: No valid .txt files found in '{data_dir}'. Dataset will be empty.")
            self.norm_params = {'global_mean': 0.0, 'global_std': 1.0}
            return
        self.norm_params = self._calculate_norm_params()

    def _prepare_samples(self, data_dir: str):
        for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
            try:
                filename = os.path.basename(filepath)
                phi_match = re.search(r'Phi_(\d+)p?(\d*)', filename)
                u_match = re.search(r'u_(\d+)p?(\d*)', filename)
                if phi_match and u_match:
                    phi = float(f"{phi_match.group(1)}.{phi_match.group(2) or '0'}")
                    u = float(f"{u_match.group(1)}.{u_match.group(2) or '0'}")
                    self.samples.append((filepath, (phi, u)))
            except (ValueError, IndexError):
                print(f"Warning: Skipping file with unexpected format: {filename}")

    def _calculate_norm_params(self) -> Dict[str, float]:
        all_data = [pd.read_csv(f[0], sep='\t', header=None).values.astype(np.float32) for f in self.samples]
        if not all_data: return {'global_mean': 0.0, 'global_std': 1.0}
        full_dataset = np.concatenate(all_data, axis=0)
        mean, std = full_dataset.mean(), full_dataset.std()
        return {'global_mean': mean, 'global_std': std}

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath, params = self.samples[idx]
        data = pd.read_csv(filepath, sep='\t', header=None).values.astype(np.float32)
        if len(data) > self.sequence_length: data = data[:self.sequence_length]
        elif len(data) < self.sequence_length:
            padding = np.tile(data[-1], (self.sequence_length - len(data), 1))
            data = np.vstack([data, padding])
        
        data = data[::self.downsample_step]
        norm_mean, norm_std = self.norm_params['global_mean'], self.norm_params['global_std']
        normalized_data = (data - norm_mean) / (norm_std + 1e-8)
        return torch.from_numpy(normalized_data).float(), torch.tensor(params, dtype=torch.float32)

    def denormalize_data(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        norm_mean, norm_std = self.norm_params['global_mean'], self.norm_params['global_std']
        return normalized_tensor * (norm_std + 1e-8) + norm_mean

# --- 2. AUTOMATED FORMULA BUILDER (No changes) ---
def analyze_and_build_formula(samples: List[Tuple[str, Tuple[float, float]]], seq_len: int, downsample_step: int, decomposition_period: int = 20):
    print("\n--- Starting Automated Formula Analysis ---")
    if not samples:
        print("No training data to analyze. Returning empty formula models.")
        return {}
        
    params = []
    features = {'heat_amp': [], 'heat_freq': [], 'heat_slope': [], 'heat_intercept': [],
                'pressure_amp': [], 'pressure_freq': [], 'pressure_slope': [], 'pressure_intercept': [],
                'time_slope': [], 'time_intercept': []}

    for filepath, (phi, u) in samples:
        df = pd.read_csv(filepath, sep='\t', header=None, names=['Heat', 'Time', 'Pressure'])
        df = df[::downsample_step].iloc[:seq_len]
        params.append([phi, u])
        time_axis = np.arange(len(df))

        # Analyze Heat and Pressure with decomposition
        for signal_name, key_prefix in [('Heat', 'heat'), ('Pressure', 'pressure')]:
            series = df[signal_name]
            # Ensure period is less than the series length
            current_period = min(decomposition_period, len(series) // 2) 
            if current_period < 2: current_period = 2 # minimum valid period
            
            try:
                # Handle short or constant series
                if len(series) <= current_period * 2 or series.nunique() == 1:
                    raise ValueError("Series is too short or constant for decomposition")

                decomp = seasonal_decompose(series, model='additive', period=current_period)
                
                # Analyze Trend (Slope and Intercept)
                trend = decomp.trend.dropna()
                if len(trend) >= 2:
                    slope, intercept = np.polyfit(trend.index, trend, 1)
                else:
                    slope, intercept = 0.0, trend.mean() if not trend.empty else series.mean()
                
                features[f'{key_prefix}_slope'].append(slope)
                features[f'{key_prefix}_intercept'].append(intercept)

                # Analyze Seasonality (Amplitude and Frequency)
                seasonal = decomp.seasonal.dropna()
                if not seasonal.empty and seasonal.nunique() > 1:
                    amp = (seasonal.max() - seasonal.min()) / 2
                    _, Pxx = signal.periodogram(seasonal, fs=1.0)
                    freq = np.argmax(Pxx) / len(Pxx) if len(Pxx) > 0 else 0
                else:
                    amp, freq = 0.0, 0.0
                    
                features[f'{key_prefix}_amp'].append(amp)
                features[f'{key_prefix}_freq'].append(freq)

            except Exception as e:
                # Fallback: simple linear fit on the raw series
                print(f"Warning: Could not decompose {signal_name} in {os.path.basename(filepath)}. Using linear fit. Error: {e}")
                slope, intercept = np.polyfit(time_axis, series, 1)
                features[f'{key_prefix}_slope'].append(slope)
                features[f'{key_prefix}_intercept'].append(intercept)
                features[f'{key_prefix}_amp'].append(0)
                features[f'{key_prefix}_freq'].append(0)

        # Analyze Time with a linear fit
        slope, intercept = np.polyfit(time_axis, df['Time'], 1)
        features['time_slope'].append(slope)
        features['time_intercept'].append(intercept)

    # --- Build regression models to generalize the formula ---
    formula_models = {}
    X = np.array(params)
    print("\n--- Derived Formula Coefficients (Parameter = C1*phi + C2*u + Intercept) ---")
    for key, y in features.items():
        model = LinearRegression()
        model.fit(X, np.array(y))
        formula_models[key] = model
        print(f"  {key:<18}: C1={model.coef_[0]:.4f}, C2={model.coef_[1]:.4f}, Intercept={model.intercept_:.4f}")
    
    print("--- Analysis Complete ---")
    return formula_models

# --- 3. TRANSFORMER BUILDING BLOCKS (No changes) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x, **kwargs):
        res = x; x = self.fn(x, **kwargs); x += res; return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(nn.Linear(emb_size, expansion * emb_size), nn.SiLU(), nn.Dropout(drop_p), nn.Linear(expansion * emb_size, emb_size))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=5, drop_p=0., forward_expansion=4, forward_drop_p=0.):
        super().__init__(ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size), MultiHeadAttention(emb_size, num_heads, drop_p), nn.Dropout(drop_p))), ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size), FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p), nn.Dropout(drop_p))))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# --- 4. GAN MODELS (No changes) ---
class Generator(nn.Module):
    def __init__(self, seq_len=100, channels=3, latent_dim=100, embed_dim=192, c_dim=2, depth=6, num_heads=4, **kwargs):
        super().__init__()
        self.latent_dim, self.c_dim, self.seq_len, self.embed_dim = latent_dim, c_dim, seq_len, embed_dim
        self.cond_embed = nn.Sequential(nn.Linear(self.c_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = TransformerEncoder(depth=depth, emb_size=embed_dim, num_heads=num_heads, **kwargs)
        self.deconv_head = nn.Sequential(nn.Conv1d(embed_dim, channels, kernel_size=1), nn.Tanh())

    def forward(self, z, c):
        x = self.l1(z).reshape(-1, self.seq_len, self.embed_dim) + self.pos_embed
        x = x + self.cond_embed(c).unsqueeze(1)
        x = self.blocks(x)
        return self.deconv_head(x.permute(0, 2, 1))

class Discriminator(nn.Module):
    def __init__(self, seq_len=100, channels=3, embed_dim=256, c_dim=2, depth=7, num_heads=4, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.cond_embed = nn.Sequential(nn.Linear(self.c_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.project_in = nn.Linear(channels, embed_dim)
        self.transformer = TransformerEncoder(depth=depth, emb_size=embed_dim, num_heads=num_heads, **kwargs)
        self.cnn_head = nn.Sequential(nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv1d(embed_dim * 2, 1, kernel_size=1))

    def forward(self, x, c):
        x = x.permute(0, 2, 1)
        x_proj = self.project_in(x)
        x_combined = x_proj + self.cond_embed(c).unsqueeze(1)
        transformed_features = self.transformer(x_combined)
        return self.cnn_head(transformed_features.permute(0, 2, 1))

# --- 5. TRAINING ORCHESTRATOR (Contains the main modifications) ---
class FlameGANTrainer:
    def __init__(self, train_dataset: FlameDataset, val_dataset: FlameDataset, formula_models: Dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.formula_models = formula_models
        self.seq_len = self.train_dataset.final_seq_len
        self.generator = None
        self.discriminator = None

    def build_model(self, latent_dim, embed_dim_g, depth_g, embed_dim_d, depth_d, num_heads):
        self.generator = Generator(seq_len=self.seq_len, latent_dim=latent_dim, embed_dim=embed_dim_g, depth=depth_g, num_heads=num_heads).to(self.device)
        self.discriminator = Discriminator(seq_len=self.seq_len, embed_dim=embed_dim_d, depth=depth_d, num_heads=num_heads).to(self.device)
        print("Data-Driven Formula-Guided WGAN-GP models built successfully.")

    def _calculate_formula_signal(self, conditions: torch.Tensor) -> torch.Tensor:
        cond_np = conditions.cpu().numpy()
        # Use a normalized time axis [0, 1] for stable sine/cosine
        time_axis = torch.linspace(0, 1, self.seq_len, device=self.device).unsqueeze(0)

        def predict_param(key):
            pred = self.formula_models[key].predict(cond_np)
            return torch.from_numpy(pred).float().to(self.device).unsqueeze(1)

        heat_amp, heat_freq = predict_param('heat_amp'), predict_param('heat_freq')
        heat_slope, heat_intercept = predict_param('heat_slope'), predict_param('heat_intercept')
        pressure_amp, pressure_freq = predict_param('pressure_amp'), predict_param('pressure_freq')
        pressure_slope, pressure_intercept = predict_param('pressure_slope'), predict_param('pressure_intercept')
        time_slope, time_intercept = predict_param('time_slope'), predict_param('time_intercept')
        
        # Build signals
        # Use a scaled time_axis for the trend part to match np.polyfit behavior
        trend_time_axis = torch.arange(0, self.seq_len, device=self.device).float().unsqueeze(0)
        
        heat = (heat_slope * trend_time_axis + heat_intercept) + heat_amp * torch.sin(2 * torch.pi * heat_freq * self.seq_len * time_axis)
        pressure = (pressure_slope * trend_time_axis + pressure_intercept) + pressure_amp * torch.cos(2 * torch.pi * pressure_freq * self.seq_len * time_axis)
        time_signal = time_slope * trend_time_axis + time_intercept
        
        unnormalized_signal = torch.stack([heat, time_signal, pressure], dim=1)
        
        # Normalize the *entire* signal at the end
        mean = self.train_dataset.norm_params['global_mean']
        std = self.train_dataset.norm_params['global_std']
        normalized_signal = (unnormalized_signal - mean) / (std + 1e-8)
        
        return normalized_signal

    def _gradient_penalty(self, real_samples, fake_samples, conditions):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, conditions)
        grad_outputs = torch.ones_like(d_interpolates, requires_grad=False)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _validate(self, val_loader):
        self.generator.eval(); self.discriminator.eval()
        val_w_dist = []
        if not val_loader: return 0.0
        with torch.no_grad():
            for real_data, params in val_loader:
                real_data_input, conditions = real_data.to(self.device).permute(0, 2, 1), params.to(self.device)
                noise = torch.randn(real_data.size(0), self.generator.latent_dim, device=self.device) * 0.1
                fake_data = self.generator(noise, conditions)
                d_real_score = self.discriminator(real_data_input, conditions)
                d_fake_score = self.discriminator(fake_data, conditions)
                w_dist = d_real_score.mean() - d_fake_score.mean()
                val_w_dist.append(w_dist.item())
        return np.mean(val_w_dist) if val_w_dist else 0.0
        
    def train(self, train_loader, val_loader, epochs, lr_g, lr_d, gp_weight=10, 
              n_critic_steps=5, patience=25, lambda_formula=0.1, checkpoint_path='best_model.pth.tar'):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
        best_val_w_dist = float('inf'); epochs_no_improve = 0
        print(f"\n--- Starting Data-Driven Formula-Guided Training (λ_formula={lambda_formula}) ---")
        
        for epoch in range(epochs):
            self.generator.train(); self.discriminator.train()
            g_loss_val, d_loss_val, formula_loss_val = 0.0, 0.0, 0.0
            for real_data, params in train_loader:
                real_data_input, conditions = real_data.to(self.device).permute(0, 2, 1), params.to(self.device)
                current_batch_size = real_data_input.size(0)
                
                for _ in range(n_critic_steps):
                    optimizer_d.zero_grad()
                    noise = torch.randn(current_batch_size, self.generator.latent_dim, device=self.device) * 0.1
                    fake_data = self.generator(noise, conditions)
                    d_real_score = self.discriminator(real_data_input, conditions)
                    d_fake_score = self.discriminator(fake_data.detach(), conditions)
                    gp = self._gradient_penalty(real_data_input, fake_data.detach(), conditions)
                    d_loss = d_fake_score.mean() - d_real_score.mean() + gp_weight * gp
                    d_loss.backward(); optimizer_d.step()
                    d_loss_val = d_loss.item()
                
                optimizer_g.zero_grad()
                noise = torch.randn(current_batch_size, self.generator.latent_dim, device=self.device) * 0.1
                fake_data_g = self.generator(noise, conditions)
                d_fake_score_g = self.discriminator(fake_data_g, conditions)
                g_adv_loss = -d_fake_score_g.mean()
                
                g_loss = g_adv_loss
                if self.formula_models:
                    formula_signal = self._calculate_formula_signal(conditions)
                    formula_loss = F.mse_loss(fake_data_g, formula_signal)
                    g_loss = g_adv_loss + (lambda_formula * formula_loss)
                    formula_loss_val = formula_loss.item()

                g_loss.backward(); optimizer_g.step()
                g_loss_val = g_loss.item()

            avg_val_w_dist = self._validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | G_loss: {g_loss_val:.4f} (Adv: {g_adv_loss.item():.4f}, Formula: {formula_loss_val:.4f}), D_loss: {d_loss_val:.4f} | Val W-Dist: {avg_val_w_dist:.4f}")
            if not np.isnan(avg_val_w_dist) and abs(avg_val_w_dist) < abs(best_val_w_dist):
                print(f"=> Val W-Dist improved from {best_val_w_dist:.4f} to {avg_val_w_dist:.4f}. Saving model...")
                best_val_w_dist = avg_val_w_dist
                epochs_no_improve = 0
                torch.save({'generator_state_dict': self.generator.state_dict(), 'formula_models': self.formula_models}, checkpoint_path)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {patience} epochs."); break
        print("--- Training complete ---")

    def generate_flame_data(self, conditions: torch.Tensor) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(conditions.size(0), self.generator.latent_dim, device=self.device) * 0.1
            generated_data_norm = self.generator(noise, conditions).permute(0, 2, 1)
            return self.train_dataset.denormalize_data(generated_data_norm).cpu().numpy()

    # --- MODIFIED 'visualize_comparison' FUNCTION ---
    def visualize_comparison(self, real_path: str, formula_signal: np.ndarray, generated_data: np.ndarray, 
                             phi: float, u: float, plots_dir: str):
        
        # 1. Load and slice the Real Data (as requested)
        try:
            real_data_full = pd.read_csv(real_path, sep='\t', header=None).values
            real_data = real_data_full[::self.train_dataset.downsample_step, :][:50]
        except Exception as e:
            print(f"Error reading real data {real_path}: {e}")
            return

        # 2. Slice the Formula and Generated data to the same length
        formula_data_sliced = formula_signal[0, :50, :]
        gen_data_sliced = generated_data[0, :50, :]
        
        # 3. Create the 3x3 plot grid
        fig, axes = plt.subplots(3, 3, figsize=(24, 12), dpi=100)
        title = (f'Visual Comparison for φ={phi:.2f}, u={u:.2f}\n'
                 f'"Formula_after_GAN" (Col 3) = "Formula_before_GAN" (Col 2) + Learned Hidden Patterns')
        fig.suptitle(title, fontsize=16)
        
        feature_names = ['Heat', 'Time', 'Pressure']
        
        # Calculate global min/max for consistent Y-axis
        global_min = min(np.min(real_data), np.min(formula_data_sliced), np.min(gen_data_sliced))
        global_max = max(np.max(real_data), np.max(formula_data_sliced), np.max(gen_data_sliced))
        y_buffer = (global_max - global_min) * 0.1
        y_lims = (global_min - y_buffer, global_max + y_buffer)

        for i, feature in enumerate(feature_names):
            ax_real, ax_formula, ax_gen = axes[i, 0], axes[i, 1], axes[i, 2]

            # Plot 1: Real Data
            ax_real.plot(real_data[:, i], color='blue', label='Real')
            ax_real.set_ylabel(feature, fontsize=12)
            ax_real.set_ylim(y_lims)
            
            # Plot 2: Formula "Blueprint" (Your "formula_before_GAN")
            ax_formula.plot(formula_data_sliced[:, i], color='green', linestyle=':', label='Blueprint (Formula)')
            ax_formula.set_ylim(y_lims)

            # Plot 3: Final Generated Data (Your "formula_after_GAN")
            ax_gen.plot(gen_data_sliced[:, i], color='red', linestyle='--', label='Final GAN Output')
            ax_gen.set_ylim(y_lims)

            # Set titles and legends
            if i == 0:
                ax_real.set_title('1. Real Data (Ground Truth)', fontsize=14)
                ax_formula.set_title('2. "Blueprint" (Formula Only)', fontsize=14)
                ax_gen.set_title('3. Final GAN Output (Blueprint + Patterns)', fontsize=14)
            
            ax_real.legend(loc='upper right')
            ax_formula.legend(loc='upper right')
            ax_gen.legend(loc='upper right')

        axes[-1, 0].set_xlabel('Time Step')
        axes[-1, 1].set_xlabel('Time Step')
        axes[-1, 2].set_xlabel('Time Step')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        # Save the plot to the specified directory
        plot_filename = os.path.join(plots_dir, f'comparison_phi_{str(phi).replace(".","p")}_u_{str(u).replace(".","p")}.png')
        plt.savefig(plot_filename)
        # plt.show() # Commented out to avoid blocking script execution
        plt.close(fig) # Close the figure to free memory


# --- 6. MAIN EXECUTION SCRIPT (Modified) ---
def main():
    TRAIN_DATA_DIR = r"D:\Data Gen\Subset - Data"
    VAL_DATA_DIR = r"D:\Data Gen\Validation - Data"

    # --- NEW: Define output directories ---
    PLOTS_DIR = r"D:\Data Gen\Plots"
    DATASETS_DIR = r"D:\Data Gen\Datasets"
    
    # --- NEW: Create directories if they don't exist ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    best_params = {
        'lr_g': 2e-7, 'lr_d': 2e-7, 'latent_dim': 100,
        'embed_dim_g': 192, 'depth_g': 6,
        'embed_dim_d': 256, 'depth_d': 7,
        'num_heads': 4, 'lambda_formula': 0.15 
    }
    
    train_dataset = FlameDataset(data_dir=TRAIN_DATA_DIR)
    val_dataset = FlameDataset(data_dir=VAL_DATA_DIR)
    
    if len(train_dataset) == 0:
        print("Error: No training data found. Exiting.")
        return
        
    if len(train_dataset.samples) > 0 and len(val_dataset.samples) > 0:
        val_dataset.norm_params = train_dataset.norm_params
    
    formula_models = analyze_and_build_formula(
        samples=train_dataset.samples, 
        seq_len=train_dataset.final_seq_len,
        downsample_step=train_dataset.downsample_step
    )

    print(f"\nData split: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(32, len(val_dataset)), shuffle=False) if len(val_dataset) > 0 else None

    trainer = FlameGANTrainer(train_dataset, val_dataset, formula_models)
    trainer.build_model(
        latent_dim=best_params['latent_dim'],
        embed_dim_g=best_params['embed_dim_g'], depth_g=best_params['depth_g'],
        embed_dim_d=best_params['embed_dim_d'], depth_d=best_params['depth_d'],
        num_heads=best_params['num_heads']
    )

    checkpoint_file = 'best_autoformula_gan.pth.tar'
    trainer.train(
        train_loader, val_loader, epochs=200, 
        lr_g=best_params['lr_g'], lr_d=best_params['lr_d'],
        patience=15, lambda_formula=best_params['lambda_formula'],
        checkpoint_path=checkpoint_file
    )

    if os.path.exists(checkpoint_file):
        print(f"\nLoading best model from '{checkpoint_file}' for generation...")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=trainer.device)
            trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
            trainer.formula_models = checkpoint['formula_models']
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Generating with last trained model.")
    else:
        print("\nWarning: No checkpoint file found. Generating with last trained model.")

    print(f"\n--- Generating and Visualizing Data ---")
    print(f"Plots will be saved to: {os.path.abspath(PLOTS_DIR)}")
    print(f"Generated datasets will be saved to: {os.path.abspath(DATASETS_DIR)}")
    
    test_params = [(1.0, 0.5), (1.2, 0.3), (0.8, 0.2)]
    
    if train_dataset.samples and trainer.formula_models:
        all_real_params = np.array([p for _, p in train_dataset.samples])
        
        for phi, u in test_params:
            print(f"\nProcessing for φ={phi}, u={u}")
            conditions_tensor = torch.tensor([[phi, u]], dtype=torch.float32).to(trainer.device)
            
            # --- This is your "formula_after_GAN" ---
            generated_data = trainer.generate_flame_data(conditions=conditions_tensor)
            
            # --- This is your "formula_before_GAN" ---
            with torch.no_grad():
                # 1. Get the normalized formula signal
                formula_signal_norm = trainer._calculate_formula_signal(conditions_tensor).permute(0, 2, 1)
                # 2. Denormalize it
                formula_signal_denorm = trainer.train_dataset.denormalize_data(formula_signal_norm).cpu().numpy()

            # --- Save the generated data to the Datasets folder ---
            dataset_filename = os.path.join(DATASETS_DIR, f'generated_phi_{str(phi).replace(".","p")}_u_{str(u).replace(".","p")}.csv')
            np.savetxt(dataset_filename, generated_data[0], delimiter='\t', header='Heat,Time,Pressure', comments='')
            
            # Find the closest real data file for comparison
            param_diffs = np.sum((all_real_params - np.array([phi, u]))**2, axis=1)
            closest_idx = np.argmin(param_diffs)
            closest_filepath, _ = train_dataset.samples[closest_idx]
            
            # --- Pass all 3 datasets to the visualizer ---
            trainer.visualize_comparison(
                real_path=closest_filepath, 
                formula_signal=formula_signal_denorm,  # "formula_before_GAN"
                generated_data=generated_data,           # "formula_after_GAN"
                phi=phi, 
                u=u,
                plots_dir=PLOTS_DIR
            )
    elif not trainer.formula_models:
        print("Error: Formula models were not created. Cannot generate data.")
    else:
        print("Skipping generation as no training samples were found.")

if __name__ == "__main__":
    main()
