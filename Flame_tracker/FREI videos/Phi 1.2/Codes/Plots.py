import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import os
import cv2

# --- 1. SCRIPT CONFIGURATION ---

# --- Input/Output Paths ---
DATASETS_DIR = r"D:\FREI_videos_Flame_tracking\Phi_1p2\Phi_1p2_u_0p4_C001H001S0001\Datasets_Intensity"
PLOTS_DIR = r"D:\FREI_videos_Flame_tracking\Phi_1p2\Phi_1p2_u_0p4_C001H001S0001\Plots_Intensity"

ORIGINAL_RAW_FILE = os.path.join(DATASETS_DIR, "Phi_1p2_u_0p4_C001H001S0001.csv")
SCALED_DATA_FILE = os.path.join(DATASETS_DIR, "Phi_1p2_u_0p4_C001H001S0001_5sec_Intensity.csv")

PLOT_OUTPUT_DIR = PLOTS_DIR
DATA_OUTPUT_DIR = DATASETS_DIR

# --- Analysis Parameters ---
TARGET_DURATION_SEC = 5.0
FLAME_SLICE = slice(None)

# --- Star Normalization Constants ---
T_STAR = 0.000275013  # Characteristic Time
P_STAR = 0.110         # Characteristic Position
V_STAR = 399.981       # Characteristic Velocity


# --- 2. MATHEMATICAL MODEL DEFINITIONS ---

def saturating_exponential(t, A, k, t0, C):
    """ Models the position curve: P(t) = A * (1 - exp(-k * (t - t0))) + C """
    safe_exp_arg = -k * (t - t0)
    safe_exp_arg = np.clip(safe_exp_arg, -700, 700)
    return A * (1 - np.exp(safe_exp_arg)) + C

def velocity_from_exponential(t, A, k, t0, C):
    """ The first derivative (Velocity) of the saturating exponential. """
    safe_exp_arg = -k * (t - t0)
    safe_exp_arg = np.clip(safe_exp_arg, -700, 700)
    return A * k * np.exp(safe_exp_arg)

def acceleration_from_exponential(t, A, k, t0, C):
    """ The second derivative (Acceleration) of the saturating exponential. """
    safe_exp_arg = -k * (t - t0)
    safe_exp_arg = np.clip(safe_exp_arg, -700, 700)
    return -A * (k**2) * np.exp(safe_exp_arg)


# --- 3. HELPER FUNCTIONS ---

def normalize_0_to_1(data):
    """Scales a numpy array to the 0-1 range."""
    min_val, max_val = data.min(), data.max()
    range_val = max_val - min_val
    if range_val > 1e-9:
        return (data - min_val) / range_val
    else:
        return np.full_like(data, 0.5)

def create_formula_image(params_list, quantity, filename, equivalent_params=None):
    """Generates an image containing only the formulas for each flame."""
    if equivalent_params is not None:
        n_formulas = 1
        if 'Normalized' in filename:
            title = f'Equivalent Normalized Formula ({quantity.capitalize()})'
        else:
            title = f'Equivalent Formula for All Flames ({quantity.capitalize()})'
    else:
        n_formulas = len(params_list)
        title = f'Fitted Formulas for Each Flame ({quantity.capitalize()})'
        
    fig_height = max(4, n_formulas * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    formulas = []
    
    if equivalent_params is not None:
        A, k, t0, C = equivalent_params
        if 'Normalized' in filename:
            formula_map = {
                'position': f'$P_{{norm, eq}}(t_{{norm}}) = {A:.2f} \\left(1 - e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}\\right) + {C:.2f}$',
                'velocity': f'$V_{{norm, eq}}(t_{{norm}}) = {A*k:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$',
                'acceleration': f'$A_{{norm, eq}}(t_{{norm}}) = {-A*k**2:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$'
            }
        else:
            formula_map = {
                'position': f'$P_{{eq}}(t_{{norm}}) = {A:.2f} \\left(1 - e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}\\right) + {C:.2f}$ (mm)',
                'velocity': f'$V_{{eq}}(t_{{norm}}) = {A*k:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$ (mm/s)',
                'acceleration': f'$A_{{eq}}(t_{{norm}}) = {-A*k**2:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$ (mm/s$^2$)'
            }
        formulas.append(formula_map[quantity])
    else:
        for params in params_list:
            flame_id = params['Flame_ID']
            A, k, t0, C = params['A'], params['k'], params['t0'], params['C']
            formula_map = {
                'position': f'$P_{{{flame_id}}}(t) = {A:.2f} \\left(1 - e^{{-{k:.2f}(t - {t0:.2f})}}\\right) + {C:.2f}$ (mm)',
                'velocity': f'$V_{{{flame_id}}}(t) = {A*k:.2f} \\, e^{{-{k:.2f}(t - {t0:.2f})}}$ (mm/s)',
                'acceleration': f'$A_{{{flame_id}}}(t) = {-A*k**2:.2f} \\, e^{{-{k:.2f}(t - {t0:.2f})}}$ (mm/s$^2$)'
            }
            formulas.append(formula_map[quantity])
            
    full_text = '\n'.join(formulas)
    ax.text(0.5, 0.5, full_text, ha='center', va='center', fontsize=12, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(title, fontsize=16); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=300); plt.close(fig)
    print(f"Saved formula image: {os.path.basename(filename)}")

def create_star_norm_formula_image(params_list, filename):
    """Generates an image of the transformed 'Star Normalized' position formulas."""
    n_formulas = len(params_list)
    title = 'Fitted "Star Normalized" Formulas (Position)'
    fig_height = max(4, n_formulas * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    formulas = []
    
    for params in params_list:
        flame_id = params['Flame_ID']
        A, k, t0, C = params['A'], params['k'], params['t0'], params['C']
        
        A_star = A / P_STAR
        k_star = k * 100.0  # Note: This k_star is an arbitrary scaling
        t0_star = t0 / T_STAR
        C_star = C / P_STAR
        
        formula = f'$P_{{{flame_id}}}^{{\\star}}({{t}}^{{\\star}}) = {A:.2f} \\left(1 - e^{{-{k_star:.2f}({{t}}^{{\\star}} - {t0_star:.4f})}}\\right) + {C_star:.2f}$'
        formulas.append(formula)

    full_text = '\n'.join(formulas)
    ax.text(0.5, 0.5, full_text, ha='center', va='center', fontsize=12, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(title, fontsize=16); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=300); plt.close(fig)
    print(f"Saved formula image: {os.path.basename(filename)}")


# --- 4. MAIN EXECUTION ---

def main():
    """
    Main pipeline for flame data processing.
    1. Pre-processes (scales) the raw data file.
    2. Fits a 3-parameter saturating exponential curve to each individual flame.
    3. Fits an Exponential curve to the aggregated normalized data.
    4. Generates all plots, parameter files, and videos.
    """
    
    # --- PART 1: DATA TRANSFORMATION (PRE-PROCESSING) ---
    print("--- Part 1: Starting Data Transformation ---")
    transformation_success = False
    try:
        print(f"Reading original file: {ORIGINAL_RAW_FILE}...")
        df_transform = pd.read_csv(ORIGINAL_RAW_FILE)

        if 'Timestamp_s' in df_transform.columns:
            original_max_time = df_transform['Timestamp_s'].max()
            
            if original_max_time > 0:
                scaling_factor = TARGET_DURATION_SEC / original_max_time
                
                print(f"Original max time: {original_max_time:.4f}s")
                print(f"Target max time:   {TARGET_DURATION_SEC}s")
                print(f"Scaling factor:    {scaling_factor:.6f}")

                df_scaled = df_transform.copy()
                df_scaled['Timestamp_s'] = df_scaled['Timestamp_s'] * scaling_factor
                
                os.makedirs(os.path.dirname(SCALED_DATA_FILE), exist_ok=True)
                df_scaled.to_csv(SCALED_DATA_FILE, index=False)
                
                print(f"\nSuccessfully created new scaled file:")
                print(f"{SCALED_DATA_FILE}")
                print(f"New max time is: {df_scaled['Timestamp_s'].max():.4f}s")
                transformation_success = True
                
            else:
                print("Error: Max time in file is 0. Cannot scale.")
        else:
            print("Error: 'Timestamp_s' column not found in the file.")

    except FileNotFoundError:
        print(f"FATAL Error: File not found at {ORIGINAL_RAW_FILE}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data transformation: {e}")
        return

    # --- PART 2: ANALYSIS (Only if Part 1 succeeded) ---
    if not transformation_success:
        print("\n--- Script Halted ---")
        print("Analysis was not performed because data transformation failed.")
        return

    print("\n--- Part 2: Starting Curve Fitting and Analysis ---")
    try:
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
        
        OUTPUT_PARAMS_FILENAME = os.path.join(DATA_OUTPUT_DIR, 'parameters_individual.csv')
        OUTPUT_EQUIVALENT_PARAMS_FILENAME = os.path.join(DATA_OUTPUT_DIR, 'parameters_equivalent_normalized.csv')

        print(f"Reading scaled data from: '{SCALED_DATA_FILE}'")
        df = pd.read_csv(SCALED_DATA_FILE)
        df = df[['Frame_ID', 'Flame_ID', 'Timestamp_s', 'Position_MM']]
        
        flame_ids = sorted(df['Flame_ID'].unique())[FLAME_SLICE]
        n_flames = len(flame_ids)
        print(f"\nProcessing {n_flames} selected flames...")
        
        legend_cols = 1 if n_flames < 30 else 2
        legend_fontsize = 'small' if n_flames < 50 else 'x-small'
        right_adjust = 0.85 if legend_cols == 1 else 0.75 
        
        print(f"--- Legend settings: {legend_cols} column(s), fontsize: '{legend_fontsize}' ---")

        all_params_data = []
        all_flame_data = {}

        # --- PART A: Perform Curve Fitting (SATURATING 3-PARAM FIT) ---
        print("\n--- Part A: Fitting Individual Flames (Saturating 3-Param Fit) ---")
        for flame_id in flame_ids:
            flame_data = df[df['Flame_ID'] == flame_id].copy()
            x_data = flame_data['Timestamp_s'].to_numpy()
            y_data = flame_data['Position_MM'].to_numpy()
            
            if len(x_data) < 6:
                print(f"Skipping Flame {flame_id}: not enough data points (need at least 6).")
                continue

            # --- MODIFIED: Use the average of the last 5 points per your request ---
            P_saturated = np.mean(y_data[-5:])
            P_min = y_data.min()

            if (P_saturated - P_min) < 1e-6: # Handle flat data
                print(f"Skipping Flame {flame_id}: Data is flat.")
                all_flame_data[flame_id] = {'x': x_data, 'y': y_data, 'params': None}
                continue

            # P(t) = A * (1 - exp(-k(t-t0))) + C
            # We fix A + C = P_saturated, so A = P_saturated - C
            # We will fit for (k, t0, C)
            def exp_model_saturating(t, k, t0, C):
                A = P_saturated - C
                if A <= 0: A = 1e-6 
                return saturating_exponential(t, A, k, t0, C)

            # Guesses for k, t0, C
            total_time = x_data.max() - x_data.min()
            guess_k = 4.0 / total_time if total_time > 0 else 1.0
            
            mid_rise_y = (P_saturated + P_min) / 2.0
            mid_rise_idx = np.argmin(np.abs(y_data - mid_rise_y))
            guess_t0 = x_data[mid_rise_idx]
            
            guess_C = P_min 
            
            initial_guesses = [guess_k, guess_t0, guess_C]
            
            # Bounds
            bounds_min = [1e-3, x_data.min() - total_time, -np.inf]
            bounds_max = [500,  x_data.max() + total_time, P_min]
            bounds = (bounds_min, bounds_max)
            
            safe_guess_k = np.clip(initial_guesses[0], bounds_min[0], bounds_max[0])
            safe_guess_t0 = np.clip(initial_guesses[1], bounds_min[1], bounds_max[1])
            safe_guess_C = np.clip(initial_guesses[2], bounds_min[2], bounds_max[2])
            safe_p0 = [safe_guess_k, safe_guess_t0, safe_guess_C]

            try:
                # --- MODIFIED: Removed 'loss=soft_l1' ---
                # This uses the default 'linear' (Least Squares) loss.
                # It will force the curve to fit the flat end points aggressively.
                popt_fit, _ = curve_fit(exp_model_saturating, 
                                        x_data, y_data, 
                                        p0=safe_p0, 
                                        bounds=bounds, 
                                        maxfev=10000)
                
                k_fit, t0_fit, C_fit = popt_fit
                A_fit = P_saturated - C_fit
                popt = [A_fit, k_fit, t0_fit, C_fit]
                
                param_row = {'Flame_ID': flame_id, 'A': popt[0], 'k': popt[1], 't0': popt[2], 'C': popt[3]}
                all_params_data.append(param_row)
                all_flame_data[flame_id] = {'x': x_data, 'y': y_data, 'params': popt}
                
            except (RuntimeError, ValueError) as e:
                print(f"Could not find a 3-param fit for Flame {flame_id}: {e}")
                all_flame_data[flame_id] = {'x': x_data, 'y': y_data, 'params': None}

        # --- PART B: Generate All Outputs ---
        print("\n--- Part B: Generating All Output Files ---")
        
        params_df = pd.DataFrame(all_params_data)
        params_df.to_csv(OUTPUT_PARAMS_FILENAME, index=False)
        print(f"Saved individual flame parameters: {os.path.basename(OUTPUT_PARAMS_FILENAME)}")

        if all_params_data:
            create_formula_image(all_params_data, 'position', os.path.join(PLOT_OUTPUT_DIR, 'formulas_individual_position.png'))
            create_formula_image(all_params_data, 'velocity', os.path.join(PLOT_OUTPUT_DIR, 'formulas_individual_velocity.png'))
            create_formula_image(all_params_data, 'acceleration', os.path.join(PLOT_OUTPUT_DIR, 'formulas_individual_acceleration.png'))
            create_star_norm_formula_image(all_params_data, os.path.join(PLOT_OUTPUT_DIR, 'formulas_star_normalized_position.png'))
        
        std_size = (12, 8)
        large_size = (15, 9)

        figs = {
            'all_pos': plt.subplots(figsize=std_size),
            'all_vel': plt.subplots(figsize=std_size),
            'all_accel': plt.subplots(figsize=std_size),
            'norm_pos': plt.subplots(figsize=large_size),
            'norm_vel': plt.subplots(figsize=large_size),
            'norm_accel': plt.subplots(figsize=large_size),
            'star_norm_pos': plt.subplots(figsize=large_size),
            'star_norm_vel': plt.subplots(figsize=large_size),
            'star_norm_accel': plt.subplots(figsize=large_size)
        }
        
        n_cols = 4; n_rows = math.ceil(n_flames/n_cols) if n_flames > 0 else 1
        fig_sub_p, axes_p = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5), constrained_layout=True)
        fig_sub_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5), constrained_layout=True)
        fig_sub_a, axes_a = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5), constrained_layout=True)
        axes_p, axes_v, axes_a = axes_p.flatten(), axes_v.flatten(), axes_a.flatten()

        for i, flame_id in enumerate(flame_ids):
            data = all_flame_data.get(flame_id)
            ax_p, ax_v, ax_a = axes_p[i], axes_v[i], axes_a[i]
            if not data or data['params'] is None:
                for ax in [ax_p, ax_v, ax_a]: ax.text(0.5, 0.5, 'Fit Failed', ha='center', va='center', transform=ax.transAxes)
                continue
            x_orig, y_orig, popt = data['x'], data['y'], data['params']
            t_min, t_max = x_orig.min(), x_orig.max()
            time_smooth = np.linspace(t_min, t_max, 200)
            
            if t_max > t_min:
                norm_time_smooth = (time_smooth - t_min) / (t_max - t_min)
            else:
                norm_time_smooth = np.zeros_like(time_smooth)
            
            flame_label = f'Flame {flame_id}'
            
            pos_smooth = saturating_exponential(time_smooth, *popt)
            vel_smooth = velocity_from_exponential(time_smooth, *popt)
            accel_smooth = acceleration_from_exponential(time_smooth, *popt)
            
            figs['all_pos'][1].plot(time_smooth, pos_smooth, color='black', alpha=0.5)
            figs['all_vel'][1].plot(time_smooth, vel_smooth, color='black', alpha=0.5)
            figs['all_accel'][1].plot(time_smooth, accel_smooth, color='black', alpha=0.5)
            
            ax_p.scatter(x_orig, y_orig, s=10); ax_p.plot(time_smooth, pos_smooth, color='red')
            ax_p.set(title=flame_label, xlabel='Time (s)', ylabel='Position (mm)'); ax_p.grid(True)
            ax_p.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
            ax_p.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
            
            ax_v.plot(time_smooth, vel_smooth, color='black')
            ax_v.set(title=flame_label, xlabel='Time (s)', ylabel='Velocity (mm/s)'); ax_v.grid(True)
            ax_v.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
            ax_v.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
            
            ax_a.plot(time_smooth, accel_smooth, color='green')
            ax_a.axhline(0, color='k', lw=0.7, ls=':'); ax_a.set(title=flame_label, xlabel='Time (s)', ylabel=r'Acceleration (mm/s$^2$)'); ax_a.grid(True)
            ax_a.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
            ax_a.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
            
            pos_norm_y = normalize_0_to_1(pos_smooth)
            vel_norm_y = normalize_0_to_1(vel_smooth)
            accel_norm_y = normalize_0_to_1(accel_smooth)
            
            figs['norm_pos'][1].plot(norm_time_smooth, pos_norm_y, label=flame_label, alpha=0.7)
            figs['norm_vel'][1].plot(norm_time_smooth, vel_norm_y, label=flame_label, alpha=0.7)
            figs['norm_accel'][1].plot(norm_time_smooth, accel_norm_y, label=flame_label, alpha=0.7)

            time_scaled_star = time_smooth / T_STAR
            pos_star_norm = pos_smooth / P_STAR
            
            vel_scaled_ynorm = vel_smooth / V_STAR
            accel_scaled_ynorm = normalize_0_to_1(accel_smooth)

            figs['star_norm_pos'][1].plot(time_scaled_star, pos_star_norm, label=flame_label, alpha=0.7)
            figs['star_norm_vel'][1].plot(time_scaled_star, vel_scaled_ynorm, label=flame_label, alpha=0.7)
            figs['star_norm_accel'][1].plot(time_scaled_star, accel_scaled_ynorm, label=flame_label, alpha=0.7)
            
            all_flame_data[flame_id]['smooth'] = {
                'time': time_smooth, 'pos': pos_smooth, 'vel': vel_smooth, 'accel': accel_smooth,
                'norm_time': norm_time_smooth,
                'pos_norm_y': pos_norm_y, 'vel_norm_y': vel_norm_y, 'accel_norm_y': accel_norm_y,
                'star_time': time_scaled_star,
                'star_pos': pos_star_norm,
                'star_vel': vel_scaled_ynorm, 'star_accel': accel_scaled_ynorm
            }

        plot_configs = {
            'all_pos': ('cumulative_plots_position.png', 'All Fitted Position Curves', 'Time (s)', 'Position (mm)', False),
            'all_vel': ('cumulative_plots_velocity.png', 'All Fitted Velocity Curves', 'Time (s)', 'Velocity (mm/s)', False),
            'all_accel': ('cumulative_plots_acceleration.png', 'All Fitted Acceleration Curves', 'Time (s)', r'Acceleration (mm/s$^2$)', False),
            
            'norm_pos': ('plot_normalized_position.png', 'Normalized Position (Min - Max)', 'Normalized Time (0 to 1)', 'Normalized Position (0 to 1)', True),
            'norm_vel': ('plot_normalized_velocity.png', 'Normalized Velocity (Min - Max)', 'Normalized Time (0 to 1)', 'Normalized Velocity (0 to 1)', True),
            'norm_accel': ('plot_normalized_acceleration.png', 'Normalized Acceleration (Min - Max)', 'Normalized Time (0 to 1)', 'Normalized Acceleration (0 to 1)', True),
            
            'star_norm_pos': ('plot_star_normalized_position.png', 'Star Normalized Position', f'Scaled Time (t/{T_STAR})', f'Star Normalized Position (P / {P_STAR})', True),
            'star_norm_vel': ('plot_star_normalized_velocity.png', 'Star Normalized Velocity', f'Scaled Time (t/{T_STAR})', f'Star Normalized Velocity (V / {V_STAR})', True),
            'star_norm_accel': ('plot_star_normalized_acceleration.png', 'Star Normalized Acceleration', f'Scaled Time (t/{T_STAR})', 'Normalized Acceleration (0 to 1)', True),
        }
        
        for name, (fname, title, xlabel, ylabel, legend) in plot_configs.items():
            fig, ax = figs[name]; ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(True)
            if 'accel' in name: ax.axhline(0, color='k', lw=0.8, ls='--')
            
            if legend:
                ax.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=legend_cols)
                fig.subplots_adjust(right=right_adjust)
                
            fig.savefig(os.path.join(PLOT_OUTPUT_DIR, fname), dpi=300); print(f"Saved plot: {fname}")
        
        for fig, axes, name in [(fig_sub_p, axes_p, "position"), (fig_sub_v, axes_v, "velocity"), (fig_sub_a, axes_a, "acceleration")]:
            for i in range(n_flames, len(axes)): fig.delaxes(axes[i])
            fig.suptitle(f'Individual Flame {name.capitalize()} Subplots', fontsize=16)
            fname = os.path.join(PLOT_OUTPUT_DIR, f'plot_subplots_{name}.png')
            fig.savefig(fname, dpi=300); print(f"Saved plot: {os.path.basename(fname)}"); plt.close(fig)
        plt.close('all')

        # --- Part C: Equivalent Curve Fitting and Plotting ---
        print("\n--- Part C: Equivalent Curve Fitting and Plotting ---")
        all_x_norm_agg, all_y_agg = [], []
        for flame_id in flame_ids:
            data = all_flame_data.get(flame_id)
            if data and data['params'] is not None:
                x_orig, y_orig = data['x'], data['y']; t_min, t_max = x_orig.min(), x_orig.max()
                if t_max > t_min:
                    all_x_norm_agg.extend((x_orig - t_min) / (t_max - t_min))
                    
                    y_range = y_orig.max() - y_orig.min()
                    if y_range > 1e-9:
                        y_norm = (y_orig - y_orig.min()) / y_range
                    else:
                        y_norm = np.full_like(y_orig, 0.5)
                    all_y_agg.extend(y_norm)
        
        if all_x_norm_agg:
            try:
                x_data_eq = np.array(all_x_norm_agg)
                y_data_eq = np.array(all_y_agg)
                sort_indices = np.argsort(x_data_eq)
                x_data_eq = x_data_eq[sort_indices]
                y_data_eq = y_data_eq[sort_indices]
                
                if x_data_eq.size == 0:
                    raise ValueError("No aggregated data to fit.")

                C_eq_fit = 0.0
                A_eq_fit = 1.0

                def exp_model_eq_2_param(t, k, t0):
                    return saturating_exponential(t, A_eq_fit, k, t0, C_eq_fit)

                mid_rise_y_eq = (A_eq_fit / 2.0) + C_eq_fit
                mid_rise_idx_eq = np.argmin(np.abs(y_data_eq - mid_rise_y_eq))
                guess_t0_eq = x_data_eq[mid_rise_idx_eq]
                
                dx = np.diff(x_data_eq)
                dy = np.diff(y_data_eq)
                slopes_eq = np.divide(dy, dx, out=np.zeros_like(dy), where=dx!=0)
                max_slope_eq = np.quantile(slopes_eq[slopes_eq > 0], 0.95) if (slopes_eq > 0).any() else 0
                guess_k_eq = max_slope_eq / A_eq_fit if A_eq_fit > 0 else 2.0
                if guess_k_eq < 0.1: guess_k_eq = 2.0
                
                initial_guesses_eq_2_param = [guess_k_eq, guess_t0_eq]
                bounds_eq_2_param = ([1e-3, 0.0], [50, 1.0])
                
                print(f"Robust Guesses for Equivalent Fit (2-param): k={guess_k_eq:.2f}, t0={guess_t0_eq:.2f}")
                print(f"Locked-in values for Equivalent Fit: A={A_eq_fit:.2f}, C={C_eq_fit:.2f}")

                popt_eq_2_param, _ = curve_fit(exp_model_eq_2_param, x_data_eq, y_data_eq, 
                                                p0=initial_guesses_eq_2_param, 
                                                bounds=bounds_eq_2_param, 
                                                maxfev=100000)
                
                k_eq_fit, t0_eq_fit = popt_eq_2_param
                popt_eq = [A_eq_fit, k_eq_fit, t0_eq_fit, C_eq_fit]
                
                print(f"Successfully fitted equivalent curve. Coefficients: A={popt_eq[0]:.2f}, k={popt_eq[1]:.2f}, t0={popt_eq[2]:.2f}, C={popt_eq[3]:.2f}")
                
                create_formula_image(None, 'position', os.path.join(PLOT_OUTPUT_DIR, 'formula_equivalent_Normalized_position.png'), equivalent_params=popt_eq)
                create_formula_image(None, 'velocity', os.path.join(PLOT_OUTPUT_DIR, 'formula_equivalent_Normalized_velocity.png'), equivalent_params=popt_eq)
                create_formula_image(None, 'acceleration', os.path.join(PLOT_OUTPUT_DIR, 'formula_equivalent_Normalized_acceleration.png'), equivalent_params=popt_eq)

                eq_params_df = pd.DataFrame([{'A': popt_eq[0], 'k': popt_eq[1], 't0': popt_eq[2], 'C': popt_eq[3]}])
                eq_params_df.to_csv(OUTPUT_EQUIVALENT_PARAMS_FILENAME, index=False)
                print(f"Saved equivalent curve parameters: {os.path.basename(OUTPUT_EQUIVALENT_PARAMS_FILENAME)}")

                fig_eq, (ax_p, ax_v, ax_a) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
                fig_eq.suptitle('Equivalent Flame Trajectory (Normalized Time and Position)', fontsize=18)
                t_smooth_norm = np.linspace(0, 1, 400)
                
                ax_p.scatter(all_x_norm_agg, all_y_agg, s=5, alpha=0.1)
                ax_p.plot(t_smooth_norm, saturating_exponential(t_smooth_norm, *popt_eq), 'r-', lw=2.5)
                ax_v.plot(t_smooth_norm, velocity_from_exponential(t_smooth_norm, *popt_eq), 'k-', lw=2.5)
                ax_a.plot(t_smooth_norm, acceleration_from_exponential(t_smooth_norm, *popt_eq), 'g-', lw=2.5)
                
                ax_a.axhline(0, color='k', lw=0.8, ls='--')
                
                for ax, title, ylabel in [(ax_p, 'Position', 'Normalized Position (0 to 1)'), 
                                          (ax_v, 'Velocity', 'Normalized Velocity (from fit)'), 
                                          (ax_a, 'Acceleration', 'Normalized Acceleration (from fit)')]:
                    ax.set(title=title, ylabel=ylabel); ax.grid(True)
                ax_a.set_xlabel('Normalized Time (0 to 1)')
                
                fname = os.path.join(PLOT_OUTPUT_DIR, 'plot_equivalent-curves.png')
                fig_eq.savefig(fname, dpi=300); print(f"Saved plot: {os.path.basename(fname)}"); plt.close(fig_eq)
            except Exception as e:
                print(f"Could not fit or plot an equivalent curve: {e}")

        # --- PART D: Video Generation ---
        print("\n--- Part D: Generating Animated Plots ---")
        VIDEO_FPS = 2.0
        
        video_plot_configs = {
            'norm_pos':     {'x': 'norm_time', 'y': 'pos_norm_y'},
            'norm_vel':     {'x': 'norm_time', 'y': 'vel_norm_y'},
            'norm_accel':   {'x': 'norm_time', 'y': 'accel_norm_y'},
            'star_norm_pos':   {'x': 'star_time', 'y': 'star_pos'},
            'star_norm_vel':   {'x': 'star_time', 'y': 'star_vel'},
            'star_norm_accel': {'x': 'star_time', 'y': 'star_accel'}
        }

        base_fig_for_size = plt.figure(figsize=large_size)
        base_fig_for_size.subplots_adjust(right=right_adjust) 
        base_fig_for_size.canvas.draw()
        frame_width = int(base_fig_for_size.get_figwidth() * base_fig_for_size.dpi)
        frame_height = int(base_fig_for_size.get_figheight() * base_fig_for_size.dpi)
        frame_size = (frame_width, frame_height)
        plt.close(base_fig_for_size)
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        for plot_name, keys in video_plot_configs.items():
            fname, title, xlabel, ylabel, _ = plot_configs[plot_name]
            
            video_filename = os.path.join(PLOT_OUTPUT_DIR, f"video_{fname.replace('plot_', '').replace('.png', '.mp4')}")
            out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, frame_size)
            print(f"Creating video: {os.path.basename(video_filename)}")

            fig, ax = plt.subplots(figsize=large_size)
            
            for i in range(1, len(flame_ids) + 1):
                ax.clear()
                current_flame_ids = flame_ids[:i]
                
                for flame_id in current_flame_ids:
                    data = all_flame_data.get(flame_id)
                    if data and data['params'] is not None:
                        smooth_data = data['smooth']
                        x_key, y_key = keys['x'], keys['y']
                        
                        if x_key in smooth_data and y_key in smooth_data:
                            ax.plot(smooth_data[x_key], smooth_data[y_key], label=f'Flame {flame_id}', alpha=0.7)
                
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(True)
                if 'accel' in plot_name: ax.axhline(0, color='k', lw=0.8, ls='--')
                
                ax.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=legend_cols)
                fig.subplots_adjust(right=right_adjust)
                
                fig.canvas.draw()
                img_buf = fig.canvas.buffer_rgba()
                frame = np.asarray(img_buf)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                out.write(frame_bgr)

            out.release()
            plt.close(fig)
            
        print("--- Video generation complete ---")

    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
    
    finally:
        print("\n\n--- Script Finished ---")


# Standard Python practice to run the main function
if __name__ == "__main__":
    main()
