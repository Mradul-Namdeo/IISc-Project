import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import os

# --- Mathematical Model Definitions ---

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

def create_formula_image(params_list, quantity, filename, equivalent_params=None):
    """Generates an image containing only the formulas for each flame."""
    if equivalent_params is not None:
        n_formulas = 1
        title = f'Equivalent Formula for All Flames ({quantity.capitalize()})'
    else:
        n_formulas = len(params_list)
        title = f'Fitted Formulas for Each Flame ({quantity.capitalize()})'
    fig_height = max(4, n_formulas * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    formulas = []
    if equivalent_params is not None:
        A, k, t0, C = equivalent_params
        formula_map = {
            'position': f'$P_{{eq}}(t_{{norm}}) = {A:.2f} \\left(1 - e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}\\right) + {C:.2f}$',
            'velocity': f'$V_{{eq}}(t_{{norm}}) = {A*k:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$',
            'acceleration': f'$A_{{eq}}(t_{{norm}}) = {-A*k**2:.2f} \\, e^{{-{k:.2f}(t_{{norm}} - {t0:.2f})}}$'
        }
        formulas.append(formula_map[quantity])
    else:
        for params in params_list:
            flame_id = params['Flame_ID']
            A, k, t0, C = params['A'], params['k'], params['t0'], params['C']
            formula_map = {
                'position': f'$P_{{{flame_id}}}(t) = {A:.2f} \\left(1 - e^{{-{k:.2f}(t - {t0:.2f})}}\\right) + {C:.2f}$',
                'velocity': f'$V_{{{flame_id}}}(t) = {A*k:.2f} \\, e^{{-{k:.2f}(t - {t0:.2f})}}$',
                'acceleration': f'$A_{{{flame_id}}}(t) = {-A*k**2:.2f} \\, e^{{-{k:.2f}(t - {t0:.2f})}}$'
            }
            formulas.append(formula_map[quantity])
    full_text = '\n'.join(formulas)
    ax.text(0.5, 0.5, full_text, ha='center', va='center', fontsize=12, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(title, fontsize=16); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=300); plt.close(fig)
    print(f"Saved formula image: {os.path.basename(filename)}")

# --- Main Execution ---
try:
    # --- Configuration ---
    INPUT_FILENAME = 'flame_trajectory_data_final36.csv'
    FLAME_SLICE = slice(0, 18)   # slice(NONE) for all flames

    output_dir = os.path.dirname(INPUT_FILENAME)
    if output_dir == '': output_dir = '.'
    print(f"Input file: '{INPUT_FILENAME}'")
    print(f"All outputs will be saved in: '{os.path.abspath(output_dir)}'")
    
    OUTPUT_PARAMS_FILENAME = os.path.join(output_dir, 'parameters_individual.csv')
    OUTPUT_EQUIVALENT_PARAMS_FILENAME = os.path.join(output_dir, 'parameters_equivalent.csv')

    df = pd.read_csv(INPUT_FILENAME)
    flame_ids = sorted(df['Flame_ID'].unique())[FLAME_SLICE]
    n_flames = len(flame_ids)
    print(f"\nProcessing {n_flames} selected flames...")
    
    all_params_data = []
    all_flame_data = {}

    # --- PART A: Perform Curve Fitting for each flame ---
    print("\n--- Part A: Fitting Individual Flames ---")
    for flame_id in flame_ids:
        flame_data = df[df['Flame_ID'] == flame_id].copy()
        x_data = flame_data['Timestamp_s'].to_numpy()
        y_data = flame_data['Position_CM'].to_numpy()
        if len(x_data) < 2:
            print(f"Skipping Flame {flame_id}: not enough data points.")
            continue
        guess_C = y_data.min(); guess_A = y_data.max() - y_data.min()
        slopes = np.diff(y_data)/np.diff(x_data); max_slope_idx = np.argmax(slopes)
        guess_t0 = x_data[max_slope_idx]; guess_k = max(0.1, slopes[max_slope_idx]/guess_A if guess_A > 1e-6 else 1.0)
        initial_guesses = [guess_A, guess_k, guess_t0, guess_C]
        time_range = x_data.max() - x_data.min()
        bounds = ([0, 1e-3, x_data.min() - time_range, 0], [guess_A*2 + 1e-6, 50, x_data.max() + time_range, y_data.max() + 1e-6])
        try:
            popt, _ = curve_fit(saturating_exponential, x_data, y_data, p0=initial_guesses, bounds=bounds, maxfev=10000)
            param_row = {'Flame_ID': flame_id, 'A': popt[0], 'k': popt[1], 't0': popt[2], 'C': popt[3]}
            all_params_data.append(param_row)
            all_flame_data[flame_id] = {'x': x_data, 'y': y_data, 'params': popt}
        except (RuntimeError, ValueError) as e:
            print(f"Could not find a good fit for Flame {flame_id}: {e}")
            all_flame_data[flame_id] = {'x': x_data, 'y': y_data, 'params': None}

    # --- PART B: Generate All Outputs ---
    print("\n--- Part B: Generating All Output Files ---")
    
    params_df = pd.DataFrame(all_params_data)
    params_df.to_csv(OUTPUT_PARAMS_FILENAME, index=False)
    print(f"Saved individual flame parameters: {os.path.basename(OUTPUT_PARAMS_FILENAME)}")

    if all_params_data:
        create_formula_image(all_params_data, 'position', os.path.join(output_dir, 'formulas_individual_position.png'))
        create_formula_image(all_params_data, 'velocity', os.path.join(output_dir, 'formulas_individual_velocity.png'))
        create_formula_image(all_params_data, 'acceleration', os.path.join(output_dir, 'formulas_individual_acceleration.png'))
    
    # --- Plot Generation Setup ---
    figs = {name: plt.subplots(figsize=(12, 8)) for name in ['all_pos', 'all_vel', 'all_accel', 'norm_pos', 'norm_vel', 'norm_accel']}
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
        norm_time_smooth = (time_smooth - t_min) / (t_max - t_min)
        flame_label = f'Flame {flame_id}'
        
        figs['all_pos'][1].plot(time_smooth, saturating_exponential(time_smooth, *popt), color='black', alpha=0.5)
        figs['all_vel'][1].plot(time_smooth, velocity_from_exponential(time_smooth, *popt), color='black', alpha=0.5)
        figs['all_accel'][1].plot(time_smooth, acceleration_from_exponential(time_smooth, *popt), color='black', alpha=0.5)
        
        # Subplots
        ax_p.scatter(x_orig, y_orig, label='Data', s=10); ax_p.plot(time_smooth, saturating_exponential(time_smooth, *popt), color='red', label='Fit')
        ax_p.set(title=flame_label, xlabel='Time (s)', ylabel='Position (cm)'); ax_p.legend(); ax_p.grid(True)
        ax_v.plot(time_smooth, velocity_from_exponential(time_smooth, *popt), color='black', label='Fit')
        ax_v.set(title=flame_label, xlabel='Time (s)', ylabel='Velocity (cm/s)'); ax_v.legend(); ax_v.grid(True)
        ax_a.plot(time_smooth, acceleration_from_exponential(time_smooth, *popt), color='green', label='Fit')
        ax_a.axhline(0, color='k', lw=0.7, ls=':'); ax_a.set(title=flame_label, xlabel='Time (s)', ylabel=r'Acceleration (cm/s$^2$)'); ax_a.legend(); ax_a.grid(True)
        
        # Normalized plots (with labels)
        figs['norm_pos'][1].plot(norm_time_smooth, saturating_exponential(time_smooth, *popt), label=flame_label, alpha=0.7)
        figs['norm_vel'][1].plot(norm_time_smooth, velocity_from_exponential(time_smooth, *popt), label=flame_label, alpha=0.7)
        figs['norm_accel'][1].plot(norm_time_smooth, acceleration_from_exponential(time_smooth, *popt), label=flame_label, alpha=0.7)

    # --- Set legend=False for "all flames" plots ---
    plot_configs = {
        'all_pos': ('plot_all-flames_position.png', 'All Fitted Position Curves', 'Time (s)', 'Position (cm)', False),
        'all_vel': ('plot_all-flames_velocity.png', 'All Fitted Velocity Curves', 'Time (s)', 'Velocity (cm/s)', False),
        'all_accel': ('plot_all-flames_acceleration.png', 'All Fitted Acceleration Curves', 'Time (s)', r'Acceleration (cm/s$^2$)', False),
        'norm_pos': ('plot_normalized_position.png', 'Normalized Position (from Curve Fit)', 'Normalized Time (0 to 1)', 'Position (cm)', True),
        'norm_vel': ('plot_normalized_velocity.png', 'Normalized Velocity (from Curve Fit)', 'Normalized Time (0 to 1)', 'Velocity (cm/s)', True),
        'norm_accel': ('plot_normalized_acceleration.png', 'Normalized Acceleration (from Curve Fit)', 'Normalized Time (0 to 1)', r'Acceleration (cm/s$^2$)', True),
    }
    for name, (fname, title, xlabel, ylabel, legend) in plot_configs.items():
        fig, ax = figs[name]; ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(True)
        if 'accel' in name: ax.axhline(0, color='k', lw=0.8, ls='--')
        if legend: ax.legend(fontsize='small')
        fig.savefig(os.path.join(output_dir, fname), dpi=300); print(f"Saved plot: {fname}")
    
    for fig, axes, name in [(fig_sub_p, axes_p, "position"), (fig_sub_v, axes_v, "velocity"), (fig_sub_a, axes_a, "acceleration")]:
        for i in range(n_flames, len(axes)): fig.delaxes(axes[i])
        fig.suptitle(f'Individual Flame {name.capitalize()} Subplots', fontsize=16)
        fname = os.path.join(output_dir, f'plot_subplots_{name}.png')
        fig.savefig(fname, dpi=300); print(f"Saved plot: {os.path.basename(fname)}")
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
                all_y_agg.extend(y_orig)
    
    if all_x_norm_agg:
        try:
            popt_eq, _ = curve_fit(saturating_exponential, all_x_norm_agg, all_y_agg, maxfev=10000)
            print(f"Successfully fitted equivalent curve. Coefficients: A={popt_eq[0]:.2f}, k={popt_eq[1]:.2f}, t0={popt_eq[2]:.2f}, C={popt_eq[3]:.2f}")
            
            create_formula_image(None, 'position', os.path.join(output_dir, 'formula_equivalent_position.png'), equivalent_params=popt_eq)
            create_formula_image(None, 'velocity', os.path.join(output_dir, 'formula_equivalent_velocity.png'), equivalent_params=popt_eq)
            create_formula_image(None, 'acceleration', os.path.join(output_dir, 'formula_equivalent_acceleration.png'), equivalent_params=popt_eq)

            eq_params_df = pd.DataFrame([{'A': popt_eq[0], 'k': popt_eq[1], 't0': popt_eq[2], 'C': popt_eq[3]}])
            eq_params_df.to_csv(OUTPUT_EQUIVALENT_PARAMS_FILENAME, index=False)
            print(f"Saved equivalent curve parameters: {os.path.basename(OUTPUT_EQUIVALENT_PARAMS_FILENAME)}")

            fig_eq, (ax_p, ax_v, ax_a) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
            fig_eq.suptitle('Equivalent Flame Trajectory (Normalized Time)', fontsize=18)
            t_smooth_norm = np.linspace(0, 1, 400)
            ax_p.scatter(all_x_norm_agg, all_y_agg, s=5, alpha=0.1, label='All Raw Data Points')
            ax_p.plot(t_smooth_norm, saturating_exponential(t_smooth_norm, *popt_eq), 'r-', lw=2.5, label='Equivalent Curve')
            ax_v.plot(t_smooth_norm, velocity_from_exponential(t_smooth_norm, *popt_eq), 'k-', lw=2.5, label='Equivalent Curve')
            ax_a.plot(t_smooth_norm, acceleration_from_exponential(t_smooth_norm, *popt_eq), 'g-', lw=2.5, label='Equivalent Curve')
            ax_a.axhline(0, color='k', lw=0.8, ls='--')
            for ax, title, ylabel in [(ax_p, 'Position', 'Position (cm)'), (ax_v, 'Velocity', 'Velocity (cm/s)'), (ax_a, 'Acceleration', r'Acceleration (cm/s$^2$)')]:
                ax.set(title=title, ylabel=ylabel); ax.legend(); ax.grid(True)
            ax_a.set_xlabel('Normalized Time (0 to 1)')
            fname = os.path.join(output_dir, 'plot_equivalent-curves.png')
            fig_eq.savefig(fname, dpi=300); print(f"Saved plot: {os.path.basename(fname)}"); plt.close(fig_eq)
        except Exception as e:
            print(f"Could not fit or plot an equivalent curve: {e}")

    print("\n\n--- Script Finished ---")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
