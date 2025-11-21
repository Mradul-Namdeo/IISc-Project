import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math

# ----------------------------------------------------------
# Parameters (defined globally as they are constant)
# ----------------------------------------------------------
T_u = 450.0
T_b = 2220.0
T_w = 1800.0
sigma = T_u / T_b
phi   = T_w / T_b
eta   = 0.059
Ea = 31000.0
R  = 8.314
N  = Ea / (R * T_u)
Le = 0.9
omega = 0.627

# ----------------------------------------------------------
# Helper functions (defined globally to be used by the main function)
# ----------------------------------------------------------
def _compute_Tfs(V):
    return 1.0 / (1.0 - (2.0 / N) * np.log(V))

def _compute_lambdas(V):
    sqrt_term = np.sqrt(V**2 / 4.0 + omega)
    l1 = -(V / 2.0) - sqrt_term
    l2 = -(V / 2.0) + sqrt_term
    return l1, l2, l2 - l1

def _residual(xf_val, V, l1, l2, H, Tfs):
    term1 = H * Tfs
    term2 = -omega * np.exp(l2 * xf_val) * (
                (sigma / l2) * np.exp(-l2 * xf_val) +
                ((phi - sigma) / (eta + l2)) * np.exp(-(eta + l2) * xf_val)
            )
    term3 = -omega * np.exp(l1 * xf_val) * (
                (sigma / l1) * (1.0 - np.exp(-l1 * xf_val)) +
                ((phi - sigma) / (eta + l1)) * (1.0 - np.exp(-(eta + l1) * xf_val)) -
                (phi / l1)
            )
    term4 = -(1.0 - sigma) * V
    return term1 + term2 + term3 + term4

def _compute_chi1(x_val, V, l1, l2):
    T1 = (sigma / (l2 * (V + 2 * l1 + l2))) * (np.exp(-(V + 2 * l1 + l2) * x_val) - 1.0)
    T2 = (phi / ((eta + l2) * (V + 2 * l1 + eta + l2))) * (np.exp(-(V + 2 * l1 + eta + l2) * x_val) - 1.0)
    T3 = (sigma / ((eta + l2) * (V + 2 * l1 + eta + l2))) * (np.exp(-(V + 2 * l1 + eta + l2) * x_val) - 1.0)
    return omega * np.exp(l1 * x_val) * (T1 + T2 - T3)

def _compute_chi2(x_val, V, l1, l2):
    R1 = (sigma / (l1 * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
    R2 = (sigma / (l1 * (V + 2 * l2 + l1))) * (np.exp(-(V + 2 * l2 + l1) * x_val) - 1.0)
    R3 = ((phi - sigma) / ((eta + l1) * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
    R4 = ((phi - sigma) / ((eta + l1) * (V + 2 * l2 + eta + l1))) * (np.exp(-(V + 2 * l2 + eta + l1) * x_val) - 1.0)
    R5 = (phi / (l1 * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
    return omega * np.exp(l2 * x_val) * (-R1 + R2 - R3 + R4 + R5)

# ----------------------------------------------------------
# Main Function to Generate Subplots
# ----------------------------------------------------------
def generate_profile_subplots(V_values):
    """
    Generates a grid of subplots for temperature and concentration profiles,
    one for each velocity value provided in a list.
    """
    num_plots = len(V_values)
    if num_plots == 0:
        print("V_values list is empty. Nothing to plot.")
        return

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False)
    axes = axes.flatten()

    xvals = np.arange(-50.01, 50.01, 0.01)

    for idx, V in enumerate(V_values):
        ax = axes[idx]
        l1, l2, H = _compute_lambdas(V)
        Tfs = _compute_Tfs(V)

        try:
            xf_sol = float(fsolve(_residual, 5.0, args=(V, l1, l2, H, Tfs)))
        except (RuntimeError, ValueError):
            xf_sol = np.nan

        if np.isnan(xf_sol):
            ax.text(0.5, 0.5, f"No solution for V={V:.2f}", ha="center", va="center", fontsize=12)
            ax.set_title(f"V = {V:.2f} (Failed)", color='red')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)
            continue
        
        # --- Calculations for plotting ---
        chi1_xfs = _compute_chi1(xf_sol, V, l1, l2)
        chi2_xfs = _compute_chi2(xf_sol, V, l1, l2)
        T = np.empty_like(xvals)
        C1 = np.empty_like(xvals)
        for k, x_val in enumerate(xvals):
            C1[k] = 1.0 - np.exp(-V * Le * (x_val - xf_sol))
            T1 = (Tfs + chi1_xfs) * np.exp(l1 * (x_val - xf_sol)) - _compute_chi1(x_val, V, l1, l2)
            T2 = (Tfs + chi2_xfs) * np.exp(l2 * (x_val - xf_sol)) - _compute_chi2(x_val, V, l1, l2)
            T[k] = T1 if x_val >= xf_sol else T2
        C1[C1 < 0.0] = np.nan

        # --- Plotting ---
        ax2 = ax.twinx()
        ax.plot(xvals, T, 'b-', linewidth=1.5, label="T(K)")
        ax2.plot(xvals, C1, 'r-', linewidth=1.0, label="C1")
        ax.axvline(xf_sol, color='k', linestyle='--', linewidth=1)
        ax.text(0.05, 0.9, f"xf_sol = {xf_sol:.2f}", transform=ax.transAxes,
                fontsize=10, color="black", fontweight="bold")
        ax.set_title(f"V = {V:.2f}")
        ax.grid(True)
        ax.set_ylim([0, 1])

        # --- NEW: Add Axis Labels ---
        # Get current row and column
        row_num = idx // cols
        col_num = idx % cols

        # Add labels only to the outer plots to avoid clutter
        if row_num == rows - 1 or (rows == 1 and num_plots < cols): # Last row
             ax.set_xlabel('Distance (x)', fontweight='bold')
        if col_num == 0: # First column
            ax.set_ylabel('T (K)', color='b', fontweight='bold')
        if col_num == cols -1: # Last column
            ax2.set_ylabel('C₁', color='r', fontweight='bold')


    # Turn off any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # --- Adjust spacing between plots ---
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    plt.show()

velocities_from_script = [0.060210448, 0.115155503, 0.165521664, 1.851468048, 3.006908463, 3.5, 3.9, 4.006908463, 4.5]
generate_profile_subplots(V_values=velocities_from_script)
