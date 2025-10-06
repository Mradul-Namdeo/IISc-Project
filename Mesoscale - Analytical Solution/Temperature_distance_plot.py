import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def generate_temperature_distance_plot(V):
    """
    Calculates and plots temperature and concentration profiles based on a given average velocity.

    This function models a physical system (flame propagation) by first finding a
    critical position (xf_sol) using scipy's fsolve, and then calculating the temperature (T)
    and concentration (C1) profiles across a spatial domain. It then visualizes these
    profiles on a dual-axis plot.

    Args:
        V (float): The average velocity, a critical input for the model.
    """
    # ----------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------
    T_u = 450.0      # Unburnt gas temperature
    T_b = 2220.0     # Burnt gas temperature
    T_w = 1800.0     # Up-stream wall temperature
    sigma = T_u / T_b  # Tu / Tb
    phi = T_w / T_b    # Tw / Tb
    eta = 0.059      # wall-temperature decay exponent

    Ea = 31000.0     # Activation energy
    R = 8.314        # Gas constant
    N = Ea / (R * T_u) # Reduced activation energy
    Le = 0.9         # Lewis number
    Nu = 3.66        # Nusselt number
    Dth = 1e-4       # Thermal diffusivity
    Ub = 0.4         # Flame speed
    Pe = (0.2 * 0.005) / Dth # Peclet number
    omega = 0.627    # Heat-loss parameter

    # ----------------------------------------------------------
    # Tfs and H calculated based on the input V
    # ----------------------------------------------------------
    Tfs = 1.0 / (1.0 - (2.0 / N) * np.log(V))
    lambda1 = -(V / 2.0) - np.sqrt(V**2 / 4.0 + omega)
    lambda2 = -(V / 2.0) + np.sqrt(V**2 / 4.0 + omega)
    H = lambda2 - lambda1

    # ----------------------------------------------------------
    # Residual function for fsolve
    # ----------------------------------------------------------
    def residual(xf_val):
        term1 = H * Tfs
        term2 = -omega * np.exp(lambda2 * xf_val) * (
                    (sigma / lambda2) * np.exp(-lambda2 * xf_val) +
                    ((phi - sigma) / (eta + lambda2)) * np.exp(-(eta + lambda2) * xf_val)
                )
        term3 = -omega * np.exp(lambda1 * xf_val) * (
                    (sigma / lambda1) * (1.0 - np.exp(-lambda1 * xf_val)) +
                    ((phi - sigma) / (eta + lambda1)) * (1.0 - np.exp(-(eta + lambda1) * xf_val)) -
                    (phi / lambda1)
                )
        term4 = -(1.0 - sigma) * V
        return term1 + term2 + term3 + term4

    try:
        xf_sol = float(fsolve(residual, 5.0))
        print(f"Calculated xf_sol = {xf_sol}")
    except Exception as e:
        print(f"Could not find a solution for xf_sol with V={V}. Error: {e}")
        return

    # ----------------------------------------------------------
    # Helper functions chi1, chi2
    # ----------------------------------------------------------
    def compute_chi1(x_val, l1, l2):
        T1 = (sigma / (l2 * (V + 2 * l1 + l2))) * (np.exp(-(V + 2 * l1 + l2) * x_val) - 1.0)
        T2 = (phi / ((eta + l2) * (V + 2 * l1 + eta + l2))) * (np.exp(-(V + 2 * l1 + eta + l2) * x_val) - 1.0)
        T3 = (sigma / ((eta + l2) * (V + 2 * l1 + eta + l2))) * (np.exp(-(V + 2 * l1 + eta + l2) * x_val) - 1.0)
        F2 = T1 + T2 - T3
        return omega * np.exp(l1 * x_val) * F2

    def compute_chi2(x_val, l1, l2):
        R1 = (sigma / (l1 * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
        R2 = (sigma / (l1 * (V + 2 * l2 + l1))) * (np.exp(-(V + 2 * l2 + l1) * x_val) - 1.0)
        R3 = ((phi - sigma) / ((eta + l1) * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
        R4 = ((phi - sigma) / ((eta + l1) * (V + 2 * l2 + eta + l1))) * (np.exp(-(V + 2 * l2 + eta + l1) * x_val) - 1.0)
        R5 = (phi / (l1 * (V + 2 * l2))) * (np.exp(-(V + 2 * l2) * x_val) - 1.0)
        G2 = -R1 + R2 - R3 + R4 + R5
        return omega * np.exp(l2 * x_val) * G2

    # ----------------------------------------------------------
    # Spatial solution
    # ----------------------------------------------------------
    xvals = np.arange(-50.01, 50.01, 0.01)

    chi1_xfs = compute_chi1(xf_sol, lambda1, lambda2)
    chi2_xfs = compute_chi2(xf_sol, lambda1, lambda2)

    T = np.empty_like(xvals)
    C1 = np.empty_like(xvals)

    for k, x_val in enumerate(xvals):
        chi1_x = compute_chi1(x_val, lambda1, lambda2)
        chi2_x = compute_chi2(x_val, lambda1, lambda2)

        C1[k] = 1.0 - np.exp(-V * Le * (x_val - xf_sol))
        
        T1 = (Tfs + chi1_xfs) * np.exp(lambda1 * (x_val - xf_sol)) - chi1_x
        T2 = (Tfs + chi2_xfs) * np.exp(lambda2 * (x_val - xf_sol)) - chi2_x

        T[k] = T1 if x_val >= xf_sol else T2

    C1[C1 < 0.0] = np.nan

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xvals, T, 'b-', linewidth=2.5, label='T (K)')
    ax1.set_xlabel('x', fontweight='bold', fontsize=13)
    ax1.set_ylabel('T (K)', color='b', fontweight='bold', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([0, 1])
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(xvals, C1, 'r-', linewidth=2.5, label='C₁')
    ax2.set_ylabel('C₁', color='r', fontweight='bold', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.axvline(xf_sol, color='k', linestyle='--', linewidth=2, label=f'xf_sol = {xf_sol:.2f}')
    
    # Add a single legend for all lines
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=13)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_weight('bold')

    plt.title(f'Temperature and C₁ profiles for V = {V}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


# Example usage
original_V = 0.060210448
generate_temperature_distance_plot(V=original_V)

# You can also try another value for V

