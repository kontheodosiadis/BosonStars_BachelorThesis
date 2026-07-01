# %% Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
from tabulate import tabulate

plt.rcParams.update({
    "font.family": "serif",          # Tell Matplotlib to use a serif font family
    "font.serif": ["DejaVu Serif"],  # Force its built-in serif font (guaranteed to exist)
    "mathtext.fontset": "cm",        # Use Computer Modern (LaTeX font) for math expressions
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 22
})

# %% Boson Star Mass-Curve Solver

# %%% Initialization
eps = 1e-5
accu = 1e-3

# %%%% System Parameters
Lambda = 0.0

sigma0_start = eps
sigma0_end = 1.0
points = 1501
sigma0_list = np.linspace(sigma0_start, sigma0_end, points)

# Fail safe
consecutive_fails = 0
max_fails = 5

# %%%% #A: Initial Setup
x_min = eps
x_max = 125.0
x_guess = np.linspace(x_min, x_max, 1000)

# Eigenfunction guesses
y_guess = np.zeros((4, x_guess.size))
y_guess[0] = np.ones_like(x_guess)  # A
y_guess[1] = np.ones_like(x_guess)
width_guess = max(8.0, 1.0 / sigma0_start)  # B
y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess) ** 2)  # sigma
y_guess[3] = -2 * sigma0_start * (x_guess / width_guess ** 2) * np.exp(-(x_guess / width_guess) ** 2)  # phi

# Eigenvalue guess
Omega_guess_val = np.sqrt(1.0 - sigma0_start ** 2)
Omega_guess = [Omega_guess_val]

# %%%% Storage Arrays
M_list = []
N_list = []
Omega_list = []
valid_sigma0 = []
sigma0inf_off = []

# %%%% #1: Define the ODE System
def fun(x, y, p):
    Omega = p[0]
    A, B, sigma, phi = y[0], y[1], y[2], y[3]

    Omega2_B = (Omega ** 2) / B
    Lambda_term = (Lambda / 2) * sigma ** 4
    phi2_A = (phi ** 2) / A

    A_prime = (A * (1 - A) / x) + x * (A ** 2) * ((Omega2_B + 1) * sigma ** 2 + Lambda_term + phi2_A)
    B_prime = (B * (A - 1) / x) + x * A * B * ((Omega2_B - 1) * sigma ** 2 - Lambda_term + phi2_A)

    friction_term = (2 / x) + (B_prime / (2 * B)) - (A_prime / (2 * A))
    force_term = A * ((Omega2_B - 1) * sigma - Lambda * sigma ** 3)

    phi_prime = -friction_term * phi - force_term

    return np.vstack((A_prime, B_prime, phi, phi_prime))

# %%% Loop over a range of central field values

for i, sigma_0 in enumerate(sigma0_list):

    # %%%% #2: Dynamic Boundary Conditions
    def bc(ya, yb, p):
        Omega = p[0]
        decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega ** 2))

        return np.array([
            ya[0] - 1.0,  # A(0) = 1
            ya[2] - sigma_0,  # sigma(0) = current sigma_0
            ya[3] - 0.0,  # phi(0) = 0
            yb[1] - 1.0 / yb[0],  # B(inf) = 1 (Asymptotic flatness)
            yb[3] + decay_rate * yb[2]  # Robin BC for scalar field decay at inf
        ])

    # %%%% #3: Solve BVP
    res = solve_bvp(fun, bc, x_guess, y_guess, p=Omega_guess, tol=1e-5, max_nodes=50000)

    # %%%%% #3.1: Checking for success
    if res.success:
        # %%%%% #3.2: Checking for nodes
        sigma = res.y[2]
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)

        if nodes == 0:
            consecutive_fails = 0  # Reset on success
            # %%%%% #3.3: Checking for sufficient Accuracy for σ(∞) → 0
            err_sigma = abs(res.y[2, -1])
            if err_sigma > accu:
                sigma0inf_off.append([
                    i,
                    sigma_0,
                    f"{err_sigma:.2e}"
                ])

            # %%%%% #4: Extracting the Data
            M = (res.x[-1] / 2) * (1 - 1 / res.y[0, -1])
            integrand = (res.x**2) * np.sqrt(res.y[0] / res.y[1]) * res.p[0] * (res.y[2]**2)
            N = simpson(integrand, x=res.x)
            
            M_list.append(M)
            N_list.append(N)
            Omega_list.append(res.p[0])
            valid_sigma0.append(sigma_0)

            # %%%%% #B: Update guesses from solution for the next iteration
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p

            # Print progress every points/10 steps
            if i % (points // 10) == 0:
                print(f"Step {i+1:3d}/{points} | sigma_0: {sigma_0:.3f} | Mass M: {M:.4f} | Omega: {res.p[0]:.4f}")
        else:
            print(f"\nExcited state at index {i}, sigma_0 = {sigma_0:.4f}")
            consecutive_fails += 1

            if consecutive_fails >= max_fails:
                print("Too many consecutive failures. Halting sweep.")
                break
    else:
        print(f"\nSolver broke at index {i}, sigma_0 = {sigma_0:.4f}")
        print(res.message)
        consecutive_fails += 1

        if consecutive_fails >= max_fails:
            print("Too many consecutive failures. Halting sweep.")
            break

M_list = np.array(M_list)
N_list = np.array(N_list)
valid_sigma0 = np.array(valid_sigma0)

# %% #5: Locate Binding Energy Boundaries
if len(M_list) > 0:
    # Calculate Binding Energy
    Eb_list = M_list - N_list
    
    # Identify transition where Eb becomes positive (ignoring numerical noise near 0)
    unbounded_mask_check = (Eb_list > 0) & (valid_sigma0 > 0.05)
    
    if np.any(unbounded_mask_check):
        idx_trans = np.argmax(unbounded_mask_check)
        
        # Interpolate exact zero-crossing for clean boundaries
        sigma0_trans = np.interp(0, 
                                 [Eb_list[idx_trans-1], Eb_list[idx_trans]], 
                                 [valid_sigma0[idx_trans-1], valid_sigma0[idx_trans]])
    else:
        # Fallback: Star remains entirely bound in this range
        sigma0_trans = valid_sigma0[-1] + 1.0

    print("\n--- Bounded State Report ---")
    if sigma0_trans <= valid_sigma0[-1]:
        print(f"Zero Binding Energy Crossing Found: Eb = 0 at sigma_0 = {sigma0_trans:.4f}")
        branch_data = [
            ["Bound", f"({valid_sigma0[0]:.4f}, {sigma0_trans:.4f})", "Eb < 0 (Green)"],
            ["Unbound", f"({sigma0_trans:.4f}, {valid_sigma0[-1]:.4f})", "Eb > 0 (Red)"]
        ]
    else:
        print("Star remains completely bounded (Eb < 0) within the scanned range.")
        branch_data = [
            ["Bound", f"({valid_sigma0[0]:.4f}, {valid_sigma0[-1]:.4f})", "Eb < 0 (Green)"]
        ]
        
    print(tabulate(branch_data, headers=["State", "sigma_0 Range", "Condition"], tablefmt="simple"))

    #6: Plotting Results
    plt.figure(figsize=(10, 6))

    # Split data masks based on the binding energy boundary
    bounded_mask = valid_sigma0 <= sigma0_trans
    unbounded_mask = valid_sigma0 >= sigma0_trans

    # Plot Bounded State (Green) and Unbounded State (Red) for M
    plt.plot(valid_sigma0[bounded_mask], M_list[bounded_mask], color='forestgreen', linewidth=2.5)
    plt.plot(valid_sigma0[unbounded_mask], M_list[unbounded_mask], color='firebrick', linewidth=2.5)
    
    # Plot N (Dash-dot)
    plt.plot(valid_sigma0[bounded_mask], N_list[bounded_mask], color='forestgreen', linewidth=2.5, linestyle='-.')
    plt.plot(valid_sigma0[unbounded_mask], N_list[unbounded_mask], color='firebrick', linewidth=2.5, linestyle='-.')

    # Plot Eb (Dotted)
    custom_ddd = (0, (3, 1.5, 1, 1.5, 1, 1.5))
    plt.plot(valid_sigma0[bounded_mask], Eb_list[bounded_mask], color='forestgreen', linewidth=2.5, linestyle=custom_ddd)
    plt.plot(valid_sigma0[unbounded_mask], Eb_list[unbounded_mask], color='firebrick', linewidth=2.5, linestyle=custom_ddd)
    
    # Background shading & text placement
    min_val = np.min(Eb_list)
    y_text_position = min_val * 5  
    
    if sigma0_trans <= valid_sigma0[-1]:
        plt.axvspan(valid_sigma0[0], sigma0_trans, color='forestgreen', alpha=0.12, zorder=0)
        plt.axvspan(sigma0_trans, valid_sigma0[-1], color='firebrick', alpha=0.12, zorder=0)
        
        mid_bounded = (valid_sigma0[0] + sigma0_trans) / 2
        mid_unbounded = (sigma0_trans + valid_sigma0[-1]) / 2
        
        plt.text(mid_bounded, y_text_position, "Bound\n$(E_b < 0)$", color='forestgreen', 
                 weight='bold', ha='center', va='center', fontsize=16,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.4'))
                 
        plt.text(mid_unbounded, y_text_position, "Unbound\n$(E_b > 0)$", color='firebrick', 
                 weight='bold', ha='center', va='center', fontsize=16,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.4'))

        # Mark the precise zero-crossing
        plt.scatter(sigma0_trans, 0, color='black', marker='x', s=70, zorder=5)
        plt.scatter(sigma0_trans, M_list[idx_trans], color = 'black', marker='x', s=70, zorder=5)
        plt.axvline(sigma0_trans, color='black', linestyle='--', alpha=0.3, zorder=1, lw=2.5)
        
    else:
        plt.axvspan(valid_sigma0[0], valid_sigma0[-1], color='forestgreen', alpha=0.12, zorder=0)
        mid_bounded = (valid_sigma0[0] + valid_sigma0[-1]) / 2
        plt.text(mid_bounded, y_text_position, "Bound\n$(E_b < 0)$", color='forestgreen', 
                 weight='bold', ha='center', va='center', fontsize=16,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.4'))

    # Visual baseline for Eb = 0
    plt.axhline(0, color='black', linewidth=1.5, alpha=0.6, zorder=1)
    
    # --- COLOR-BLIND FRIENDLY LEGEND HANDLES ---
    legend_handles = [
        Line2D([0], [0], color='#333333', linewidth=2.5, linestyle='-', label='$M/(M_{Pl}^2/m)$'),
        Line2D([0], [0], color='#333333', linewidth=2.5, linestyle='-.', label='$N/(M_{Pl}^2/m^2)$'),
        Line2D([0], [0], color='#333333', linewidth=2.5, linestyle=custom_ddd, label='$E_b/(M_{Pl}^2/m)$')
    ]
    
    # Styling Plot (y-axis drops below 0 to accommodate the Eb curve)
    plt.xlim(sigma0_start, sigma0_end)
    max_val = max(np.max(M_list), np.max(N_list))
    plt.ylim(min_val * 9, max_val * 1.14)
    plt.xlabel("Central Field $\sigma_c$")
    plt.ylabel("Dimensionless Quantities")
    plt.grid(alpha=0.6)
    
    plt.legend(handles=legend_handles, loc='upper right', fontsize=16)
    plt.show()
else:
    print("\nNo valid data to analyze or plot.")

# %% % #7: Accuracy Report
headers = ["Index", "σ0 Value", "σ(inf) Error"]
print("\n--- Accuracy Report (Values > accu) ---")
if sigma0inf_off:
    print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
else:
    print("All final values within accuracy limits at infinity!")