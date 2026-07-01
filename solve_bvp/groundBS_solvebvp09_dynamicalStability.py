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
sigma0_end = 0.6
points = 1001
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

#%% #5: Locate Maximum Mass & Separate Branches
if len(M_list) > 0:
    idx_max = np.argmax(M_list)
    sigma0_max = valid_sigma0[idx_max]
    M_max = M_list[idx_max]
    N_max = N_list[idx_max]

    print("\n--- Stability Branch Report ---")
    print(f"Maximum Mass Found: M_max = {M_max:.4f} at sigma_0 = {sigma0_max:.4f}")

    # Formulate intervals for table display
    branch_data = [
        ["S-branch", f"({valid_sigma0[0]:.4f}, {sigma0_max:.4f})", "Stable (Green)"],
        ["U-branch", f"({sigma0_max:.4f}, {valid_sigma0[-1]:.4f})", "Unstable (Red)"]
    ]
    print(tabulate(branch_data, headers=["Branch", "sigma_0 Range", "Condition"], tablefmt="simple"))

    # #6: Plotting Results
    plt.figure(figsize=(10, 6))

    # Split data masks based on the turning point
    stable_mask = valid_sigma0 <= sigma0_max
    unstable_mask = valid_sigma0 >= sigma0_max

    # Plot Stable Branch (Green) and Unstable Branch (Red) - [Removed label tags here]
    plt.plot(valid_sigma0[stable_mask], M_list[stable_mask], color='forestgreen', linewidth=2.5)
    plt.plot(valid_sigma0[unstable_mask], M_list[unstable_mask], color='firebrick', linewidth=2.5)
    
    # For number of particles - [Removed label tags here]
    plt.plot(valid_sigma0[stable_mask], N_list[stable_mask], color='forestgreen', linewidth=2.5, linestyle='-.')
    plt.plot(valid_sigma0[unstable_mask], N_list[unstable_mask], color='firebrick', linewidth=2.5, linestyle='-.')

    # Shading backgrounds
    plt.axvspan(valid_sigma0[0], sigma0_max, color='forestgreen', alpha=0.12, zorder=0)
    plt.axvspan(sigma0_max, valid_sigma0[-1], color='firebrick', alpha=0.12, zorder=0)

    # Calculate midpoints for text label coordinates
    mid_stable = (valid_sigma0[0] + sigma0_max) / 2
    mid_unstable = (sigma0_max + valid_sigma0[-1]) / 2
    
    # Place text directly in the middle of each interval region
    y_text_position = N_max * 1.1  
    
    plt.text(mid_stable, y_text_position, "S-branch\n(Stable)", color='forestgreen', 
             weight='bold', ha='center', va='center', fontsize=16,
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.4'))
             
    plt.text(mid_unstable, y_text_position, "U-branch\n(Unstable)", color='firebrick', 
             weight='bold', ha='center', va='center', fontsize=16,
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.4'))

    # Mark the precise turning point
    plt.scatter(sigma0_max, M_max, color='black', s=70, zorder=5)
    plt.scatter(sigma0_max, N_max, color='black', s=70, zorder=5)
    
    # Connect it with the horizontal axis
    plt.axvline(sigma0_max, color='black', linestyle='--', alpha=0.3, zorder=1, lw=2.5)
    
    # --- COLOR-BLIND FRIENDLY LEGEND HANDLES ---
    # We create neutral black handles that match the linestyles of the variables
    legend_handles = [
        Line2D([0], [0], color='#333333', linewidth=2.5, linestyle='-', label='$M/(M_{Pl}^2/m)$'),
        Line2D([0], [0], color='#333333', linewidth=2.5, linestyle='-.', label='$N/(M_{Pl}^2/m^2)$')
    ]
    
    # Styling Plot
    plt.xlim(sigma0_start, sigma0_end)
    plt.ylim(0, N_max * 1.22)
    plt.xlabel("Central Field $\sigma_c$")
    plt.ylabel("Dimensionless Quantities")
    plt.grid(alpha=0.6)
    
    # Pass the custom handles explicitly to the legend
    plt.legend(handles=legend_handles, loc = 'lower right', fontsize=16)
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

