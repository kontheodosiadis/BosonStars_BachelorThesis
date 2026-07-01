#%% Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    "legend.fontsize": 14,
    "figure.titlesize": 22
})

#%% Boson Star Mass-Curve Solver

#%%% Initialization
eps = 1e-5
accu = 1e-3

#%%%% System Parameters
Lambda = 0.0

sigma0_start = eps
sigma0_end   = 1.0  # Adjusted slightly; very high sigma_0 often requires shrinking x_max
points = 1501
sigma0_list = np.linspace(sigma0_start, sigma0_end, points)

# Fail safe
consecutive_fails = 0
max_fails = 5

#%%%% #A: Initial Setup
x_min = eps
x_max = 125.0
x_guess = np.linspace(x_min, x_max, 1000)

# Eigenfunction guesses
y_guess = np.zeros((4, x_guess.size))
y_guess[0] = np.ones_like(x_guess)                                                                            # A
y_guess[1] = np.ones_like(x_guess)
width_guess = max(8.0, 1.0 / sigma0_start)                                                                            # B
y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess)**2)                               # sigma
y_guess[3] = -2 * sigma0_start * (x_guess / width_guess**2) * np.exp(-(x_guess / width_guess)**2) # phi

# Eigenvalue guess
Omega_guess_val = np.sqrt(1.0 - sigma0_start**2)
Omega_guess = [Omega_guess_val]

#%%%% Storage Arrays
M_list = []
N_list = []
# Omega_list = []
valid_sigma0 = []
sigma0inf_off = []

#%%%% #1: Define the ODE System
def fun(x, y, p):
    Omega = p[0]
    A, B, sigma, phi = y[0], y[1], y[2], y[3]
    
    Omega2_B = (Omega**2) / B
    Lambda_term = (Lambda / 2) * sigma**4
    phi2_A = (phi**2) / A
    
    A_prime = (A * (1 - A) / x) + x * (A**2) * ((Omega2_B + 1) * sigma**2 + Lambda_term + phi2_A)
    B_prime = (B * (A - 1) / x) + x * A * B * ((Omega2_B - 1) * sigma**2 - Lambda_term + phi2_A)
    
    friction_term = (2 / x) + (B_prime / (2 * B)) - (A_prime / (2 * A))
    force_term = A * ((Omega2_B - 1) * sigma - Lambda * sigma**3)
    
    phi_prime = -friction_term * phi - force_term
    
    return np.vstack((A_prime, B_prime, phi, phi_prime))

#%%% Loop over range of central field values

for i, sigma_0 in enumerate(sigma0_list):
    
    #%%%% #2: Dynamic Boundary Conditions
    def bc(ya, yb, p):
        Omega = p[0]
        decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
        
        return np.array([
            ya[0] - 1.0,                 # A(0) = 1
            ya[2] - sigma_0,             # sigma(0) = current sigma_0
            ya[3] - 0.0,                 # phi(0) = 0
            # Standard Condition
            # yb[1] - 1.0,                 # B(inf) = 1 (Asymptotic flatness)
            # More accurate condition 
            yb[1] - 1.0/yb[0],
            # yb[0] * yb[1] - 1.0,       # A*B = 1 in vacuum (Exact Asymptotic flatness)
            yb[3] + decay_rate * yb[2]   # Robin BC for scalar field decay at inf
        ])
        
    #%%%% #3: Solve BVP
    res = solve_bvp(fun, bc, x_guess, y_guess, p=Omega_guess, tol=1e-5, max_nodes=50000)
    
    #%%%%% #3.1: Checking for success
    if res.success:
        #%%%%% #3.2: Checking for nodes
        sigma = res.y[2]
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
        
        if nodes==0:
            consecutive_fails = 0  # Reset on success
            #%%%%% #3.3: Checking for sufficient Accuracy for σ(∞) → 0
            err_sigma = abs(res.y[2, -1])
            if err_sigma > accu:
                sigma0inf_off.append([
                    i, 
                    sigma_0, 
                    f"{err_sigma:.2e}"
                ])
                
            #%%%%% #4: Extracting the Data

            M = (res.x[-1]/2) * (1 - 1/res.y[0, -1])
            N_integrand = (res.x**2) * np.sqrt(res.y[0] / res.y[1]) * res.p[0] * (res.y[2]**2)
            N = simpson(N_integrand, x=res.x)
            
            M_list.append(M)
            N_list.append(N)
            # Omega_list.append(res.p[0])
            valid_sigma0.append(sigma_0)
            
            #%%%%% #B: Update guesses from solution for the next iteration
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p
            
            # Print progress every points/10 steps
            if i % (points//10) == 0:
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

# Turning into NumPy arrays for better handling
M_list = np.array(M_list)
N_list = np.array(N_list)
valid_sigma0 = np.array(valid_sigma0)

# Binding Energy
Eb_list = M_list - N_list


#%%% #5: Plotting Results

fig, ax = plt.subplots(figsize=(10, 5))

dyn_crit = M_list.argmax()
energ_crit = abs(Eb_list[dyn_crit:]).argmin() + dyn_crit

# Definition of intervals matching the fates of your stars
all_intervals = [
    {"start": 0, "end": dyn_crit, "color": 'limegreen'},
    {"start": dyn_crit, "end": energ_crit, "color": 'darkblue'},
    {"start": energ_crit, "end": len(valid_sigma0)-1, "color": 'firebrick'}
]

custom_ddd = (0, (3, 1.5, 1, 1.5, 1, 1.5))

# --- 1. Plot Data Curves & Shaded Background Fates ---
for interval in all_intervals:
    i = interval["start"]
    o = interval["end"]
    
    # Using o+1 to eliminate physical gap styling issues between colored line joins
    ax.plot(valid_sigma0[i:o+1], M_list[i:o+1], color=interval["color"], linewidth=2.5)
    ax.plot(valid_sigma0[i:o+1], N_list[i:o+1], color=interval["color"], linewidth=2.5, linestyle="-.")
    ax.plot(valid_sigma0[i:o+1], Eb_list[i:o+1], color=interval["color"], linewidth=2.5, linestyle=custom_ddd)
    
    ax.axvspan(valid_sigma0[i], valid_sigma0[o], color=interval["color"], alpha=0.08, zorder=0)

# Solid zero baseline line
ax.axhline(y=0, color='gray', linewidth=1.2, alpha=0.5, zorder=1)

# Soft vertical guide lines at critical transition thresholds
ax.axvline(x=valid_sigma0[dyn_crit], color='gray', ls='--', lw=2.5, alpha=0.9, zorder=2)
ax.axvline(x=valid_sigma0[energ_crit], color='gray', ls='--', lw=2.5, alpha=0.9, zorder=2)

# --- 2. Dynamic Placement of Mid-Region Text Annotations ---
# Calculate precise horizontal centers for all three fate columns
x_reg1 = (sigma0_start + valid_sigma0[dyn_crit]) / 2
x_reg2 = (valid_sigma0[dyn_crit] + valid_sigma0[energ_crit]) / 2
x_reg3 = (valid_sigma0[energ_crit] + sigma0_end) / 2

# Scale Y limits dynamically to keep text boxes well within margins
y_max_val = max(M_list.max(), N_list.max())
y_min_val = Eb_list.min()
ax.set_ylim(y_min_val * 4, y_max_val * 1.1)

y_top = max(M_list.max(), N_list.max()) * 0.46
y_bottom = y_top - 0.13

# Text box edge & background parameters
bbox_style = dict(boxstyle="round,pad=0.5", fc="white", ec="#E2E8F0", alpha=0.95, lw=1)
color_stable = '#2E7D32'    # Deep Green
color_unstable = '#A52A2A'  # Muted Dark Red

# Top Annotations: Stability Properties
ax.text(x_reg1, y_top, "S-branch\n(Stable)", color=color_stable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg2, y_top, "U-branch\n(Unstable)", color=color_unstable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg3, y_top, "U-branch\n(Unstable)", color=color_unstable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)

# Bottom Annotations: Energy Binding Bindings
ax.text(x_reg1, y_bottom, "Bound\n($E_b < 0$)", color=color_stable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg2, y_bottom, "Bound\n($E_b < 0$)", color=color_stable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg3, y_bottom, "Unbound\n($E_b > 0$)", color=color_unstable, 
         fontsize=14, fontweight='bold', ha='center', va='center', bbox=bbox_style)

# --- 3. High-Contrast Anchor Points ---
ax.scatter(valid_sigma0[dyn_crit], M_list[dyn_crit], color='black', marker='o', 
            s=90, zorder=6, linewidths=1.5)
ax.scatter(valid_sigma0[dyn_crit], N_list[dyn_crit], color='black', marker='o', 
            s=90, zorder=6, linewidths=1.5)
ax.scatter(valid_sigma0[energ_crit], Eb_list[energ_crit], color='black', marker='x', 
            s=90, zorder=6, linewidths=2.5)

# --- 4. Clean Layout Formatting & Simple Internal Legend ---
ax.set_xlabel(r"Central Field $\sigma_c$")
ax.set_ylabel("Dimensionless Quantities")
ax.set_xlim(sigma0_start, sigma0_end)

# Neutral line references for the simple geometric legend representation
line_M  = mlines.Line2D([], [], color='#4A5568', linestyle='-',  linewidth=2.5, label=r'$M / (M_{Pl}^2 / m)$')
line_N  = mlines.Line2D([], [], color='#4A5568', linestyle='-.', linewidth=2.5, label=r'$N / (M_{Pl}^2 / m^2)$')
line_Eb = mlines.Line2D([], [], color='#4A5568', linestyle=custom_ddd, linewidth=2.5, label=r'$E_b / (M_{Pl}^2 / m)$')

ax.legend(handles=[line_M, line_N, line_Eb], loc='upper right')

ax.grid(alpha=0.6) 
plt.tight_layout()
plt.show()

#%%% #6: Accuracy Report
headers = ["Index", "σ0 Value", "σ(inf) Error"]
print("\n--- Accuracy Report (Values > accu) ---")
if sigma0inf_off:
    print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
else:
    print("All final values within accuracy limits at infinity!")