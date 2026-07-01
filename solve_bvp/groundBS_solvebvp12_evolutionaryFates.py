#%% Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
from tabulate import tabulate

plt.rcParams.update({
    "font.family": "serif",          
    "font.serif": ["DejaVu Serif"],  
    "mathtext.fontset": "cm",        
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
sigma0_end   = 1.00  
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
y_guess[0] = np.ones_like(x_guess)                                                                            
y_guess[1] = np.ones_like(x_guess)
width_guess = max(8.0, 1.0 / sigma0_start)                                                                            
y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess)**2)                               
y_guess[3] = -2 * sigma0_start * (x_guess / width_guess**2) * np.exp(-(x_guess / width_guess)**2) 

# Eigenvalue guess
Omega_guess_val = np.sqrt(1.0 - sigma0_start**2)
Omega_guess = [Omega_guess_val]

#%%%% Storage Arrays
M_list = []
N_list = []
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
            ya[0] - 1.0,                 
            ya[2] - sigma_0,             
            ya[3] - 0.0,                 
            yb[1] - 1.0/yb[0],
            yb[3] + decay_rate * yb[2]   
        ])
        
    #%%%% #3: Solve BVP
    res = solve_bvp(fun, bc, x_guess, y_guess, p=Omega_guess, tol=1e-5, max_nodes=50000)
    
    #%%%%% #3.1: Checking for success
    if res.success:
        sigma = res.y[2]
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
        
        if nodes==0:
            consecutive_fails = 0  
            err_sigma = abs(res.y[2, -1])
            if err_sigma > accu:
                sigma0inf_off.append([i, sigma_0, f"{err_sigma:.2e}"])
                
            #%%%%% #4: Extracting the Data
            M = (res.x[-1]/2) * (1 - 1/res.y[0, -1])
            N_integrand = (res.x**2) * np.sqrt(res.y[0] / res.y[1]) * res.p[0] * (res.y[2]**2)
            N = simpson(N_integrand, x=res.x)
            
            M_list.append(M)
            N_list.append(N)
            valid_sigma0.append(sigma_0)
            
            # Update guesses
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p
            
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
        consecutive_fails += 1
        if consecutive_fails >= max_fails:
            print("Too many consecutive failures. Halting sweep.")
            break

# Turning into NumPy arrays
M_list = np.array(M_list)
N_list = np.array(N_list)
valid_sigma0 = np.array(valid_sigma0)

# Binding Energy
Eb_list = M_list - N_list


#%%% #5: Plotting Results

fig, ax = plt.subplots(figsize=(10, 5))

dyn_crit = M_list.argmax()
energ_crit = abs(Eb_list[dyn_crit:]).argmin() + dyn_crit

# Definition of intervals
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
    
    ax.plot(valid_sigma0[i:o+1], M_list[i:o+1], color=interval["color"], linewidth=2.5, linestyle=":")
    ax.plot(valid_sigma0[i:o+1], N_list[i:o+1], color=interval["color"], linewidth=2.5, linestyle="-.")
    ax.plot(valid_sigma0[i:o+1], Eb_list[i:o+1], color=interval["color"], linewidth=2.5, linestyle=custom_ddd)
    
    ax.axvspan(valid_sigma0[i], valid_sigma0[o], color=interval["color"], alpha=0.08, zorder=0)

# Guide lines
ax.axhline(y=0, color='gray', linewidth=1.2, alpha=0.5, zorder=1)
ax.axvline(x=valid_sigma0[dyn_crit], color='gray', ls='--', lw=2.5, alpha=0.9, zorder=2)
ax.axvline(x=valid_sigma0[energ_crit], color='gray', ls='--', lw=2.5, alpha=0.9, zorder=2)

# --- 2. Dynamic Placement of Mid-Region Text Annotations ---
x_reg1 = (sigma0_start + valid_sigma0[dyn_crit]) / 2
x_reg2 = (valid_sigma0[dyn_crit] + valid_sigma0[energ_crit]) / 2
x_reg3 = (valid_sigma0[energ_crit] + sigma0_end) / 2

y_max_val = max(M_list.max(), N_list.max())
y_min_val = Eb_list.min()
ax.set_ylim(y_min_val * 4, y_max_val * 1.1)

y_top = y_max_val * 0.36

bbox_style = dict(boxstyle="round,pad=0.5", fc="white", ec="#E2E8F0", alpha=0.95, lw=1)

# Three explicit text colors for the three fate regions
color_stable = '#2E7D32'    # Deep Green
color_collapse = '#1565C0'  # Deep Blue
color_disperse = '#A52A2A'  # Muted Dark Red

# Top Annotations: Star Fates
ax.text(x_reg1, y_top, "Stable", color=color_stable, 
         fontsize=16, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg2, y_top, "Unstable\nCollapse to BH", color=color_collapse, 
         fontsize=16, fontweight='bold', ha='center', va='center', bbox=bbox_style)
ax.text(x_reg3, y_top, "Unstable\nDisperse Away", color=color_disperse, 
         fontsize=16, fontweight='bold', ha='center', va='center', bbox=bbox_style)

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