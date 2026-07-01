#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
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

#%% Boson Star Mass-Curve Solver

#%%% Initialization
eps = 1e-5
accu = 1e-3

#%%%% System Parameters
Lambda = 0.0

sigma0_start = eps
sigma0_end   = 2.0
points = 2001
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
y_guess[0] = np.ones_like(x_guess)                                                             # A
y_guess[1] = np.ones_like(x_guess)
width_guess = max(8.0, 1.0 / sigma0_start)                                                             # B
y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess)**2)                               # sigma
y_guess[3] = -2 * sigma0_start * (x_guess / width_guess**2) * np.exp(-(x_guess / width_guess)**2) # phi

# Eigenvalue guess
Omega_guess_val = np.sqrt(1.0 - sigma0_start**2)
Omega_guess = [Omega_guess_val]

#%%%% Storage Arrays
M_list = []
Omega_list = []
R_list = []
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

#%%% Loop over a range of central field values

for i, sigma_0 in enumerate(sigma0_list):
    
    #%%%% #2: Dynamic Boundary Conditions
    def bc(ya, yb, p):
        Omega = p[0]
        decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
        
        return np.array([
            ya[0] - 1.0,                 # A(0) = 1
            ya[2] - sigma_0,             # sigma(0) = current sigma_0
            ya[3] - 0.0,                 # phi(0) = 0
            yb[1] - 1.0/yb[0],           # B(inf) = 1 (Asymptotic flatness)
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
            else:
                #%%%%% #4: Extracting the Data
                M = (res.x[-1]/2) * (1 - 1/res.y[0, -1])
                
                # Radius containing 95% of total mass
                M_profile = (res.x / 2) * (1 - 1 / res.y[0])
                idx_95 = np.where(M_profile >= 0.95 * M)[0]
                R = res.x[idx_95[0]] if len(idx_95) > 0 else res.x[-1]
                
                M_list.append(M)
                Omega_list.append(res.p[0])
                R_list.append(R)
                valid_sigma0.append(sigma_0)
            
            #%%%%% #B: Update guesses from solution for the next iteration
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p
            
            # Print progress every points/10 steps
            if i % (points//10) == 0:
                print(f"Step {i+1:3d}/{points} | sigma_0: {sigma_0:.3f} | Mass M: {M:.4f} | Radius R: {R:.2f} | Omega: {res.p[0]:.4f}")
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

#%%% #5: Plotting Results
fig, ax1 = plt.subplots(figsize=(6, 6))

#5.4: Plotting M vs R with Zoom Inset
ax1.plot(R_list, M_list, color='#36454F', linewidth=2.5)
ax1.set_ylabel("Total Mass $\mathcal{M}$")
ax1.set_xlabel("Radius $R$ (95% of Total Mass)")
# ax1.set_title("Mass vs Radius")
ax1.grid(True, alpha=0.8)
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 0.67) # Adjust y limit for clean appearance

# [Start of replacement section for arrows]

# Add well-aligned, uniformly spaced arrows using quiver
R_arr = np.array(R_list)
M_arr = np.array(M_list)

# 1. Define physical axis ranges to handle the massive aspect ratio difference
R_range = 50.0 - 0.0     # From ax1.set_xlim(0, 50)
M_range = 0.67 - (-0.03) # From ax1.set_ylim(-0.03, 0.67)

# 2. Calculate arc length in "visual space" for perfectly even arrow spacing
vis_dx = np.diff(R_arr) / R_range
vis_dy = np.diff(M_arr) / M_range
vis_dist = np.sqrt(vis_dx**2 + vis_dy**2)
cum_dist = np.insert(np.cumsum(vis_dist), 0, 0)

# We want exactly 9 arrows spaced evenly along the visual curve
num_arrows = 9
# Avoid placing arrows at the very first or very last point
target_dists = np.linspace(0, cum_dist[-1], num_arrows + 2)[1:-1] 
arrow_indices = [np.searchsorted(cum_dist, d) for d in target_dists]

# Extract coordinates for the arrows
arrow_Rs = R_arr[arrow_indices]
arrow_Ms = M_arr[arrow_indices]

# 3. Tangent vector calculation via gradients
dR_grad = np.gradient(R_arr)
dM_grad = np.gradient(M_arr)

# Normalize vectors in "visual space" so they all have the same apparent length
vis_u = dR_grad / R_range
vis_v = dM_grad / M_range
vis_mag = np.sqrt(vis_u**2 + vis_v**2)

u_vis_unit = vis_u / vis_mag
v_vis_unit = vis_v / vis_mag

# 4. Convert back to data coordinates
# Set arrow length to exactly 3.5% of the plot's width/height
arrow_length_fraction = 0.035 
u_arrows = (u_vis_unit * R_range * arrow_length_fraction)[arrow_indices]
v_arrows = (v_vis_unit * M_range * arrow_length_fraction)[arrow_indices]

# Plot the arrows!
ax1.quiver(arrow_Rs, arrow_Ms, u_arrows, v_arrows,
           angles='xy', scale_units='xy', scale=1.0, pivot='mid',
           color='#36454F', width=0.008, headwidth=4.5, headlength=5, headaxislength=4,
           label=None, zorder=5)

# [End of replacement section for arrows]

# Inset Zoom Axis for the Loop
# Placing it in the upper right corner
ax_ins = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
ax_ins.plot(R_list, M_list, color='#36454F', linewidth=2)

# Define zoom limits (Focusing on the spiral)
idx_max = np.argmax(M_list)
R_max = R_list[idx_max]
M_max = M_list[idx_max]

idx_spiral = np.argmin(M_list[idx_max:])
R_spiral = R_list[idx_spiral]
M_spiral = M_list[idx_spiral]

DeltaR = (R_max - R_spiral) * 0.5
DeltaM = (M_max - M_spiral)/3

# Set the limits for the zoomed-in view
ax_ins.set_xlim(R_spiral - DeltaR, R_spiral + DeltaR) 
ax_ins.set_ylim(M_spiral - DeltaM, M_spiral + DeltaM)

# Clean up inset ticks
ax_ins.tick_params(axis='both', which='major', labelsize=8)
ax_ins.grid(True, alpha=0.8)

# Automatically draw the box and connector lines on the main plot
ax1.indicate_inset_zoom(ax_ins, edgecolor="gray")

# Formatting the general figure
plt.tight_layout()
plt.show()

#%%% #6: Accuracy Report
headers = ["Index", "σ0 Value", "σ(inf) Error"]
print("\n--- Accuracy Report (Values > accu) ---")
if sigma0inf_off:
    print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
else:
    print("All final values within accuracy limits at infinity!")