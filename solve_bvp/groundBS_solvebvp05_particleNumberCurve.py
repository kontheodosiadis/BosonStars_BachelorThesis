#%% Modules
import numpy as np
import matplotlib.pyplot as plt
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
    "legend.fontsize": 15,
    "figure.titlesize": 22
})

#%% Boson Star Mass-Curve Solver

#%%% Initialization

#%%%% System Parameters
eps = 1e-5
accu = 1e-3

Lambda = 0.0

sigma0_start = eps
sigma0_end   = 1.0  # Adjusted slightly; very high sigma_0 often requires shrinking x_max
points = 1001
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
y_guess[0] = np.ones_like(x_guess)                                                            # A
y_guess[1] = np.ones_like(x_guess)
width_guess = max(8.0, 1.0 / sigma0_start)                                                            # B
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
            integrand = (res.x**2) * np.sqrt(res.y[0] / res.y[1]) * res.p[0] * (res.y[2]**2)
            N = simpson(integrand, x=res.x)
            
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
        
#%%%% Finding where M = N
# Avoid equality at the beginning --> equality after max
ref_idx = np.argmax(M_list)

# Convert lists to numpy arrays for vector math
M_arr = np.array(M_list[ref_idx:])
N_arr = np.array(N_list[ref_idx:])
sigma_arr = np.array(valid_sigma0[ref_idx:])

# Calculate the absolute difference between the two datasets
diff = np.abs(M_arr - N_arr)

# Find the index where the difference is minimized
closest_idx = np.argmin(diff)

# Extract the values at this index
crossing_sigma = sigma_arr[closest_idx]
crossing_M = M_arr[closest_idx]
crossing_N = N_arr[closest_idx]

print(f"\nCentral Field (sigma_c): {crossing_sigma:.3f}")
print(f"Mass (M): {crossing_M:.3f} | Particle Number (N): {crossing_N:.3f}")
print(f"Absolute Difference: {diff[closest_idx]:.2e}")

#%%% #5: Plotting Results

fig, ax1 = plt.subplots(figsize=(8, 6))

# Define your line colors
color_M = '#2F4F4F'  # Charcoal 
color_N = '#1A4A54'  # Deep Teal/Petrol

# Plot 1: Mass on the Left Axis (All text and labels are default black)
ax1.set_xlabel(r"Central Field $\sigma_c$")
ax1.set_ylabel(r"Total Mass $\mathcal{M}$") # Removed color=color_M
line_M, = ax1.plot(valid_sigma0, M_list, color=color_M, linewidth=2.5, label=r"Mass $\mathcal{M}$")
ax1.set_xlim(sigma0_start, sigma0_end)
ax1.grid(True)

# Instantiate a second axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot 2: Particle Number on the Right Axis (All text and labels are default black)
ax2.set_ylabel(r"Total Number of Particles $\mathcal{N}$") # Removed color=color_N
line_N, = ax2.plot(valid_sigma0, N_list, color=color_N, linewidth=2.5, linestyle='--', label=r"Particle Number $\mathcal{N}$")

# Enforce the Same Scale
min_val = min(min(M_list), min(N_list))
max_val = max(max(M_list), max(N_list))
padding = (max_val - min_val) * 0.05
shared_ylim = (0, max_val + padding)

# Apply the identical limits to both axes
ax1.set_ylim(shared_ylim)
ax2.set_ylim(shared_ylim)

# Combine legends from both axes
lines = [line_M, line_N]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

#plt.title(f'Mass & Particle Number Relationship with Central Field $(Λ={Lambda})$', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

#%%% #6: Accuracy Report
headers = ["Index", "σ0 Value", "σ(inf) Error"]
print("\n--- Accuracy Report (Values > accu) ---")
if sigma0inf_off:
    print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
else:
    print("All final values within accuracy limits at infinity!")