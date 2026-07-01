#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_bvp, simpson
from tabulate import tabulate
import os

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

#%%%% System Parameters
Lambda = 0.0

sigma0_start = 0.005
sigma0_end   = 0.5  
points = 500
sigma0_list = np.linspace(sigma0_start, sigma0_end, points)

eps = 1e-5
accu = 1e-3

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

#%%%% Storage Arrays for Animation & Static Plots
save_every = 5 # Save every 5 steps for a smooth animation (100 frames total)
successful_solutions = [] 
sigma0inf_off = []

# Values to keep as persistent background lines (from thesis script)
save_sigma0 = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5]
save_indices = [np.argmin(np.abs(sigma0_list - target)) for target in save_sigma0]
persistent_data = [] 
colors_persistent = plt.cm.tab10(np.linspace(0, 1, len(save_indices)))

#%%%% #1: Define the ODE System
def fun(x, y, p):
    # Eigenfunctions
    A = y[0]
    B = y[1]
    sigma = y[2]
    v = y[3]
    
    # Eigenvalue: Solver parameter
    Omega = p[0]
    
    # Einstein-Klein-Gordon Equations
    common_term = (v**2/A) + (Omega**2/B)*sigma**2
    Lambda_term = Lambda*sigma**4/2
    dAdx = x*A**2 *(common_term + sigma**2 + Lambda_term) - (A-1)*A/x
    dBdx = x*A*B * (common_term - sigma**2 - Lambda_term) + (A-1)*B/x
    dsigmadx = v
    dvdx = -(2/x + 0.5*(dBdx/B - dAdx/A))*v - A*((-1 + Omega**2/B)*sigma - Lambda*sigma**3)
    
    return np.vstack((dAdx, dBdx, dsigmadx, dvdx))

#%%% Loop over range of central field values
print("***Progress Report***")
for i, sigma_0 in enumerate(sigma0_list):
    #%%%% #2: Dynamic Boundary Conditions (Using yb[1] - 1.0/yb[0] from static script)
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
    
    if res.success:
        Omega = res.p[0]
        x = res.x
        A = res.y[0]
        B = res.y[1]
        sigma = res.y[2]
        
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
        
        if nodes == 0:
            consecutive_fails = 0 
            
            err_sigma = abs(sigma[-1])
            if err_sigma > accu:
                sigma0inf_off.append([i, sigma_0, f"{err_sigma:.2e}"])
                
            # Extracting the Data
            M_list = (x/2) * (1 - 1/A)
            M_total = M_list[-1]
            
            # Number of particles (from static script, for terminal updates)
            N_integrand = (x**2) * np.sqrt(A / B) * Omega * (sigma**2)
            N = simpson(N_integrand, x=x)

            # Check if this is a thesis value we want to keep in the background
            if i in save_indices:
                idx = save_indices.index(i)
                persistent_data.append({
                    'x': x,
                    'sigma': sigma,
                    'M_list': M_list,
                    'sigma_0': sigma_0,
                    'M': M_total,
                    'color': colors_persistent[idx]
                })
                print(f"\n[Saved Profile] σ0 : {sigma_0:.2f} | Omega: {Omega:.6f} | Mass: {M_total:.3f} | Particles: {N:.3f}") 
            
            # Save for animation frames
            if i % save_every == 0:
                successful_solutions.append({
                    'x': x,
                    'sigma': sigma,
                    'M_list': M_list,
                    'sigma_0': sigma_0,
                    'M': M_total
                })
            
            # Update guesses
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p
            
            # Progress report
            if i % (points//10) == 0:
                print(f"Step {i+1:3d}/{points} | sigma_0: {sigma_0:.3f} | Mass M: {M_total:.4f} | Omega: {Omega:.4f}")
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

#%%% #4: Animation Setup
print("\n***Generating Animation***")

# Make sure to update this directory to match your machine!
save_folder = r"C:\Users\hp\Desktop\1.University\A1.Semesters\8.Semester\Thesis\1.Codes\1.2.Development_Lab\Ground_State\Animations"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

gif_filename = 'Varying_Sigmac.gif'
gif_path = os.path.join(save_folder, gif_filename)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
#fig.suptitle(f'Boson Star Evolution $(\Lambda={Lambda})$', fontsize=18, fontweight='bold')

# Plot the persistent thesis curves in the background first
for data in persistent_data:
    sigma0_val = data['sigma_0']
    M_val = data['M']
    
    # Background Scalar Fields
    ax1.plot(data['x'], data['sigma'], 
             color=data['color'], linestyle='--', alpha=0.8, lw=2,
             label=f'$\sigma_c={sigma0_val:.2f}$')
    
    # Background Masses
    ax2.plot(data['x'], data['M_list'], 
             color=data['color'], linestyle='--', alpha=0.8, lw=2)
             #label=rf'$\mathcal{M}={M_val:.3f}$')

# Create the solid lines that will animate over the top
line1, = ax1.plot([], [], lw=2.5, color='black')  
line2, = ax2.plot([], [], lw=2.5, color='black')  

max_sigma = np.max([np.max(d['sigma']) for d in successful_solutions])
max_M = np.max([np.max(d['M_list']) for d in successful_solutions])

# Format Axis 1 (Scalar Field) using limits from static script
ax1.set_xlim(0, 20) 
ax1.set_ylim(0, max_sigma * 1.05)
ax1.set_ylabel("Scalar Field $\sigma$")
ax1.set_xlabel("Radius $x$")
ax1.set_title("Scalar Field Profiles")
ax1.grid(True)

# Format Axis 2 (Mass) using limits from static script
ax2.set_xlim(0, 50) 
ax2.set_ylim(0, max_M * 1.05)
ax2.set_ylabel("Internal Mass $\mathcal{M}$")
ax2.set_xlabel("Radius $x$")
ax2.set_title("Mass Profiles")
ax2.grid(True)

plt.subplots_adjust(left=0.1, right=0.95, top=0.92, wspace=0.38)

def update(frame):
    data = successful_solutions[frame]
    
    # Update Scalar Field line
    line1.set_data(data['x'], data['sigma'])
    line1.set_label(f'Current\n$\sigma_c={data["sigma_0"]:.3f}$')
    ax1.legend(loc='upper right')
    
    # Update Mass line
    line2.set_data(data['x'], data['M_list'])
    #line2.set_label(f'Current $M={data["M"]:.3f}$')
    #ax2.legend(loc='lower right')
    
    return line1, line2

ani = FuncAnimation(
    fig, 
    update, 
    frames=len(successful_solutions), 
    blit=False, 
    interval=100  # 100ms per frame to make the 100 frames take about 10 seconds
)

# Save the GIF
ani.save(gif_path, writer='pillow', fps=10)

print(f"Animation successfully saved as: {gif_filename}")
plt.show()

#%%% #5: Accuracy Report
headers = ["Index", "σ0 Value", "σ(inf) Error"]
print("\n***Accuracy Report (Values > accu)***")
if sigma0inf_off:
    print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
else:
    print("All final values within accuracy limits at infinity!")