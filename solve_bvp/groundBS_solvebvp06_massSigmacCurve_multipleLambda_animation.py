import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_bvp
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
    "legend.fontsize": 14,
    "figure.titlesize": 22
})

#% Boson Star Mass-Curve Evolution with Lambda

#%% 1. Initialization & Parameters
Lambda_list = [0.0, 1.0, 10.0, 30.0, 100.0, 200.0, 300.0]
colors = plt.cm.tab10(np.linspace(0, 1, len(Lambda_list)))

sigma0_start = 0.005
sigma0_end   = 0.505
sigma_points = 201
sigma0_list = np.linspace(sigma0_start, sigma0_end, sigma_points)

eps = 1e-5
accu = 1e-2

# This will hold the dictionary of curves for each Lambda
successful_lambda_sweeps = [] 

#%% 2. Physics Sweep
print("*** Starting Physics Sweep ***")

for j, Lambda in enumerate(Lambda_list):
    print(f"\n--- Processing Lambda = {Lambda:.0f} (Step {j+1}/{len(Lambda_list)}) ---")
    
    # Reset system bounds and guesses for each new Lambda
    x_min = eps
    x_max = 125.0
    
    width_guess = max(8.0, 1.0 / sigma0_start)
    Omega_guess_val = np.sqrt(1.0 - sigma0_start**2)
    
    x_guess = np.linspace(x_min, x_max, 1000)
    y_guess = np.zeros((4, x_guess.size))
    y_guess[0] = np.ones_like(x_guess)                                                   
    y_guess[1] = np.ones_like(x_guess)                                                   
    y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess)**2)                               
    y_guess[3] = -2 * sigma0_start * (x_guess / width_guess**2) * np.exp(-(x_guess / width_guess)**2) 
    Omega_guess = [Omega_guess_val]
    
    # Temporary storage for the current Lambda sweep
    current_M_list = []
    current_Omega_list = []
    current_valid_sigma0 = []
    sigma0inf_off = []
    
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
    
    for i, sigma0 in enumerate(sigma0_list):
        def bc(ya, yb, p):
            Omega = p[0]
            decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
            return np.array([
                ya[0] - 1.0,                 
                ya[2] - sigma0,              
                ya[3] - 0.0,                 
                yb[1] - 1.0,                 
                yb[3] + decay_rate * yb[2]   
            ])
            
        res = solve_bvp(fun, bc, x_guess, y_guess, p=Omega_guess, tol=1e-5, max_nodes=50000)
        
        if not res.success:
            print(f"Solver broke at index {i}, sigma_0 = {sigma0:.4f}. Stopping this Lambda sweep.")
            break
            
        # Accuracy check
        err_sigma = abs(res.y[2, -1])
        if err_sigma > accu:
            sigma0inf_off.append([i, sigma0, f"{err_sigma:.2e}"])
            
        # Extract macroscopic data
        M = (res.x[-1]/2) * (1 - 1/res.y[0, -1])
        
        current_M_list.append(M)
        current_Omega_list.append(res.p[0])
        current_valid_sigma0.append(sigma0)
        
        # Continuation step
        x_guess = res.x
        y_guess = res.y
        Omega_guess = res.p
        
        if i % (sigma_points//5) == 0:
            print(f"  Step {i:3d}/{sigma_points} | sigma_0: {sigma0:.3f} | Mass M: {M:.4f} | Omega: {res.p[0]:.4f}")

    # Pack the completed sweep into the dictionary
    successful_lambda_sweeps.append({
        'Lambda': Lambda,
        'sigma0_array': current_valid_sigma0,
        'M_array': current_M_list,
        'Omega_array': current_Omega_list,
        'color': colors[j],
        'errors': sigma0inf_off
    })

#%% 3. Animation Setup
print("\n*** Generating Animation ***")

save_folder = r"C:\Users\hp\Desktop\1.University\A1.Semesters\8.Semester\Thesis\1.Codes\1.2.Development_Lab\Ground_State\Animations"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

gif_filename = 'MassCurves.gif'
gif_path = os.path.join(save_folder, gif_filename)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#fig.suptitle('Boson Star Macroscopic Properties vs Central Field', fontsize=16, fontweight='bold')

# Find absolute maximums to keep axes completely static
max_M_global = np.max([np.max(d['M_array']) for d in successful_lambda_sweeps])
max_s0_global = np.max([np.max(d['sigma0_array']) for d in successful_lambda_sweeps])

# Setup Axis 1 (Mass)
ax1.set_xlim(0, max_s0_global * 1.05)
ax1.set_ylim(0, max_M_global * 1.1)
ax1.set_ylabel("Total Mass $\mathcal{M}$")
ax1.set_xlabel("Central Field $\sigma_c$")
ax1.set_title("Total Mass vs Central Field")
ax1.grid(True, alpha=0.5)

# Setup Axis 2 (Omega)
ax2.set_xlim(0, max_s0_global * 1.05)
ax2.set_ylim(0.4, 1.05) # Omega usually ranges from ~0.5 to 1.0
ax2.set_ylabel("Eigenfrequency $\Omega$")
ax2.set_xlabel("Central Field $\sigma_c$")
ax2.set_title("Frequency vs Central Field")
ax2.grid(True, alpha=0.5)

fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.15, wspace=0.2)

# Initialize the empty moving lines (removed labels here so they don't clutter the legend)
line1, = ax1.plot([], [], lw=2.5)
line2, = ax2.plot([], [], lw=2.5)

drawn = set()

def update(frame):
    data = successful_lambda_sweeps[frame]

    line1.set_data(data['sigma0_array'], data['M_array'])
    line1.set_color(data['color'])

    line2.set_data(data['sigma0_array'], data['Omega_array'])
    line2.set_color(data['color'])

    if frame not in drawn:
        ax1.plot(data['sigma0_array'], data['M_array'],
            color=data['color'], alpha=0.6, lw=1.5, linestyle='--', label=fr"$\Lambda = {data['Lambda']:.0f}$")

        ax2.plot(data['sigma0_array'], data['Omega_array'],
            color=data['color'], alpha=0.6, lw=1.5, linestyle='--')

        drawn.add(frame)

    ax1.legend(loc='upper right')

    return line1, line2

ani = FuncAnimation(
    fig, 
    update, 
    frames=len(successful_lambda_sweeps), 
    blit=False, 
    interval=1500 # 1.5 seconds per frame
)

# Bind animation to figure to prevent garbage collection deletion in some IDEs
fig.ani = ani 
fig.ani.save(gif_path, writer='pillow', fps=1)

print(f"Animation successfully saved as: {gif_filename}")
plt.tight_layout()
plt.show()

#%% 4. Final Accuracy Report Printout
print("\n*** Final Accuracy Report ***")
for d in successful_lambda_sweeps:
    if d['errors']:
        print(f"\nValues > accu for Lambda = {d['Lambda']}:")
        print(tabulate(d['errors'], headers=["Index", "σ0 Value", "σ(inf) Error"], tablefmt='fancygrid'))