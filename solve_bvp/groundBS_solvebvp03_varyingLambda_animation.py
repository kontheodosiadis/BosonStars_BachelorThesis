import numpy as np
from scipy.integrate import solve_bvp, simpson
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Matplotlib configuration for publication-quality styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# Fixed System Parameters
sigma_c = 0.1  
eps = 1e-5
x_max = 40.0
n = 1000
x_mesh = np.linspace(eps, x_max, n)

# Denser parameter list for a fluid animation sweep
points = 151
Lambda_list = np.linspace(0.0, 300.0, points)

# Specific benchmark profiles to preserve as persistent background lines
save_Lambda = [0.0, 10.0, 20.0, 50.0, 150.0, 300.0]
save_indices = [np.argmin(np.abs(Lambda_list - target)) for target in save_Lambda]

# Initial analytical guess configuration for the base case (Lambda = 0)
y_guess = np.zeros((4, x_mesh.size))
y_guess[0] = np.ones_like(x_mesh)
y_guess[1] = np.ones_like(x_mesh)
width_guess = max(8.0, 1.0 / sigma_c) 
y_guess[2] = sigma_c * np.exp(-(x_mesh / width_guess)**2) 
y_guess[3] = -2 * sigma_c * (x_mesh / width_guess**2) * np.exp(-(x_mesh / width_guess)**2)

Omega_guess_val = np.sqrt(1.0 - sigma_c**2)
Omega_guess = [Omega_guess_val]

successful_solutions = []
persistent_data = []

# Einstein-Klein-Gordon System Function
def field_equations(x, y, p, current_Lambda):
    A, B, sigma, v = y[0], y[1], y[2], y[3]
    Omega = p[0]
    
    common_term = (v**2 / A) + (Omega**2 / B) * sigma**2
    Lambda_term = current_Lambda * sigma**4 / 2
    
    dAdx = x * A**2 * (common_term + sigma**2 + Lambda_term) - (A - 1) * A / x
    dBdx = x * A * B * (common_term - sigma**2 - Lambda_term) + (A - 1) * B / x
    dsigmadx = v
    dvdx = -(2 / x + 0.5 * (dBdx / B - dAdx / A)) * v - A * ((-1 + Omega**2 / B) * sigma - current_Lambda * sigma**3)
    
    return np.vstack((dAdx, dBdx, dsigmadx, dvdx))

print("--- Sweeping Parametric Space ---")
for i, Lambda_val in enumerate(Lambda_list):
    def boundary_conditions(ya, yb, p):
        Omega = p[0]
        decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
        return np.array([
            ya[0] - 1.0,                 
            ya[2] - sigma_c,             
            ya[3] - 0.0,                 
            yb[1] - 1.0 / yb[0],                 
            yb[3] + decay_rate * yb[2]   
        ])
    
    # Solve BVP
    res = solve_bvp(lambda x, y, p: field_equations(x, y, p, Lambda_val), 
                    boundary_conditions, x_mesh, y_guess, p=Omega_guess, tol=1e-6, max_nodes=50000)
    
    if res.success:
        Omega_sol = res.p[0]
        x_sol = res.x
        A_sol = res.y[0]
        sigma_sol = res.y[2]
        
        # Verify node structure for the ground state
        nodes = np.sum((sigma_sol[:-1] * sigma_sol[1:]) < 0)
        if nodes == 0:
            M_list = (x_sol / 2) * (1 - 1 / A_sol)
            M_total = M_list[-1]
            
            frame_data = {
                'x': x_sol,
                'sigma': sigma_sol,
                'M_list': M_list,
                'Lambda': Lambda_val,
                'M': M_total
            }
            
            if i in save_indices:
                persistent_data.append(frame_data)
                print(f"[Stored Reference] \u039b = {Lambda_val:5.1f} | Asymptotic Mass = {M_total:.4f}")
            
            successful_solutions.append(frame_data)
            
            # Step continuation criteria forward (homotopy)
            x_mesh = res.x
            y_guess = res.y
            Omega_guess = res.p
            
            # Print periodic progress to the console
            if i % 10 == 0:
                print(f"Success: \u039b = {Lambda_val:5.1f} | Mass = {M_total:.4f} | Omega = {Omega_sol:.4f}")
    else:
        # If it fails, print the message so it doesn't fail silently
        print(f"Solver failed at \u039b = {Lambda_val:.1f}: {res.message}")

print("\n--- Compiling Animation Frames ---")

# Failsafe: Check if we actually have frames to animate
if len(successful_solutions) == 0:
    print("Error: The solver did not find any successful solutions. Cannot render animation.")
else:
    save_folder = r"C:\Users\hp\Desktop\1.University\A1.Semesters\8.Semester\Thesis\1.Codes\1.2.Development_Lab\Ground_State\Animations"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    gif_path = os.path.join(save_folder, 'Varying_Lambda.gif')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    
    # Generate gradient reference spectrum for static backgrounds
    color_map = plt.cm.viridis(np.linspace(0.2, 0.85, len(persistent_data)))
    for idx, data in enumerate(persistent_data):
        ax1.plot(data['x'], data['sigma'], color=color_map[idx], linestyle='--', alpha=0.5, lw=2,
                 label=f'$\Lambda={data["Lambda"]:.1f}$')
        ax2.plot(data['x'], data['M_list'], color=color_map[idx], linestyle='--', alpha=0.5, lw=2)
    
    # Initialize responsive focal lines
    line1, = ax1.plot([], [], lw=2.5, color='black')  
    line2, = ax2.plot([], [], lw=2.5, color='black')  
    
    # Configuration for Scalar Field axis
    ax1.set_xlim(0, 30) 
    ax1.set_ylim(0, sigma_c * 1.15)
    ax1.set_ylabel("Scalar Field $\sigma$")
    ax1.set_xlabel("Radius $x$")
    ax1.set_title("Scalar Field Profiles")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Configuration for Internal Mass axis
    max_M = np.max([np.max(d['M_list']) for d in successful_solutions])
    ax2.set_xlim(0, 30) 
    ax2.set_ylim(0, max_M * 1.05)
    ax2.set_ylabel("Internal Mass $\mathcal{M}$")
    ax2.set_xlabel("Radius $x$")
    ax2.set_title("Mass Profiles")
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, wspace=0.35)
    
    def animate(frame):
        data = successful_solutions[frame]
        line1.set_data(data['x'], data['sigma'])
        line1.set_label(f'Current\n$\Lambda={data['Lambda']:.1f}$')
        ax1.legend(loc='upper right')
        
        line2.set_data(data['x'], data['M_list'])
        #fig.suptitle(f"Boson Star Profile Adaptation: $\Lambda = {data['Lambda']:.1f}$  (Fixed $\sigma_c = {sigma_c}$)", fontsize=15)
        return line1, line2
    
    # Bind animation to fig.ani to prevent garbage collection deletion
    fig.ani = FuncAnimation(fig, animate, frames=len(successful_solutions), blit=False, interval=60)
    fig.ani.save(gif_path, writer='pillow', fps=13)
    
    print(f"File compilation complete. Output written to:\n{gif_path}")
    plt.show()