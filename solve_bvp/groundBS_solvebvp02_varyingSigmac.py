#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.integrate import simpson

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

#%%%% Storage Arrays for Visualization
save_sigma0 = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5]
save_indices = [np.argmin(np.abs(sigma0_list - target)) for target in save_sigma0]
sigma0inf_off = []

#%%%% Initializing the plot
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
# fig2, ax = plt.subplots()

colors = plt.cm.tab10(np.linspace(0,1,len(sigma0_list)))

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
#print("***Progress Report***")
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
    
    if res.success:
        # Unpacking the solution
        # Eigenvalue    
        Omega = res.p[0]
        
        # Eigenfunctions
        x = res.x
        A = res.y[0]
        B = res.y[1]
        sigma = res.y[2]
        v = res.y[3]
        
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
        
        if nodes == 0:
            consecutive_fails = 0 
            
            err_sigma = abs(res.y[2, -1])
            if err_sigma > accu:
                sigma0inf_off.append([i, sigma_0, f"{err_sigma:.2e}"])
                          
            # Mass
            M = (x/2) * (1 - 1/A)
            M_total = M[-1]
            
            # Number of particles
            N_integrand = (x**2) * np.sqrt(A / B) * Omega * (sigma**2)
            N = simpson(N_integrand, x=x)
            
            # Save over interval (not clatter the plot) 
            if i in save_indices:
                print(f"\nσ0 : {sigma0_list[i]:.2f} | Omega: {Omega:.6f} | Nodes: {nodes} | Mass: {M[-1]:.3f} | Particles: {N:.3f}") 
                
                #%%%% #4.2: Validity Test
            
                print(f"σ(inf) = {sigma[-1]:.6e}")
                print(f"A(inf) = {A[-1]:.6e}")
                print(f"B(eps) = {B[0]:.3e}")
                print(f"B(inf) = {B[-1]:.3e}")
                
                #%%% #5: Plots 
                #%%%% #5.1: Plotting the scalar field solution on the first axis
                ax1.plot(x, sigma, label=f'$σ_c={sigma_0:.2f}$', color=colors[i], lw=2.5)
            
                #%%%% #5.2: Plotting the mass solution on the second axis
                ax2.plot(x, M, label=f'M={M_total:.3f}', color=colors[i], lw = 2.5)
                
                #%%%% #5.3: Plotting the scalar field slopes
                # ax.plot(x, v, label=f'$\sigma_c={sigma_0:.2f}$', color = colors[i], lw=2.5)
                
            # Update guesses
            x_guess = res.x
            y_guess = res.y
            Omega_guess = res.p
            
        #     # Progress report
        #     if i % (points//10) == 0:
        #         print(f"Step {i+1:3d}/{points} | sigma_0: {sigma_0:.3f} | Mass M: {M_total:.4f} | Omega: {res.p[0]:.4f}")
        # else:
        #     print(f"\nExcited state at index {i}, sigma_0 = {sigma_0:.4f}")
        #     consecutive_fails += 1
        #     if consecutive_fails >= max_fails:
        #         print("Too many consecutive failures. Halting sweep.")
        #         break
    else:
        print(f"\nSolver broke at index {i}, sigma_0 = {sigma_0:.4f}")
        print(res.message)
        consecutive_fails += 1
        if consecutive_fails >= max_fails:
            print("Too many consecutive failures. Halting sweep.")
            break

ax1.set_ylabel("Scalar Field $\sigma$")
ax1.set_xlabel("Radius x")
ax1.set_title("Scalar Field Profiles")
ax1.set_xlim(0, 20)
ax1.legend()
ax1.grid(True)
        
ax2.set_xlim(0, 50)
ax2.set_ylim(0)
ax2.set_ylabel("Internal Mass $\mathcal{M}$")
ax2.set_xlabel("Radius x")
ax2.set_title("Mass Profiles")
#ax2.legend()
ax2.grid(True)
        
# Adjust layout to prevent overlapping labels
#fig1.suptitle(f'Boson Star Profiles: Scalar Fields & Masses $(\Lambda={Lambda})$', fontsize=16, fontweight='bold')
fig1.subplots_adjust(wspace=0.38)

# ax.set_ylabel("Scalar Field Slope $\sigma'$")
# ax.set_xlabel("Radius $x$")
# #ax.set_title("Scalar Field Slope Profile")
# ax.set_xlim(0, 40)
# ax.legend()
# ax.grid(True)

# fig2.suptitle(f'Scalar Field Slope Profile $(\Lambda={Lambda})$', fontsize=16, fontweight='bold')
plt.show()