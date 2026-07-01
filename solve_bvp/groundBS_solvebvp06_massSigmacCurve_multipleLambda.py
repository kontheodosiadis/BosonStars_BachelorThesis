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
eps = 1e-5
accu = 1e-2

#%%% Initialization

#%%%% System Parameters

#%%%%% List of Λ Values
# Lambda_start  = 0.0
# Lambda_end    = 300.0
# Lambda_points = 10
# Lambda_list = np.linspace(Lambda_start, Lambda_end, Lambda_points)
Lambda_list = [0.0, 1.0, 10.0, 30.0, 100.0, 200.0, 300.0]
colors = plt.cm.tab10(np.linspace(0,1,len(Lambda_list)))

#%%%%% List of σ0 Values
sigma0_start = eps
sigma0_end   = 0.6
sigma_points = 601
sigma0_list = np.linspace(sigma0_start, sigma0_end, sigma_points)

consecutive_fails = 0
max_fails = 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#%%%%% Storage Arrays
Mmax_list = []
sigma0_Mmax_list = []

#%%% Looping over the Λ values
for j, Lambda in enumerate(Lambda_list):
    print(f"\n***Lamba = {Lambda:.0f} (Step {j+1}/{len(Lambda_list)})***")
    # Initial Setup for 1st guess
    x_min = eps
    x_max = 125.0
    
    # Generate the very first guess for sigma0_start
    width_guess = max(8.0, 1.0 / sigma0_start)
    Omega_guess_val = np.sqrt(1.0 - sigma0_start**2)
    
    x_guess = np.linspace(x_min, x_max, 1000)
    y_guess = np.zeros((4, x_guess.size))
    y_guess[0] = np.ones_like(x_guess)                                                            # A
    y_guess[1] = np.ones_like(x_guess)                                                            # B
    y_guess[2] = sigma0_start * np.exp(-(x_guess / width_guess)**2)                               # sigma
    y_guess[3] = -2 * sigma0_start * (x_guess / width_guess**2) * np.exp(-(x_guess / width_guess)**2) # phi
    Omega_guess = [Omega_guess_val]
    
    # Storage Arrays
    M_list = []
    Omega_list = []
    valid_sigma0 = []
    sigma0inf_off = []
    
    #%%%% #1: ODE System
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
    
    #%%% Looping over σ0 values 
    
    consecutive_fails = 0  # Reset every Λ value
    for i, sigma0 in enumerate(sigma0_list):
        
        #%%%% #2: Boundary Conditions (Dynamic for current sigma0) ---
        def bc(ya, yb, p):
            Omega = p[0]
            decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
            
            return np.array([
                ya[0] - 1.0,                 # A(0) = 1
                ya[2] - sigma0,              # sigma(0) = current sigma_0
                ya[3] - 0.0,                 # phi(0) = 0
                yb[1] - 1.0/yb[0],           # B(inf) = 1 (Asymptotic flatness)
                yb[3] + decay_rate * yb[2]   # Robin BC for scalar field decay at inf
            ])
            
        #%%%% #3: Solve BVP
        res = solve_bvp(fun, bc, x_guess, y_guess, p=Omega_guess, tol=1e-5, max_nodes=50000)
        
        x = res.x
        A = res.y[0]
        B = res.y[1]
        sigma = res.y[2]
        
        #%%%% #4: Validity Tests
        #%%%%% Test #1: Solver is successful
        if res.success:
            #%%%%% Test #2: 0 nodes solution
            sigma = res.y[2]
            nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
            
            if nodes==0:
                consecutive_fails = 0  # Reset on success
                
                #%%%% #5: Update guesses from solution for the NEXT iteration
                x_guess = res.x
                y_guess = res.y
                Omega_guess = res.p
                
                #%%%%% Test #3: Accuracy for σ(∞) → 0
                err_sigma = abs(res.y[2, -1])
                if err_sigma > accu:
                    sigma0inf_off.append([
                        i, 
                        sigma0, 
                        f"{err_sigma:.2e}"
                    ])
                else:
                    #%%%%% #6: Extracting the Data
                    M = (res.x[-1]/2) * (1 - 1/res.y[0, -1])
                    
                    M_list.append(M)
                    Omega_list.append(res.p[0])
                    valid_sigma0.append(sigma0)
                    # Print progress every points/5 steps
                    if i % (sigma_points//5) == 0:
                        print(f"Step {i+1:3d}/{sigma_points} | sigma_0: {sigma0:.3f} | Mass M: {M:.4f} | Omega: {res.p[0]:.4f}")
    
            else:
                print(f"\nExcited state at index {i}, sigma_0 = {sigma0:.4f}")
                consecutive_fails += 1
                
                if consecutive_fails >= max_fails:
                    print("Too many consecutive failures. Halting sweep.")
                    break
        else:
            print(f"\nSolver broke at index {i}, sigma_0 = {sigma0:.4f}")
            print(res.message)
            consecutive_fails += 1
            
            if consecutive_fails >= max_fails:
                print("Too many consecutive failures. Halting sweep.")
                break
    
    #%%%% Pinpointing the max
    max_idx = np.argmax(M_list)
    Mmax_list.append(M_list[max_idx])
    sigma0_Mmax_list.append(valid_sigma0[max_idx])
    
    #%%%% #7: Plotting the results
    
    # Plot Mass vs Central Density
    ax1.plot(valid_sigma0, M_list, color=colors[j], label=f"Λ={Lambda:.0f}", lw=2.5)
    
    # Plot Eigenfrequency vs Central Density
    ax2.plot(valid_sigma0, Omega_list, color=colors[j], label=f"Λ={Lambda:.0f}", lw=2.5)    
    
    #%%%% #8: Accuracy Report
    headers = ["Index", "σ0 Value", "σ(inf) Error"]
    print("\n--- Accuracy Report (Values > accu) ---")
    if sigma0inf_off:
        print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
    else:
        print("All final values within accuracy limits at infinity!")
    
print("\n Maximum Mass Locations:")
for i in range(len(Lambda_list)):
    print(f"Lambda: {Lambda_list[i]} | Mmax: {Mmax_list[i]:.3f} | σ0(Mmax): {sigma0_Mmax_list[i]:.3f}")


ax1.set_ylabel("Total Mass $\mathcal{M}$")
ax1.set_xlabel("Central Field $\sigma_c$")
ax1.set_title("Total Mass vs Central Field")
ax1.set_xlim(sigma0_start, sigma0_end)
ax1.set_ylim(0)
ax1.grid(True)

ax2.set_ylabel("Eigenfrequency $\Omega$")
ax2.set_xlabel("Central Field $\sigma_c$")
ax2.set_title("Frequency vs Central Field")
ax2.set_xlim(sigma0_start, sigma0_end)
ax2.grid(True)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=7, frameon=False)

#fig.suptitle("Mass & Eigenfrequency Relationship with Central Field", fontsize=16, fontweight='bold')

plt.tight_layout() 
plt.show()