#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy import stats
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
    "legend.fontsize": 12,
    "figure.titlesize": 22
})

#%% Boson Star Mass-Curve Solver

#%%% Initialization

#%%%% System Parameters

#%%%%% List of Λ Values
Lambda_start  = 0.0
Lambda_end    = 300.0
Lambda_points = 21
Lambda_list = np.linspace(Lambda_start, Lambda_end, Lambda_points)
# Lambda_list = [0.0, 1.0, 10.0, 30.0, 100.0, 200.0, 300.0]

#%%%%% List of σ0 Values
sigma0_start = 0.005
sigma0_end   = 0.6
sigma_points = 100
# !!! For better sigma plot more points are necessary 
# Better loacate the max → less numerical instability
sigma0_list = np.linspace(sigma0_start, sigma0_end, sigma_points)

eps = 1e-5
accu = 1e-2

consecutive_fails = 0
max_fails = 5

# Storage Array
Mmax_list = []
sigma0_where_Mmax = []

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
                print(f"Excited state at index {i}, sigma_0 = {sigma0:.4f}")
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
    
    #%%%% #7: Data (maximum mass) for plot
    M_list = np.array(M_list)
    is_Mmax = M_list.argmax()
    Mmax_list.append(M_list[is_Mmax])
    sigma0_where_Mmax.append(valid_sigma0[is_Mmax])
    print(f"Lambda: {Lambda:.3f} | Max Mass: {max(M_list):.3f}")
    
    #%%%% #8: Accuracy Report
    headers = ["Index", "σ0 Value", "σ(inf) Error"]
    print("\n--- Accuracy Report (Values > accu) ---")
    if sigma0inf_off:
        print(tabulate(sigma0inf_off, headers=headers, tablefmt='fancygrid'))
    else:
        print("All final values within accuracy limits at infinity!")
    
#%%%% #9: Plotting the results
fig, ax1 = plt.subplots(figsize=(7, 6))

# Evolution of Critical Mass over Λ
ax1.scatter(Lambda_list, Mmax_list, color = '#36454F', label = 'Generated Data')
ax1.set_ylabel("Maximum Total Mass $\mathcal{M}_{max}$")
ax1.set_xlabel("Self-Interaction Strength $\Lambda$")

# Performing Linear Regression
# Transforming the lists into np.arrays for better handling
Lambda_list = np.array(Lambda_list)
Mmax_list = np.array(Mmax_list)

# Performing a linear regression and printing the results
regression = stats.linregress(Lambda_list, Mmax_list**2)

print(f"Equation: M_crit = sqrt({regression.slope:.4f} * $\Lambda$ + {regression.intercept:.4f})")
print(f"R-squared: {regression.rvalue**2:.6f}")

# Plotting the result
Lambda_plot = np.linspace(Lambda_start, Lambda_end, 1001)
Mmax_regression = np.sqrt(regression.slope * Lambda_plot + regression.intercept)
ax1.plot(Lambda_plot, Mmax_regression, color = '#36454F', label=rf'Regression $R^2 = {regression.rvalue**2:.6f}$' + '\n' + 
               rf'$\mathcal{{M}}_{{max}} = \sqrt{{{regression.slope:.3f}\Lambda+{regression.intercept:.3f}}}$', lw=2.5)

# Theoretical for Λ>>1 from Shapiro
Mmax_theoretical = 0.22 * np.sqrt(Lambda_plot)
ax1.plot(Lambda_plot, Mmax_theoretical, color = '#8C9DA8', label = 'Theoretical for $\Lambda \gg 1$\n $\mathcal{{M}}_{{max}}=0.22\sqrt{\Lambda}$', linestyle = "--", linewidth = 2.5)
ax1.legend()
ax1.grid(True)

# Insert Plot: Error between Regression & Shapiro Theoretical
ax_inset = ax1.inset_axes([0.45, 0.1, 0.4, 0.3])
threshold = 200
mask = Lambda_plot >= threshold

error_val = abs(Mmax_regression[mask]/Mmax_theoretical[mask] - 1) * 100

ax_inset.plot(Lambda_plot[mask], error_val, color='gray', linestyle='-')

ax_inset.set_xlim(threshold)
ax_inset.set_ylim(0)

ax_inset.set_title(f"Percent Error: Regression Vs Theory ($\Lambda > {threshold}$)", fontsize=12)
ax_inset.set_ylabel("% Error", fontsize=11)
ax_inset.set_xlabel("$\Lambda$", fontsize=11)
ax_inset.tick_params(labelsize=11)
ax_inset.grid(True, alpha=0.3)

#fig.suptitle("Maximum Mass-Self-Interaction Relationship", fontsize=16, fontweight='bold')

# ax1.indicate_inset_zoom(ax_inset, edgecolor="black")
plt.tight_layout() 
plt.show()