#%% Modules
import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
import matplotlib.pyplot as plt

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


#%% Python BVP Solver Methodology

#%%% Parameters
sigma_c = 0.1  # Fixed central scalar field value (adjust as needed)
Lambda_list = np.array([0.0, 10.0, 20.0, 50.0, 150.0, 300.0]) # Varying self-interaction parameter
eps = 1e-5

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 6.5), sharex=True)
fig2, ax = plt.subplots()

colors = plt.cm.tab10(np.linspace(0, 1, len(Lambda_list)))

for i in range(len(Lambda_list)):
    Lambda = Lambda_list[i]
    
    #%%% #1: Define the ODE system function
    def fun(x, y, p): # Number of unknowns: 5 (4 y + 1 p)
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
    
    #%%% #2: Define the boundary conditions (as residues)
    def bc(ya, yb, p):
        Omega = p[0]
        decay_rate = np.sqrt(np.maximum(1e-10, 1.0 - Omega**2))
        
        resA = ya[0]-1.0                    # A(0) = 1 asymptotic flatness
        resB = yb[1]-1.0/yb[0]              # B(∞) = 1 asymptotic flatness
        ressigma0 = ya[2]-sigma_c           # σ(0) = σ_c
        resv = ya[3] - 0.0                  # v(0) = σ'(0) = 0
        
        # Robin boundary condition v(xmax) = σ'(xmax) = -sqrt(1-Omega^2) * σ(xmax) 
        # Describes the asymptotic decay of the field at xmax
        resdecay = yb[3] + decay_rate * yb[2]
        return np.array([resA, resB, ressigma0, resv, resdecay]) # 5 BCs for 5 unknowns
    
    #%%% #3: Provide an initial mesh & guess
    x_max = 40
    n = 1000
    x_mesh = np.linspace(eps, x_max, n)
    
    # Eigenfunction guesses
    y_guess = np.zeros((4, x_mesh.size))
    y_guess[0] = np.ones_like(x_mesh)
    y_guess[1] = np.ones_like(x_mesh)
    width_guess = max(8.0, 1.0 / sigma_c) 
    y_guess[2] = sigma_c * np.exp(-(x_mesh / width_guess)**2) 
    y_guess[3] = -2 * sigma_c * (x_mesh / width_guess**2) * np.exp(-(x_mesh / width_guess)**2)
    
    # Eigenvalue guess
    Omega_guess_val = np.sqrt(1.0 - sigma_c**2)
    Omega_guess = [Omega_guess_val] 
        
    #%%% #4: Solve & Evaluate
    print(f"Running solver for Λ = {Lambda}...")
    print(f"Using x_max = {x_max:.1f} and Omega guess = {Omega_guess_val:.4f}")
    
    sol = solve_bvp(fun, bc, x_mesh, y_guess, p=Omega_guess, max_nodes=50000, tol=1e-7)
    
    print("Status: ", sol.status)
    if sol.status != 0:
        print(sol.message)   
            
    if sol.success: # only if the solver converges
        # Eigenvalue    
        Omega = sol.p[0]
        
        # Eigenfunctions
        x = sol.x
        A = sol.y[0]
        B = sol.y[1]
        sigma = sol.y[2]
        v = sol.y[3]
        
        # Mass
        M = (x/2) * (1 - 1/A)
        
        # Number of particles
        N_integrand = (x**2) * np.sqrt(A / B) * Omega * (sigma**2)
        N = simpson(N_integrand, x=x)
        
        #%%%% #4.1: Ground state check with number of nodes (Bolzano Theorem)
        nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
        
        print(f"Solver converged! Omega: {Omega:.6f} | Nodes: {nodes} | Mass: {M[-1]:.3f} | Particles: {N:.3f}") 
        
        #%%%% #4.2: Validity Test
        print(f"σ(inf) = {sigma[-1]:.6e}")
        print(f"A(inf) = {A[-1]:.6e}")
        print(f"B(eps) = {B[0]:.3e}")
        print(f"B(inf) = {B[-1]:.3e}")
        print("-" * 40)
        
        #%%% #5: Plots 
        #%%%% #5.1: Plotting the scalar field solution on the first axis
        ax1.plot(x, sigma, label=f'$\Lambda={Lambda}$', color=colors[i], lw=2.5)
    
        #%%%% #5.2: Plotting the mass solution on the second axis
        ax2.plot(x, M, label=f'M={M[-1]:.3f}', color=colors[i], lw = 2.5)
        
        #%%%% #5.3: Plotting the scalar field slopes
        ax.plot(x, v, label=f'$\Lambda={Lambda}$', color = colors[i], lw=2.5)


ax1.set_ylabel("Scalar Field $\sigma$")
ax1.set_xlabel("Radius $x$")
ax1.set_title("Scalar Field Profiles")
ax1.set_xlim(0, 30)
ax1.legend()
ax1.grid(True)
        
ax2.set_xlim(0, 30)
ax2.set_ylim(0)
ax2.set_ylabel("Internal Mass $\mathcal{M}$")
ax2.set_xlabel("Radius $x$")
ax2.set_title("Mass Profiles")
#ax2.legend()
ax2.grid(True)
        
# Adjust layout to prevent overlapping labels
#fig1.suptitle(f'Boson Star Profiles: Scalar Fields & Masses $(\sigma_c={sigma_c})$', fontsize=16, fontweight='bold')
fig1.subplots_adjust(wspace=0.38)

ax.set_ylabel("Scalar Field Slope $\sigma'$")
ax.set_xlabel("Radius $x$")
#ax.set_title("Scalar Field Slope Profile")
ax.set_xlim(0, 40)
ax.legend()
ax.grid(True)

#fig2.suptitle(f'Scalar Field Slope Profile $(\sigma_c={sigma_c})$', fontsize=16, fontweight='bold')
plt.show()