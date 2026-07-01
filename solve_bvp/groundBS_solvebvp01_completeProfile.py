#%% Modules
import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Appearence formatting for Thesis

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
Lambda = 0.0
sigma0 = 0.1
eps = 1e-5

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
    resB = yb[1]-1.0/yb[0]              # B(∞) = 1 asymptotic flatness → B(xmax)=1/A(xmax) more accurate
    ressigma0 = ya[2]-sigma0            # σ(0) = σ0
    resv = ya[3] - 0.0                  # v(0) = σ'(0) = 0
    
    # Dirichlet boundary condition σ(∞) = 0
    # To work well it requires xmax = ∞ 
    # ressigmainf = yb[2] - 0.0
    
    # Robin boundary condition v(xmax) = σ'(xmax) = -sqrt(1-Omega^2) * σ(xmax)
    # Describes the asymptotic decay of the field at xmax
    resdecay = yb[3] + decay_rate * yb[2]
    return np.array([resA, resB, ressigma0, resv, resdecay]) # 5 BCs for 5 unknowns

#%%% #3: Provide an initial mesh & guess (taking advantage from the physics of a boson star)
x_max = 65
n = 1000
x_mesh = np.linspace(eps, x_max, n)

# Eigenfunction guesses
y_guess = np.zeros((4, x_mesh.size))
y_guess[0] = np.ones_like(x_mesh)
y_guess[1] = np.ones_like(x_mesh)
width_guess = max(8.0, 1.0 / sigma0) # σ0↑ the decay is quicker (narrower gaussian)
y_guess[2] = sigma0 * np.exp(-(x_mesh / width_guess)**2) # gaussian for asymptotic decay to 0
y_guess[3] = -2 * (x_mesh / width_guess**2) * y_guess[2]
# Eigenvalue guess
Omega_guess_val = np.sqrt(1.0 - sigma0**2)
Omega_guess = [Omega_guess_val] # σ0 → 0: Ω → 1

    
#%%% #4: Solve & Evaluate
print(f"Running solver for σ0 = {sigma0}...")
print(f"Using x_max = {x_max:.1f} and Omega guess = {Omega_guess_val:.4f}")

sol = solve_bvp(fun, bc, x_mesh, y_guess, p=Omega_guess, max_nodes=50000, tol=1e-7)

print("Status: ", sol.status)
if sol.status != 0:
    print(sol.message)   
        
if sol.success: # only is the solver converges
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
    
    #%%%% #4.1: Ground state check with number of nodes (Bolzano Thorem)
    nodes = np.sum((sigma[:-1] * sigma[1:]) < 0)
    
    print(f"Solver converged! Omega: {Omega:.6f} | Nodes: {nodes} | Mass: {M[-1]:.3f} | Particles: {N:.3f}") 
    
    #%%%% #4.2: Validity Test

    print(f"σ(inf) = {sigma[-1]:.6e}")
    print(f"A(inf) = {A[-1]:.6e}")
    print(f"B(eps) = {B[0]:.3e}")
    print(f"B(inf) = {B[-1]:.3e}")
    
    #%%%%% Checking 10 last values of scalar field & Mass
    print("\nPosition 10 last values")
    print(x[-10:])
    
    
    print("\nScalar Field 10 last values:")
    print(sigma[-10:])
    print("% Difference from absolute last value:")
    print(100 * (sigma[-10:] - sigma[-1]) / sigma[-1])
    
    print("\nMass 10 last values:")
    print(M[-10:])
    print("% Difference from absolute last value:")
    print(100 * (M[-10:] - M[-1]) / M[-1])
    
    #%%% #5: Plotting
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
        
    #5.1: Metric Components A & B vs radius x (Top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, A, color='#56B4E9', lw = 2.5, label=r'A(x)')
    ax1.plot(x, B, color='#D55E00', lw = 2.5, label=f'B(x) | B(0)={B[0]:.3f}')
    ax1.set_xlim(0)
    ax1.set_xlabel("Radius x")
    ax1.set_ylabel("Metric Functions")
    ax1.set_title("Metric Profile")
    ax1.legend()
    ax1.grid(True)
        
    #5.2: Scalar field profile (Bottom Left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, sigma, color='#003f5C', lw = 2.5, label=f'$σ_c={sigma0}$')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_xlim(0)
    ax2.set_xlabel("Radius x")
    ax2.set_ylabel("Scalar Field $\sigma$")
    ax2.set_title("Scalar Field Profile")
    ax2.legend(fontsize=17)
    ax2.grid(True)
        
    #5.3: Mass profile (Bottom Right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, M, label=rf"Mass $\mathcal{{M}}={M[-1]:.3f}$", color='#222222', lw=2.5)
    ax3.set_xlim(0)
    ax3.set_ylim(0)
    ax3.set_ylabel("Internal Mass $\mathcal{M}$")
    ax3.set_xlabel("Radius x")
    ax3.set_title("Mass Profile")
    ax3.legend()
    ax3.grid(True)
        
    #fig.suptitle(f'Boson Star Profile: Metric, Scalar Field & Mass ($\Lambda={Lambda}$)', fontweight='bold')
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.show()