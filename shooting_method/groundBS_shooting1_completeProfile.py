#%% Modules
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Shooting Method Methodology

#%%% System Parameters
Lambda = 0.0
sigma_0 = 0.25
# Dimensionless radius interval for solver
x_min = 1e-5    # Start of the solver
x_max = 2000.0  # Arbitarily high; the solver will cut them on its own

#%%% #1: Define the ODE System
def fun(x, y, omega):
    A, B, sigma, v = y
    
    omega2_B = (omega**2) / B
    Lambda_term = (Lambda / 2) * sigma**4
    v2_A = (v**2) / A
    
    A_prime = (A * (1 - A) / x) + x * (A**2) * ((omega2_B + 1) * sigma**2 + Lambda_term + v2_A)
    B_prime = (B * (A - 1) / x) + x * A * B * ((omega2_B - 1) * sigma**2 - Lambda_term + v2_A)
    
    friction_term = (2 / x) + (B_prime / (2 * B)) - (A_prime / (2 * A))
    force_term = A * ((omega2_B - 1) * sigma - Lambda * sigma**3)
    
    v_prime = -friction_term * v - force_term
    
    return [A_prime, B_prime, v, v_prime]

#%%% #2: Define Stopping Events
def event_node(x, y, omega):
    # Triggers when sigma (y[2]) crosses 0 → node = excited state
    return y[2]
event_node.terminal = True
event_node.direction = -1 # Only catch going from positive to negative

def event_diverge(x, y, omega):
    # Triggers when v (y[3]) crosses 0 from negative to positive → scalar field rises
    # This means the field bottomed out and is growing again.
    return y[3]
event_diverge.terminal = True
event_diverge.direction = 1 # Only catch going from negative to positive

#%%% #3: Shooting method with Bisection

#%%%% #3.1: Bracketing the omega (ω) for the bisection

def omega_state(omega_guess, sigma_val):
    y0 = [1.0, 1.0, sigma_val, 0.0]
    sol = solve_ivp(fun, [x_min, x_max], y0, args=(omega_guess,),
        events=[event_node, event_diverge],
        rtol=1e-5, atol=1e-7, # Looser tolerances for fast scanning
        method='RK45')
    
    if len(sol.t_events[0]) > 0:
        return "HIGH" # Triggered event_node
    elif len(sol.t_events[1]) > 0:
        return "LOW"  # Triggered event_diverge
    else:
        return "PERFECT" # Extremely rare to hit this exactly

def omega_bracket(sigma_val, start_omega=0, step=0.1, omega_limit=10.0):
    print(f"\nScanning for valid omega bracket for σ0 = {sigma_val}...")
    current_omega = start_omega
    current_state = omega_state(current_omega, sigma_val)
    
    # Starting from ω=0, I make sure that I start from LOW
    # Assuming standard behavior where a low omega diverges:
    while True:
        next_omega = current_omega + step
        next_state = omega_state(next_omega, sigma_val)
        
        if next_state != current_state:
            print(f"Bracket found: [{current_omega:.4f}, {next_omega:.4f}]")
            return current_omega, next_omega
        
        current_omega = next_omega
        
        # Safety net to prevent infinite loops
        if current_omega > omega_limit:
            raise ValueError("Could not find a bracket. Check equations or increase max limit.")

#%%%% #3.2: Bisection method
omega_low, omega_high = omega_bracket(sigma_0)
tolerance = 1e-13

print(f"\nStarting Shooting Method for σ0 = {sigma_0}...")

# Bisection Method
total_loops = 51

for iteration in range(total_loops): # 100 iterations yields incredible precision
    # Omega guess in the middle   
    omega_guess = (omega_low + omega_high) / 2.0
    
    # Initial conditions: [A(0)=1✓, B(0), sigma(0)=σ0✓, v(0)=0✓]
    # We temporarily set B(0) = 1.0 (for the shooting)
    y0 = [1.0, 1.0, sigma_0, 0.0]
    
    sol = solve_ivp(fun, [x_min, x_max], y0, args=(omega_guess,),
        events=[event_node, event_diverge],
        rtol=1e-8, atol=1e-10, # Tight internal tolerances
        dense_output=True,
        method='RK45')
    
    if len(sol.t_events[0]) > 0:
        # Event 0: Crossed zero (Node). The pull was TOO STRONG.
        if iteration % (total_loops//10) == 0:
            print(f"Iteration {iteration:3d}/{total_loops-1}: omega (ω): {omega_guess:.6f} | Too HIGH")
        # Lowering the high value of omega      
        omega_high = omega_guess

    elif len(sol.t_events[1]) > 0:
        # Event 1: Diverged (Well). The pull was TOO WEAK.
        if iteration % (total_loops//10) == 0:
            print(f"Iteration {iteration:3d}/{total_loops-1}: omega (ω): {omega_guess:.6f} | Too LOW")
        # Increasing the low value of omega
        omega_low = omega_guess
        
    else:
        print("Perfect fit found!")
        break

#%%% #4: Final Integration and Rescaling
omega_final = (omega_low + omega_high) / 2.0
print("Bisection complete. Extracting final profile...")
    
# Define a gentle event to capture the tail without hitting numerical noise
def event_tail(x, y, omega):
    # Triggers when the scalar field drops to a negligible value (1e-7)
    return y[2] - 1e-7
event_tail.terminal = True
event_tail.direction = -1
    
# Run one last time using ONLY the tail event
sol_final = solve_ivp(fun, [x_min, x_max], [1.0, 1.0, sigma_0, 0.0], args=(omega_final,),
    events=[event_tail], # Replaced the strict bisection events
    rtol=1e-10, atol=1e-12 # Slightly tighter tolerances for the final run
    )
    
x = sol_final.t
A = sol_final.y[0]
B_temp = sol_final.y[1]
sigma = sol_final.y[2]
    
# The Rescaling Trick
B_inf = B_temp[-1]
B_true = B_temp / B_inf
Omega_true = omega_final / np.sqrt(B_inf)
    
print(f"True Eigenvalue Omega (Ω): {Omega_true:.6f}")
print(f"σ(inf) = {sigma[-1]:.6e}")
print(f"A(inf) = {A[-1]:.6e}")
print(f"B(eps) = {B_true[0]:.3f}")
print(f"B(inf) = {B_true[-1]:.3e}")
    
M = (x/2) * (1 - 1/A)
print(f"Approximation to total mass M={M[-1]:.6f}")
    
#%%% #5: Plotting
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    
# --- Top Row: A and B together ---
# gs[0, :] tells this subplot to take all columns in row 0
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x, A, color='cornflowerblue', label='A(x)')
ax1.plot(x, B_true, color='indianred', label=f'B(x) | B(0)={B_true[0]:.3f}')
ax1.set_xlim(0)
ax1.set_ylabel("Metric Functions")
ax1.set_title("Metric Profile", fontsize=14)
ax1.legend()
ax1.grid(True)
    
# --- Bottom Left: Scalar Field ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x, sigma, color='forestgreen', label=f'$σ_0={sigma_0}$')
ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax2.set_xlim(0)
ax2.set_xlabel("Normalized Distance x")
ax2.set_ylabel("Scalar Field σ")
ax2.set_title("Scalar Field Profile", fontsize=14)
ax2.legend()
ax2.grid(True)
    
# --- Bottom Right: Internal Mass ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x, M, label=f'M={M[-1]:.3f}', color='forestgreen')
ax3.set_xlim(0)
ax3.set_ylim(0)
ax3.set_ylabel("Internal Mass $M/(M_{Pl}^2/m)$")
ax3.set_xlabel("Normalized Distance x")
ax3.set_title("Mass Profile", fontsize=14)
ax3.legend()
ax3.grid(True)
    
# Adjust layout and add a global title
fig.suptitle(f'Boson Star Analysis: Metric, Scalar Field & Mass ($\Lambda={Lambda}$)', 
             fontsize=16, fontweight='bold')
plt.tight_layout() # Adjust rect to keep title from overlapping
plt.show()