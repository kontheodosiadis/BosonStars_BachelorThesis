#%% Modules
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Shooting Method Methodology
eps = 1e-05
#%%% System Parameters
Lambda_list = [0.0, 1.0, 10.0, 30.0, 100.0, 200.0, 300.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(Lambda_list))) # Viridis handles multiple curves beautifully

#%%%%% List of σ0 Values
sigma0_start = eps
sigma0_end   = 0.6
sigma_points = 200  # Reduced slightly for speed; change back to 601 for publication-grade curves
sigma0_list = np.linspace(sigma0_start, sigma0_end, sigma_points)
x_min = 1e-5    # Start of the solver
x_max = 2000.0  # Arbitrarily high

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#%%% Shooting Method for a range of σ0 values
for idx, Lambda in enumerate(Lambda_list):
    # FIX 1: Reset tracking lists for each distinct Lambda curve
    M_list = []
    Omega_list = []
    
    print(f"\n--- Processing Lambda = {Lambda} ---")
    
    for i, sigma_0 in enumerate(sigma0_list):
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
            return y[2]
        event_node.terminal = True
        event_node.direction = -1 
        
        def event_diverge(x, y, omega):
            return y[3]
        event_diverge.terminal = True
        event_diverge.direction = 1 
        
        #%%%% #3.1: Bracketing omega
        def omega_state(omega_guess, sigma_val):
            y0 = [1.0, 1.0, sigma_val, 0.0]
            sol = solve_ivp(fun, [x_min, x_max], y0, args=(omega_guess,),
                            events=[event_node, event_diverge],
                            rtol=1e-5, atol=1e-7, 
                            method='RK45')
            if len(sol.t_events[0]) > 0: return "HIGH" 
            elif len(sol.t_events[1]) > 0: return "LOW"  
            return "PERFECT"
        
        def omega_bracket(sigma_val, start_omega=0, step=0.1, omega_limit=10.0):
            current_omega = start_omega
            current_state = omega_state(current_omega, sigma_val)
            while True:
                next_omega = current_omega + step
                next_state = omega_state(next_omega, sigma_val)
                if next_state != current_state:
                    return current_omega, next_omega
                current_omega = next_omega
                if current_omega > omega_limit:
                    raise ValueError("Could not find a bracket.")
        
        #%%%% #3.2: Bisection method
        omega_low, omega_high = omega_bracket(sigma_0)
        
        # FIX 2: Optimized loop count (30 loops gives full Float64 precision limits)
        total_loops = 30 
        
        for iteration in range(total_loops): 
            omega_guess = (omega_low + omega_high) / 2.0
            y0 = [1.0, 1.0, sigma_0, 0.0]
            sol = solve_ivp(fun, [x_min, x_max], y0, args=(omega_guess,),
                            events=[event_node, event_diverge],
                            rtol=1e-7, atol=1e-9, 
                            method='RK45')
            
            if len(sol.t_events[0]) > 0:
                omega_high = omega_guess
            elif len(sol.t_events[1]) > 0:
                omega_low = omega_guess
            else:
                break
        
        #%%% #4: Final Integration and Rescaling
        omega_final = (omega_low + omega_high) / 2.0
            
        def event_tail(x, y, omega):
            return y[2] - 1e-7
        event_tail.terminal = True
        event_tail.direction = -1
            
        sol_final = solve_ivp(fun, [x_min, x_max], [1.0, 1.0, sigma_0, 0.0], args=(omega_final,),
                              events=[event_tail], rtol=1e-8, atol=1e-10)
        
        M = (sol_final.t[-1]/2) * (1 - 1/sol_final.y[0, -1])
        M_list.append(M)
        
        Omega_true = omega_final / np.sqrt(sol_final.y[0, -1] * sol_final.y[1, -1])
        Omega_list.append(Omega_true)
        
        if i % (sigma_points//5) == 0:
            print(f"  Step {i+1:3d}/{sigma_points} | sigma_0: {sigma_0:.3f} | Mass M: {M:.4f}")
       
    # FIX 3: Dynamic plotting using color map arrays and labeling for legends
    ax1.plot(sigma0_list, M_list, color=colors[idx], linewidth=2, label=f"$\Lambda = {Lambda}$")
    ax2.plot(sigma0_list, Omega_list, color=colors[idx], linewidth=2, label=f"$\Lambda = {Lambda}$")

# Formatting the M vs σ0 Diagram
ax1.set_ylabel("Total Mass $M/(M_{Pl}^2/m)$", fontsize=12)
ax1.set_xlabel("Central Field $\sigma_0$", fontsize=12)
ax1.set_xlim(sigma0_start, sigma0_end)
ax1.set_ylim(0, None)
ax1.set_title("Boson Star Mass Curve", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend() # Show the labels

# Formatting the Ω vs σ0 Diagram
ax2.set_ylabel("Eigenfrequency $\Omega$", fontsize=12)
ax2.set_xlabel("Central Field $\sigma_0$", fontsize=12)
ax2.set_xlim(sigma0_start, sigma0_end)
ax2.set_title("Frequency vs Central Density", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend() # Show the labels

# Formatting the general figure
fig.suptitle('Shooting Method: Family of Boson Star Solutions (CSW Profile)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()