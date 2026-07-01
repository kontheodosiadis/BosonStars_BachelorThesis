# A Numerical Study of Ground-State Boson Stars with Quartic Self-Interaction
**Bachelor Thesis by Konstantinos Theodosiadis**

---

## Overview
This repository contains the algorithms and final documentation for my Bachelor's thesis, which presents a numerical study of ground-state boson stars with quartic self-interaction.

* `Bachelor_Thesis.pdf`: The complete, final written thesis document.
* `/solve_bvp` and `/shooting_method`: A collection of Python scripts used to solve the eigenvalue boundary value problems (BVPs) of the Einstein-Klein-Gordon equations and visualize the results.

---

## Python Scripts

The numerical analysis is split into two methods: Python's built-in `solve_bvp` solver and the shooting method.

### 1. Built-in Python `solve_bvp` Solver
These scripts utilize `scipy.integrate.solve_bvp` to numerically solve the differential equations and map out the configurations of the boson stars.

* `groundBS_solvebvp01_completeProfile`: Core solver that numerically solves the EKG equations with Python's built-in `solve_bvp` and generates a full profile.
* `groundBS_solvebvp02_varyingSigmac`: Computes scalar field and mass distribution profiles for varying values of the central field parameter $\sigma_c$.
* `groundBS_solvebvp02_varyingSigmac_animation`: GIF animation showing the evolution of scalar field and mass distribution profiles as $\sigma_c$ varies.
* `groundBS_solvebvp03_varyingLambda`: Computes scalar field and mass distribution profiles for varying values of the self-interaction strength parameter $\Lambda$.
* `groundBS_solvebvp03_varyingLambda_animation`: GIF animation showing the evolution of scalar field and mass distribution profiles as $\Lambda$ varies.
* `groundBS_solvebvp04_massSigmacCurve`: Computes the total mass-central field relationship for a given self-interaction $\Lambda$.
* `groundBS_solvebvp05_particleNumberCurve`: Plots the total particle number-central field relationship alongside the total mass-central field curve.
* `groundBS_solvebvp06_massSigmacCurve_multipleLambda`: Compiles total mass-central field curves across multiple self-interaction strengths (various $\Lambda$).
* `groundBS_solvebvp06_massSigmacCurve_multipleLambda_animation`: GIF animation comparing total mass-central field curves across different $\Lambda$ strengths.
* `groundBS_solvebvp07_maxMass_movement`: Tracks the global maximum mass $M_{max}$ as the self-interaction strength $\Lambda$ increases.
* `groundBS_solvebvp08_maxMassLambda_fitting`: Compiles a $(\Lambda, M_{max})$ dataset and fits the data points to the theoretical relation $M_{max} = \sqrt{a \cdot \Lambda + b}$.
* `groundBS_solvebvp09_dynamicalStability`: Determines the dynamical stability boundary and explicitly specifies stable vs. unstable regions.
* `groundBS_solvebvp10_energeticStability`: Determines the energetic stability boundary and explicitly specifies bound vs. unbound regions.
* `groundBS_solvebvp11_stabilityBoundaries`: Maps both dynamical and energetic stability regions simultaneously.
* `groundBS_solvebvp11_stabilityRegions`: Color-coded visualization showcasing simultaneous dynamical and energetic stability regions.
* `groundBS_solvebvp12_evolutionaryFates`: Illustrates the three distinct evolutionary fates across the different stability regions.
* `groundBS_solvebvp13_MRdiagram`: Generates the Mass-Radius ($M$-$R$) diagram for a configured boson star.
* `groundBS_solvebvp14_MRdiagramStability`: Analyzes and highlights stability regions directly on the $M$-$R$ diagram.

### 2. Shooting Method
These scripts solve the boundary value problem utilizing the shooting method.

* `groundBS_shooting01_completeProfile`: Core solver that numerically solves the EKG equations using a custom shooting method approach to generate the full star profile.
* `groundBS_shooting02_massSigmacCurve`: Computes the total mass-central field relationship for a given self-interaction $\Lambda$ using the shooting method.
