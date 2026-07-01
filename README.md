# A Numerical Study of Ground-State Boson Stars with Quartic Self-Interaction
Bachelor Thesis by Konstantinos Theodosiadis

## Bachelor Thesis
The core of this repository is the bachelor thesis document.

## Python Scripts
This is supplied by the python scripts.

### Built-in Python solve_bvp Solver
* **groundBS_solvebvp01_completeProfile**: core solver that numerically solves the EKG equations with Python's built-in solve_bvp and generates a full profile
* **groundBS_solvebvp02_varyingSigmac**: scalar field and mass distribution profiles for varying the central field parameter $\sigma_c$
* **groundBS_solvebvp02_varyingSigmac_animation**: gif animation for scalar field and mass distribution profiles for varying the central field parameter $\sigma_c$
* **groundBS_solvebvp03_varyingLambda**: scalar field and mass distribution profiles for varying the self-interaction strength parameter $\Lambda$
* **groundBS_solvebvp03_varyingLambda**: gif animation for scalar field and mass distribution profiles for varying the self-interaction strength parameter $\Lambda$
* **groundBS_solvebvp04_massSigmacCurve**: total mass-central field relationship for a given self-interaction $\Lambda$
* **groundBS_solvebvp05_particleNumberCurve**: total particle number-central field relationship drawn together with the total mass-central field curve
* **groundBS_solvebvp06_massSigmacCurve_multipleLambda**: total mass-central field curves for different self-interaction strengths (various $\Lambda$)
* **groundBS_solvebvp06_massSigmacCurve_multipleLambda_animation**: gif animation for total mass-central field curves for different self-interaction strengths (various $\Lambda$)
* **groundBS_solvebvp07_maxMass_movement**: global maximum $\mathcal{M}_\text{max}$ tracking for increasing self-interaction $\Lambda$
* **groundBS_solvebvp08_maxMassLambda_fitting**: compiling a ($\Lambda$, $\mathcal{M}_\text{max}$) dataset, and fitting the data point at the relation $\mathcal{M}_\text{max} = \sqrt{a \cdot \Lambda + b}$
* **groundBS_solvebvp09_dynamicalStability**: dynamical stability boundary and specifying stability-instability regions
* **groundBS_solvebvp10_energeticStability**: energetic stability boundary and specifying bound-unbound regions
* **groundBS_solvebvp11_stabilityBoundaries**: simultaneous dynamical and energetic stability regions
* **groundBS_solvebvp11_stabilityRegions**: simultaneous dynamcial and energetic stability regions (color-coded)
* **groundBS_solvebvp12_evolutionaryFates**: three evolutionary fates for the stability regions
* **groundBS_solvebvp13_MRdiagram**: M-R diagram for a boson star
* **groundBS_solvebvp14_MRdiagramStability**: stability based on the M-R diagram

### Shooting Method
* **groundBS_shooting01_completeProfile**: core solver that numerically solves the EKG equations with Python's built-in solve_bvp and generates a full profile
* **groundBS_shooting02_massSigmacCurve**: total mass-central field relationship for a given self-interaction $\Lambda$
