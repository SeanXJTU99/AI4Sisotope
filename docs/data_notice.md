# Data Anonymization Notice

## What Is Anonymized

All atomic coordinates in template files and mock data have been
replaced with randomly perturbed values:

```python
np.random.normal(0, 0.1, size=coords.shape)
```

This applies to:
- `abacus/templates/*/STRU.template` — all atomic positions
- `data/mock/bf3_dimer.xyz` — all atomic coordinates
- Any inline coordinate arrays in Python scripts

## What Is Preserved (Real Values)

The following are production-tuning parameters kept at their actual
values as proof of model calibration capability:

### ABACUS Functional Parameters
- SCAN+rVV10: `rvv10_b=15.7`, `rvv10_c=0.0093`
- PBE0-D3(BJ): `exx_fraction=0.25`, `d3_version=3`
- SCF convergence: `scf_thr=1e-7`, `scf_nmax=200`
- Mixing: `mixing_beta=0.3`, `mixing_ndim=8`
- Smearing: `smearing_method=gaussian`, `smearing_sigma=0.002`

### DeePMD-kit Training Hyperparameters
- DPA-2 architecture: repformer layers, dimensions, attention config
- Loss prefactors: `start_pref_f=1000`, `start_pref_e=0.02`
- Learning rate: exponential decay, `start_lr=1e-3`, `decay_steps=5000`
- DP-LONG: `ewald_beta=0.4`, `rcut_lr=12.0`

### PIMD Configuration
- Beads count: P=64 (production value for BF3 at 145 K)
- Thermostat: PILE_G with `tau=100 fs`, `lambda=0.5`
- PIGLET GLE matrix: 6x6 parameters from Ceriotti et al. (PRL 2012)
- Timestep: 0.25 fs

### FEP Parameters
- Zwanzig exponential averaging formula
- Block size: 20 blocks for uncertainty estimation
- Physical constants: kB, hbar at CODATA 2018 values

## Why This Matters

Isotope fractionation coefficient (alpha) prediction is exquisitely
sensitive to functional choice and PIMD convergence parameters.
Getting these right is 90% of the difficulty. The coordinate values
themselves matter far less for demonstrating the methodology.

## Data Source

Training data used to produce the validated force fields comes from
proprietary industrial isotope separation facilities. The datasets
are not included in this repository per data usage agreements.

This repository provides:
1. Complete code for the full pipeline
2. Configuration templates with real tuning parameters
3. Mock data for format demonstration and testing

Researchers can substitute their own DFT-generated training data
following the exact same pipeline configuration.
