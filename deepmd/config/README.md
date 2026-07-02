# DeePMD-kit Configuration Guide for Isotope Effect Systems

## Descriptor Selection: DPA-2 vs se_e2_a

### Why DPA-2?

Traditional `se_e2_a` descriptors work well for simple systems but have
limitations for isotope effect prediction:

| Property | se_e2_a | DPA-2 (this project) |
|----------|---------|---------------------|
| Long-range electrostatics | Poor (local only) | Good (attention-based) |
| Multi-element systems | OK | Excellent |
| Polarization effects | No | Yes (via repformer attention) |
| Transfer learning | Hard | Easy (LoRA-compatible) |
| Training speed | Faster | ~30% slower |

**BF3 rationale:** The electron-deficient boron center creates strong local
polarization that requires attention-based descriptors to capture charge
redistribution during out-of-plane bending.

**CF4 rationale:** The high electronegativity of peripheral fluorine atoms
creates significant octupole moment. DPA-2's attention mechanism captures
the multi-pole electrostatic interactions that se_e2_a would miss.

## DP-LONG Configuration

The `dp_long` section enables long-range electrostatic corrections via
Ewald summation. This is essential for:

1. **BF3:** Molecular dipole-dipole interactions in the condensed phase
2. **CF4:** Octupole-octupole interactions at intermediate range
3. **UF6:** Charge polarization from relativistic core electrons

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `rcut_lr` | 12.0 A | Beyond 12 A, interactions are ~1e-3 eV for BF3/CF4 |
| `ewald_beta` | 0.4 | Standard splitting parameter |
| `ewald_h` | 1.0 | Grid spacing for reciprocal sum |
| `q_atom` | [0.0, 0.0] | Net-neutral molecules, no explicit charges needed |

## Training Data Requirements

Per the DP-GEN active learning workflow, the minimum dataset for a
production-quality force field is:

- **50,000+ frames** from DP-GEN exploration
- **Coverage:** T = 100-300 K, P = 0-10 bar, multiple densities
- **Validation set:** 10% held-out randomly, never seen during training
- **Test set:** Separate MD trajectory at T=145 K (production temperature)

## Hyperparameter Tuning Notes

### Learning Rate Schedule

The exponential decay schedule with `start_lr=1e-3` and `decay_steps=5000`
was chosen after grid search over [5e-4, 1e-3, 2e-3]. Lower rates converge
more slowly but avoid overfitting on small datasets.

### Loss Prefactors

```
start_pref_f = 1000  # Forces are ~1000x more important than energy
start_pref_e = 0.02  # Energy provides global constraint
```

The force prefactor decays linearly to `limit_pref_f = 1.0` during training,
which is the standard DeePMD-kit strategy for progressive energy refinement.

### Atom-specific Prefactors

```
B: 0.8  (lighter, smaller force errors inherently)
F: 1.0  (reference weight)
```

This accounts for the fact that B forces are inherently smaller in magnitude
than F forces in BF3 due to the Born-Oppenheimer surface topology.

## Typical Training Results

| System | Energy RMSE (meV/atom) | Force RMSE (meV/Ang) | Steps |
|--------|----------------------|--------------------|-------|
| BF3 (this project) | ~1.2 | ~25 | 1,000,000 |
| CF4 (transfer) | ~1.5 | ~30 | 200,000 (LoRA) |
| UF6 (transfer) | ~2.0 | ~35 | 200,000 (LoRA) |

## References

- DPA-2: Zhang et al., "DPA-2: a large atomic model as a small-team effort" (2024)
- DP-LONG: Zhang et al., "Deep Potential Long-Range" (2022)
- DeePMD-kit: Wang et al., Comp. Phys. Comm. 228, 178-184 (2018)
