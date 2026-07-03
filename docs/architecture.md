# System Architecture

## Full Pipeline Overview

```mermaid
flowchart TB
    subgraph DFT["1. DFT Data Production (ABACUS)"]
        A1[BF3: SCAN+rVV10]
        A2[CF4: PBE0-D3(BJ)]
        A3[UF6: PBE0-D3(BJ) + relativistic PP]
        A1 & A2 & A3 --> B[ABACUS SCF]
    end

    subgraph AL["2. Active Learning (DP-GEN)"]
        B --> C[Exploration MD<br/>model deviation]
        C --> D{deviation > trust_lo?}
        D -->|yes| E[ABACUS Labeling]
        E --> F[Training Set Update]
        F --> G[DPA-2 Training]
        G --> C
    end

    subgraph ML["3. Force Field (DeePMD-kit)"]
        G --> H[DPA-2 + DP-LONG<br/>frozen_model.pb]
        H --> I[Validation<br/>E < 2 meV/atom<br/>F < 50 meV/Ang]
    end

    subgraph QM["4. Quantum Dynamics (i-PI + PIMD)"]
        H --> J[PIMD P=64<br/>PILE/PIGLET thermostat]
        J --> K[Liquid Phase<br/>32-molecule cell]
        J --> L[Gas Phase<br/>isolated molecule]
    end

    subgraph FEP["5. Free Energy (FEP)"]
        K --> M[Mass Mutation<br/>10B -> 11B]
        L --> M
        M --> N[DeltaA_liquid]
        M --> O[DeltaA_gas]
        N & O --> P[DeltaDeltaA = A_l - A_g]
    end

    subgraph CAL["6. Prediction & Calibration"]
        P --> Q[alpha = exp(-DDA/kT)]
        Q --> R[Compare with experiment]
        R --> S[Cross-system generalization<br/>BF3 -> CF4 -> UF6]
    end

    S -.->|feedback| A1
```

## Layer Details

### 1. ABACUS DFT Data Production

**Purpose:** Generate high-quality first-principles reference data for force field training.

**Why ABACUS?**
- Native NAO basis sets: faster than plane-wave for molecular systems
- Deep integration with DeepModeling ecosystem (DeePKS, DeepH, DP-GEN)
- Open-source, no license restrictions for industrial deployment

**Functional Selection Rationale:**

| System | Functional | Reason |
|--------|-----------|--------|
| BF3 | SCAN+rVV10 | Strongly constrained meta-GGA; rVV10 captures interlayer dispersion critical for planar molecule stacking |
| CF4 | PBE0-D3(BJ) | Hybrid functional needed for accurate C-F bond polarization; D3(BJ) for octupole-octupole dispersion |
| UF6 | PBE0-D3(BJ) + relativistic PP | Heavy element (Z=92) requires scalar relativistic corrections via fully-relativistic pseudopotential |

**Convergence Criteria:**
- SCF threshold: 1e-7 eV (tight, needed for force accuracy)
- Force convergence: 0.01 eV/Ang
- k-points: Gamma 2x2x2 for 15-20 A cubic boxes
- Energy cutoff: 100 Ry (BF3 LCAO), 120 Ry (CF4 PW)

### 2. DP-GEN Active Learning

**Exploration:** Short NVT trajectories (5000 steps, 0.5 fs) at 100-300 K with current model ensemble.
**Candidate selection:** Frames with max force deviation between trust_lo=0.05 and trust_hi=0.15 eV/Ang.
**Labeling:** ABACUS single-point calculations on selected candidates.
**Convergence:** Typically 8-15 iterations for rigid molecules (BF3, CF4).

### 3. DPA-2 Force Field

**Architecture:**
- Descriptor: repinit (8-dim type embedding) + repformer (6 layers, 128-dim G1, 64-dim G2)
- Attention: gated attention with 32-dim hidden, sqrt(n_neighbor) normalization
- Fitting net: 3-layer [240, 240, 240] with ResNet
- Type embedding: [32, 32] learned per-element

**DP-LONG:** Ewald-based long-range electrostatic correction (rcut_lr=12.0 A, beta=0.4).

**Training:** 1M steps, exponential LR decay (1e-3 -> 5e-8), batch_size=4.

### 4. i-PI Path Integral Molecular Dynamics

**PIMD Parameters:**
- Beads: P=64 (converged for BF3 nu2 mode ~691 cm-1 at 145 K)
- Timestep: 0.25 fs (stable for light-element path integral)
- Thermostat: PILE_G (tau=100 fs, lambda=0.5)
- Total steps: 500k production (after 50k equilibration)

**Phase Separation:**
- Liquid: 32-molecule periodic cell at experimental density
- Gas: Isolated molecule in large box (>20 A)

### 5. FEP Free Energy Perturbation

**Method:** Mass mutation via Zwanzig exponential averaging.

**Formula:**
```
Delta H_spring = 0.5 * Delta(mass) * (P*kT/hbar)^2 * sum_s (r_s - r_{s+1})^2
Delta A = -kT * ln( <exp(-Delta H_spring / kT)> )
```

**Error Estimation:** Block averaging (20 blocks) with bootstrap.

### 6. Alpha Prediction and Calibration

**Core Formula:** ln(alpha) = -(DeltaA_liquid - DeltaA_gas) / (k_B T)

**Classification:**
- alpha > 1: Inverse isotope effect (heavy in gas phase)
- alpha < 1: Normal isotope effect (light in gas phase)
- |alpha-1| < 2*sigma: Uncertain

## Computational Resource Estimates

| Stage | GPU-Hours | Wall Time | Hardware |
|-------|-----------|-----------|----------|
| ABACUS labeling (per iteration) | 0 (CPU) | 2-4 h | 16-core CPU |
| DPA-2 training (1M steps) | 48 | 12 h | 4x V100 |
| PIMD liquid (500k steps) | 24 | 6 h | 1x V100 |
| PIMD gas (500k steps) | 2 | 0.5 h | 1x V100 |
| FEP analysis | 0 (CPU) | 5 min | 1-core CPU |
| **Total per system** | **~74** | **~20 h** | mixed |
