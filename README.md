# AI4S Isotope -- Cross-Scale Prediction Platform for Anomalous Isotope Effects

AI-driven prediction of low-temperature condensed-phase isotope fractionation
coefficients by training neural network potential energy surfaces that replace
traditional DFT, reducing O(N^3) to O(N) while preserving quantum mechanical
accuracy via path integral molecular dynamics (PIMD).

**2024.11 -- 2025.06**

## Architecture

```mermaid
flowchart TB
    subgraph DFT["1. DFT Data (ABACUS)"]
        A1[BF3: SCAN+rVV10]
        A2[CF4: PBE0-D3(BJ)]
        A3[UF6: relativistic PP]
    end
    subgraph AL["2. Active Learning (DP-GEN)"]
        B1[Exploration MD] --> B2[Labeling] --> B3[Training]
    end
    subgraph ML["3. Force Field (DeePMD-kit)"]
        C1[DPA-2 + DP-LONG] --> C2[Validation]
    end
    subgraph QM["4. Quantum Dynamics (i-PI + PIMD)"]
        D1[Liquid PIMD P=64] & D2[Gas PIMD P=64] --> D3[FEP Mass Mutation]
    end
    subgraph OUT["5. Prediction"]
        E1[alpha = exp(-DDA/kT)] --> E2[Experimental Calibration]
    end
    DFT --> AL --> ML --> QM --> OUT
    E2 -.->|feedback| DFT
```

## Target Systems

| System | Central Isotope | Key Mode | Functional | Beads |
|--------|----------------|----------|------------|-------|
| BF3 | 10B/11B | nu2 out-of-plane bend (~691 cm-1) | SCAN+rVV10 | P=64 |
| CF4 | 12C/13C | nu3 stretch / nu4 bend | PBE0-D3(BJ) | P=48 |
| UF6 | 235U/238U | nu3 stretch / nu4 bend | PBE0-D3(BJ) + rel. PP | P=16 |

## Project Structure

```
ai4s-isotope/
  abacus/
    templates/        ABACUS input templates (BF3, CF4, UF6)
    scripts/          Data extraction and convergence checking
  dp_gen/             DP-GEN active learning pipeline
    exploration/      MD exploration and candidate selection
    labeling/         ABACUS job submission and data collection
  deepmd/             DeePMD-kit force field training
    train/            DPA-2 config, training scripts, LoRA fine-tuning
    validate/         Model validation and learning curves
    deploy/           LAMMPS deployment input
    config/           Descriptor and hyperparameter guide
  pimd/               Path integral molecular dynamics
    ipi/              i-PI input files and launch scripts
    convergence/      Beads count (P) convergence testing
    fep/              Free energy perturbation calculations
    analysis/         VACF power spectrum, mode decomposition
  calibration/        Experimental calibration and comparison
  generalization/     Cross-system transfer (BF3 -> CF4, UF6)
  tests/              Unit and integration tests
  data/mock/          Mock data for demonstration
  docs/               Architecture and data documentation
```

## Quick Start

```bash
# Build the Docker image
docker build -t ai4s-isotope .

# Run the container
docker run --gpus all -it -p 31415:31415 ai4s-isotope

# Full workflow (inside container)
cd /workspace/ai4s-isotope
bash dp_gen/run_dpgen.sh                    # Stage 1-2: DFT data + active learning
bash deepmd/train/train.sh                   # Stage 3: DPA-2 training
bash pimd/ipi/run_pimd.sh                    # Stage 4: PIMD simulation
python pimd/fep/calculate_alpha.py           # Stage 5: alpha prediction
python calibration/compare_exp_vs_theory.py  # Stage 6: calibration
```

## Workflow Stages

### 1. ABACUS DFT Data Production
High-quality first-principles reference data using SCAN+rVV10 (BF3) or
PBE0-D3(BJ) (CF4, UF6) functionals with pseudopotentials. UF6 uses
fully-relativistic pseudopotential for U (Z=92).

### 2. DP-GEN Active Learning
Autonomous exploration-labeling-training loop. Model deviation flags
under-sampled configurations for targeted DFT labeling. Typically
converges in 8-15 iterations for rigid molecules.

### 3. DPA-2 Force Field Training
Deep Potential Attention-2 architecture with DP-LONG long-range
electrostatic correction. 1M training steps, exponential LR decay.
Target accuracy: <2 meV/atom energy, <50 meV/Ang force.

### 4. PIMD Quantum Dynamics
Path integral MD with P=64 beads (BF3), PILE_G thermostat.
Separate simulations for liquid (32-molecule cell) and gas
(isolated molecule) phases. i-PI handles path integral integration;
DeePMD-kit provides real-time forces.

### 5. FEP Free Energy Perturbation
Mass mutation via Zwanzig exponential averaging computes Helmholtz
free energy difference for isotope substitution in both phases.
Block averaging with 20 blocks estimates statistical uncertainty.

### 6. Calibration and Prediction
alpha = exp(-(DeltaA_liquid - DeltaA_gas) / kT). Predicted alpha
compared against industrial distillation column measurements.
Cross-system generalization via LoRA fine-tuning.

## Data Anonymization Notice

**Training data is sourced from proprietary industrial isotope separation
facilities and is not included in this repository.** All atomic coordinates
in template files are randomly perturbed (`np.random.normal(0, 0.1, ...)`).
Functional parameters, PIMD configurations, and thermostat settings are
preserved at their actual production values. See `docs/data_notice.md` for details.

## Physical Mechanism

For BF3, the inverse vapor pressure isotope effect (IVPIE) is driven by
the nu2 out-of-plane umbrella bending mode. In the liquid phase, steric
hindrance from neighboring molecules causes a blue-shift of nu2, creating
a larger zero-point energy penalty for 10B (lighter isotope). This makes
11BF3 thermodynamically favored in the gas phase -- it enriches at the
top of the distillation column, matching industrial observations.

## Tech Stack

| Component | Software | Version |
|-----------|----------|---------|
| DFT Engine | ABACUS | 3.7 |
| Force Field | DeePMD-kit (DPA-2 + DP-LONG) | 2.2 |
| Active Learning | DP-GEN | 0.11 |
| PIMD Engine | i-PI | 2.6 |
| Thermostats | PILE / PIGLET (GLE) | -- |
| Free Energy | FEP (Zwanzig averaging) | -- |
| Analysis | VACF, Power Spectrum, Mode Decomp | -- |
| Container | Docker (CUDA 12.1) | -- |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_fep.py -v
python -m pytest tests/test_dp_gen_workflow.py -v
```

## Documentation

- `docs/architecture.md` -- Full system architecture and design rationale
- `docs/data_notice.md` -- Data anonymization policy
- `docs/workflow_tutorial.ipynb` -- Step-by-step tutorial
- `deepmd/config/README.md` -- Force field configuration guide

## License

MIT
