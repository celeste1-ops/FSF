# Feeling State Framework (FSF) — Reproducibility Code

This repository contains the computational implementation and reproducibility materials for:

**A Mechanistic Nonlinear Dynamical Framework for Feeling-State Evolution: Bifurcation, Trapped States, and Recovery Thresholds**  
Suning Gu, preprint version 1.0 (2026).

The manuscript PDF is intentionally not duplicated in this repository. The repository is designed to be linked from the preprint's **Code Availability** statement.

## Repository contents

```text
fsf_model.py                 canonical model equations, baseline parameters,
                             integrators, Jacobian, fixed-point and branch utilities
fig01_...py to fig12_...py  one script for each numerical paper figure
run_all_figures.py           regenerate all 12 numerical figures
run_verification.py          run all automated verification checks
verification/                independent analytic, numerical, and release checks
identifiability/             Table 1 / Figure 9 simulate-and-recover pipeline
figures/                     packaged outputs for numerical Figures 1–12
MANIFEST.md                  reported result -> generating/checking script
REPRODUCIBILITY_CHECKS.md    summary and scope of completed checks
requirements.txt             compatible minimum direct dependencies
requirements-lock.txt        audited direct numerical package versions
CITATION.cff                 machine-readable citation metadata
LICENSE                      MIT License
SHA256SUMS.txt               checksums for the public release files
```

The explanatory model-overview artwork is included in the paper but is not included here because it is not a numerical output and is not required to reproduce any result.

## Python and installation

**Python 3.11 or later is required for the audited locked dependencies.** The audited rerun used Python 3.13.5.

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the audited direct package versions:

```bash
pip install -r requirements-lock.txt
```

Alternatively, install compatible minimum direct versions:

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` records the direct numerical packages used in the audited rerun; it is not a complete operating-system-level environment image.

## Regenerate all numerical figures

From the repository root:

```bash
python run_all_figures.py
```

Individual examples:

```bash
python fig01_recovery_cliff.py
python fig04_basin.py
python fig08_entry_exit.py
python fig10_fold.py
python fig12_csd_fold.py
```

Each script writes its output into `figures/`. Regenerating all figures may take several minutes; Figure 10 is the most time-consuming script.

## Run verification checks

```bash
python run_verification.py
```

The runner executes independent analytic checks, mechanism-specific checks, manuscript-claim checks, and release-integrity checks. Important reported values use numerical assertions with explicit tolerances; a mismatch exits with an error. The release-integrity stage also compares the exact analytic Jacobian with centered finite differences at 200 reproducibly sampled legal state/parameter points and verifies every entry in `SHA256SUMS.txt`.

## Parameter-recovery experiment

Quick smoke test:

```bash
python identifiability/run_table1.py --smoke
```

Exactly one of `--smoke` or `--full` is required; unknown or misspelled options are rejected.

Full experiment used for Table 1 and Figure 9:

```bash
python identifiability/run_table1.py --full
python fig09_identifiability.py
```

The full optimization is substantially slower. The packaged `N=100` result files are included in `identifiability/`, with metadata for `N`, noise level, and observed channels, so Figure 9 can be regenerated without rerunning the full optimization.

## Numerical conventions

- Deterministic ODEs use `solve_ivp(..., method="LSODA")`, `rtol=1e-10`, `atol=1e-12`, and `max_step=1.0`.
- Piecewise interventions are split at parameter/input discontinuities.
- Stochastic state simulations use Euler–Maruyama with `dt=0.1`; seeds are specified in the corresponding scripts.
- Deterministic integration does not clamp states. The stochastic extension applies the projection rules described in the paper.
  

## Citation

Preprint DOI: https://doi.org/10.5281/zenodo.21383204

Gu, S. (2026). *A Mechanistic Nonlinear Dynamical Framework for Feeling-State Evolution: Bifurcation, Trapped States, and Recovery Thresholds* (Version 1.0) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.21383204

Machine-readable citation metadata are provided in `CITATION.cff`.

## License

The software and associated repository materials are released under the **MIT License**. See `LICENSE`.
