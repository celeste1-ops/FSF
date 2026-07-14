# Reproducibility checks

The public release was checked for the following:

1. All Python files compile successfully.
2. The automated verification suite exits successfully only when numerical assertions pass.
3. The local threshold, leading-order target, fixed points, equilibrium `S` values, fold locations, fold nondegeneracy quantities, and `lambda` threshold match the manuscript values within stated tolerances.
4. The exact analytic Jacobian in `fsf_model.py` is automatically compared with centered finite differences at 200 reproducibly sampled legal state/parameter points; the release fails if the maximum absolute error exceeds `1e-7`.
5. Figure 8 contains a matched no-pulse control: from `(A_M(0), C_0)=(4.0, 0.72)` with background input 1, the no-pulse trajectory recovers while the specified pulse produces trapping.
6. The Figure 8 exit-intervention panel uses the separately stated deeper initial condition `(4.0, 0.60)`.
7. The `mu=0.60` grid check recovers at all 900 tested initial states by `T=500`.
8. All 12 numerical figure scripts run successfully and write their documented outputs.
9. The packaged parameter-recovery files are the `N=100` outputs and include `N`, `noise`, and `channels` metadata. Smoke tests write ignored filenames and do not overwrite them.
10. Cache files, virtual environments, smoke outputs, temporary logs, duplicate manuscripts, explanatory artwork, and internal revision records are excluded.
11. `SHA256SUMS.txt` records checksums for the release files, and the automated release-integrity check verifies every listed digest.

The full `N=100 × 4` optimization is computationally expensive. The public package contains the saved full-run arrays; release checks validate their structure and regenerate Figure 9 from them. The saved arrays are not a substitute for rerunning the full optimization when an independent end-to-end replication is required.
