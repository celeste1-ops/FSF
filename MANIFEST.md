# Reproducibility manifest

`fsf_model.py` is the canonical source for the model equations and baseline parameters used by the figure generators. `verification/verify_analytic_values.py` deliberately re-implements selected equations independently, using a numerical Jacobian, to provide a non-circular cross-check.

## Analytic and numerical checks

| Result | Reported value / status | Generating or checking file |
|---|---:|---|
| Local threshold `C*` | 0.288889 | `verification/verify_analytic_values.py` |
| Leading-order conflict target | 0.433232 | `verification/verify_analytic_values.py` |
| Trapped node | `(8.0775, 0.4304)` | `verification/verify_analytic_values.py` |
| Trapped full-system `S*` | 13.8037 | `verification/verify_manuscript_claims.py` |
| Saddle | `(0.7851, 0.4147)` | `verification/verify_analytic_values.py` |
| Saddle full-system `S*` | 1.3785 | `verification/verify_manuscript_claims.py` |
| Exact `mu` fold | `mu_c=0.55895`, `A=4.4414` | `fig10_fold.py`; verification scripts |
| Exact `lambda` fold | `lambda_c=0.08587` | verification scripts |
| Fold nondegeneracy | `p^T F_mu≈-0.3520`, `p^T D²F(q,q)≈-0.0138` | `verification/verify_manuscript_claims.py` |
| Baseline recovery cliff | `C_0≈0.7270` | `fig01_recovery_cliff.py`; `verification/verify_analytic_values.py` |
| `mu=0.60` recovery grid | 900/900 recover by `T=500` | `verification/verify_manuscript_claims.py` |
| Figure 8 matched control | no-pulse recovery; pulse trapping | `fig08_entry_exit.py`; `verification/verify_manuscript_claims.py` |
| Exact analytic Jacobian | centered finite-difference agreement at 200 legal random points; max error must be `<1e-7` | `verification/verify_release_integrity.py` |

## Numerical figures

| Paper figure | Main content | Script | Packaged output |
|---|---|---|---|
| Figure 1 | recovery cliff | `fig01_recovery_cliff.py` | `figures/fig1_recovery_cliff.png` |
| Figure 2 | robustness across parameter settings | `fig02_robustness.py` | `figures/fig2_robustness.png` |
| Figure 3 | nearby initial conditions, divergent fates | `fig03_trajectories.py` | `figures/fig3_trajectories.png` |
| Figure 4 | finite-time basin map | `fig04_basin.py` | `figures/fig4_basin.png` |
| Figure 5 | descriptive transient fluctuations | `fig05_fluctuations.py` | `figures/fig5_fluctuations.png` |
| Figure 6 | noisy endpoint divergence | `fig06_fate_divergence.py` | `figures/fig6_fate_divergence.png` |
| Figure 7 | intervention tests | `fig07_interventions.py` | `figures/fig7_interventions.png` |
| Figure 8 | matched pulse-entry control and exit intervention | `fig08_entry_exit.py` | `figures/fig8_entry_exit.png` |
| Figure 9 | parameter recovery / identifiability | `identifiability/run_table1.py`; `fig09_identifiability.py` | `figures/fig9_identifiability.png` |
| Figure 10 | fold diagram | `fig10_fold.py` | `figures/fig10_fold.png` |
| Figure 11 | recovery cliff versus `R` | `fig11_cliff_vs_R.py` | `figures/fig11_cliff_vs_R.png` |
| Figure 12 | exploratory approach-to-fold ensemble | `fig12_csd_fold.py` | `figures/fig12_csd_fold.png` |

## Parameter-recovery data

The four packaged `N=100` result files include `true`, `est`, `N`, `noise`, and `channels` arrays. Their shapes, finiteness, parameter bounds, paired full-state parameter draws, and metadata are checked by `verification/verify_release_integrity.py`.
