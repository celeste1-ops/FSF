"""Figure 9 — parameter recovery scatter (§5.1). Reads identifiability/results_*.npz
produced by identifiability/run_table1.py --full (driver seed 42)."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['λ', 'μ', 'ε', 'η', 'ψ']
fs = np.load('identifiability/results_full-state_10.npz')
so = np.load('identifiability/results_S-only_10.npz')
if fs['true'].shape[0] != 100 or so['true'].shape[0] != 100:
    raise RuntimeError('Figure 9 requires N=100 full-run result files; rerun run_table1.py --full')
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, (dat, lab) in enumerate([(fs, 'full-state, 10% noise'), (so, 'S-only, 10% noise')]):
    for j in range(5):
        ax = axes[row, j]
        x, y = dat['true'][:, j], dat['est'][:, j]
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(x, y, 'o', ms=3, alpha=0.6, color='#1f77b4' if row == 0 else '#d62728')
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set_title(f"{NAMES[j]}  (r = {r:.2f})", fontsize=10)
        if j == 0: ax.set_ylabel(f"estimated\n({lab})", fontsize=9)
        if row == 1: ax.set_xlabel('true', fontsize=9)
        print(f"[fig09] {lab} {NAMES[j]}: r = {r:.3f}")
fig.suptitle('Simulate-and-recover, N = 100 per condition (driver seed 42)')
fig.tight_layout(); fig.savefig('figures/fig9_identifiability.png', dpi=150)
