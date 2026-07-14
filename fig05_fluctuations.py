"""Figure 5 — descriptive transient fluctuation differences (§4.3.5). Stochastic, seed = 1.
Descriptive only (see manuscript); no quoted magnitudes."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, em_simulate

SIG = (0.08, 0.02, 0.05); C0S = [0.85, 0.72, 0.60, 0.50]; WIN = 200  # 20 time units
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for k, c0 in enumerate(C0S):
    t, y = em_simulate([3.0, c0], 200, BASE, SIG, seed=1)  # common noise path (seed 1) across C0s
    A = y[0]
    rv = np.array([A[max(0, i-WIN):i+1].var() for i in range(len(A))])
    ac = np.full(len(A), np.nan)
    for i in range(WIN, len(A)):
        w = A[i-WIN:i+1]
        ac[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
    lab = f'$C_0$={c0}'
    axes[0].plot(t, A, lw=0.7, label=lab); axes[1].plot(t, rv, lw=1); axes[2].plot(t, ac, lw=1)
axes[0].set_ylabel('$A_M$'); axes[0].legend(fontsize=8, ncol=4)
axes[1].set_ylabel('rolling var'); axes[2].set_ylabel('rolling lag-1 AC'); axes[2].set_xlabel('t')
fig.suptitle('Transient fluctuation profiles across initial conditions (descriptive)')
fig.tight_layout(); fig.savefig('figures/fig5_fluctuations.png', dpi=160)
print("[fig05] generated (descriptive; common noise path, seed = 1, across all four C0s)")
