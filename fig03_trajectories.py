"""Figure 3 — basin-outcome divergence from near-identical C0 (§4.3.3). Deterministic.
Quoted: C0=0.74 recovers; 0.71 and 0.50 trapped, final A ~ 8.06."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, simulate

te = np.linspace(0, 200, 801)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for c0, col in [(0.74, '#2ca02c'), (0.71, '#d62728'), (0.50, '#9467bd')]:
    s = simulate([3.0, c0], 200, BASE, t_eval=te)
    axes[0].plot(s.t, s.y[0], color=col, label=f'$C_0$={c0} (final $A_M$={s.y[0,-1]:.2f})')
    axes[1].plot(s.t, s.y[1], color=col)
    print(f"[fig03] C0={c0}: final A = {s.y[0,-1]:.3f}, final C = {s.y[1,-1]:.3f}")
axes[0].set_xlabel('t'); axes[0].set_ylabel('$A_M$'); axes[0].legend(fontsize=8)
axes[1].set_xlabel('t'); axes[1].set_ylabel('$C$')
fig.suptitle('A 0.03 difference in $C_0$ places trajectories in opposite basins')
fig.tight_layout(); fig.savefig('figures/fig3_trajectories.png', dpi=160)
