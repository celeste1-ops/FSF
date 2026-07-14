"""Figure 2 — cliff robustness across regimes (§4.3.2). Deterministic, seed-free.
Quoted: baseline 0.73, lam=0.25->0.83, mu=0.15->0.90, u=8->0.78, R=0.3->0.67."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, final_A, cliff

regimes = [("baseline", {}), (r"$\lambda=0.25$", {'lam': 0.25}), (r"$\mu=0.15$", {'mu': 0.15}),
           (r"$u_{eff}=8$", {'u_eff': 8.0}), (r"$R=0.3$", {'R': 0.3})]
C0s = np.linspace(0.4, 0.99, 60)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for name, over in regimes:
    p = dict(BASE); p.update(over)
    ax.plot(C0s, [final_A(c, p) for c in C0s], lw=1.4, label=f"{name} (cliff {cliff(p):.2f})")
    print(f"[fig02] {name:<14} cliff = {cliff(p):.4f}")
ax.set_xlabel('$C_0$'); ax.set_ylabel('final $A_M$ (t = 200)')
ax.set_title('Recovery cliff across parameter regimes'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/fig2_robustness.png', dpi=160)
