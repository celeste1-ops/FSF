"""Figure 1 — recovery cliff (§4.3.1). Deterministic, seed-free.
Quoted numbers: cliff C0 ~ 0.727 (bisection); trapped final A ~ 8."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, final_A, cliff

C0s = np.linspace(0.05, 0.95, 91)
fin = [final_A(c, BASE) for c in C0s]
b = cliff(BASE)
print(f"[fig01] cliff bisection C0* = {b:.4f}  (manuscript 0.727)")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(C0s, fin, 'o-', ms=3, lw=1, color='#1f77b4')
ax.axvline(b, ls='--', c='crimson', label=f'cliff C0* = {b:.3f}')
ax.set_xlabel('initial cognitive stability  $C_0$'); ax.set_ylabel('final $A_M$ (t = 200)')
ax.set_title('Recovery cliff (separatrix crossing in initial-condition space)')
ax.legend(); fig.tight_layout(); fig.savefig('figures/fig1_recovery_cliff.png', dpi=160)
