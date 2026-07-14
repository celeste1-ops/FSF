"""Figure 11 — recovery cliff vs structural discrimination R (§3.5, §4.3.2).
Quoted: ~0.64 (R=0.2) rising to ~0.81 (R=1.2); 0.666/0.727/0.774/0.813 at 0.3/0.6/0.9/1.2."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, cliff

Rs = [0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
cs = []
for r in Rs:
    p = dict(BASE); p['R'] = r
    c = cliff(p); cs.append(c); print(f"[fig11] R={r}: cliff = {c:.4f}")
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(Rs, cs, 'o-', c='#1f77b4')
ax.set_xlabel('structural discrimination  $R$'); ax.set_ylabel('recovery cliff  $C_0^*$')
ax.set_title('Cliff rises with $R$ across the tested range')
fig.tight_layout(); fig.savefig('figures/fig11_cliff_vs_R.png', dpi=160)
