"""Figure 6 — noise-induced finite-horizon outcome divergence (§4.3.6).
Seeds 0-49; sigma = (0.06, 0.025, 0.06); C0 = 0.62, A0 = 3; classify at T = 200, A < 0.5 = recovered.
Quoted: 88% (44/50) trapped-class/non-recovered, 12% recovered."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, em_simulate

SIG = (0.06, 0.025, 0.06)
fig, ax = plt.subplots(figsize=(8, 4.8))
rec = 0
for seed in range(50):
    t, y = em_simulate([3.0, 0.62], 200, BASE, SIG, seed=seed)
    r = y[0, -1] < 0.5; rec += r
    ax.plot(t, y[0], lw=0.6, alpha=0.7, color=('#2ca02c' if r else '#ff7f0e'))
print(f"[fig06] recovered {rec}/50, trapped {50-rec}/50  (manuscript: 6/50 recovered, 44/50 trapped)")
# Wilson 95% CI for the trapped-class proportion.
n = 50; k = 50 - rec; z = 1.959963984540054
phat = k / n
den = 1 + z*z/n
center = (phat + z*z/(2*n)) / den
half = z * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / den
print(f"[fig06] Wilson 95% CI for trapped-class proportion = [{center-half:.3%}, {center+half:.3%}]")
ax.set_xlabel('t'); ax.set_ylabel('$A_M$')
ax.set_title(f'50 realizations from identical IC ($C_0$=0.62): {50-rec}/50 trapped-classified at T=200')
fig.tight_layout(); fig.savefig('figures/fig6_fate_divergence.png', dpi=160)
