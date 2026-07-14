"""Figure 4 — finite-time outcome map approximating basin structure (§4.3.4).
Deterministic; 30x30 = 900-point grid; classified at t=200, A<0.5 = recovered."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, simulate, Cstar, interior_fixed_point

A0s = np.linspace(0.2, 9.8, 30); C0s = np.linspace(0.02, 0.98, 30)
M = np.zeros((30, 30))
for i, a0 in enumerate(A0s):
    for j, c0 in enumerate(C0s):
        M[j, i] = 1.0 if simulate([a0, c0], 200, BASE).y[0, -1] < 0.5 else 0.0
sA, sC, _ = interior_fixed_point(BASE, [0.8, 0.41])
fig, ax = plt.subplots(figsize=(7, 5))
ax.pcolormesh(A0s, C0s, M, cmap=matplotlib.colors.ListedColormap(['#f4b6c2', '#b8e0b8']), shading='nearest')
ax.axhline(Cstar(BASE), ls=':', c='k', label=f'$C^*$ = {Cstar(BASE):.3f}')
ax.plot(sA, sC, 'kx', ms=10, mew=2, label='saddle (§3.4)')
ax.set_xlabel('$A_M(0)$'); ax.set_ylabel('$C_0$'); ax.legend(loc='lower right', fontsize=8)
ax.set_title('Finite-time outcome map (green = recovered at t = 200)')
fig.tight_layout(); fig.savefig('figures/fig4_basin.png', dpi=160)
print(f"[fig04] 900-point grid done; recovered fraction = {M.mean():.3f}; saddle at ({sA:.3f},{sC:.3f})")
