"""Figure 10 — existence boundary of the trapped state (§3.5).

The interior equilibrium branch is parameterized by A_M rather than followed
with unconstrained fsolve continuation. This avoids accidental convergence to
the A_M = 0 quiescent equilibrium manifold.

Quoted: node and saddle branches annihilate at mu_c = 0.5590, A ~ 4.44.
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from fsf_model import BASE, Cbar_exact, jac2, mu_on_interior_branch

EPS = 1e-7
res = minimize_scalar(
    lambda A: -mu_on_interior_branch(A, BASE),
    bounds=(EPS, BASE['A_max'] - EPS), method='bounded',
    options={'xatol': 1e-13}
)
if not res.success:
    raise RuntimeError(f"branch maximum solve failed: {res.message}")
Af = float(res.x)
mu_c = float(-res.fun)
Cf = float(Cbar_exact(Af, BASE))
print(f"[fig10] exact branch maximum: mu_c = {mu_c:.5f} at A = {Af:.4f}, C = {Cf:.4f}")

# Dense exact branch parameterization. For fixed lambda, every positive
# interior equilibrium satisfies C=Cbar_exact(A) and mu=mu(A).
As = np.linspace(EPS, BASE['A_max'] - EPS, 12000)
mus = np.array([mu_on_interior_branch(A, BASE) for A in As])
Cs = np.array([Cbar_exact(A, BASE) for A in As])

# Plot the range used in the manuscript figure. The low-A branch enters the
# physical positive-mu panel at A=0 near mu≈0.185, so it does not extend to 0.1.
show = (mus >= 0.10) & (mus <= mu_c + 1e-8)
low = show & (As < Af)
high = show & (As > Af)

# Verify stability labels numerically on representative branch samples.
def check_branch(mask, expected):
    idx = np.flatnonzero(mask)
    for i in idx[::max(1, len(idx)//30)]:
        p = dict(BASE); p['mu'] = mus[i]
        ev = np.linalg.eigvals(jac2(As[i], Cs[i], p)).real
        if expected == 'saddle' and not (ev.min() < 0 < ev.max()):
            raise RuntimeError(f"unexpected low-A branch stability at A={As[i]:.6g}: {ev}")
        if expected == 'node' and not np.all(ev < 0):
            raise RuntimeError(f"unexpected high-A branch stability at A={As[i]:.6g}: {ev}")
check_branch(low, 'saddle')
check_branch(high, 'node')

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(mus[high], As[high], '-', c='#1f77b4', lw=2, label='trapped node (stable)')
ax.plot(mus[low], As[low], '--', c='#d62728', lw=2, label='saddle (unstable)')
ax.plot(mu_c, Af, 'ko', ms=7, label=f'fold  $\\mu_c$ = {mu_c:.4f}')
ax.axvline(BASE['mu'], ls=':', c='gray', label='baseline $\\mu$ = 0.3')
ax.set_xlim(0.10, 0.58)
ax.set_xlabel('integration efficiency  $\\mu$'); ax.set_ylabel('equilibrium $A_M$')
ax.set_title('Fold: trapped state exists only below $\\mu_c$'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/fig10_fold.png', dpi=160)
