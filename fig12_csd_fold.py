"""Figure 12 — critical slowing down as mu approaches the fold (§4.3.5).
EM initialized at the mu-specific trapped node, dt=0.1, T=1500, burn-in 500, sigma=(0.05, 0.005, 0), seeds 0-9,
escape criterion A_M < 1 after burn-in. Quoted survivor-conditioned summaries: var 0.027->0.149, lag-10 (10 time units) AC ~0.60->0.89,
5/10 escapes at mu=0.55; dominant eigenvalue -0.049 -> -0.010."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, em_simulate, interior_fixed_point

MUS = [0.35, 0.40, 0.45, 0.50, 0.55]; BURN = 5000  # steps (=500 time units)
res = {}
for mu in MUS:
    p = dict(BASE); p['mu'] = mu
    An, Cn, ev = interior_fixed_point(p, [7.5, 0.43])
    vs, acs, esc = [], [], 0
    for seed in range(10):
        t, y = em_simulate([An, Cn], 1500, p, (0.05, 0.005, 0.0), seed=seed)
        A = y[0, BURN:]
        if A.min() < 1.0:
            esc += 1; continue
        vs.append(A.var())
        acs.append(np.corrcoef(A[:-100], A[100:])[0, 1])  # lag = 10 time units = 100 steps
    res[mu] = (np.mean(vs) if vs else np.nan, np.mean(acs) if acs else np.nan, esc, ev[-1], vs, acs)
    print(f"[fig12] mu={mu}: var={res[mu][0]:.3f}, AC10={res[mu][1]:.2f}, escapes={esc}/10, dom eig={ev[-1]:+.4f}")
fig, ax1 = plt.subplots(figsize=(7.5, 4.6)); ax2 = ax1.twinx()
mus = list(res)
for m in mus:
    ax1.plot([m]*len(res[m][4]), res[m][4], 'o', c='#1f77b4', ms=4, alpha=0.5)
    ax2.plot([m]*len(res[m][5]), res[m][5], 's', c='#d62728', ms=4, alpha=0.5)
ax1.plot(mus, [res[m][0] for m in mus], '-', c='#1f77b4', lw=2, label='survivor var $A_M$')
ax2.plot(mus, [res[m][1] for m in mus], '--', c='#d62728', lw=2, label='survivor lag-10 AC')
ax1.axvline(0.5590, ls=':', c='k'); ax1.text(0.5595, ax1.get_ylim()[1]*0.9, '$\\mu_c$', fontsize=10)
ax1.set_xlabel('$\\mu$'); ax1.set_ylabel('survivor-conditioned variance of $A_M$', color='#1f77b4')
ax2.set_ylabel('lag-10 autocorrelation', color='#d62728')
fig.suptitle('CSD as $\\mu$ approaches the fold (survivor-conditioned; 10-seed exploratory ensemble)')
fig.tight_layout(); fig.savefig('figures/fig12_csd_fold.png', dpi=160)
