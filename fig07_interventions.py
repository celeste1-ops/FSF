"""Figure 7 — intervention tests (§4.3.7). Deterministic, seed-free.
Protocol (stated here and in manuscript): constant u_eff = 5 (baseline), start (A0, C0) = (3.0, 0.65).
(a) Test A single-target boosts: C+0.15 @t=30; C+0.5 @t=30; mu->0.8 on [30,90]; R->0.9 from t=30; E->0.8 from t=30.
(b) Test B mu->0.8 windows: early [20,80], mid [60,120], late [100,160]. Quoted: late fails, final A ~ 3.7.
(c) Test C u_eff(t) patterns from low-activation start (A0=0.5, C0=0.75), equal total exposure."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, simulate_piecewise, simulate

T = 200; te = np.linspace(0, T, 2001)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) Test A — C boosts are state kicks, not parameter windows
def run_ckick(dC, tk):
    s1 = simulate([3.0, 0.65], tk, BASE, t_eval=te[te <= tk])
    y = [s1.y[0, -1], min(s1.y[1, -1] + dC, 0.999), s1.y[2, -1]]
    s2 = simulate(y, T - tk, BASE, t_eval=te[te <= T - tk])
    return np.concatenate([s1.t, s2.t + tk]), np.concatenate([s1.y, s2.y], axis=1)
runs_a = [("C +0.15 @30", run_ckick(0.15, 30)), ("C +0.50 @30", run_ckick(0.50, 30))]
for lab, win in [(r"$\mu\to0.8$ [30,90]", [(30, 90, {'mu': 0.8})]),
                 (r"$R\to0.9$ from 30", [(30, T, {'R': 0.9})]),
                 (r"$E\to0.8$ from 30", [(30, T, {'E': 0.8})])]:
    runs_a.append((lab, simulate_piecewise([3.0, 0.65], T, BASE, win, t_eval=te)))
for lab, (tt, yy) in runs_a:
    rec = yy[0][-1] < 0.5
    axes[0].plot(tt, yy[0], lw=1.4, ls='-' if rec else '--', label=f"{lab} → {'recovers' if rec else f'trapped ({yy[0][-1]:.1f})'}")
    print(f"[fig07a] {lab}: final A = {yy[0][-1]:.3f}")
axes[0].set_title('(a) Test A — single-target interventions'); axes[0].legend(fontsize=7)

# (b) Test B — timing of the mu boost
for lab, w in [("early [20,80]", (20, 80)), ("mid [60,120]", (60, 120)), ("late [100,160]", (100, 160))]:
    tt, yy = simulate_piecewise([3.0, 0.65], T, BASE, [(w[0], w[1], {'mu': 0.8})], t_eval=te)
    rec = yy[0][-1] < 0.5
    axes[1].plot(tt, yy[0], lw=1.4, ls='-' if rec else '--', label=f"{lab} → final A={yy[0][-1]:.2f}")
    print(f"[fig07b] {lab}: final A = {yy[0][-1]:.3f}")
axes[1].set_title(r'(b) Test B — timing of $\mu\to0.8$'); axes[1].legend(fontsize=7)

# (c) Test C — u_eff(t) delivery patterns, equal total exposure (60 u·t.u. each)
def u_rep(t):  return 6.0 if any(a <= t <= a + 2 for a in (10, 25, 40, 55, 70)) else 0.0
def u_single(t): return 6.0 if 10 <= t <= 20 else 0.0
def u_sus(t): return 2.0 if 10 <= t <= 40 else 0.0
S_end = {}
for lab, uf in [("repeated short pulses", u_rep), ("one longer pulse", u_single), ("sustained lower-amplitude", u_sus)]:
    breaks = {
        "repeated short pulses": [10, 12, 25, 27, 40, 42, 55, 57, 70, 72],
        "one longer pulse": [10, 20],
        "sustained lower-amplitude": [10, 40],
    }[lab]
    tt, yy = simulate_piecewise([0.5, 0.75], 120, BASE, [], u_fn=uf,
                                t_eval=np.linspace(0, 120, 1201), breakpoints=breaks)
    axes[2].plot(tt, yy[2], lw=1.4, label=lab); S_end[lab] = yy[2]
labels = list(S_end)
sp = max(np.max(np.abs(S_end[labels[i]] - S_end[labels[j]]))
         for i in range(len(labels)) for j in range(i + 1, len(labels)))
print(f"[fig07c] max pairwise |S| spread across patterns = {sp:.4f} (near-overlap claim)")
axes[2].set_title('(c) Test C — feeling S under delivery patterns'); axes[2].legend(fontsize=8)
for ax, yl in zip(axes, ['$A_M$', '$A_M$', '$S$']):
    ax.set_xlabel('t'); ax.set_ylabel(yl)
fig.tight_layout(); fig.savefig('figures/fig7_interventions.png', dpi=160)
