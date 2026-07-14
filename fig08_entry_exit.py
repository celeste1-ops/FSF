"""Figure 8 — matched entry control and exit-intervention test (§4.3.8).
Deterministic, seed-free.

Panel (a), entry: matched initial condition (A0, C0) = (4.0, 0.72),
background u_eff = 1. The control has no high pulse; the treatment adds
u_eff = 8 on t in [10,15]. The control recovers and the pulse trajectory traps.

Panel (b), exit: deeper initial condition (A0, C0) = (4.0, 0.60), with the
same pulse. Intervention package mu -> 0.8 and lambda -> 0.05 is applied early
[40,100] or late [120,180]. Additional checks retain the original protocol:
mu-only, E+R persistence, and lambda-only duration scan.
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fsf_model import BASE, simulate_piecewise

T_MAIN = 250

def u_pulse(t):
    return 8.0 if 10 <= t <= 15 else 1.0

def u_background(t):
    return 1.0

def run(ic, windows, T=T_MAIN, pulse=True):
    uf = u_pulse if pulse else u_background
    breaks = [10, 15] if pulse else []
    return simulate_piecewise(ic, T, BASE, windows, u_fn=uf,
                              t_eval=np.linspace(0, T, T * 10 + 1),
                              breakpoints=breaks)

def recovery_time(ts, As, threshold=0.5):
    idx = np.where(As < threshold)[0]
    return float(ts[idx[0]]) if len(idx) else None

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

# Panel (a): matched causal control for entry.
entry_ic = [4.0, 0.72]
for label, pulse in [("background only", False), ("background + pulse", True)]:
    tt, yy = run(entry_ic, [], pulse=pulse)
    rec = recovery_time(tt, yy[0])
    suffix = f"recovery t≈{rec:.1f}" if rec is not None else f"trapped (final {yy[0,-1]:.2f})"
    axes[0].plot(tt, yy[0], lw=1.6, label=f"{label}: {suffix}")
    print(f"[fig08-entry] {label}: rec time={rec}, final A={yy[0,-1]:.6f}, final C={yy[1,-1]:.6f}")
axes[0].axvspan(10, 15, alpha=0.22)
axes[0].text(12.5, 7.8, 'pulse window', fontsize=8, ha='center', va='top', rotation=90)
axes[0].set_title('(a) Matched entry test: pulse changes the long-run outcome')
axes[0].set_xlabel('t'); axes[0].set_ylabel('$A_M$'); axes[0].legend(fontsize=8)

# Panel (b): exit from a deeper trapped trajectory.
exit_ic = [4.0, 0.60]
exit_cases = [
    ("no intervention", []),
    (r"early package [40,100]: $\mu\to0.8,\lambda\to0.05$", [(40, 100, {'mu': 0.8, 'lam': 0.05})]),
    (r"$\mu\to0.8$ alone [40,100]", [(40, 100, {'mu': 0.8})]),
    (r"late package [120,180]", [(120, 180, {'mu': 0.8, 'lam': 0.05})]),
]
for label, windows in exit_cases:
    tt, yy = run(exit_ic, windows, pulse=True)
    rec = recovery_time(tt, yy[0])
    suffix = f"recovery t≈{rec:.1f}" if rec is not None else f"trapped (final {yy[0,-1]:.2f})"
    axes[1].plot(tt, yy[0], lw=1.4, label=f"{label}: {suffix}")
    print(f"[fig08-exit] {label}: rec time={rec}, final A={yy[0,-1]:.6f}")
axes[1].axvspan(10, 15, alpha=0.22)
axes[1].set_title('(b) Exit test: return to background does not reverse trapping')
axes[1].set_xlabel('t'); axes[1].set_ylabel('$A_M$'); axes[1].legend(fontsize=7.4)

# Package + E,R timing comparison (quoted in manuscript).
t0, y0 = run(exit_ic, [(40, 100, {'mu': 0.8, 'lam': 0.05})])
t1, y1 = run(exit_ic, [(40, 100, {'mu': 0.8, 'lam': 0.05, 'E': 0.8, 'R': 0.9})])
print(f"[fig08-check] early package recovery={recovery_time(t0,y0[0])}")
print(f"[fig08-check] early package + E,R recovery={recovery_time(t1,y1[0])}")

# Lambda-alone duration scan, separate extended runs.
t60, y60 = run(exit_ic, [(40, 100, {'lam': 0.05})], T=400, pulse=True)
print(f"[fig08-check] lambda-only 60-unit window: final A={y60[0,-1]:.6f}")
Lstar = None
for L in range(60, 131):
    tt, yy = run(exit_ic, [(40, 40 + L, {'lam': 0.05})], T=400, pulse=True)
    if yy[0,-1] < 0.5:
        Lstar = L
        break
print(f"[fig08-check] lambda-only minimal successful integer duration={Lstar}")

# E+R persistence, 400-unit window, T=500.
tER, yER = run(exit_ic, [(40, 440, {'E': 0.8, 'R': 0.9})], T=500, pulse=True)
print(f"[fig08-check] E+R 400-unit window: final A={yER[0,-1]:.6f}")

fig.tight_layout()
fig.savefig('figures/fig8_entry_exit.png', dpi=160)
