"""Table 1 / Figure 9 driver — simulate-and-recover (Section 5.1).

SPEC (documented constants used by this packaged driver):
  - free parameters + uniform prior ranges: lam [0.05,0.35], mu [0.10,0.60],
    eps [0.00,0.10], eta [0.002,0.020], psi [0.005,0.040]; others at BASE
  - trajectory: T = 60, sampled every 0.5 time units (121 points/channel)
  - initial condition, KNOWN during fitting: (A0, C0, S0) = (3.0, 0.65, 0.0)
  - observation noise: Gaussian, sd = noise_level x channel SD (heteroscedastic)
  - objective: least squares on residuals standardized by each channel's scale
  - 5 optimizer restarts (scipy least_squares, bounded), keep best cost
  - driver seed 42; the three full-state noise conditions share the same
    true-parameter draws (paired design); S-only follows the same generator
    sequence continued (independent draws)
Usage:  python run_table1.py --smoke      (N=6, one condition; minutes)
        python run_table1.py --full       (N=100 x 4 conditions; hours)
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsf_model import BASE, simulate

NAMES = ['lam', 'mu', 'eps', 'eta', 'psi']
LO = np.array([0.05, 0.10, 0.00, 0.002, 0.005])
HI = np.array([0.35, 0.60, 0.10, 0.020, 0.040])
T, DT = 60.0, 0.5
TE = np.arange(0, T + 1e-9, DT)
IC = [3.0, 0.65, 0.0]

def traj(theta):
    p = dict(BASE); p.update(dict(zip(NAMES, theta)))
    return simulate(IC, T, p, t_eval=TE).y  # (3, 121)

def fit(obs, scales, channels, rng, restarts=5):
    def resid(th):
        y = traj(th)[channels]
        return ((y - obs) / scales[:, None]).ravel()
    best = None
    for _ in range(restarts):
        x0 = rng.uniform(LO, HI)
        r = least_squares(resid, x0, bounds=(LO, HI), xtol=1e-10, ftol=1e-10)
        if not np.all(np.isfinite(r.x)) or not np.isfinite(r.cost):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError('all optimizer restarts failed')
    return best.x

def run_condition(n_trials, noise, channels, rng_draw, rng_fit):
    rows = []
    for i in range(n_trials):
        th = rng_draw.uniform(LO, HI)
        y = traj(th)[channels]
        sd = y.std(axis=1); sd[sd < 1e-9] = 1e-9
        obs = y + noise * sd[:, None] * rng_draw.standard_normal(y.shape)
        est = fit(obs, sd, channels, rng_fit)
        rows.append((th, est))
    return rows

def metrics(rows):
    tr = np.array([r[0] for r in rows]); es = np.array([r[1] for r in rows])
    out = {}
    for j, nm in enumerate(NAMES):
        r = np.corrcoef(tr[:, j], es[:, j])[0, 1]
        nrmse = np.sqrt(np.mean((es[:, j] - tr[:, j]) ** 2)) / (HI[j] - LO[j])
        bias = np.mean(es[:, j] - tr[:, j])
        out[nm] = (r, nrmse, bias)
    return out

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the FSF Table 1 / Figure 9 simulate-and-recover experiment.',
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true',
                      help='run a quick N=6 full-state 10%% smoke test')
    mode.add_argument('--full', action='store_true',
                      help='run the complete N=100 x 4 experiment')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    full = args.full
    N = 100 if full else 6
    conds = ([('full-state 5%', 0.05, [0, 1, 2]), ('full-state 10%', 0.10, [0, 1, 2]),
              ('full-state 20%', 0.20, [0, 1, 2]), ('S-only 10%', 0.10, [2])]
             if full else [('full-state 10% (smoke)', 0.10, [0, 1, 2])])
    rng_draw = np.random.default_rng(42); rng_fit = np.random.default_rng(4242)
    shared = [rng_draw.uniform(LO, HI) for _ in range(N)]  # paired draws across noise conds
    for name, noise, ch in conds:
        t0 = time.time()
        rows = []
        for i in range(N):
            th = shared[i] if ch == [0, 1, 2] else rng_draw.uniform(LO, HI)
            y = traj(th)[ch]
            sd = y.std(axis=1); sd[sd < 1e-9] = 1e-9
            obs = y + noise * sd[:, None] * rng_draw.standard_normal(y.shape)
            est = fit(obs, sd, ch, rng_fit)
            rows.append((th, est))
        m = metrics(rows)
        tag = name.split(' (')[0].replace(' ', '_').replace('%', '')
        if not full:
            tag += '_smoke'
        out_path = Path(__file__).resolve().parent / f"results_{tag}.npz"
        np.savez(out_path, true=np.array([r[0] for r in rows]),
                 est=np.array([r[1] for r in rows]), N=N, noise=noise, channels=np.array(ch))
        print(f"\n[{name}]  N={N}  ({time.time()-t0:.0f}s)")
        for nm in NAMES:
            r, nr, b = m[nm]
            print(f"  {nm:>4}:  r={r:.2f}  nRMSE={nr:.2f}  bias={b:+.3f}")


def verdict(r, nrmse):
    """Author-defined working criterion (manuscript Section 5.1)."""
    if r >= 0.9 and nrmse <= 0.12:
        return 'well recovered'
    if r >= 0.6 and nrmse <= 0.25:
        return 'weakly recovered'
    return 'poorly recovered'
