"""FSF model — canonical definition.

Every figure and verification script in this repository imports the model
from this file. If an equation or baseline parameter ever changes, it
changes here and nowhere else.

Model: manuscript Sections 2.3-2.7.
Manuscript: see the preprint citation in README.md.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

BASE = dict(u_eff=5.0, lam=0.15, A_max=10.0, mu=0.3, eps=0.02, R=0.6, w0=0.2,
            E=0.3, K_w=1.0, eta=0.008, psi=0.015, kappa=0.05, gamma=1.0,
            a=0.1, b=0.05)


# ---------------------------------------------------------------- equations
def W_of(A, p):
    """Integration openness (Section 2.5)."""
    return (A + p['w0'] + p['E']) / (A + p['w0'] + p['E'] + p['K_w'])


def D_of(A, C, p, u=None):
    """Unintegrated conflict (Section 2.4). u overrides p['u_eff']."""
    uu = p['u_eff'] if u is None else u
    return A * (1 - C) * (1 + p['gamma'] * uu)


def rhs3(t, y, p, u_fn=None):
    """Full 3D right-hand side (Sections 2.3, 2.6, 2.7).

    u_fn : optional callable t -> u_eff(t) for time-varying input;
           when None, constant p['u_eff'] is used.
    """
    A, C, S = y
    u = p['u_eff'] if u_fn is None else u_fn(t)
    D = D_of(A, C, p, u)
    W = W_of(A, p)
    dA = p['lam'] * A * (1 - C) * (1 - A / p['A_max']) \
        - (p['mu'] * C + p['eps']) * A / (1 + A)
    dC = p['eta'] * p['R'] * W * D * (1 - C) \
        - p['psi'] * D * (W * p['R'] + (1 - W) * p['kappa']) * C / (1 + C)
    dS = -p['a'] * S + p['b'] * D
    return [dA, dC, dS]


def rhs2(t, y, p):
    """Planar (A_M, C) core; S is slaved and does not feed back (Section 3.1)."""
    dA, dC, _ = rhs3(t, [y[0], y[1], 0.0], p)
    return [dA, dC]


# ---------------------------------------------------------------- integrators
def _require_success(sol, context):
    if not sol.success:
        raise RuntimeError(f"{context} failed: {sol.message}")


def simulate(y0, T, p, u_fn=None, t_eval=None, method='LSODA'):
    """Deterministic run. y0 = (A0, C0, S0) or (A0, C0)."""
    y0 = list(y0) + [0.0] * (3 - len(y0))
    sol = solve_ivp(rhs3, [0, T], y0, args=(p, u_fn), method=method,
                    rtol=1e-10, atol=1e-12, t_eval=t_eval, max_step=1.0)
    _require_success(sol, "deterministic integration")
    return sol


def simulate_piecewise(y0, T, p, windows, u_fn=None, t_eval=None, breakpoints=None):
    """Deterministic run with piecewise-constant parameter overrides.

    windows : list of (t_start, t_end, {param: value}) intervention windows.
    breakpoints : optional additional times at which integration must be split,
                  useful for discontinuities in a time-varying u_fn.

    Integration is split at every window boundary and optional breakpoint.
    Duplicate boundary samples are removed from the returned arrays.
    """
    extra = set() if breakpoints is None else set(map(float, breakpoints))
    edges = sorted({0.0, float(T)} | {float(w[0]) for w in windows} |
                   {float(w[1]) for w in windows} | extra)
    edges = [e for e in edges if 0.0 <= e <= T]
    ts_all, ys_all = [], []
    y = list(y0) + [0.0] * (3 - len(y0))
    for seg_i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if b <= a:
            continue
        pk = dict(p)
        for (w0_, w1_, over) in windows:
            if w0_ <= a and b <= w1_:
                pk.update(over)
        te = None
        if t_eval is not None:
            te = np.asarray(t_eval)
            te = te[(te >= a) & (te <= b)]
            if len(te) == 0 or te[0] > a:
                te = np.concatenate([[a], te])
            if te[-1] < b:
                te = np.concatenate([te, [b]])
        sol = solve_ivp(rhs3, [a, b], y, args=(pk, u_fn), method='LSODA',
                        rtol=1e-10, atol=1e-12, t_eval=te, max_step=1.0)
        _require_success(sol, f"piecewise integration on [{a}, {b}]")
        # Drop the repeated left boundary after the first segment.
        sl = slice(None) if seg_i == 0 else slice(1, None)
        ts_all.append(sol.t[sl])
        ys_all.append(sol.y[:, sl])
        y = sol.y[:, -1]
    return np.concatenate(ts_all), np.concatenate(ys_all, axis=1)


def em_simulate(y0, T, p, sig, seed, dt=0.1, u_fn=None):
    """Euler-Maruyama stochastic run (Section 4.1).

    sig : (sigma_A, sigma_C, sigma_S) additive noise amplitudes.
    Per-step clamping: A_M >= 0, C in [0.001, 0.999], S >= 0.
    One rng per realization; three N(0,1) draws per step in (A, C, S) order.
    """
    if dt <= 0 or T < 0:
        raise ValueError("Require dt > 0 and T >= 0")
    n = int(round(T / dt))
    if not np.isclose(n * dt, T, rtol=0.0, atol=1e-12):
        raise ValueError("T must be an integer multiple of dt in em_simulate")
    rng = np.random.default_rng(seed)
    y = np.array(list(y0) + [0.0] * (3 - len(y0)), dtype=float)
    out = np.empty((3, n + 1))
    out[:, 0] = y
    sd = np.sqrt(dt) * np.asarray(sig, dtype=float)
    for k in range(n):
        f = rhs3(k * dt, y, p, u_fn)
        y = y + np.asarray(f) * dt + sd * rng.standard_normal(3)
        y[0] = max(y[0], 0.0)
        y[1] = min(max(y[1], 0.001), 0.999)
        y[2] = max(y[2], 0.0)
        out[:, k + 1] = y
    t = np.linspace(0, n * dt, n + 1)
    return t, out


# ---------------------------------------------------------------- analysis
def jac2(A, C, p):
    """Exact Jacobian of the autonomous planar core for constant p['u_eff']."""
    Q = 1 + p['gamma'] * p['u_eff']
    g = p['w0'] + p['E']
    den = A + g + p['K_w']
    W = (A + g) / den
    W_A = p['K_w'] / den ** 2

    D = A * (1 - C) * Q
    D_A = (1 - C) * Q
    D_C = -A * Q

    B = p['kappa'] + W * (p['R'] - p['kappa'])
    B_A = W_A * (p['R'] - p['kappa'])
    X = p['eta'] * p['R'] * W * (1 - C) - p['psi'] * B * C / (1 + C)
    X_A = p['eta'] * p['R'] * W_A * (1 - C) \
        - p['psi'] * B_A * C / (1 + C)
    X_C = -p['eta'] * p['R'] * W - p['psi'] * B / (1 + C) ** 2

    fA_A = p['lam'] * (1 - C) * (1 - 2 * A / p['A_max']) \
        - (p['mu'] * C + p['eps']) / (1 + A) ** 2
    fA_C = -p['lam'] * A * (1 - A / p['A_max']) \
        - p['mu'] * A / (1 + A)
    fC_A = D_A * X + D * X_A
    fC_C = D_C * X + D * X_C
    return np.array([[fA_A, fA_C], [fC_A, fC_C]], dtype=float)


def interior_fixed_point(p, guess, residual_tol=1e-8):
    """Solve rhs2 = 0 from a guess and require a physical interior root."""
    x, info, ier, msg = fsolve(lambda z: rhs2(0.0, z, p), guess,
                               full_output=True, xtol=1e-11)
    res = np.linalg.norm(info['fvec'], ord=np.inf)
    if ier != 1 or not np.all(np.isfinite(x)) or res > residual_tol:
        raise RuntimeError(f"fixed-point solve failed: {msg}; residual={res:.3e}")
    if not (x[0] > 1e-10 and 0.0 < x[1] < 1.0):
        raise RuntimeError(
            f"solver converged to a non-interior equilibrium ({x[0]:.6g}, {x[1]:.6g}); "
            "use an interior-branch parameterization or a different bracket/guess"
        )
    ev = np.linalg.eigvals(jac2(x[0], x[1], p))
    ev = ev[np.argsort(ev.real)]
    return x[0], x[1], ev


def Cstar(p):
    """Local transverse-stability threshold (Section 3.2)."""
    den = p['lam'] + p['mu']
    if den <= 0:
        raise ValueError("Cstar requires lam + mu > 0")
    return (p['lam'] - p['eps']) / den


def Cbar_kappa0(p):
    """kappa = 0 conflict-driven equilibrium (Section 3.5)."""
    if p['eta'] <= 0:
        raise ValueError("Cbar_kappa0 requires eta > 0")
    return (-p['psi'] + np.sqrt(p['psi'] ** 2 + 4 * p['eta'] ** 2)) / (2 * p['eta'])


def Cbar_exact(A, p):
    """Exact kappa > 0 C-nullcline C(A) in the positive-coefficient regime."""
    w = W_of(A, p)
    a_ = p['eta'] * p['R'] * w
    b_ = p['psi'] * (w * p['R'] + (1 - w) * p['kappa'])
    if a_ <= 0:
        raise ValueError("Cbar_exact requires eta * R * W(A) > 0")
    return (-b_ + np.sqrt(b_ * b_ + 4 * a_ * a_)) / (2 * a_)


def mu_on_interior_branch(A, p):
    """Exact mu(A) along the positive interior equilibrium branch."""
    C = Cbar_exact(A, p)
    return (p['lam'] * (1 - C) * (1 - A / p['A_max']) * (1 + A) - p['eps']) / C


def lambda_on_interior_branch(A, p):
    """Exact lambda(A) along the positive interior equilibrium branch."""
    C = Cbar_exact(A, p)
    den = (1 - C) * (1 - A / p['A_max']) * (1 + A)
    if den <= 0:
        raise ValueError("lambda_on_interior_branch requires 0 < A < A_max and C < 1")
    return (p['mu'] * C + p['eps']) / den


def final_A(C0, p, A0=3.0, T=200.0):
    """Endpoint A_M of a deterministic run (recovery-cliff classifier input)."""
    return simulate([A0, C0], T, p).y[0, -1]


def cliff(p, A0=3.0, T=200.0, lo=0.30, hi=0.99, thr=0.5, xtol=1e-4):
    """Recovery-cliff location by bisection: A_M(T) = thr crossing in C0."""
    flo = final_A(lo, p, A0, T) - thr
    fhi = final_A(hi, p, A0, T) - thr
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError(
            f"cliff bracket does not straddle a crossing: f(lo)={flo:.3g}, f(hi)={fhi:.3g}"
        )
    return brentq(lambda c: final_A(c, p, A0, T) - thr, lo, hi, xtol=xtol)


def fold_in(p, key, guess, residual_tol=1e-8):
    """Locate a candidate fold in parameter `key`: solve rhs2 = 0 and det J = 0."""
    def sys(x):
        pk = dict(p)
        pk[key] = x[2]
        return [*rhs2(0.0, x[:2], pk), np.linalg.det(jac2(x[0], x[1], pk))]

    x, info, ier, msg = fsolve(sys, guess, full_output=True, xtol=1e-11)
    res = np.linalg.norm(info['fvec'], ord=np.inf)
    if ier != 1 or not np.all(np.isfinite(x)) or res > residual_tol:
        raise RuntimeError(f"fold solve failed: {msg}; residual={res:.3e}")
    if not (x[0] > 0 and 0 < x[1] < 1):
        raise RuntimeError(f"fold solve returned non-interior state: {x}")
    return x  # (A_fold, C_fold, key_c)
