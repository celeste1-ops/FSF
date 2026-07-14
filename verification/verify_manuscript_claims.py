"""Additional manuscript-claim verification.

Covers numerical claims that are not figure outputs:
- full 3D equilibrium S values;
- fold nondegeneracy quantities and second eigenvalue;
- exact interior-branch extrema for mu and lambda;
- the mu=0.60 recovery-grid check;
- cumulative-conflict and |Delta C| ratios for the high-mu comparison.

Run from any working directory:
    python verification/verify_manuscript_claims.py
"""
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsf_model import (BASE, Cbar_exact, D_of, fold_in, interior_fixed_point,
                       jac2, lambda_on_interior_branch, mu_on_interior_branch,
                       rhs2, rhs3, simulate, simulate_piecewise)


def fold_nondegeneracy():
    A, C, mu = fold_in(BASE, 'mu', [4.5, 0.43, 0.56])
    p = dict(BASE); p['mu'] = mu
    J = jac2(A, C, p)
    evals, vr = np.linalg.eig(J)
    i0 = int(np.argmin(np.abs(evals)))
    q = np.real(vr[:, i0]); q /= np.linalg.norm(q)
    # Fix the remaining sign ambiguity so signed values are reproducible.
    if q[0] < 0:
        q = -q
    evals_l, vl = np.linalg.eig(J.T)
    pvec = np.real(vl[:, int(np.argmin(np.abs(evals_l)))])
    pvec /= pvec @ q

    F_mu = np.array([-C * A / (1 + A), 0.0])
    trans = float(pvec @ F_mu)
    h = 3e-4
    z = np.array([A, C])
    F0 = np.asarray(rhs2(0.0, z, p))
    d2 = (np.asarray(rhs2(0.0, z + h*q, p)) - 2*F0
          + np.asarray(rhs2(0.0, z - h*q, p))) / h**2
    quad = float(pvec @ d2)
    other = float(np.real(evals[np.argmax(np.abs(evals))]))
    return A, C, mu, other, trans, quad


def branch_extrema():
    eps = 1e-8
    rmu = minimize_scalar(lambda A: -mu_on_interior_branch(A, BASE),
                          bounds=(eps, BASE['A_max']-eps), method='bounded',
                          options={'xatol': 1e-13})
    rla = minimize_scalar(lambda A: lambda_on_interior_branch(A, BASE),
                          bounds=(eps, BASE['A_max']-eps), method='bounded',
                          options={'xatol': 1e-13})
    if not (rmu.success and rla.success):
        raise RuntimeError('branch-extremum optimization failed')
    return rmu.x, -rmu.fun, rla.x, rla.fun


def stationary_point_counts(n=200001):
    """Dense-grid check of the manuscript's single-stationary-point claims."""
    eps = 1e-6
    A = np.linspace(eps, BASE['A_max'] - eps, n)
    mu_vals = np.array([mu_on_interior_branch(a, BASE) for a in A])
    la_vals = np.array([lambda_on_interior_branch(a, BASE) for a in A])

    def count_sign_changes(vals):
        d = np.diff(vals)
        # Ignore exact zeros from floating point; propagate nearest nonzero sign.
        sg = np.sign(d)
        nz = np.flatnonzero(sg)
        if len(nz) == 0:
            return 0
        first = nz[0]
        sg[:first] = sg[first]
        for i in range(first + 1, len(sg)):
            if sg[i] == 0:
                sg[i] = sg[i - 1]
        return int(np.sum(sg[1:] * sg[:-1] < 0))

    return count_sign_changes(mu_vals), count_sign_changes(la_vals)


def recovery_grid_mu060():
    p = dict(BASE); p['mu'] = 0.60
    A0s = np.linspace(0.0, BASE['A_max'], 30)
    C0s = np.linspace(0.0, 1.0, 30)
    end = []
    for A0 in A0s:
        for C0 in C0s:
            end.append(simulate([A0, C0], 500.0, p).y[0, -1])
    end = np.asarray(end)
    return int(np.sum(end < 0.5)), len(end), float(end.max())



def figure8_control_checks():
    def upulse(t):
        return 8.0 if 10 <= t <= 15 else 1.0
    def ubg(t):
        return 1.0
    te = np.linspace(0.0, 250.0, 2501)
    t0, y0 = simulate_piecewise([4.0, 0.72], 250.0, BASE, [], u_fn=ubg,
                                t_eval=te, breakpoints=[])
    t1, y1 = simulate_piecewise([4.0, 0.72], 250.0, BASE, [], u_fn=upulse,
                                t_eval=te, breakpoints=[10, 15])
    idx = np.where(y0[0] < 0.5)[0]
    rt = float(t0[idx[0]]) if len(idx) else None
    return rt, float(y0[0,-1]), float(y1[0,-1])

def conflict_integral(C0, mu, T=5000.0):
    p = dict(BASE); p['mu'] = mu
    def rhs4(t, y):
        drift = rhs3(t, y[:3], p)
        return [*drift, D_of(y[0], y[1], p)]
    sol = solve_ivp(rhs4, [0, T], [3.0, C0, 0.0, 0.0], method='LSODA',
                    rtol=1e-10, atol=1e-12, max_step=1.0)
    if not sol.success:
        raise RuntimeError(sol.message)
    return float(sol.y[1, -1]), float(sol.y[3, -1])


def assert_close(name, actual, expected, atol):
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name}: expected {expected}, obtained {actual}")


if __name__ == '__main__':
    # Full-system equilibrium S values.
    eq_results = {}
    for guess, name in [([8.0, 0.43], 'trapped'), ([0.8, 0.41], 'saddle')]:
        A, C, ev = interior_fixed_point(BASE, guess)
        S = BASE['b'] * D_of(A, C, BASE) / BASE['a']
        eq_results[name] = (A, C, S)
        print(f"{name:8s} full equilibrium = (A={A:.4f}, C={C:.4f}, S={S:.4f}); eig={ev}")
    assert_close('trapped S*', eq_results['trapped'][2], 13.8037, 7e-4)
    assert_close('saddle S*', eq_results['saddle'][2], 1.3785, 7e-4)

    A, C, mu, other, trans, quad = fold_nondegeneracy()
    print(f"fold = (A={A:.5f}, C={C:.5f}, mu={mu:.5f})")
    print(f"fold second eigenvalue = {other:+.4f}  (target about -0.1168)")
    print(f"p^T F_mu = {trans:+.4f}  (q_A>0 orientation; target about -0.352)")
    print(f"p^T D2F(q,q) = {quad:+.4f}  (target about -0.0138)")
    assert_close('fold A', A, 4.44144, 2e-5)
    assert_close('fold C', C, 0.42828, 2e-5)
    assert_close('fold mu', mu, 0.55895, 2e-5)
    assert_close('fold second eigenvalue', other, -0.1168, 2e-4)
    assert_close('fold transversality', trans, -0.3520, 3e-4)
    assert_close('fold quadratic coefficient', quad, -0.0138, 3e-4)

    Amu, muc, Ala, lac = branch_extrema()
    print(f"mu(A) global max = {muc:.5f} at A={Amu:.5f}")
    print(f"lambda(A) global min = {lac:.5f} at A={Ala:.5f}")
    assert_close('mu branch maximum', muc, 0.55895, 2e-5)
    assert_close('lambda branch minimum', lac, 0.08587, 2e-5)
    nmu, nla = stationary_point_counts()
    print(f"dense-grid derivative sign changes: mu(A)={nmu}, lambda(A)={nla} (target 1 each)")
    if (nmu, nla) != (1, 1):
        raise RuntimeError('single-stationary-point check failed')

    nrec, ntot, maxA = recovery_grid_mu060()
    print(f"mu=0.60 grid recovery at T=500: {nrec}/{ntot}; max endpoint A={maxA:.3e}")
    if (nrec, ntot) != (900, 900) or maxA >= 1e-10:
        raise AssertionError('mu=0.60 recovery-grid check failed')

    rt8, A_no, A_pulse = figure8_control_checks()
    print(f"Figure 8 matched control: no-pulse recovery t={rt8:.1f}, final A={A_no:.3e}; pulse final A={A_pulse:.6f}")
    if not (rt8 is not None and abs(rt8 - 69.6) < 0.2 and A_no < 0.5 and A_pulse > 7.5):
        raise RuntimeError('Figure 8 matched entry-control check failed')

    for C0 in [0.75, 0.80]:
        C3, I3 = conflict_integral(C0, 0.3)
        C7, I7 = conflict_integral(C0, 0.7)
        dcr = abs(C3-C0) / abs(C7-C0)
        cr = I3/I7
        print(f"C0={C0:.2f}: conflict ratio mu0.3/mu0.7={cr:.3f}; |Delta C| ratio={dcr:.3f}")
        if not (cr > 1.0 and dcr > 1.0):
            raise AssertionError('high-mu conflict/Delta-C comparison failed')

    print('Manuscript-claim checks passed.')
