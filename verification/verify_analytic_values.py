"""Independent deterministic checks for the principal FSF numerical values.

This auxiliary verifier intentionally re-implements the planar equations with
independent parameter names and a centered finite-difference Jacobian. It is
therefore a cross-check of, rather than a call back into, ``fsf_model.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

P = dict(u_eff=5.0, lam=0.15, Amax=10.0, mu=0.3, eps=0.02, R=0.6,
         w0=0.2, E=0.3, Kw=1.0, eta=0.008, psi=0.015,
         kappa=0.05, gamma=1.0)


def W(A, p):
    return (A + p['w0'] + p['E']) / (A + p['w0'] + p['E'] + p['Kw'])


def D(A, C, p):
    return A * (1 - C) * (1 + p['gamma'] * p['u_eff'])


def dA(A, C, p):
    return (p['lam'] * A * (1-C) * (1 - A/p['Amax'])
            - (p['mu']*C + p['eps']) * A/(1+A))


def dC(A, C, p):
    d, w = D(A, C, p), W(A, p)
    return (p['eta']*p['R']*w*d*(1-C)
            - p['psi']*d*(w*p['R'] + (1-w)*p['kappa'])*C/(1+C))


def rhs(t, y, p):
    return [dA(y[0], y[1], p), dC(y[0], y[1], p)]


def jac(A, C, p, h=1e-7):
    J = np.zeros((2, 2))
    J[0, 0] = (dA(A+h, C, p) - dA(A-h, C, p)) / (2*h)
    J[0, 1] = (dA(A, C+h, p) - dA(A, C-h, p)) / (2*h)
    J[1, 0] = (dC(A+h, C, p) - dC(A-h, C, p)) / (2*h)
    J[1, 1] = (dC(A, C+h, p) - dC(A, C-h, p)) / (2*h)
    return J


def eq(x, p):
    return [dA(x[0], x[1], p), dC(x[0], x[1], p)]


def assert_close(name, actual, expected, atol):
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name}: expected {expected}, obtained {actual}")


def main() -> int:
    lam, eps, Amax = P['lam'], P['eps'], P['Amax']

    c_star = (lam-eps)/(lam+P['mu'])
    cbar0 = (-P['psi'] + np.sqrt(P['psi']**2 + 4*P['eta']**2)) / (2*P['eta'])
    G = (Amax+1)**2/(4*Amax)
    muc0 = (G*lam*(1-cbar0)-eps)/cbar0
    assert_close('C*', c_star, 0.2888888889, 1e-9)
    assert_close('C-bar(kappa=0)', cbar0, 0.4332320125, 1e-9)
    assert_close('leading mu_c^(0)', muc0, 0.547446, 1e-6)

    trap = fsolve(eq, [8, .43], args=(P,))
    saddle = fsolve(eq, [.8, .41], args=(P,))
    trap_ev = np.sort(np.linalg.eigvals(jac(*trap, P)).real)
    saddle_ev = np.sort(np.linalg.eigvals(jac(*saddle, P)).real)
    assert_close('trapped node', trap, [8.0775, 0.4304], 5e-5)
    assert_close('trapped eigenvalues', trap_ev, [-0.2283, -0.0546], 8e-5)
    assert_close('saddle', saddle, [0.7851, 0.4147], 5e-5)
    assert_close('saddle eigenvalues', saddle_ev, [-0.0137, 0.0275], 8e-5)

    def fold_mu(x):
        p = dict(P); p['mu'] = x[2]
        return [dA(x[0], x[1], p), dC(x[0], x[1], p),
                np.linalg.det(jac(x[0], x[1], p))]

    def fold_lam(x):
        p = dict(P); p['lam'] = x[2]
        return [dA(x[0], x[1], p), dC(x[0], x[1], p),
                np.linalg.det(jac(x[0], x[1], p))]

    xm = fsolve(fold_mu, [4.5, .43, .56])
    xl = fsolve(fold_lam, [4.5, .43, .086])
    assert_close('mu fold [A,C,mu]', xm, [4.44144, 0.42828, 0.55895], 2e-5)
    assert_close('lambda fold', xl[2], 0.08587, 2e-5)

    def final_A(C0, p):
        sol = solve_ivp(rhs, [0, 200], [3.0, C0], args=(p,), method='LSODA',
                        rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[0, -1]

    def cliff(p):
        return brentq(lambda c: final_A(c, p) - 0.5, 0.3, 0.99, xtol=1e-5)

    c_base = cliff(P)
    assert_close('baseline cliff', c_base, 0.7270, 6e-4)
    r_expected = [(0.3, 0.6664), (0.6, 0.7270), (0.9, 0.7746), (1.2, 0.8130)]
    r_actual = []
    for Rv, expected in r_expected:
        p = dict(P); p['R'] = Rv
        value = cliff(p)
        assert_close(f'cliff R={Rv}', value, expected, 7e-4)
        r_actual.append((Rv, value))

    print(f"C*={c_star:.6f}; C-bar={cbar0:.6f}; leading mu_c={muc0:.6f}")
    print(f"trapped={tuple(trap.round(4))}; saddle={tuple(saddle.round(4))}")
    print(f"mu_c={xm[2]:.5f} at A={xm[0]:.4f}; lambda_c={xl[2]:.5f}")
    print(f"baseline cliff={c_base:.4f}; R sweep={[(r, round(v,4)) for r,v in r_actual]}")
    print("Independent analytic-value checks passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
