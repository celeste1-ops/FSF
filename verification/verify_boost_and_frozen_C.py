"""Assertions for the frozen-C and boosted-parameter comparisons."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsf_model import BASE, simulate, Cbar_exact, interior_fixed_point, fold_in


def assert_close(name, actual, expected, atol):
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name}: expected {expected}, obtained {actual}")


def main() -> int:
    expected_frozen = {0.3: 0.7369, 0.6: 0.6575, 0.9: 0.5410}
    observed = {}
    for R, expected in expected_frozen.items():
        p = dict(BASE); p['R'] = R
        value = float(simulate([3.0, 0.80], 5000, p).y[1, -1])
        assert_close(f'frozen final C at R={R}', value, expected, 7e-4)
        observed[R] = value

    At, Ct, _ = interior_fixed_point(BASE, [8.0, 0.43])
    pb = dict(BASE); pb['R'] = 0.9; pb['E'] = 0.8
    cbar_base = Cbar_exact(At, BASE)
    cbar_boost = Cbar_exact(At, pb)
    assert_close('baseline C-bar', cbar_base, 0.4304, 6e-5)
    assert_close('boosted C-bar', cbar_boost, 0.4314, 7e-5)

    Ab, Cb, _ = interior_fixed_point(pb, [8.0, 0.43])
    assert_close('boosted trapped A', Ab, 8.068, 8e-4)
    assert_close('boosted trapped C', Cb, 0.4314, 8e-5)
    xf = fold_in(pb, 'mu', [4.5, 0.43, 0.55])
    assert_close('boosted fold mu_c', xf[2], 0.5544, 7e-5)
    assert_close('boosted fold A', xf[0], 4.468, 2e-3)

    print('frozen final C:', {r: round(v, 4) for r, v in observed.items()})
    print(f'baseline/boosted C-bar={cbar_base:.4f}/{cbar_boost:.4f}')
    print(f'boosted trapped=({Ab:.3f}, {Cb:.4f}); boosted mu_c={xf[2]:.4f}')
    print('Boost and frozen-C checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
