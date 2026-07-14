"""Release-structure and saved-result integrity checks."""
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import sys
from contextlib import redirect_stderr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsf_model import BASE, jac2, rhs2
from identifiability.run_table1 import parse_args as parse_table1_args

ID = ROOT / 'identifiability'
FIG = ROOT / 'figures'

EXPECTED = {
    'results_full-state_5.npz': (100, 0.05, np.array([0, 1, 2])),
    'results_full-state_10.npz': (100, 0.10, np.array([0, 1, 2])),
    'results_full-state_20.npz': (100, 0.20, np.array([0, 1, 2])),
    'results_S-only_10.npz': (100, 0.10, np.array([2])),
}

LO = np.array([0.05, 0.10, 0.00, 0.002, 0.005])
HI = np.array([0.35, 0.60, 0.10, 0.020, 0.040])
EXPECTED_FIGURES = [
    'fig1_recovery_cliff.png', 'fig2_robustness.png', 'fig3_trajectories.png',
    'fig4_basin.png', 'fig5_fluctuations.png', 'fig6_fate_divergence.png',
    'fig7_interventions.png', 'fig8_entry_exit.png', 'fig9_identifiability.png',
    'fig10_fold.png', 'fig11_cliff_vs_R.png', 'fig12_csd_fold.png',
]


def _centered_fd_jacobian(A: float, C: float, p: dict[str, float]) -> np.ndarray:
    """Centered finite-difference Jacobian of rhs2 at one legal state point."""
    x = np.array([A, C], dtype=float)
    out = np.empty((2, 2), dtype=float)
    for j in range(2):
        h = 1e-6 * max(1.0, abs(x[j]))
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        fp = np.asarray(rhs2(0.0, xp, p), dtype=float)
        fm = np.asarray(rhs2(0.0, xm, p), dtype=float)
        out[:, j] = (fp - fm) / (2.0 * h)
    return out


def _check_analytic_jacobian(n_points: int = 200, tol: float = 1e-7) -> float:
    """Compare jac2 with centered finite differences over random legal states/parameters."""
    rng = np.random.default_rng(20260714)
    max_err = 0.0
    for _ in range(n_points):
        p = dict(BASE)
        p.update({
            'u_eff': rng.uniform(0.0, 10.0),
            'lam': rng.uniform(0.05, 0.35),
            'A_max': rng.uniform(5.0, 20.0),
            'mu': rng.uniform(0.10, 0.60),
            'eps': rng.uniform(0.0, 0.10),
            'R': rng.uniform(0.20, 1.20),
            'w0': rng.uniform(0.0, 1.0),
            'E': rng.uniform(0.0, 1.0),
            'K_w': rng.uniform(0.20, 2.0),
            'eta': rng.uniform(0.002, 0.020),
            'psi': rng.uniform(0.005, 0.040),
            'kappa': rng.uniform(0.0, 0.20),
            'gamma': rng.uniform(0.20, 2.0),
        })
        A = rng.uniform(1e-4, p['A_max'] - 1e-4)
        C = rng.uniform(1e-4, 1.0 - 1e-4)
        exact = jac2(A, C, p)
        numeric = _centered_fd_jacobian(A, C, p)
        err = float(np.max(np.abs(exact - numeric)))
        max_err = max(max_err, err)
    if max_err > tol:
        raise AssertionError(
            f'analytic Jacobian finite-difference mismatch: max error={max_err:.3e}, '
            f'tolerance={tol:.1e}'
        )
    return max_err



def _check_table1_cli_contract() -> None:
    """Require exact long options; argparse abbreviations and invalid mode combinations must fail."""
    smoke = parse_table1_args(['--smoke'])
    full = parse_table1_args(['--full'])
    if not (smoke.smoke and not smoke.full):
        raise AssertionError('--smoke did not select smoke mode exactly')
    if not (full.full and not full.smoke):
        raise AssertionError('--full did not select full mode exactly')

    invalid = ([], ['--smok'], ['--ful'], ['--unknown'], ['--smoke', '--full'])
    for argv in invalid:
        with redirect_stderr(io.StringIO()):
            try:
                parse_table1_args(argv)
            except SystemExit as exc:
                if exc.code == 0:
                    raise AssertionError(f'invalid CLI arguments exited successfully: {argv}')
            else:
                raise AssertionError(f'invalid CLI arguments were accepted: {argv}')

def _check_sha256_manifest() -> int:
    """Verify every file listed in SHA256SUMS.txt (the checksum file excludes itself)."""
    manifest = ROOT / 'SHA256SUMS.txt'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    count = 0
    for raw in manifest.read_text(encoding='utf-8').splitlines():
        if not raw.strip():
            continue
        digest, rel = raw.split(maxsplit=1)
        rel = rel.strip()
        path = ROOT / rel.removeprefix('./')
        if not path.is_file():
            raise FileNotFoundError(f'checksum entry is missing: {rel}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f'checksum mismatch: {rel}')
        count += 1
    return count


def main() -> int:
    true_full = []
    for filename, (n, noise, channels) in EXPECTED.items():
        path = ID / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as z:
            required = {'true', 'est', 'N', 'noise', 'channels'}
            if set(z.files) != required:
                raise AssertionError(f'{filename}: keys {z.files}, expected {sorted(required)}')
            true = z['true']; est = z['est']
            if true.shape != (n, 5) or est.shape != (n, 5):
                raise AssertionError(f'{filename}: unexpected array shapes')
            if not (np.all(np.isfinite(true)) and np.all(np.isfinite(est))):
                raise AssertionError(f'{filename}: non-finite values')
            if not (np.all(true >= LO) and np.all(true <= HI)):
                raise AssertionError(f'{filename}: true parameters outside declared ranges')
            if not (np.all(est >= LO - 1e-12) and np.all(est <= HI + 1e-12)):
                raise AssertionError(f'{filename}: estimates outside optimizer bounds')
            if int(z['N']) != n or not np.isclose(float(z['noise']), noise):
                raise AssertionError(f'{filename}: metadata mismatch')
            if not np.array_equal(z['channels'], channels):
                raise AssertionError(f'{filename}: channel metadata mismatch')
            if filename.startswith('results_full-state'):
                true_full.append(true)

    if not all(np.array_equal(true_full[0], x) for x in true_full[1:]):
        raise AssertionError('full-state conditions do not share paired true-parameter draws')

    missing = [name for name in EXPECTED_FIGURES if not (FIG / name).is_file()]
    if missing:
        raise FileNotFoundError(f'missing packaged figures: {missing}')
    if (FIG / 'model_overview.png').exists() or (FIG / 'model_overview.svg').exists():
        raise AssertionError('explanatory overview should not be in the numerical reproducibility package')

    gitignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
    for pattern in ('__pycache__/', '*.py[cod]', '.venv/', 'identifiability/*_smoke.npz'):
        if pattern not in gitignore:
            raise AssertionError(f'.gitignore is missing required pattern: {pattern}')

    _check_table1_cli_contract()
    max_jac_err = _check_analytic_jacobian()
    checksum_count = _check_sha256_manifest()
    print('Strict Table 1 CLI check passed; abbreviations and invalid mode combinations are rejected.')
    print(f'Analytic Jacobian check passed at 200 random legal points; max error={max_jac_err:.3e}.')
    print(f'SHA256 verification passed for {checksum_count} release files.')
    print('Saved-result metadata, paired draws, figure set, and release structure passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
