"""Regenerate all 12 numerical paper figures in sequence."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "fig01_recovery_cliff.py",
    ROOT / "fig02_robustness.py",
    ROOT / "fig03_trajectories.py",
    ROOT / "fig04_basin.py",
    ROOT / "fig05_fluctuations.py",
    ROOT / "fig06_fate_divergence.py",
    ROOT / "fig07_interventions.py",
    ROOT / "fig08_entry_exit.py",
    ROOT / "fig09_identifiability.py",
    ROOT / "fig10_fold.py",
    ROOT / "fig11_cliff_vs_R.py",
    ROOT / "fig12_csd_fold.py",
]


def main() -> int:
    (ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        print(f"\n=== Running {script.name} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nAll numerical figure scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
