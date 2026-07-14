"""Run all lightweight FSF verification scripts in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "verification" / "verify_analytic_values.py",
    ROOT / "verification" / "verify_boost_and_frozen_C.py",
    ROOT / "verification" / "verify_manuscript_claims.py",
    ROOT / "verification" / "verify_release_integrity.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nAll verification scripts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
