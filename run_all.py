"""One-command reproduction: run all scenarios, the event-study module,
and build all tables and figures.

Usage (from repo root):  python run_all.py
Expected runtime: see README.md.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENARIOS = ["S1", "S2", "S3", "S6", "S4", "S5", "W1", "W2", "W3", "W6", "W4"]

if __name__ == "__main__":
    for name in SCENARIOS:
        print(f"=== scenario {name} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "src" / "simulate.py"),
                        "--scenario", name, "--reps", "500"], check=True)
    print("=== event-study module ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / "simulate.py"),
                    "--eventstudy", "--reps", "200"], check=True)
    print("=== tables and figures ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / "tables_figures.py")],
                   check=True)
    print("Done. See output/.")
