"""Single entry point that regenerates all sample data and fixtures.

Run from the C3-Model-Validation folder:
    python scripts/make_fixtures.py

Each defective fixture is verified by verify_fixtures.py.
"""
from pathlib import Path
import subprocess
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    "prepare_data.py",       # splits, eval.csv, spare.csv, train row hashes
    "train_good_model.py",   # fixtures/good
    "make_defective_fixtures.py",  # leaky, overfit, biased, ... variants
    "verify_fixtures.py",    # confirms each defect is really present
]

for step in STEPS:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SCRIPTS_DIR / step)], check=True)

print("\nAll fixtures generated.")
