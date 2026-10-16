"""Compare all saved runs for one dataset (best score per method and family).

Usage:  python scripts/compare_baselines.py diabetes
"""
import json
import sys

import pandas as pd

from c2_automl.config import OUTPUT_DIR

dataset_id = sys.argv[1]
rows = []
for path in sorted((OUTPUT_DIR / dataset_id).glob("*_summary.json")):
    s = json.loads(path.read_text())
    rows.append({"run_id": s["run_id"], "method": s["method"], "trials": s["total_trials"],
                 "best_family": s["best_family"], "best_val": round(s["best_val"], 4)})

if not rows:
    sys.exit(f"No runs found in {OUTPUT_DIR / dataset_id}")
print(pd.DataFrame(rows).sort_values("best_val", ascending=False).to_string(index=False))
