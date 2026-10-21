"""
PRODUCTION STREAM SIMULATOR

Turns data/processed/future_stream.csv into one folder per scenario:

    data/scenarios/<scenario>/windows.csv       production rows, with a 'window' column and the true label 'y'
    data/scenarios/<scenario>/ground_truth.json what was injected, where, and what C4 should conclude

Labels are stored with the rows, but the C4 runner only "releases" them
`label_delay` windows later (see configs/monitoring.yaml).

Run from the project root:
    python -m src.simulator.generate_scenarios
"""
import numpy as np
import pandas as pd

from src.common.config import DATA_SCENARIOS, LABEL, load_config, save_json
from src.bootstrap.data import load_split
from src.simulator import drift_injector as inj


def make_windows(df: pd.DataFrame, window_size: int, seed: int) -> pd.DataFrame:
    """Shuffle once, then cut into full windows of `window_size` rows."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_windows = len(df) // window_size
    df = df.iloc[: n_windows * window_size].copy()
    df["window"] = np.arange(len(df)) // window_size
    return df


def strength_for(window: int, start: int, ramp: int) -> float:
    """0 before drift; ramps up linearly over `ramp` windows (0 = sudden)."""
    if window < start:
        return 0.0
    if not ramp:
        return 1.0
    return min(1.0, (window - start + 1) / ramp)


def build_scenario(name: str, s: dict, base: pd.DataFrame, start: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    pool_after = base[base["window"] >= start]          # rows available for label-shift resampling
    out = []
    for w_idx, w in base.groupby("window", sort=True):
        strength = strength_for(w_idx, start, s.get("ramp_windows", 0))
        if s["type"] == "none" or strength == 0:
            w2 = w.copy()
        elif s["type"] == "label_shift":
            w2 = inj.label_shift(w, strength, s, rng, pool_after)
        else:
            w2 = inj.INJECTORS[s["type"]](w, strength, s, rng)
        w2["injected"] = strength > 0
        out.append(w2)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    mon = load_config("monitoring")
    sc = load_config("scenarios")
    seed, start = sc["seed"], sc["drift_start"]

    base = make_windows(load_split("future_stream"), mon["window_size"], seed)
    n_windows = int(base["window"].nunique())
    print(f"Future stream: {len(base):,} rows -> {n_windows} windows of {mon['window_size']}")

    for name, s in sc["scenarios"].items():
        df = build_scenario(name, s, base, start, seed)
        folder = DATA_SCENARIOS / name
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / "windows.csv", index=False)

        truth = {
            "scenario": name,
            "drift_type": s["type"],
            "description": s.get("description", ""),
            "drift_start_window": start if s["type"] != "none" else None,
            "n_windows": n_windows,
            "settings": {k: v for k, v in s.items() if not k.startswith("expected")},
            "expected_diagnosis": s["expected_diagnosis"],
            "expected_action": s["expected_action"],
            "seed": seed,
        }
        save_json(truth, folder / "ground_truth.json")

        before = df[df["window"] < start][LABEL].mean()
        after = df[df["window"] >= start][LABEL].mean()
        print(f"  {name:<24} saved  (label share before {before:.2f}, after {after:.2f})")

    print("Next step:  python -m src.c4.monitoring.calibration")


if __name__ == "__main__":
    main()
