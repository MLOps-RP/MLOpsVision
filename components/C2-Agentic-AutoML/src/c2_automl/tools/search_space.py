"""Hyperparameter search spaces (trusted tool). Spaces are plain dicts so they can be adapted."""
import copy
import math
from typing import Optional

import optuna
from optuna.distributions import (BaseDistribution, CategoricalDistribution, FloatDistribution,
                                  IntDistribution)

from ..config import load_yaml


def load_base_space(family: str) -> dict:
    """The static search space for one family (from configs/search_spaces.yaml)."""
    return copy.deepcopy(load_yaml("search_spaces.yaml")[family])


def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """Ask Optuna for one value per hyperparameter inside the given ranges."""
    params = {}
    for name, r in space.items():
        if r["type"] == "int":
            params[name] = trial.suggest_int(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "float":
            params[name] = trial.suggest_float(name, r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "cat":
            params[name] = trial.suggest_categorical(name, r["choices"])
        else:
            raise ValueError(f"Unknown range type for {name}: {r['type']}")
    return params


# ---------- helpers for warm-starting a new round ----------
def to_distributions(space: dict) -> dict[str, BaseDistribution]:
    dists = {}
    for name, r in space.items():
        if r["type"] == "int":
            dists[name] = IntDistribution(r["low"], r["high"], log=r.get("log", False))
        elif r["type"] == "float":
            dists[name] = FloatDistribution(r["low"], r["high"], log=r.get("log", False))
        else:
            dists[name] = CategoricalDistribution(r["choices"])
    return dists


def in_space(params: dict, space: dict) -> bool:
    """True if params has exactly the space's parameters and every value is inside its range."""
    if set(params) != set(space):
        return False
    for name, r in space.items():
        v = params[name]
        if r["type"] == "cat":
            if v not in r["choices"]:
                return False
        elif not (r["low"] <= v <= r["high"]):
            return False
    return True


def clip_to_space(params: dict, space: dict) -> dict:
    """Move old best params inside the new ranges (used to warm-start the next round)."""
    clipped = {}
    for name, r in space.items():
        if name not in params:
            continue                      # new parameter (e.g. class_weight): let Optuna choose
        v = params[name]
        if r["type"] == "cat":
            if v in r["choices"]:
                clipped[name] = v
        else:
            v = min(max(v, r["low"]), r["high"])
            clipped[name] = int(round(v)) if r["type"] == "int" else float(v)
    return clipped


# ---------- Agent 2 Level 4: adaptive ranges ----------
def _fix_range(name: str, r: dict, limits: dict) -> None:
    """Clip a numeric range to its hard limits and make sure low < high."""
    lo_lim, hi_lim = limits.get(name, [-math.inf, math.inf])
    low = min(max(r["low"], lo_lim), hi_lim)
    high = min(max(r["high"], lo_lim), hi_lim)
    if r["type"] == "int":
        low, high = int(round(low)), int(round(high))
        if low >= high:
            if high - 1 >= lo_lim:
                low = high - 1
            else:
                high = low + 1
    elif low >= high:
        low = high * 0.5 if high > 0 else high - 1
        if low < lo_lim:                  # squeezed against the lower limit: open the range upwards
            low = lo_lim
            high = min(hi_lim, low * 2 if low > 0 else low + 1)
    r["low"], r["high"] = low, high


def _describe(name: str, side: str, old, new, reason: str) -> str:
    fmt = (lambda v: f"{v:g}") if isinstance(old, float) or isinstance(new, float) else str
    return f"{name}.{side} {fmt(old)} -> {fmt(new)} ({reason})"


def range_edge_hits(params: dict, space: dict, tolerance: float) -> list[tuple[str, str]]:
    """Numeric params whose best value lies in the outer `tolerance` share of the range."""
    hits = []
    for name, r in space.items():
        if r["type"] == "cat" or name not in params or r["high"] <= r["low"]:
            continue
        v, low, high = params[name], r["low"], r["high"]
        if r.get("log") and low > 0:
            pos = (math.log(v) - math.log(low)) / (math.log(high) - math.log(low))
        else:
            pos = (v - low) / (high - low)
        if pos <= tolerance:
            hits.append((name, "low"))
        elif pos >= 1 - tolerance:
            hits.append((name, "high"))
    return hits


def adapt_space(space: dict, actions: list[str], family: str, rules: dict,
                best_params: Optional[dict] = None,
                context: Optional[dict] = None) -> tuple[dict, list[str]]:
    """Return (new space, list of human-readable changes). Pure function: the input is not modified.

    1. Apply every action from Agent 1 (multiply ranges or add parameters).
    2. Range-edge rule: extend a range whose best value sits at its edge,
       but only for parameters that no action touched this round.
    3. Clip everything to the hard limits and keep low < high.
    """
    context = context or {}
    limits = rules["limits"]
    new = copy.deepcopy(space)
    changes, touched = [], set()

    for action in actions:
        effect = rules["actions"].get(action)
        if effect is None:
            continue                                    # e.g. drop_family: handled by Agent 1
        for name, spec in effect.get("add_params", {}).get(family, {}).items():
            if name in new:
                continue
            spec = dict(spec)
            if spec.get("high") == "auto":              # xgboost scale_pos_weight
                ratio = context.get("imbalance_ratio")
                if context.get("task_type") != "binary" or not ratio:
                    continue
                spec["high"] = round(min(1.0 / ratio, limits[name][1]), 2)
            new[name] = spec
            touched.add(name)
            changes.append(f"add {name} {spec.get('choices', [spec.get('low'), spec.get('high')])} ({action})")
        for name, mult in effect.items():
            if name == "add_params" or name not in new or new[name]["type"] == "cat":
                continue
            r, old = new[name], dict(new[name])
            for side in ("low", "high"):
                if side in mult:
                    r[side] = r[side] * mult[side]
            _fix_range(name, r, limits)
            touched.add(name)
            for side in ("low", "high"):
                if r[side] != old[side]:
                    changes.append(_describe(name, side, old[side], r[side], action))

    if best_params:
        edge = rules["edge"]
        for name, side in range_edge_hits(best_params, space, edge["tolerance"]):
            if name in touched:
                continue                                # diagnosis actions win over the edge rule
            r, old = new[name], new[name][side]
            if r["type"] == "int":
                r[side] = (max(old + 1, math.ceil(old * edge["factor"])) if side == "high"
                           else min(old - 1, math.floor(old / edge["factor"])))
            else:
                r[side] = old * edge["factor"] if side == "high" else old / edge["factor"]
            _fix_range(name, r, limits)
            if r[side] != old:
                changes.append(_describe(name, side, old, r[side], "best value at range edge"))
    return new, changes
