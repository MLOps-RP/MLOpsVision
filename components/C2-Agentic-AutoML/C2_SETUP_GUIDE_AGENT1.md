# C2 – Project Setup Guide (Starting with Agent 1)

> This guide covers **Roadmap Step 0** (project setup) and builds **Agent 1 – Adaptive Model Generation, Levels 1–3** (§13.2 of [C2_IMPLEMENTATION_GUIDE.md](C2_IMPLEMENTATION_GUIDE.md)).
> Tools: **Python + `venv` + `pip`** (no uv). Written for **macOS (Apple Silicon)**; Windows/Linux differences are noted.
>
> At the end you will have:
> - a working Python project,
> - dummy C1 data (Adult, Pima Diabetes, imbalanced Adult),
> - Agent 1 that **profiles the dataset → picks model families → trains them → diagnoses problems → plans the next round**,
> - 13 passing unit tests.

---

## Contents

- [Part A – Project Setup](#part-a--project-setup)
  - [A1. Install the prerequisites](#a1-install-the-prerequisites)
  - [A2. Create the folder structure](#a2-create-the-folder-structure)
  - [A3. Create the virtual environment](#a3-create-the-virtual-environment)
  - [A4. Install the libraries](#a4-install-the-libraries)
  - [A5. Make the code importable (`pyproject.toml`)](#a5-make-the-code-importable-pyprojecttoml)
  - [A6. `.gitignore`](#a6-gitignore)
  - [A7. Connect VS Code and Jupyter](#a7-connect-vs-code-and-jupyter)
  - [A8. Check that everything works](#a8-check-that-everything-works)
- [Part B – Foundation Agent 1 Needs](#part-b--foundation-agent-1-needs)
  - [B1. Config files](#b1-config-files)
  - [B2. Config loader](#b2-config-loader)
  - [B3. Data contracts (schemas)](#b3-data-contracts-schemas)
  - [B4. Dummy C1 output](#b4-dummy-c1-output)
  - [B5. Trusted tools: models, training, profile, signals](#b5-trusted-tools-models-training-profile-signals)
- [Part C – Agent 1, Level by Level](#part-c--agent-1-level-by-level)
  - [C1. Level 1 – Eligible families](#c1-level-1--eligible-families)
  - [C2. Level 2 – Profile-aware selection](#c2-level-2--profile-aware-selection)
  - [C3. Level 3 – Diagnosis + plan](#c3-level-3--diagnosis--plan)
  - [C4. The full Agent 1 file](#c4-the-full-agent-1-file)
  - [C5. Unit tests](#c5-unit-tests)
  - [C6. Run Agent 1 on real data](#c6-run-agent-1-on-real-data)
- [Part D – Commit, Troubleshooting, Next Steps](#part-d--commit-troubleshooting-next-steps)

---

## Part A – Project Setup

### A1. Install the prerequisites

| Tool | Why | Install (macOS) |
|---|---|---|
| Homebrew | Package manager for Mac | See https://brew.sh |
| **Python 3.11** | Project language (3.12 / 3.13 also work) | `brew install python@3.11` |
| **libomp** | ⚠️ **Required** by XGBoost and LightGBM on Mac. Without it, `import xgboost` fails with *"libomp.dylib could not be loaded"* | `brew install libomp` |
| Git | Version control | `brew install git` (or Xcode tools) |
| VS Code | Editor | + extensions **Python**, **Jupyter**, **YAML** |

Check:

```bash
python3.11 --version     # Python 3.11.x
brew list libomp         # should list files, not an error
```

> **Windows:** install Python 3.11 from python.org and tick *"Add to PATH"*. libomp is not needed.
> **Linux:** `sudo apt install python3.11 python3.11-venv libgomp1`.

### A2. Create the folder structure

From the repository root (`MLOpsVision/`):

```bash
cd components/C2-Agentic-AutoML

mkdir -p configs data/raw data/c1_output \
         src/c2_automl/io src/c2_automl/tools src/c2_automl/agents \
         scripts notebooks outputs tests

touch src/c2_automl/__init__.py src/c2_automl/io/__init__.py \
      src/c2_automl/tools/__init__.py src/c2_automl/agents/__init__.py
```

Result:

```
C2-Agentic-AutoML/
├── configs/                 # YAML settings (no hard-coded numbers in code)
├── data/
│   ├── raw/                 # downloaded files (not in git)
│   └── c1_output/           # dummy C1 output (not in git)
├── src/c2_automl/           # ALL real code (a Python package)
│   ├── io/                  # reading C1 data
│   ├── tools/               # trusted computation: training, metrics, signals
│   └── agents/              # agent decisions
├── scripts/                 # things you run from the terminal
├── notebooks/               # exploration + thesis plots only
├── outputs/                 # logs, models (not in git)
└── tests/                   # pytest
```

### A3. Create the virtual environment

A virtual environment keeps this project's libraries separate from the rest of your computer.

```bash
# still inside components/C2-Agentic-AutoML
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Your terminal prompt now starts with `(.venv)`.
**Run `source .venv/bin/activate` every time you open a new terminal for this project.**

### A4. Install the libraries

Create **`requirements.txt`** (what the code needs):

```text
pandas>=2.2
numpy>=1.26
pyarrow>=15.0
openpyxl>=3.1
scikit-learn>=1.5
xgboost>=2.1
lightgbm>=4.5
catboost>=1.2.7
optuna>=4.0
pydantic>=2.7
pyyaml>=6.0
```

Create **`requirements-dev.txt`** (tools for you, not for the code):

```text
pytest>=8.0
jupyterlab>=4.0
ipykernel>=6.29
matplotlib>=3.8
```

Install:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

> Libraries for later steps (do **not** install yet): `langgraph`, `langchain-core` (Step 3/5), `mlflow`, `shap` (Step 4), `fastapi`, `uvicorn` (Step 8). Add them to `requirements.txt` when you reach those steps.

### A5. Make the code importable (`pyproject.toml`)

This lets scripts, tests and notebooks use `from c2_automl... import ...` from any folder.

Create **`pyproject.toml`**:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "c2-automl"
version = "0.1.0"
description = "C2 - Agentic AutoML Model Generation and Selection Engine"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Install your own package in **editable** mode (code changes apply immediately, no reinstall needed):

```bash
pip install -e .
```

### A6. `.gitignore`

Create **`components/C2-Agentic-AutoML/.gitignore`**:

```text
.venv/
__pycache__/
*.egg-info/
.ipynb_checkpoints/
.pytest_cache/
.env
.DS_Store

data/raw/
data/c1_output/
outputs/
catboost_info/
mlruns/
```

> Data and outputs are **not** committed. Anyone can rebuild them with `python scripts/build_dummy_data.py`.

### A7. Connect VS Code and Jupyter

1. In VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `components/C2-Agentic-AutoML/.venv/bin/python`.
2. Register the environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name c2-automl --display-name "Python (C2 AutoML)"
   ```
3. In any notebook, pick the kernel **"Python (C2 AutoML)"** (top right).

### A8. Check that everything works

```bash
python -c "import pandas, sklearn, xgboost, lightgbm, catboost, optuna, pydantic; print('all imports OK')"
```

✅ If you see `all imports OK`, the setup is done. (If XGBoost fails → `brew install libomp`.)

Commit:

```bash
git add components/C2-Agentic-AutoML
git commit -m "C2: project setup (venv, requirements, package skeleton)"
```

---

## Part B – Foundation Agent 1 Needs

Agent 1 makes **decisions**. It needs some **trusted tools** to give it facts (dataset profile, training results, signals). We build those first.

```
dummy C1 data ──► task_profile ──► Agent 1: select families
                                         │
                        train_eval (CV) ◄┘
                                │
                         signals.py ──► Agent 1: diagnose ──► Agent 1: plan next round
```

### B1. Config files

All thresholds live in YAML, not in code. You will tune and justify them in your thesis.

**`configs/default.yaml`**

```yaml
seed: 42
cv_folds: 5
test_size: 0.2

# Main metric per task type (scikit-learn scorer names; higher is always better)
metrics:
  binary: roc_auc
  multiclass: f1_macro
  regression: neg_root_mean_squared_error

# Used by Agent 1 to describe the dataset (task profile)
profile:
  small_rows: 1000          # fewer rows than this = "small" dataset
  large_rows: 200000        # more rows than this = "large" dataset
  many_categorical_ratio: 0.3

# Used later by Agent 1 Level 4 and Agent 2
budget:
  total_trials: 150
  max_rounds: 4
```

**`configs/model_registry.yaml`** – which model families exist and which tasks they support:

```yaml
families:
  logistic_regression:
    tasks: [binary, multiclass]
    role: simple_baseline
  ridge:
    tasks: [regression]
    role: simple_baseline
  random_forest:
    tasks: [binary, multiclass, regression]
  xgboost:
    tasks: [binary, multiclass, regression]
  lightgbm:
    tasks: [binary, multiclass, regression]
  catboost:
    tasks: [binary, multiclass, regression]
    good_for: many_categoricals
```

**`configs/diagnosis_rules.yaml`** – Agent 1's rules (§10.5 of the main guide):

```yaml
# Starting thresholds for the binary metric (ROC-AUC).
# Tuning and justifying these values is part of the research.

overfitting:
  min_gap: 0.05             # train score - validation score
  min_train_score: 0.95

underfitting:
  max_gap: 0.02
  max_train_score: 0.80

class_imbalance:
  max_imbalance_ratio: 0.20 # minority class size / majority class size
  max_minority_recall: 0.50

weak_features:
  max_lift: 0.05            # validation score - dummy-model score

unsuitable_model_family:
  min_gap_to_best: 0.03     # this family is this much worse than the best family

# What Agent 1 asks for after each diagnosis
actions:
  overfitting: [shrink_depth_range, increase_regularization]
  underfitting: [widen_depth_range, increase_n_estimators]
  class_imbalance: [enable_class_weights]
  weak_features: [refer_features_to_c1]
  unsuitable_model_family: [drop_family]
  healthy: []
```

### B2. Config loader

**`src/c2_automl/config.py`**

```python
from pathlib import Path

import yaml

# src/c2_automl/config.py -> parents[2] is the component root folder
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def load_default_config() -> dict:
    return load_yaml("default.yaml")
```

### B3. Data contracts (schemas)

These Pydantic models are the "language" between C1, C2 and C3. **Share this file with your C1 and C3 teammates.**

**`src/c2_automl/schemas.py`**

```python
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class C1Output(BaseModel):
    """What C1 gives to C2. Simulated by io/dummy_c1.py until C1 is ready."""
    dataset_id: str
    task_type: TaskType
    target: str
    data_dir: str
    feature_types: dict[str, str]                  # feature -> "numeric" | "categorical"
    fqs: dict[str, float]                          # feature -> quality score 0..1
    class_distribution: Optional[dict[str, float]] = None
    positive_class: Optional[str] = None
    decision_log: list[str] = []


class TaskProfile(BaseModel):
    n_rows: int
    n_features: int
    n_categorical: int
    categorical_ratio: float
    size_class: str                                # "small" | "medium" | "large"
    metric: str
    n_classes: Optional[int] = None
    imbalance_ratio: Optional[float] = None        # minority / majority


class DiagnosisLabel(str, Enum):
    UNDERFITTING = "underfitting"
    OVERFITTING = "overfitting"
    IMBALANCE = "class_imbalance"
    WEAK_FEATURES = "weak_features"
    UNSUITABLE_FAMILY = "unsuitable_model_family"
    HEALTHY = "healthy"


class Diagnosis(BaseModel):
    model_family: str
    labels: list[DiagnosisLabel]
    evidence: dict[str, float]
    reasons: list[str]                             # human-readable, one per label
    confidence: float = Field(default=1.0, ge=0, le=1)


class PlanDecision(BaseModel):
    """Agent 1's plan for the next round."""
    families_next_round: list[str]
    actions: dict[str, list[str]]                  # family -> actions for Agent 2
    referrals_to_c1: list[str] = []
    stop: bool = False
    decided_by: str = "rules"                      # "rules" | "rules+llm" | "fallback"
    rationale: str = ""


class ReasoningEntry(BaseModel):
    round: int
    agent: str
    step: str
    decision: dict
    evidence: dict = {}
    decided_by: str = "rules"
    rationale: str = ""
```

### B4. Dummy C1 output

C1 is not ready, so this file **simulates** it: download a dataset, split train/test, basic cleaning, and a fake Feature Quality Score (mutual information).

**`src/c2_automl/io/dummy_c1.py`**

```python
"""Simulates C1's output so C2 can be built before C1 is ready.

Only this file knows how C1 data is produced. When C1 is ready, replace it
with a real C1 client and keep the same C1Output contract.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split

from ..config import DATA_DIR, load_default_config
from ..schemas import C1Output, TaskType

# Public datasets from OpenML (downloaded automatically, then cached)
OPENML_DATASETS = {
    "adult": {"name": "adult", "version": 2, "task": TaskType.BINARY},
    "diabetes": {"name": "diabetes", "version": 1, "task": TaskType.BINARY},
}


def load_table(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file (for Kaggle datasets you download yourself)."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _feature_types(X: pd.DataFrame) -> dict[str, str]:
    return {c: "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical"
            for c in X.columns}


def _basic_clean(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_types: dict[str, str]):
    """Very simple cleaning. The real, adaptive cleaning is C1's job."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for c, kind in feature_types.items():
        if kind == "numeric":
            median = X_train[c].median()
            X_train[c] = X_train[c].fillna(median)
            X_test[c] = X_test[c].fillna(median)
        else:
            train_vals = X_train[c].astype("object").fillna("missing").astype(str)
            test_vals = X_test[c].astype("object").fillna("missing").astype(str)
            categories = sorted(train_vals.unique())
            # unseen test categories become -1
            X_train[c] = pd.Categorical(train_vals, categories=categories).codes.astype("int32")
            X_test[c] = pd.Categorical(test_vals, categories=categories).codes.astype("int32")
    return X_train, X_test


def _encode_target(y: pd.Series, task_type: TaskType, positive_class: Optional[str]):
    """Binary: positive class -> 1 (default = minority class). Multiclass: 0..k-1."""
    if task_type == TaskType.REGRESSION:
        return y.astype(float), None
    y = y.astype(str)
    if task_type == TaskType.BINARY:
        positive_class = positive_class or y.value_counts().idxmin()
        return (y == positive_class).astype(int), positive_class
    classes = sorted(y.unique())
    return y.map({c: i for i, c in enumerate(classes)}).astype(int), None


def make_imbalanced(X: pd.DataFrame, y: pd.Series, minority_share: float, seed: int):
    """Downsample class 1 so it is only `minority_share` of all rows (for testing imbalance)."""
    rng = np.random.default_rng(seed)
    majority_idx = y.index[y == 0]
    minority_idx = y.index[y == 1]
    n_keep = int(minority_share / (1 - minority_share) * len(majority_idx))
    keep = rng.choice(minority_idx, size=min(n_keep, len(minority_idx)), replace=False)
    idx = majority_idx.append(pd.Index(keep))
    return X.loc[idx], y.loc[idx]


def build_dummy_c1(df: pd.DataFrame, target: str, task_type: TaskType, dataset_id: str,
                   positive_class: Optional[str] = None,
                   minority_share: Optional[float] = None) -> C1Output:
    cfg = load_default_config()
    seed = cfg["seed"]
    out_dir = DATA_DIR / "c1_output" / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    X, y = df.drop(columns=[target]), df[target]
    y, positive_class = _encode_target(y, task_type, positive_class)
    if minority_share is not None:
        X, y = make_imbalanced(X, y, minority_share, seed)

    stratify = None if task_type == TaskType.REGRESSION else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=stratify, random_state=seed)

    feature_types = _feature_types(X_train)
    X_train, X_test = _basic_clean(X_train, X_test, feature_types)

    # Fake Feature Quality Score = mutual information, scaled to 0..1
    discrete = [feature_types[c] == "categorical" for c in X_train.columns]
    mi_fn = mutual_info_regression if task_type == TaskType.REGRESSION else mutual_info_classif
    mi = mi_fn(X_train, y_train, discrete_features=discrete, random_state=seed)
    fqs = {c: round(float(v / (mi.max() + 1e-12)), 4) for c, v in zip(X_train.columns, mi)}

    X_train.to_parquet(out_dir / "X_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_train.to_frame(target).to_parquet(out_dir / "y_train.parquet")
    y_test.to_frame(target).to_parquet(out_dir / "y_test.parquet")

    class_distribution = None
    if task_type != TaskType.REGRESSION:
        class_distribution = {str(k): round(float(v), 4)
                              for k, v in y_train.value_counts(normalize=True).items()}

    c1 = C1Output(
        dataset_id=dataset_id, task_type=task_type, target=target, data_dir=str(out_dir),
        feature_types=feature_types, fqs=fqs, class_distribution=class_distribution,
        positive_class=positive_class,
        decision_log=["DUMMY C1: median imputation, ordinal encoding, mutual-information FQS"],
    )
    (out_dir / "manifest.json").write_text(c1.model_dump_json(indent=2))
    return c1


def build_from_openml(key: str, dataset_id: Optional[str] = None,
                      minority_share: Optional[float] = None) -> C1Output:
    spec = OPENML_DATASETS[key]
    data = fetch_openml(name=spec["name"], version=spec["version"], as_frame=True)
    df = data.frame
    target = data.target.name
    return build_dummy_c1(df, target, spec["task"], dataset_id or key,
                          minority_share=minority_share)


def load_c1_output(dataset_id: str) -> C1Output:
    path = DATA_DIR / "c1_output" / dataset_id / "manifest.json"
    return C1Output.model_validate_json(path.read_text())


def load_split(c1: C1Output, split: str = "train") -> tuple[pd.DataFrame, pd.Series]:
    """split = "train" or "test". C2 must only tune on "train"."""
    d = Path(c1.data_dir)
    X = pd.read_parquet(d / f"X_{split}.parquet")
    y = pd.read_parquet(d / f"y_{split}.parquet")[c1.target]
    return X, y
```

**`scripts/build_dummy_data.py`**

```python
"""Create dummy C1 outputs for development (run once)."""
from c2_automl.io.dummy_c1 import build_from_openml

if __name__ == "__main__":
    for key in ["adult", "diabetes"]:
        c1 = build_from_openml(key)
        print(f"built {c1.dataset_id}: {c1.class_distribution}")

    # Same Adult data, but positive class is only 3% -> tests the imbalance diagnosis
    c1 = build_from_openml("adult", dataset_id="adult_imbalanced", minority_share=0.03)
    print(f"built {c1.dataset_id}: {c1.class_distribution}")
```

Run it (needs internet the first time; OpenML files are cached afterwards):

```bash
python scripts/build_dummy_data.py
```

Expected output:

```text
built adult: {'0': 0.7607, '1': 0.2393}
built diabetes: {'0': 0.6515, '1': 0.3485}
built adult_imbalanced: {'0': 0.97, '1': 0.03}
```

Each dataset now has a folder like `data/c1_output/diabetes/` containing `X_train.parquet`, `y_train.parquet`, `X_test.parquet`, `y_test.parquet` and `manifest.json` (the `C1Output`).

> **Using your own Kaggle CSV/Excel instead:**
> ```python
> from c2_automl.io.dummy_c1 import load_table, build_dummy_c1
> from c2_automl.schemas import TaskType
> df = load_table("data/raw/loan_default.csv")
> build_dummy_c1(df, target="default", task_type=TaskType.BINARY, dataset_id="loan")
> ```

### B5. Trusted tools: models, training, profile, signals

These do the **computation**. Agent 1 only **reads** their results (the "bounded orchestrator" rule, §9 of the main guide).

**`src/c2_automl/tools/models.py`** – the only place that creates models:

```python
"""Model factory: the only place that creates model objects (trusted tool)."""
from typing import Optional

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ..schemas import TaskType


def make_model(family: str, task_type: TaskType, params: Optional[dict] = None, seed: int = 42):
    is_clf = task_type != TaskType.REGRESSION
    p = dict(params or {})

    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**{"max_iter": 2000, **p}))
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if family == "random_forest":
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, **p})
    if family == "xgboost":
        cls = XGBClassifier if is_clf else XGBRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, "verbosity": 0, **p})
    if family == "lightgbm":
        cls = LGBMClassifier if is_clf else LGBMRegressor
        return cls(**{"n_jobs": -1, "random_state": seed, "verbose": -1, **p})
    if family == "catboost":
        cls = CatBoostClassifier if is_clf else CatBoostRegressor
        return cls(**{"random_seed": seed, "verbose": 0, "allow_writing_files": False, **p})
    raise ValueError(f"Unknown model family: {family}")
```

**`src/c2_automl/tools/train_eval.py`** – cross-validation that keeps **train and validation scores** (needed to detect overfitting) and **out-of-fold predictions** (needed for minority-class recall):

```python
"""Cross-validated training (trusted tool). Agent 1 reads these results; it never trains models itself."""
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..schemas import TaskType
from .models import make_model


def make_splitter(task_type: TaskType, folds: int, seed: int):
    if task_type == TaskType.REGRESSION:
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate_family(family: str, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                          metric: str, folds: int = 5, seed: int = 42,
                          params: Optional[dict] = None) -> dict:
    """Train one model family with CV. Returns train/validation scores and out-of-fold predictions."""
    scorer = get_scorer(metric)
    splitter = make_splitter(task_type, folds, seed)
    train_scores, val_scores, fit_times = [], [], []
    oof_pred = np.zeros(len(y), dtype=y.dtype)

    for train_idx, val_idx in splitter.split(X, y):
        model = make_model(family, task_type, params, seed)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - start)
        train_scores.append(scorer(model, X_tr, y_tr))
        val_scores.append(scorer(model, X_va, y_va))
        oof_pred[val_idx] = np.ravel(model.predict(X_va))

    return {
        "family": family,
        "params": params or {},
        "train_scores": [float(s) for s in train_scores],
        "val_scores": [float(s) for s in val_scores],
        "mean_train": float(np.mean(train_scores)),
        "mean_val": float(np.mean(val_scores)),
        "cv_std": float(np.std(val_scores)),
        "fit_time_s": float(np.sum(fit_times)),
        "oof_pred": oof_pred,
    }


def baseline_score(X: pd.DataFrame, y: pd.Series, task_type: TaskType, metric: str,
                   folds: int = 5, seed: int = 42) -> float:
    """Score of a 'no-skill' model. Used to measure lift (are the features useful at all?)."""
    dummy = DummyRegressor() if task_type == TaskType.REGRESSION else DummyClassifier(strategy="prior")
    scores = cross_val_score(dummy, X, y, scoring=metric, cv=make_splitter(task_type, folds, seed))
    return float(np.mean(scores))
```

**`src/c2_automl/tools/task_profile.py`** – describes the dataset for Agent 1:

```python
import pandas as pd

from ..schemas import C1Output, TaskProfile, TaskType


def profile_task(c1: C1Output, X: pd.DataFrame, y: pd.Series, cfg: dict) -> TaskProfile:
    """Describe the dataset so Agent 1 can choose suitable model families."""
    p = cfg["profile"]
    n_rows, n_features = X.shape
    n_categorical = sum(1 for t in c1.feature_types.values() if t == "categorical")

    if n_rows < p["small_rows"]:
        size_class = "small"
    elif n_rows > p["large_rows"]:
        size_class = "large"
    else:
        size_class = "medium"

    n_classes = imbalance_ratio = None
    if c1.task_type != TaskType.REGRESSION:
        counts = y.value_counts()
        n_classes = int(len(counts))
        imbalance_ratio = round(float(counts.min() / counts.max()), 4)

    return TaskProfile(
        n_rows=n_rows, n_features=n_features, n_categorical=n_categorical,
        categorical_ratio=round(n_categorical / max(n_features, 1), 4),
        size_class=size_class, metric=cfg["metrics"][c1.task_type.value],
        n_classes=n_classes, imbalance_ratio=imbalance_ratio,
    )
```

**`src/c2_automl/tools/signals.py`** – the evidence numbers Agent 1 diagnoses from:

```python
"""Evidence signals for diagnosis (trusted tool). Pure numbers, no decisions."""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from ..schemas import TaskType


def compute_signals(result: dict, y: pd.Series, task_type: TaskType, baseline: float,
                    best_val_all_families: float, fqs: dict[str, float]) -> dict[str, float]:
    fqs_values = np.array(list(fqs.values())) if fqs else np.array([1.0])
    signals = {
        "mean_train": result["mean_train"],
        "mean_val": result["mean_val"],
        "gap": result["mean_train"] - result["mean_val"],
        "cv_std": result["cv_std"],
        "baseline": baseline,
        "lift_over_baseline": result["mean_val"] - baseline,
        "gap_to_best_family": best_val_all_families - result["mean_val"],
        "mean_fqs": float(fqs_values.mean()),
        "low_fqs_fraction": float((fqs_values < 0.1).mean()),
    }
    if task_type != TaskType.REGRESSION:
        signals.update(_class_signals(y, result["oof_pred"]))
    return {k: round(float(v), 4) for k, v in signals.items()}


def _class_signals(y: pd.Series, oof_pred: np.ndarray) -> dict[str, float]:
    counts = y.value_counts()
    minority_class = counts.idxmin()
    labels = list(counts.index)
    recalls = recall_score(y, oof_pred, labels=labels, average=None, zero_division=0)
    minority_recall = float(recalls[labels.index(minority_class)])
    return {
        "imbalance_ratio": float(counts.min() / counts.max()),
        "minority_recall": minority_recall,
        "min_class_recall": float(np.min(recalls)),
    }
```

| Signal | Meaning | Used for |
|---|---|---|
| `gap` | train score − validation score | overfitting / underfitting |
| `cv_std` | how much the score changes between folds | instability (later) |
| `lift_over_baseline` | how much better than a dummy model | weak features |
| `gap_to_best_family` | how far behind the best family | unsuitable family |
| `imbalance_ratio`, `minority_recall` | class balance and how well the rare class is found | class imbalance |
| `mean_fqs`, `low_fqs_fraction` | feature quality from C1 | supporting evidence for weak features |

---

## Part C – Agent 1, Level by Level

Agent 1 is built in **one file**, one level at a time. Each level adds functions; the earlier ones are not changed.

### C1. Level 1 – Eligible families

**Goal:** list every model family that supports the task, minus any families C3 has blocked.

```python
def eligible_families(task_type: str, registry: dict, blocked: list[str] = ()) -> list[str]:
    """All model families that support this task type, minus families blocked by C3."""
    return [name for name, spec in registry["families"].items()
            if task_type in spec["tasks"] and name not in blocked]
```

✅ **Check:** `eligible_families("binary", registry)` → `['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']`.

### C2. Level 2 – Profile-aware selection

**Goal:** the first round's model families depend on the dataset (size, share of categorical features), and the reason is logged.

```python
def select_initial_families(profile: TaskProfile, c1: C1Output, registry: dict, cfg: dict,
                            blocked: list[str] = ()) -> tuple[list[str], ReasoningEntry]:
    """Choose round-1 families from the dataset profile (guide §10.3)."""
    eligible = eligible_families(c1.task_type.value, registry, blocked)
    baseline_family = "ridge" if c1.task_type.value == "regression" else "logistic_regression"
    reasons = []

    if profile.size_class == "small":
        wanted = ["random_forest", "catboost", baseline_family]
        reasons.append(f"small dataset ({profile.n_rows} rows): boosting overfits easily, "
                       "prefer random forest, catboost and a simple baseline")
    elif profile.size_class == "large":
        wanted = ["lightgbm", "xgboost"]
        reasons.append(f"large dataset ({profile.n_rows} rows): start with the fastest boosters")
    else:
        wanted = ["random_forest", "xgboost", "lightgbm"]
        reasons.append(f"medium dataset ({profile.n_rows} rows): default tree-based families")

    if profile.categorical_ratio > cfg["profile"]["many_categorical_ratio"] and "catboost" not in wanted:
        wanted.append("catboost")
        reasons.append(f"{profile.categorical_ratio:.0%} categorical features: add catboost")

    families = [f for f in wanted if f in eligible]
    if blocked:
        reasons.append(f"blocked by C3: {list(blocked)}")
    if not families:
        families = eligible
        reasons.append("no rule matched an eligible family: fall back to all eligible families")

    entry = ReasoningEntry(round=1, agent="agent1", step="select_initial_families",
                           decision={"families": families}, evidence=profile.model_dump(),
                           rationale="; ".join(reasons))
    return families, entry
```

| Dataset | Rule | Families chosen |
|---|---|---|
| Pima Diabetes (614 train rows) | small | random_forest, catboost, logistic_regression |
| Adult (39k rows, 57% categorical) | medium + many categoricals | random_forest, xgboost, lightgbm, catboost |
| > 200k rows | large | lightgbm, xgboost |

✅ **Check:** Pima and Adult get **different** lists, and `entry.rationale` explains why.

### C3. Level 3 – Diagnosis + plan

**Goal:** turn training results into **named problems backed by numbers**, then decide the next round. This is the start of your **core novelty**.

`diagnose_family()` – applies the YAML rules to one family's signals:

```python
def diagnose_family(family: str, signals: dict[str, float], rules: dict) -> Diagnosis:
    """Turn evidence signals into named failure modes using the YAML rules."""
    labels, reasons = [], []

    r = rules["overfitting"]
    if signals["gap"] > r["min_gap"] and signals["mean_train"] > r["min_train_score"]:
        labels.append(DiagnosisLabel.OVERFITTING)
        reasons.append(f"train-validation gap {signals['gap']:.3f} > {r['min_gap']} "
                       f"with train score {signals['mean_train']:.3f}")

    r = rules["underfitting"]
    if signals["gap"] < r["max_gap"] and signals["mean_train"] < r["max_train_score"]:
        labels.append(DiagnosisLabel.UNDERFITTING)
        reasons.append(f"low train score {signals['mean_train']:.3f} and small gap {signals['gap']:.3f}")

    r = rules["class_imbalance"]
    if ("imbalance_ratio" in signals and signals["imbalance_ratio"] < r["max_imbalance_ratio"]
            and signals["minority_recall"] < r["max_minority_recall"]):
        labels.append(DiagnosisLabel.IMBALANCE)
        reasons.append(f"imbalance ratio {signals['imbalance_ratio']:.3f} and "
                       f"minority recall {signals['minority_recall']:.3f}")

    r = rules["weak_features"]
    if signals["lift_over_baseline"] < r["max_lift"]:
        labels.append(DiagnosisLabel.WEAK_FEATURES)
        reasons.append(f"only {signals['lift_over_baseline']:.3f} better than a dummy model "
                       f"(mean FQS {signals['mean_fqs']:.2f})")

    r = rules["unsuitable_model_family"]
    if signals["gap_to_best_family"] >= r["min_gap_to_best"]:
        labels.append(DiagnosisLabel.UNSUITABLE_FAMILY)
        reasons.append(f"{signals['gap_to_best_family']:.3f} below the best family")

    if not labels:
        labels.append(DiagnosisLabel.HEALTHY)
        reasons.append("no rule triggered")

    return Diagnosis(model_family=family, labels=labels, evidence=signals, reasons=reasons)
```

`plan_next_round()` – combines all diagnoses into one plan for Agent 2:

```python
def plan_next_round(diagnoses: list[Diagnosis], rules: dict, round_no: int
                    ) -> tuple[PlanDecision, ReasoningEntry]:
    """Decide which families to keep and which search changes Agent 2 should make."""
    action_map = rules["actions"]
    keep, actions, referrals, notes = [], {}, [], []

    for d in diagnoses:
        labels = {lbl.value for lbl in d.labels}
        if DiagnosisLabel.UNSUITABLE_FAMILY.value in labels:
            notes.append(f"drop {d.model_family}")
            continue
        keep.append(d.model_family)
        actions[d.model_family] = sorted({a for lbl in labels for a in action_map[lbl]
                                          if a not in ("drop_family", "refer_features_to_c1")})

    if not keep:  # never drop everything: keep the best family
        best = min(diagnoses, key=lambda d: d.evidence["gap_to_best_family"])
        keep, actions[best.model_family] = [best.model_family], []
        notes.append(f"all families looked unsuitable; keep best: {best.model_family}")

    all_weak = all(DiagnosisLabel.WEAK_FEATURES in d.labels for d in diagnoses)
    if all_weak:
        referrals.append("weak_features: no model family beats the dummy baseline; "
                         "feature engineering needed in C1")
    if any(DiagnosisLabel.IMBALANCE in d.labels for d in diagnoses):
        referrals.append("class_imbalance: consider resampling (e.g. SMOTE/ADASYN) in C1")

    plan = PlanDecision(families_next_round=keep, actions=actions, referrals_to_c1=referrals,
                        stop=all_weak, rationale="; ".join(notes) or "keep all families")
    entry = ReasoningEntry(round=round_no, agent="agent1", step="plan_next_round",
                           decision=plan.model_dump(),
                           evidence={d.model_family: [l.value for l in d.labels] for d in diagnoses},
                           rationale=plan.rationale)
    return plan, entry
```

What the plan contains:

| Field | Meaning | Who uses it |
|---|---|---|
| `families_next_round` | families to keep (unsuitable ones dropped) | Agent 1, round 2 |
| `actions` | per family, e.g. `shrink_depth_range` | **Agent 2** (adaptive search ranges, Step 4) |
| `referrals_to_c1` | feature/imbalance issues for C1 | C1 (later) |
| `stop` | stop searching (e.g. every family has weak features) | LangGraph loop (Step 3/4) |
| `decided_by` | `rules` now; `rules+llm` after Step 5 | reasoning log / audit |

### C4. The full Agent 1 file

Put Levels 1–3 together in **`src/c2_automl/agents/agent1_generation.py`**:

```python
"""Agent 1 - Adaptive Model Generation.

Level 1: eligible families from the registry
Level 2: profile-aware initial selection
Level 3: rule-based diagnosis + plan for the next round

All functions are pure (no training here), so they are easy to unit test.
"""
from ..schemas import (C1Output, Diagnosis, DiagnosisLabel, PlanDecision, ReasoningEntry,
                       TaskProfile)


# ---------- Level 1 ----------
def eligible_families(task_type: str, registry: dict, blocked: list[str] = ()) -> list[str]:
    """All model families that support this task type, minus families blocked by C3."""
    return [name for name, spec in registry["families"].items()
            if task_type in spec["tasks"] and name not in blocked]


# ---------- Level 2 ----------
def select_initial_families(profile: TaskProfile, c1: C1Output, registry: dict, cfg: dict,
                            blocked: list[str] = ()) -> tuple[list[str], ReasoningEntry]:
    """Choose round-1 families from the dataset profile (guide §10.3)."""
    eligible = eligible_families(c1.task_type.value, registry, blocked)
    baseline_family = "ridge" if c1.task_type.value == "regression" else "logistic_regression"
    reasons = []

    if profile.size_class == "small":
        wanted = ["random_forest", "catboost", baseline_family]
        reasons.append(f"small dataset ({profile.n_rows} rows): boosting overfits easily, "
                       "prefer random forest, catboost and a simple baseline")
    elif profile.size_class == "large":
        wanted = ["lightgbm", "xgboost"]
        reasons.append(f"large dataset ({profile.n_rows} rows): start with the fastest boosters")
    else:
        wanted = ["random_forest", "xgboost", "lightgbm"]
        reasons.append(f"medium dataset ({profile.n_rows} rows): default tree-based families")

    if profile.categorical_ratio > cfg["profile"]["many_categorical_ratio"] and "catboost" not in wanted:
        wanted.append("catboost")
        reasons.append(f"{profile.categorical_ratio:.0%} categorical features: add catboost")

    families = [f for f in wanted if f in eligible]
    if blocked:
        reasons.append(f"blocked by C3: {list(blocked)}")
    if not families:
        families = eligible
        reasons.append("no rule matched an eligible family: fall back to all eligible families")

    entry = ReasoningEntry(round=1, agent="agent1", step="select_initial_families",
                           decision={"families": families}, evidence=profile.model_dump(),
                           rationale="; ".join(reasons))
    return families, entry


# ---------- Level 3 ----------
def diagnose_family(family: str, signals: dict[str, float], rules: dict) -> Diagnosis:
    """Turn evidence signals into named failure modes using the YAML rules."""
    labels, reasons = [], []

    r = rules["overfitting"]
    if signals["gap"] > r["min_gap"] and signals["mean_train"] > r["min_train_score"]:
        labels.append(DiagnosisLabel.OVERFITTING)
        reasons.append(f"train-validation gap {signals['gap']:.3f} > {r['min_gap']} "
                       f"with train score {signals['mean_train']:.3f}")

    r = rules["underfitting"]
    if signals["gap"] < r["max_gap"] and signals["mean_train"] < r["max_train_score"]:
        labels.append(DiagnosisLabel.UNDERFITTING)
        reasons.append(f"low train score {signals['mean_train']:.3f} and small gap {signals['gap']:.3f}")

    r = rules["class_imbalance"]
    if ("imbalance_ratio" in signals and signals["imbalance_ratio"] < r["max_imbalance_ratio"]
            and signals["minority_recall"] < r["max_minority_recall"]):
        labels.append(DiagnosisLabel.IMBALANCE)
        reasons.append(f"imbalance ratio {signals['imbalance_ratio']:.3f} and "
                       f"minority recall {signals['minority_recall']:.3f}")

    r = rules["weak_features"]
    if signals["lift_over_baseline"] < r["max_lift"]:
        labels.append(DiagnosisLabel.WEAK_FEATURES)
        reasons.append(f"only {signals['lift_over_baseline']:.3f} better than a dummy model "
                       f"(mean FQS {signals['mean_fqs']:.2f})")

    r = rules["unsuitable_model_family"]
    if signals["gap_to_best_family"] >= r["min_gap_to_best"]:
        labels.append(DiagnosisLabel.UNSUITABLE_FAMILY)
        reasons.append(f"{signals['gap_to_best_family']:.3f} below the best family")

    if not labels:
        labels.append(DiagnosisLabel.HEALTHY)
        reasons.append("no rule triggered")

    return Diagnosis(model_family=family, labels=labels, evidence=signals, reasons=reasons)


def plan_next_round(diagnoses: list[Diagnosis], rules: dict, round_no: int
                    ) -> tuple[PlanDecision, ReasoningEntry]:
    """Decide which families to keep and which search changes Agent 2 should make."""
    action_map = rules["actions"]
    keep, actions, referrals, notes = [], {}, [], []

    for d in diagnoses:
        labels = {lbl.value for lbl in d.labels}
        if DiagnosisLabel.UNSUITABLE_FAMILY.value in labels:
            notes.append(f"drop {d.model_family}")
            continue
        keep.append(d.model_family)
        actions[d.model_family] = sorted({a for lbl in labels for a in action_map[lbl]
                                          if a not in ("drop_family", "refer_features_to_c1")})

    if not keep:  # never drop everything: keep the best family
        best = min(diagnoses, key=lambda d: d.evidence["gap_to_best_family"])
        keep, actions[best.model_family] = [best.model_family], []
        notes.append(f"all families looked unsuitable; keep best: {best.model_family}")

    all_weak = all(DiagnosisLabel.WEAK_FEATURES in d.labels for d in diagnoses)
    if all_weak:
        referrals.append("weak_features: no model family beats the dummy baseline; "
                         "feature engineering needed in C1")
    if any(DiagnosisLabel.IMBALANCE in d.labels for d in diagnoses):
        referrals.append("class_imbalance: consider resampling (e.g. SMOTE/ADASYN) in C1")

    plan = PlanDecision(families_next_round=keep, actions=actions, referrals_to_c1=referrals,
                        stop=all_weak, rationale="; ".join(notes) or "keep all families")
    entry = ReasoningEntry(round=round_no, agent="agent1", step="plan_next_round",
                           decision=plan.model_dump(),
                           evidence={d.model_family: [l.value for l in d.labels] for d in diagnoses},
                           rationale=plan.rationale)
    return plan, entry
```

### C5. Unit tests

Agent 1's functions are **pure** (they don't train anything), so they can be tested with hand-made numbers in milliseconds.

**`tests/test_agent1.py`**

```python
import pytest

from c2_automl.agents.agent1_generation import (diagnose_family, eligible_families,
                                                plan_next_round, select_initial_families)
from c2_automl.config import load_default_config, load_yaml
from c2_automl.schemas import C1Output, DiagnosisLabel, TaskProfile, TaskType

CFG = load_default_config()
REGISTRY = load_yaml("model_registry.yaml")
RULES = load_yaml("diagnosis_rules.yaml")


def make_c1(task_type=TaskType.BINARY):
    return C1Output(dataset_id="t", task_type=task_type, target="y", data_dir=".",
                    feature_types={}, fqs={})


def make_profile(n_rows=5000, categorical_ratio=0.0):
    size = "small" if n_rows < 1000 else "large" if n_rows > 200000 else "medium"
    return TaskProfile(n_rows=n_rows, n_features=10, n_categorical=int(10 * categorical_ratio),
                       categorical_ratio=categorical_ratio, size_class=size, metric="roc_auc")


def make_signals(**overrides):
    s = {"mean_train": 0.90, "mean_val": 0.89, "gap": 0.01, "cv_std": 0.01, "baseline": 0.5,
         "lift_over_baseline": 0.39, "gap_to_best_family": 0.0, "mean_fqs": 0.4,
         "low_fqs_fraction": 0.1, "imbalance_ratio": 0.5, "minority_recall": 0.8}
    s.update(overrides)
    return s


# ---------- Level 1 ----------
def test_eligible_families_binary_excludes_ridge():
    fams = eligible_families("binary", REGISTRY)
    assert "ridge" not in fams and "xgboost" in fams


def test_blocked_family_removed():
    assert "catboost" not in eligible_families("binary", REGISTRY, blocked=["catboost"])


# ---------- Level 2 ----------
def test_small_dataset_avoids_lightgbm():
    fams, _ = select_initial_families(make_profile(n_rows=500), make_c1(), REGISTRY, CFG)
    assert "random_forest" in fams and "lightgbm" not in fams


def test_many_categoricals_adds_catboost():
    fams, entry = select_initial_families(make_profile(categorical_ratio=0.5), make_c1(), REGISTRY, CFG)
    assert "catboost" in fams
    assert "categorical" in entry.rationale


def test_regression_uses_ridge_baseline():
    fams, _ = select_initial_families(make_profile(n_rows=500), make_c1(TaskType.REGRESSION),
                                      REGISTRY, CFG)
    assert "ridge" in fams and "logistic_regression" not in fams


# ---------- Level 3 ----------
@pytest.mark.parametrize("overrides, expected", [
    ({"mean_train": 1.0, "gap": 0.15}, DiagnosisLabel.OVERFITTING),
    ({"mean_train": 0.70, "gap": 0.0}, DiagnosisLabel.UNDERFITTING),
    ({"imbalance_ratio": 0.03, "minority_recall": 0.2}, DiagnosisLabel.IMBALANCE),
    ({"lift_over_baseline": 0.01}, DiagnosisLabel.WEAK_FEATURES),
    ({"gap_to_best_family": 0.08}, DiagnosisLabel.UNSUITABLE_FAMILY),
    ({}, DiagnosisLabel.HEALTHY),
])
def test_diagnosis_rules(overrides, expected):
    d = diagnose_family("xgboost", make_signals(**overrides), RULES)
    assert expected in d.labels
    assert len(d.reasons) == len(d.labels)


def test_plan_drops_unsuitable_and_sends_actions():
    over = diagnose_family("random_forest", make_signals(mean_train=1.0, gap=0.15), RULES)
    bad = diagnose_family("logistic_regression", make_signals(gap_to_best_family=0.1), RULES)
    plan, _ = plan_next_round([over, bad], RULES, round_no=1)
    assert plan.families_next_round == ["random_forest"]
    assert "shrink_depth_range" in plan.actions["random_forest"]


def test_plan_refers_weak_features_to_c1_and_stops():
    weak = [diagnose_family(f, make_signals(lift_over_baseline=0.0), RULES) for f in ["a", "b"]]
    plan, _ = plan_next_round(weak, RULES, round_no=1)
    assert plan.stop and any("weak_features" in r for r in plan.referrals_to_c1)
```

Run:

```bash
pytest -q
```

Expected:

```text
.............                                                            [100%]
13 passed
```

### C6. Run Agent 1 on real data

This script runs the whole Agent 1 cycle once, using **default** model settings (Agent 2 tuning comes in Step 2).

**`scripts/run_agent1_round1.py`**

```python
"""Run Agent 1 on one dataset: profile -> select families -> train (defaults) -> diagnose -> plan.

Usage:  python scripts/run_agent1_round1.py diabetes [--folds 3]
"""
import argparse
import json

from c2_automl.agents.agent1_generation import (diagnose_family, plan_next_round,
                                                select_initial_families)
from c2_automl.config import OUTPUT_DIR, load_default_config, load_yaml
from c2_automl.io.dummy_c1 import load_c1_output, load_split
from c2_automl.tools.signals import compute_signals
from c2_automl.tools.task_profile import profile_task
from c2_automl.tools.train_eval import baseline_score, cross_validate_family


def main(dataset_id: str, folds: int):
    cfg = load_default_config()
    registry = load_yaml("model_registry.yaml")
    rules = load_yaml("diagnosis_rules.yaml")
    seed = cfg["seed"]
    log = []

    c1 = load_c1_output(dataset_id)
    X, y = load_split(c1, "train")           # never the test split here
    profile = profile_task(c1, X, y, cfg)
    print("Task profile:", profile.model_dump())

    families, entry = select_initial_families(profile, c1, registry, cfg)
    log.append(entry)
    print("\nAgent 1 selected:", families, "\nReason:", entry.rationale)

    results = {}
    for family in families:
        print(f"  training {family} ...")
        results[family] = cross_validate_family(family, X, y, c1.task_type, profile.metric, folds, seed)
    base = baseline_score(X, y, c1.task_type, profile.metric, folds, seed)
    best_val = max(r["mean_val"] for r in results.values())

    diagnoses = []
    print(f"\n{'family':<22}{'train':>8}{'val':>8}{'gap':>8}  diagnosis")
    for family, r in results.items():
        signals = compute_signals(r, y, c1.task_type, base, best_val, c1.fqs)
        d = diagnose_family(family, signals, rules)
        diagnoses.append(d)
        print(f"{family:<22}{signals['mean_train']:>8.3f}{signals['mean_val']:>8.3f}"
              f"{signals['gap']:>8.3f}  {[l.value for l in d.labels]}")
        for reason in d.reasons:
            print(f"{'':<22}- {reason}")

    plan, entry = plan_next_round(diagnoses, rules, round_no=1)
    log.append(entry)
    print("\nPlan for round 2:", json.dumps(plan.model_dump(), indent=2))

    out = OUTPUT_DIR / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent1_reasoning_log.json").write_text(
        json.dumps([e.model_dump() for e in log], indent=2, default=str))
    (out / "agent1_diagnoses.json").write_text(
        json.dumps([d.model_dump() for d in diagnoses], indent=2, default=str))
    print(f"\nSaved logs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.dataset_id, args.folds)
```

#### Run 1 – Pima Diabetes (small dataset → expect overfitting)

```bash
python scripts/run_agent1_round1.py diabetes
```

Actual output from testing this guide (takes about 10 seconds):

```text
Task profile: {'n_rows': 614, 'n_features': 8, 'n_categorical': 0, 'categorical_ratio': 0.0, 'size_class': 'small', 'metric': 'roc_auc', 'n_classes': 2, 'imbalance_ratio': 0.535}

Agent 1 selected: ['random_forest', 'catboost', 'logistic_regression'] 
Reason: small dataset (614 rows): boosting overfits easily, prefer random forest, catboost and a simple baseline
  training random_forest ...
  training catboost ...
  training logistic_regression ...

family                   train     val     gap  diagnosis
random_forest            1.000   0.811   0.189  ['overfitting']
                      - train-validation gap 0.189 > 0.05 with train score 1.000
catboost                 0.994   0.825   0.169  ['overfitting']
                      - train-validation gap 0.169 > 0.05 with train score 0.994
logistic_regression      0.842   0.833   0.008  ['healthy']
                      - no rule triggered

Plan for round 2: {
  "families_next_round": [
    "random_forest",
    "catboost",
    "logistic_regression"
  ],
  "actions": {
    "random_forest": [
      "increase_regularization",
      "shrink_depth_range"
    ],
    "catboost": [
      "increase_regularization",
      "shrink_depth_range"
    ],
    "logistic_regression": []
  },
  "referrals_to_c1": [],
  "stop": false,
  "decided_by": "rules",
  "rationale": "keep all families"
}
```

**How to read it:** Random Forest and CatBoost score almost 1.0 on the training data but only ~0.82 on validation. Agent 1 calls this **overfitting**, quotes the numbers, and asks Agent 2 to **shrink depth and add regularization** in round 2. Logistic regression is healthy.

#### Run 2 – Imbalanced Adult (3% positive class → expect imbalance)

```bash
python scripts/run_agent1_round1.py adult_imbalanced --folds 3
```

Actual output from testing this guide (diagnosis part):

```text
family                   train     val     gap  diagnosis
random_forest            1.000   0.869   0.131  ['overfitting', 'class_imbalance', 'unsuitable_model_family']
                      - train-validation gap 0.131 > 0.05 with train score 1.000
                      - imbalance ratio 0.031 and minority recall 0.233
                      - 0.047 below the best family
xgboost                  0.994   0.905   0.089  ['overfitting', 'class_imbalance']
                      - train-validation gap 0.089 > 0.05 with train score 0.994
                      - imbalance ratio 0.031 and minority recall 0.275
lightgbm                 0.992   0.914   0.077  ['overfitting', 'class_imbalance']
                      - train-validation gap 0.077 > 0.05 with train score 0.992
                      - imbalance ratio 0.031 and minority recall 0.264
catboost                 0.978   0.916   0.062  ['overfitting', 'class_imbalance']
                      - train-validation gap 0.062 > 0.05 with train score 0.978
                      - imbalance ratio 0.031 and minority recall 0.261

  "referrals_to_c1": [
    "class_imbalance: consider resampling (e.g. SMOTE/ADASYN) in C1"
  ],

Plan for round 2 (summary):
  families_next_round: ['xgboost', 'lightgbm', 'catboost']      <- random_forest dropped
  actions (each family): ['enable_class_weights', 'increase_regularization', 'shrink_depth_range']
  rationale: drop random_forest
```

**How to read it:** ROC-AUC looks fine (~0.91), but every family finds only ~25% of the rare class. Agent 1 detects **class imbalance**, drops Random Forest as **unsuitable** (0.047 behind the best family), and writes a **referral to C1** about resampling.

Both runs save `outputs/<dataset>/agent1_reasoning_log.json` and `agent1_diagnoses.json` – the start of your **auditable reasoning** output.

> Your numbers may differ slightly between library versions, but the diagnoses should be the same.

✅ **Agent 1 Levels 1–3 are done** when the tests pass and both runs show the expected diagnoses.

---

## Part D – Commit, Troubleshooting, Next Steps

### Commit

```bash
git add components/C2-Agentic-AutoML
git commit -m "C2: Agent 1 levels 1-3 (selection, diagnosis, planning) with dummy C1 data"
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoostError: libomp.dylib could not be loaded` | `brew install libomp`, then restart the terminal |
| `ModuleNotFoundError: No module named 'c2_automl'` | Activate the venv (`source .venv/bin/activate`) and run `pip install -e .` |
| `FileNotFoundError: .../manifest.json` | Run `python scripts/build_dummy_data.py` first |
| OpenML download fails | Check internet; or download the CSV from Kaggle into `data/raw/` and use `load_table` + `build_dummy_c1` |
| Notebook can't import `c2_automl` | Select the kernel **"Python (C2 AutoML)"** (step A7) |
| Training is slow on Adult | Use `--folds 3` during development |
| `python3.11: command not found` | `brew install python@3.11`, or use `python3.12` / `python3.13` everywhere instead |

### Next steps

| Next | What | Main guide |
|---|---|---|
| **Step 2 – Agent 2, Levels 1–3** | Optuna tuning with fixed ranges (Baseline B2), pruning, trial history | §11, §13.3 |
| **Step 3 – LangGraph** | Wrap these functions as graph nodes: `profile → select → hpo → evaluate → diagnose → plan` | §8 |
| **Step 4 – Agent 1 Level 4 + Agent 2 Level 4** | Agent 2 turns Agent 1's `actions` into new search ranges; loop over rounds ⭐ | §11.2, §13.2–13.3 |

> Tip: open a notebook `notebooks/01_explore_datasets.ipynb` now and look at the dummy data (`load_split`) and the JSON logs in `outputs/`. It helps you explain Agent 1's decisions to your supervisor.
