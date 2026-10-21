"""
DEPLOYMENT SIMULATOR  (stands in for C1 -> C2 -> C3 -> PROMOTE)

1. Load + clean the Adult dataset, split 40/10/10/40
2. Train v1 = Logistic Regression and v2 = XGBoost (inside one sklearn Pipeline each)
3. Evaluate both on the test split
4. Log + register both in MLflow, set aliases production -> v2, previous -> v1
5. Build and save the versioned baseline for each model

Run from the project root:
    python -m src.bootstrap.train_and_register
"""
import warnings

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import LABEL, MODEL_NAME, all_features, feature_lists, model_dir, save_json
from src.common import registry
from src.bootstrap.data import load_adult, split_and_save, describe
from src.bootstrap.baseline import build_baseline

PREPROCESSING_VERSION = "preproc_v1"
warnings.filterwarnings("ignore", category=UserWarning)


def make_preprocessor() -> ColumnTransformer:
    num, cat = feature_lists()
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])


def make_models() -> dict:
    """v1 = simple model (rollback target), v2 = strong model (production)."""
    try:
        from xgboost import XGBClassifier
        strong = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.9, colsample_bytree=0.9,
                               eval_metric="logloss", random_state=42, n_jobs=-1)
        strong_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("WARNING: xgboost is not installed - using sklearn HistGradientBoosting instead.")
        strong = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42)
        strong_name = "HistGradientBoosting"

    return {
        "v1": ("LogisticRegression", LogisticRegression(max_iter=1000)),
        "v2": (strong_name, strong),
    }


def evaluate(pipe, df: pd.DataFrame) -> dict:
    X, y = df[all_features()], df[LABEL]
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"test_accuracy": accuracy_score(y, pred),
            "test_f1": f1_score(y, pred),
            "test_roc_auc": roc_auc_score(y, proba)}


def main() -> None:
    # 1. data -------------------------------------------------------------
    df = load_adult()
    describe(df)
    parts = split_and_save(df)
    for k, v in parts.items():
        print(f"  {k:<14} {len(v):>6,} rows")
    train, val, test = parts["train"], parts["val"], parts["test"]

    # 2-5. models -------------------------------------------------------------
    registry.setup_mlflow()
    versions = {}
    for key, (model_type, estimator) in make_models().items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(train[all_features()], train[LABEL])
        metrics = evaluate(pipe, test)
        print(f"\n{key} {model_type}: " + "  ".join(f"{m}={v:.3f}" for m, v in metrics.items()))

        with mlflow.start_run(run_name=f"{key}-{model_type}") as run:
            mlflow.log_params({"model_type": model_type, "role": key,
                               "preprocessing_version": PREPROCESSING_VERSION,
                               "train_rows": len(train), "seed": 42})
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(pipe, artifact_path="model",
                                registered_model_name=MODEL_NAME,
                                serialization_format="cloudpickle")
            version = getattr(info, "registered_model_version", None) or \
                registry.version_for_run(run.info.run_id)
            version = str(version)
            versions[key] = version

            # versioned baseline, stored locally AND attached to the MLflow run
            baseline = build_baseline(pipe, val, version, model_type, PREPROCESSING_VERSION)
            baseline["test_metrics"] = metrics
            out = model_dir(version)
            save_json(baseline, out / "baseline.json")
            val[feature_lists()[0]].to_csv(out / "reference_numeric.csv", index=False)
            mlflow.log_artifact(str(out / "baseline.json"), artifact_path="c4_baseline")
            mlflow.log_artifact(str(out / "reference_numeric.csv"), artifact_path="c4_baseline")
            print(f"   registered as {MODEL_NAME} version {version}; baseline saved to {out}")

    registry.save_registry_info(versions["v1"], versions["v2"])
    registry.reset_production()
    print(f"\nAliases set: production -> version {versions['v2']}, previous -> version {versions['v1']}")
    print("Next step:  python -m src.simulator.generate_scenarios")


if __name__ == "__main__":
    main()
