"""
Everything C4 needs from MLflow, in one place.

- The tracking store is a SQLite file (mlflow.db) in the project root.
- The model registry holds "income-classifier" with two versions.
- Aliases:  production -> the model serving traffic
            previous   -> the model to roll back to
"""
import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import ROOT, MODEL_NAME, EXPERIMENT_NAME, ARTIFACTS, save_json, load_json

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
REGISTRY_FILE = ARTIFACTS / "registry.json"   # remembers which version is v1 (LogReg) and v2 (XGBoost)


def setup_mlflow() -> None:
    """Point MLflow at the project's SQLite store and make sure the experiment exists."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME,
                                 artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def version_for_run(run_id: str) -> str:
    """Find the registered model version that was created from a run."""
    for mv in client().search_model_versions(f"name='{MODEL_NAME}'"):
        if mv.run_id == run_id:
            return str(mv.version)
    raise RuntimeError(f"No registered version found for run {run_id}")


def set_alias(alias: str, version) -> None:
    client().set_registered_model_alias(MODEL_NAME, alias, str(version))


def get_alias_version(alias: str) -> str:
    return str(client().get_model_version_by_alias(MODEL_NAME, alias).version)


def load_model(version):
    """Load one model version (a full sklearn Pipeline)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def log_artifact_to_version(version, path, artifact_path: str) -> None:
    """Attach a file (e.g. thresholds.json) to the MLflow run that produced a model version."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        run_id = client().get_model_version(MODEL_NAME, str(version)).run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:   # the local copy in artifacts/ is what C4 uses, so this is optional
        print(f"   (note: could not attach {path.name} to MLflow run: {e})")


def save_registry_info(v1: str, v2: str) -> None:
    save_json({"model_name": MODEL_NAME, "v1_logreg": str(v1), "v2_xgboost": str(v2)}, REGISTRY_FILE)


def registry_info() -> dict:
    return load_json(REGISTRY_FILE)


def reset_production() -> None:
    """Put the registry back to the starting state: production = v2, previous = v1."""
    info = registry_info()
    set_alias("production", info["v2_xgboost"])
    set_alias("previous", info["v1_logreg"])


def rollback(reason: str) -> tuple[str, str]:
    """
    Move the 'production' alias to the 'previous' version.
    Returns (old_version, new_version). Raises if there is nothing to roll back to.
    """
    old = get_alias_version("production")
    new = get_alias_version("previous")
    if old == new:
        raise RuntimeError("Production is already the previous version - nothing to roll back to.")
    set_alias("production", new)
    c = client()
    c.set_model_version_tag(MODEL_NAME, new, "promoted_by", "C4 rollback")
    c.set_model_version_tag(MODEL_NAME, old, "rolled_back_reason", reason[:250])
    return old, new
