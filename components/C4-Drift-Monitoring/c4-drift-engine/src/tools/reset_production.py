"""
Put the MLflow registry back to the starting state before a demo:
    production -> v2 (XGBoost), previous -> v1 (Logistic Regression)

    python -m src.tools.reset_production
"""
from src.common import registry


def main() -> None:
    registry.reset_production()
    print(f"production -> version {registry.get_alias_version('production')}, "
          f"previous -> version {registry.get_alias_version('previous')}")


if __name__ == "__main__":
    main()
