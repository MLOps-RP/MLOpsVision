from pydantic import BaseModel


class DatasetProfile(BaseModel):
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    frac_categorical: float
    dim_ratio: float              # n_rows / n_features
    mean_abs_skew: float
    missing_ratio: float          # fraction of missing cells
    rows_with_missing_ratio: float
    n_cols_with_missing: int
    max_col_missing: float
    mar_signal: float             # max |corr(missing-indicator, other numeric col)|
    miss_target_assoc: float      # max Cramer's V(missing-indicator, target)
    n_classes: int
    imbalance_ratio: float        # max/min class count
    min_class_count: int
    minority_share: float
    class_entropy: float          # normalised to [0, 1]


class StrategyDecision(BaseModel):
    imputer: str
    balancer: str
    imputer_scores: dict[str, float] = {}
    balancer_scores: dict[str, float] = {}
    fired_rules: list[str] = []
    used_cv: bool = False
    runtime_s: float = 0.0


class FeatureScore(BaseModel):
    name: str
    P: float                      # aggregated predictive power
    S: float                      # bootstrap selection frequency
    M: float                      # missing ratio (generated: max of parents)
    R: float                      # max |Spearman| with a higher-P feature
    fqs: float
    op: str = "raw"
    parents: list[str] = []
    selected: bool = False
