"""Candidate balancing strategies (FR3). Resampling only ever runs inside Pipeline.fit."""
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks

BALANCERS = ["none", "class_weight", "smote", "smote_nc", "adasyn", "tomek", "smote_tomek"]
_K = 3  # small k so folds of tiny-minority datasets still work (loader enforces >= 20 per class)


def uses_class_weight(name: str) -> bool:
    return name == "class_weight"


def make_sampler(name: str, n_onehot: int, seed: int):
    """n_onehot = width of the leading one-hot block of the preprocessed matrix."""
    if name in ("none", "class_weight"):
        return None
    if name == "smote":
        return SMOTE(k_neighbors=_K, random_state=seed)
    if name == "smote_nc":  # ponytail: one-hot groups are treated as independent binaries; group-aware SMOTE if it matters
        if n_onehot == 0:
            return SMOTE(k_neighbors=_K, random_state=seed)
        return SMOTENC(categorical_features=list(range(n_onehot)),
                       k_neighbors=_K, random_state=seed)
    if name == "adasyn":
        return ADASYN(n_neighbors=_K, random_state=seed)
    if name == "tomek":
        return TomekLinks()
    if name == "smote_tomek":
        return SMOTETomek(smote=SMOTE(k_neighbors=_K, random_state=seed), random_state=seed)
    raise ValueError(name)
