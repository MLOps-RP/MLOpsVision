"""Set-aware greedy selection (mRMR-style) on FQS relevance, with a hard collinearity cut.
score_j = rel_j - lam * max |Spearman(j, already chosen)|; candidates with |rho| > thr to a chosen feature are skipped."""
import numpy as np
import pandas as pd

from c1.selection.rankers import prep


def corr_matrix(X: pd.DataFrame, cols: list[str], max_rows: int = 2000, seed: int = 0) -> np.ndarray:
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    Z, _ = prep(X[cols])
    C = np.abs(np.nan_to_num(pd.DataFrame(Z).rank().corr().to_numpy()))
    np.fill_diagonal(C, 0)
    return C


def greedy_select(rel: pd.Series, C: np.ndarray, k: int, lam: float = 0.5, thr: float = 0.9) -> list[str]:
    """rel indexed like C's rows. Returns up to k names (may be fewer if everything left is collinear)."""
    r, chosen, left = rel.to_numpy(), [], set(range(len(rel)))
    while left and len(chosen) < k:
        best = max(left, key=lambda j: r[j] - lam * (C[j, chosen].max() if chosen else 0.0))
        left.discard(best)
        if chosen and C[best, chosen].max() > thr:
            continue
        chosen.append(best)
    return [rel.index[j] for j in chosen]
