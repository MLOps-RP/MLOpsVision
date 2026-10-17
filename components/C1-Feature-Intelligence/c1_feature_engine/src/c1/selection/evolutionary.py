"""GA over binary feature masks (custom, no DEAP: RESEARCH R6).
fitness = inner-CV score - lam*size/n + mu*mean(FQS of subset); FQS-seeded init, elitism, time/generation budget."""
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def ga_select(Z: np.ndarray, y, fq: np.ndarray, k0: int, lam: float = 0.05, mu: float = 0.05, pop: int = 16,
              gens: int = 8, elite: int = 2, budget_s: float = 90.0, rows: int = 1500, seed: int = 0):
    """Z: n x d numeric matrix (train rows only), fq: FQS per column. Returns (bool mask, history list)."""
    rng, y, d, t0 = np.random.default_rng(seed), np.asarray(y), Z.shape[1], time.time()
    if len(Z) > rows:
        idx = rng.choice(len(Z), rows, replace=False)
        Z, y = Z[idx], y[idx]
    cv = StratifiedKFold(min(3, int(np.bincount(np.unique(y, return_inverse=True)[1]).min())), shuffle=True, random_state=seed)
    cache = {}

    def fitness(m):
        key = m.tobytes()
        if key not in cache:
            if not m.any():
                cache[key] = -np.inf
            else:
                clf = LGBMClassifier(n_estimators=40, num_leaves=15, class_weight="balanced", random_state=seed,
                                     n_jobs=1, verbose=-1)
                cv_s = cross_val_score(clf, Z[:, m], y, cv=cv, scoring="f1_macro", error_score=0.0).mean()
                cache[key] = cv_s - lam * m.sum() / d + mu * fq[m].mean()
        return cache[key]

    order = np.argsort(-fq)
    P = []
    for k in sorted({max(1, k0 // 2), k0, min(d, 2 * k0), max(1, d // 2)}):       # FQS-seeded
        m = np.zeros(d, bool)
        m[order[:k]] = True
        P.append(m)
    pr = 0.15 + 0.6 * (np.argsort(order) < max(1, d // 3))                       # biased random init
    while len(P) < pop:
        P.append(rng.random(d) < pr)
    hist = []
    for g in range(gens):
        f = np.array([fitness(m) for m in P])
        hist.append(float(f.max()))
        if time.time() - t0 > budget_s:
            break
        nxt = [P[i].copy() for i in np.argsort(-f)[:elite]]
        while len(nxt) < pop:
            a, b = (P[max(rng.choice(pop, 3), key=lambda i: f[i])] for _ in range(2))
            child = np.where(rng.random(d) < 0.5, a, b)
            child ^= rng.random(d) < 1.0 / d
            nxt.append(child)
        P = nxt
    f = np.array([fitness(m) for m in P])
    return P[int(np.argmax(f))], hist
