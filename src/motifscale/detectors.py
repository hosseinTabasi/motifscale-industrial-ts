"""Detectors sharing one interface: fit(list of ref arrays) -> score(test with m-1 ref prefix)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from .coreset import CoresetMotifDictionary, min_dist_blocked, _features
from .windows import sliding, flat
from .matrix_profile import ab_join_discord


class CoresetDetector:
    name = "MotifScale-coreset (ours)"

    def __init__(self, m, budget, radius=None, seed=0, max_motifs=128):
        self.d = CoresetMotifDictionary(m=m, budget=budget, radius=radius, normalize=False,
                                        seed=seed, max_motifs=max_motifs, target_coverage=0.99)

    def fit(self, refs):
        self.d.fit(refs); return self

    def score(self, x):
        return self.d.score(x)


class KNNFullDetector:
    """Exact 1-NN distance to every reference window (budget = 1, no dictionary)."""
    name = "1-NN all reference windows"

    def __init__(self, m):
        self.m = m

    def fit(self, refs):
        self.F = np.concatenate([_features(sliding(r, self.m), False) for r in refs]); return self

    def score(self, x):
        return min_dist_blocked(_features(sliding(x, self.m), False), self.F)[0]


class IForestDetector:
    name = "Isolation Forest (windows)"

    def __init__(self, m, seed=0):
        self.m, self.seed = m, seed

    def fit(self, refs):
        X = np.concatenate([flat(sliding(r, self.m)) for r in refs])
        self.f = IsolationForest(n_estimators=200, random_state=self.seed).fit(X); return self

    def score(self, x):
        return -self.f.score_samples(flat(sliding(x, self.m)))


class PCADetector:
    """PCA on points, T^2 + SPE combined (each scaled by its reference 99th pct)."""
    name = "PCA T2+SPE"

    def __init__(self, m, var=0.9):
        self.m, self.var = m, var

    def fit(self, refs):
        X = np.concatenate(refs)
        mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        ev = S ** 2 / (len(X) - 1)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), self.var) + 1)
        self.mu, self.P, self.ev = mu, Vt[:k].T, ev[:k]
        t2, spe = self._parts(X)
        self.s1, self.s2 = np.quantile(t2, 0.99) + 1e-9, np.quantile(spe, 0.99) + 1e-9
        return self

    def _parts(self, X):
        Z = (X - self.mu) @ self.P
        t2 = (Z ** 2 / self.ev).sum(1)
        spe = (((X - self.mu) - Z @ self.P.T) ** 2).sum(1)
        return t2, spe

    def score(self, x):
        t2, spe = self._parts(x)
        s = t2 / self.s1 + spe / self.s2
        return s[self.m - 1:]


class MPDiscordDetector:
    """Exact matrix-profile AB-join discord vs the file's own reference (per channel z-norm)."""
    name = "Matrix-profile AB-join discord"

    def __init__(self, m):
        self.m = m

    def fit(self, refs):
        self.refs = refs; return self

    def score_with_ref(self, x, ref):
        return ab_join_discord(x, ref, self.m)
