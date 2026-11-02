"""Budgeted coreset motif dictionary (the method of this repository).

Given windows w_1..w_n, a compute budget fraction b selects a pool of
n_c = ceil(b n) candidate windows (stratified in time). Distances are computed only
between all windows and this pool, so cost is O(b n^2 m) instead of O(n^2 m).
A window i is *covered* by candidate j if its RMS distance d_ij <= radius.
The dictionary S is chosen greedily to maximise the monotone submodular coverage

    f(S) = |{ i : min_{j in S} d_ij <= radius }|,

which gives the classic (1 - 1/e) guarantee w.r.t. the best dictionary of the same
size drawn from the pool. Greedy stops at `target_coverage`, at `max_motifs`, or when
the marginal gain drops below `min_gain` of n (diminishing returns).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from .windows import sliding, znorm, flat, sqdist


def _features(w: np.ndarray, normalize: bool) -> np.ndarray:
    w = znorm(w) if normalize else w
    return (flat(w) / np.sqrt(w.shape[1] * w.shape[2])).astype(np.float32)  # RMS units


def min_dist_blocked(a: np.ndarray, b: np.ndarray, block: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Nearest row of b for each row of a (RMS distance), blocked for memory."""
    d = np.empty(len(a), np.float32)
    j = np.empty(len(a), np.int64)
    bb = (b * b).sum(1)
    for s in range(0, len(a), block):
        x = a[s:s + block]
        D = (x * x).sum(1)[:, None] + bb[None, :] - 2.0 * x @ b.T
        j[s:s + block] = D.argmin(1)
        d[s:s + block] = np.sqrt(np.maximum(D[np.arange(len(x)), j[s:s + block]], 0))
    return d, j


def auto_radius(F: np.ndarray, m: int, sample: int = 1000, seed: int = 12345) -> float:
    """Noise-floor radius: median nearest-neighbour RMS distance inside a fixed random
    sample of windows (trivial matches |i-j| < m excluded). Independent of the budget."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F), min(sample, len(F)), replace=False))
    X = F[idx]
    D = (X * X).sum(1)[:, None] + (X * X).sum(1)[None, :] - 2.0 * X @ X.T
    D[np.abs(idx[:, None] - idx[None, :]) < m] = np.inf
    return float(np.sqrt(np.maximum(np.median(D.min(1)), 0)))


@dataclass
class CoresetMotifDictionary:
    m: int = 32
    budget: float = 0.05          # THE knob: fraction of windows examined as candidates
    radius: float | None = None   # RMS distance for "covered"; None = auto (data noise floor)
    target_coverage: float = 0.95
    max_motifs: int = 64
    min_gain: float = 0.002
    normalize: bool = False       # z-normalise each channel of each window (shape motifs)
    stride: int = 1
    seed: int = 0
    motifs_: np.ndarray = field(default=None, repr=False)
    stats_: dict = field(default_factory=dict, repr=False)

    def windows(self, x) -> np.ndarray:
        xs = x if isinstance(x, list) else [x]
        return np.concatenate([sliding(s, self.m, self.stride) for s in xs if len(s) >= self.m])

    def fit(self, x) -> "CoresetMotifDictionary":
        t0 = time.perf_counter()
        W = self.windows(x)
        F = _features(W, self.normalize)
        n = len(F)
        rng = np.random.default_rng(self.seed)
        if self.radius is None:
            self.radius_ = auto_radius(F, self.m)
        else:
            self.radius_ = float(self.radius)
        n_c = int(min(n, max(8, np.ceil(self.budget * n))))
        edges = np.linspace(0, n, n_c + 1).astype(int)
        cand = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])])
        cov = np.zeros((n, n_c), bool)
        bc = (F[cand] ** 2).sum(1)
        for s in range(0, n, 4096):
            x_ = F[s:s + 4096]
            D = (x_ ** 2).sum(1)[:, None] + bc[None, :] - 2.0 * x_ @ F[cand].T
            cov[s:s + 4096] = D <= self.radius_ ** 2
        covered = np.zeros(n, bool)
        chosen, gains = [], []
        while len(chosen) < self.max_motifs:
            g = cov[~covered].sum(0)
            j = int(g.argmax())
            if g[j] < max(1, self.min_gain * n):
                break
            chosen.append(j)
            gains.append(int(g[j]))
            covered |= cov[:, j]
            if covered.mean() >= self.target_coverage:
                break
        if not chosen:
            chosen = [0]
        idx = cand[chosen]
        self.motif_windows_ = W[idx]
        self.motifs_ = F[idx]
        self.motif_index_ = idx
        self.stats_ = dict(radius=self.radius_, n_windows=n, n_candidates=n_c, n_motifs=len(idx),
                           pool_coverage=float(covered.mean()), distance_evals=int(n * n_c),
                           fit_seconds=time.perf_counter() - t0, gains=gains)
        return self

    def nearest(self, x) -> tuple[np.ndarray, np.ndarray]:
        F = _features(self.windows(x), self.normalize)
        return min_dist_blocked(F, self.motifs_)

    def evaluate(self, x) -> dict:
        d, _ = self.nearest(x)
        return dict(coverage=float((d <= self.radius_).mean()), recon_rms=float(d.mean()))

    def bag_of_motifs(self, x, soft: bool = False) -> np.ndarray:
        """Histogram of nearest-motif assignments (explainable representation)."""
        d, j = self.nearest(x)
        h = np.bincount(j, minlength=len(self.motifs_)).astype(float)
        return h / max(1, h.sum())

    def score(self, x) -> np.ndarray:
        """Discord score per window = RMS distance to nearest dictionary motif."""
        return self.nearest(x)[0]

    def channel_attribution(self, x) -> np.ndarray:
        """Per-channel squared error to the nearest motif (root-cause ranking), (N, C)."""
        W = self.windows(x)
        F = _features(W, self.normalize)
        _, j = min_dist_blocked(F, self.motifs_)
        M = self.motifs_[j].reshape(len(F), W.shape[1], W.shape[2])
        return ((F.reshape(M.shape) - M) ** 2).sum(-1)
