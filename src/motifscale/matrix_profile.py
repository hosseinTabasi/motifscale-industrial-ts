"""Matrix-profile baselines: own exact STOMP (reference), stumpy exact/anytime/multivariate."""
from __future__ import annotations
import numpy as np
from .windows import sliding, znorm, flat, sqdist

try:
    import stumpy
except Exception:  # pragma: no cover
    stumpy = None


def stomp_numpy(t: np.ndarray, m: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact univariate z-normalised matrix profile by blocked brute force (O(n^2 m)).
    Used as a correctness reference for small n."""
    w = flat(znorm(sliding(t, m)))
    n = len(w)
    excl = int(np.ceil(m / 4))
    P = np.full(n, np.inf)
    I = np.zeros(n, int)
    for s in range(0, n, block):
        d = sqdist(w[s:s + block], w)
        idx = np.arange(s, min(s + block, n))
        for r, i in enumerate(idx):
            d[r, max(0, i - excl):i + excl + 1] = np.inf
        j = d.argmin(1)
        P[idx] = np.sqrt(d[np.arange(len(idx)), j])
        I[idx] = j
    return P, I


def stump_exact(t: np.ndarray, m: int) -> np.ndarray:
    return stumpy.stump(np.asarray(t, float), m)[:, 0].astype(float)


def scrump_anytime(t: np.ndarray, m: int, percentage: float, seed: int = 0) -> np.ndarray:
    """SCRIMP++-style anytime approximation: only `percentage` of diagonals evaluated."""
    np.random.seed(seed)
    a = stumpy.scrump(np.asarray(t, float), m, percentage=percentage, pre_scrump=True)
    a.update()
    return a.P_.astype(float)


def mstump_profile(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """mSTAMP: x (T, C) -> P (C, n) where row k is the profile using best k+1 dims."""
    P, I = stumpy.mstump(np.asarray(x, float).T, m)
    return P, I


def ab_join_discord(test: np.ndarray, ref: np.ndarray, m: int) -> np.ndarray:
    """Per-channel AB-join (test vs normal reference) z-norm NN distance, summed over channels.
    Returns a score per test window (window i covers test[i:i+m])."""
    s = 0.0
    for c in range(test.shape[1]):
        tc, rc = test[:, c].astype(float), ref[:, c].astype(float)
        if np.std(rc) < 1e-8:
            continue
        s = s + stumpy.stump(tc, m, T_B=rc, ignore_trivial=False)[:, 0].astype(float)
    return np.asarray(s, float)


def top_k_motifs_from_profile(P: np.ndarray, k: int, m: int) -> list[int]:
    """Greedy top-k motif starts from a matrix profile with exclusion zone m/2."""
    P = np.asarray(P, float).copy()
    out = []
    for _ in range(k):
        if not np.isfinite(P).any():
            break
        i = int(np.nanargmin(P))
        out.append(i)
        P[max(0, i - m // 2): i + m // 2 + 1] = np.inf
    return out
