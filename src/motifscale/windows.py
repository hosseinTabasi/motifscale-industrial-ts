"""Sliding windows and z-normalised distances (vectorised)."""
from __future__ import annotations
import numpy as np


def sliding(x: np.ndarray, m: int, stride: int = 1) -> np.ndarray:
    """x: (T, C) -> (N, C, m) windows ending at m-1, m-1+stride, ..."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    w = np.lib.stride_tricks.sliding_window_view(x, m, axis=0)  # (T-m+1, C, m)
    return np.ascontiguousarray(w[::stride])


def znorm(w: np.ndarray) -> np.ndarray:
    mu = w.mean(-1, keepdims=True)
    sd = w.std(-1, keepdims=True)
    return (w - mu) / np.where(sd < 1e-8, 1.0, sd)


def flat(w: np.ndarray) -> np.ndarray:
    return w.reshape(len(w), -1)


def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance between rows of a (n,d) and b (k,d)."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)
