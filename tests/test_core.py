import numpy as np
import pytest
from motifscale.windows import sliding, znorm, sqdist
from motifscale.coreset import CoresetMotifDictionary, min_dist_blocked
from motifscale.metrics import point_adjust, best_f1, detection_report
from motifscale import matrix_profile as mp


def test_sliding_shapes():
    x = np.arange(20, dtype=float).reshape(10, 2)
    w = sliding(x, 4)
    assert w.shape == (7, 2, 4) and np.allclose(w[1, 0], [2, 4, 6, 8])


def test_znorm_zero_mean_unit_std():
    w = znorm(np.random.default_rng(0).normal(3, 2, (5, 2, 16)))
    assert np.allclose(w.mean(-1), 0, atol=1e-9) and np.allclose(w.std(-1), 1, atol=1e-9)


def test_sqdist_matches_bruteforce():
    rng = np.random.default_rng(1); a, b = rng.normal(size=(6, 5)), rng.normal(size=(4, 5))
    ref = ((a[:, None] - b[None]) ** 2).sum(-1)
    assert np.allclose(sqdist(a, b), ref)


def test_numpy_stomp_matches_stumpy():
    pytest.importorskip("stumpy")
    t = np.sin(np.linspace(0, 30, 600)) + 0.1 * np.random.default_rng(0).normal(size=600)
    P, _ = mp.stomp_numpy(t, 20)
    assert np.allclose(P, mp.stump_exact(t, 20), atol=1e-4)


def test_coreset_greedy_monotone_gains_and_full_budget_covers_repeats():
    rng = np.random.default_rng(0)
    pattern = np.sin(np.linspace(0, 2 * np.pi, 20))
    x = np.tile(pattern, 30)[:, None] + 0.01 * rng.normal(size=(600, 1))
    d = CoresetMotifDictionary(m=20, budget=1.0, radius=0.2, max_motifs=40).fit(x)
    g = d.stats_["gains"]
    assert all(g[i] >= g[i + 1] for i in range(len(g) - 1))  # submodular greedy: non-increasing gains
    assert d.evaluate(x)["coverage"] >= 0.95


def test_budget_controls_candidates():
    x = np.random.default_rng(0).normal(size=(2000, 3))
    a = CoresetMotifDictionary(m=16, budget=0.01).fit(x); b = CoresetMotifDictionary(m=16, budget=0.1).fit(x)
    assert b.stats_["n_candidates"] > 5 * a.stats_["n_candidates"]
    assert b.stats_["distance_evals"] > a.stats_["distance_evals"]


def test_discord_score_detects_injected_anomaly():
    rng = np.random.default_rng(0)
    ref = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 40)[:, None] + 0.05 * rng.normal(size=(1000, 1))
    test = ref.copy(); test[500:520] += 3.0
    d = CoresetMotifDictionary(m=25, budget=0.2).fit(ref)
    s = d.score(test)
    assert 470 <= int(np.argmax(s)) <= 520


def test_channel_attribution_points_to_faulty_sensor():
    rng = np.random.default_rng(0); ref = rng.normal(size=(800, 4)) * 0.1
    test = ref.copy(); test[300:340, 2] += 2.0
    d = CoresetMotifDictionary(m=16, budget=0.2).fit(ref)
    A = d.channel_attribution(test)
    assert A[300:320].sum(0).argmax() == 2


def test_point_adjust_and_f1():
    y = np.array([0, 1, 1, 1, 0, 0, 1, 1]); p = np.array([0, 0, 1, 0, 0, 0, 0, 0], bool)
    assert point_adjust(p, y).tolist() == [False, True, True, True, False, False, False, False]
    s = y + 0.01 * np.arange(8)
    assert best_f1(s, y) == 1.0 and detection_report(s, y)["auc_pr"] == pytest.approx(1.0)


def test_min_dist_blocked():
    rng = np.random.default_rng(2); a = rng.normal(size=(50, 4)).astype(np.float32); b = a[:10]
    d, j = min_dist_blocked(a, b, block=7)
    assert np.allclose(d[:10], 0, atol=1e-3) and (j[:10] == np.arange(10)).all()
