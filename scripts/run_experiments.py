"""Runs E1-E5 and writes experiments/results/*.csv|json and figures/*.png."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from motifscale.data import load_skab, load_skab_anomaly_free, load_tep
from motifscale.coreset import CoresetMotifDictionary, _features, min_dist_blocked
from motifscale.windows import sliding
from motifscale import matrix_profile as mp
from motifscale.detectors import CoresetDetector, KNNFullDetector, IForestDetector, PCADetector, MPDiscordDetector
from motifscale.metrics import detection_report

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
RES, FIG = os.path.join(ROOT, "experiments", "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
M, B, SEEDS = CFG["window"], CFG["budget"], CFG["seeds"]
R = None if CFG["radius"] == "auto" else float(CFG["radius"])
RR = None


def timed(f, *a, **k):
    t = time.perf_counter(); out = f(*a, **k); return out, time.perf_counter() - t


def e1_scaling():
    af = load_skab_anomaly_free()
    long = np.concatenate([af] + [s.test for s in load_skab()])  # ~45k x 8 real SKAB samples
    mp.stump_exact(long[:500, 0], M); mp.scrump_anytime(long[:500, 0], M, 0.1); mp.mstump_profile(long[:300, :2], M)  # JIT warm-up
    rows = []
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        t1 = long[:n, 0]
        P, s = timed(mp.stump_exact, t1, M); rows.append(dict(n=n, method="exact STOMP (stumpy)", dims=1, seconds=s))
        _, s = timed(mp.scrump_anytime, t1, M, 0.1); rows.append(dict(n=n, method="anytime SCRIMP++ 10%", dims=1, seconds=s))
        if n <= 8000:
            _, s = timed(mp.stomp_numpy, t1, M); rows.append(dict(n=n, method="exact brute force (numpy)", dims=1, seconds=s))
        d, s = timed(CoresetMotifDictionary(m=M, budget=B, radius=R).fit, t1); rows.append(dict(n=n, method=f"coreset b={B}", dims=1, seconds=s))
        if n <= 16000:
            _, s = timed(mp.mstump_profile, long[:n], M); rows.append(dict(n=n, method="mSTAMP exact (stumpy)", dims=8, seconds=s))
        _, s = timed(CoresetMotifDictionary(m=M, budget=B, radius=R).fit, long[:n]); rows.append(dict(n=n, method=f"coreset b={B}", dims=8, seconds=s))
        print("E1", n, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e1_scaling.csv", index=False)
    plt.figure(figsize=(6, 4))
    for (mth, dm), g in df.groupby(["method", "dims"]):
        plt.loglog(g.n, g.seconds, "o-", label=f"{mth} [{dm}-d]")
    plt.xlabel("series length n"); plt.ylabel("seconds (CPU)"); plt.legend(fontsize=7); plt.title("E1 runtime scaling (SKAB)")
    plt.tight_layout(); plt.savefig(f"{FIG}/e1_scaling.png", dpi=140); plt.close()
    return df


def e2_pareto():
    x = load_skab_anomaly_free()
    W = sliding(x, M); F = _features(W, False)
    rows = []
    global RR
    RR = CoresetMotifDictionary(m=M, budget=B).fit(x).radius_
    for b in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        for sd in SEEDS:
            d, s = timed(CoresetMotifDictionary(m=M, budget=b, radius=R, seed=sd, max_motifs=64, target_coverage=1.0).fit, x)
            k = d.stats_["n_motifs"]; ev = d.evaluate(x)
            rows.append(dict(method="coreset (ours)", budget=b, seed=sd, k=k, seconds=s, **ev))
            rng = np.random.default_rng(sd); idx = rng.choice(len(F), k, replace=False)
            dd = min_dist_blocked(F, F[idx])[0]
            rows.append(dict(method="random k windows", budget=b, seed=sd, k=k, seconds=0.0, coverage=float((dd <= RR).mean()), recon_rms=float(dd.mean())))
    # exact matrix-profile top-k motifs (same k as coreset at the largest budget), and k-means centroids
    (P, _), s_mp = timed(mp.mstump_profile, x, M)
    for b in [0.005, 0.05, 0.2]:
        ks = [r["k"] for r in rows if r["method"] == "coreset (ours)" and r["budget"] == b]
        k = int(np.median(ks))
        idx = mp.top_k_motifs_from_profile(P[-1], k, M)
        dd = min_dist_blocked(F, F[idx])[0]
        rows.append(dict(method="mSTAMP top-k motifs (exact)", budget=b, seed=0, k=k, seconds=s_mp, coverage=float((dd <= RR).mean()), recon_rms=float(dd.mean())))
        from sklearn.cluster import MiniBatchKMeans
        km, s = timed(MiniBatchKMeans(k, random_state=0, n_init=3).fit, F)
        dd = min_dist_blocked(F, km.cluster_centers_.astype(np.float32))[0]
        rows.append(dict(method="k-means centroids (not real snippets)", budget=b, seed=0, k=k, seconds=s, coverage=float((dd <= RR).mean()), recon_rms=float(dd.mean())))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e2_pareto.csv", index=False)
    g = df.groupby(["method", "budget"]).agg(k=("k", "mean"), coverage=("coverage", "mean"), coverage_std=("coverage", "std"),
                                              recon_rms=("recon_rms", "mean"), seconds=("seconds", "mean")).reset_index()
    g.to_csv(f"{RES}/e2_pareto_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mth, gg in g.groupby("method"):
        ax[0].plot(gg.k, gg.coverage, "o-", label=mth); ax[1].semilogx(gg.seconds.clip(lower=1e-3), gg.coverage, "o", label=mth)
    ax[0].set_xlabel("dictionary size k"); ax[0].set_ylabel(f"coverage (RMS<= {RR:.2f}, auto)"); ax[1].set_xlabel("seconds"); ax[0].legend(fontsize=7)
    fig.suptitle("E2 representativeness vs size / compute (SKAB anomaly-free, 8-d)"); fig.tight_layout(); fig.savefig(f"{FIG}/e2_pareto.png", dpi=140); plt.close()
    # explainability figure: top motifs as real snippets
    d = CoresetMotifDictionary(m=M, budget=B, radius=R).fit(x)
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    for a, w, gn in zip(axs.ravel(), d.motif_windows_[:8], d.stats_["gains"][:8]):
        a.plot(w.T, lw=0.8); a.set_title(f"motif covers {gn} windows", fontsize=8); a.set_xticks([])
    fig.suptitle("First 8 greedy motifs (real SKAB snippets, 8 sensors)"); fig.tight_layout(); fig.savefig(f"{FIG}/motifs_skab.png", dpi=130); plt.close()
    return g


def e3_stability():
    segs = load_skab()
    groups = {"anomaly-free": [load_skab_anomaly_free()]}
    for s in segs:
        groups.setdefault(s.group, []).append(s.ref)
    rows = []
    names = list(groups)
    for sd in SEEDS:
        dicts = {g: CoresetMotifDictionary(m=M, budget=0.1, radius=R, seed=sd).fit(groups[g]) for g in names}
        for a in names:
            for b in names:
                rows.append(dict(seed=sd, dict_from=a, evaluated_on=b, **dicts[a].evaluate(groups[b])))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e3_cross_coverage.csv", index=False)
    piv = df.groupby(["dict_from", "evaluated_on"]).coverage.mean().unstack()
    piv.to_csv(f"{RES}/e3_cross_coverage_matrix.csv")
    # seed stability: mean RMS distance from each motif of seed a to nearest motif of seed b
    st = []
    for g in names:
        ds = [CoresetMotifDictionary(m=M, budget=0.1, radius=R, seed=sd).fit(groups[g]) for sd in SEEDS]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dd = min_dist_blocked(ds[i].motifs_, ds[j].motifs_)[0]
                st.append(dict(group=g, seed_a=i, seed_b=j, k_a=len(ds[i].motifs_), k_b=len(ds[j].motifs_),
                               mean_motif_match_rms=float(dd.mean()), frac_matched=float((dd <= ds[i].radius_).mean())))
    pd.DataFrame(st).to_csv(f"{RES}/e3_seed_stability.csv", index=False)
    # TEP: normal reference vs normal prefix of each test run
    ref, tsegs = load_tep()
    dt = CoresetMotifDictionary(m=CFG["window_tep"], budget=0.1, radius=R).fit(ref)
    tep = [dict(run=s.name, coverage_normal_prefix=dt.evaluate(s.test[:160])["coverage"], coverage_fault=dt.evaluate(s.test[160:])["coverage"]) for s in tsegs]
    pd.DataFrame(tep).to_csv(f"{RES}/e3_tep_coverage.csv", index=False)
    plt.figure(figsize=(5.5, 4.5)); plt.imshow(piv.values, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(label="coverage")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=30); plt.yticks(range(len(piv.index)), piv.index)
    for i in range(len(piv)):
        for j in range(len(piv)):
            plt.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", color="w", fontsize=8)
    plt.xlabel("evaluated on"); plt.ylabel("dictionary built on"); plt.title("E3 cross-regime motif coverage"); plt.tight_layout()
    plt.savefig(f"{FIG}/e3_cross_coverage.png", dpi=140); plt.close()
    return piv


def _detect_all(segs, m, budget, methods, seed=0):
    refs = [s.ref for s in segs]
    fitted = {k: (v.fit(refs) if k != "mp" else v) for k, v in methods.items()}
    out = {k: [] for k in methods}; ys = []
    thr = {}
    for k, det in fitted.items():
        if k in ("mp",):
            continue
        rs = np.concatenate([det.score(r) for r in refs]); thr[k] = float(np.quantile(rs, 0.99))
    times = {k: 0.0 for k in methods}
    for s in segs:
        x = np.concatenate([s.ref[-(m - 1):], s.test])
        ys.append(s.labels)
        for k, det in fitted.items():
            t = time.perf_counter()
            sc = det.score_with_ref(x, s.ref) if k == "mp" else det.score(x)
            times[k] += time.perf_counter() - t
            out[k].append(sc)
    y = np.concatenate(ys)
    rows = []
    for k in methods:
        sc = np.concatenate(out[k])
        rows.append(dict(method=methods[k].name, key=k, seed=seed, score_seconds=times[k], **detection_report(sc, y, thr.get(k))))
    return rows


def e4_detection():
    rows = []
    skab = load_skab()
    for sd in SEEDS:
        meth = {"coreset": CoresetDetector(M, B, R, seed=sd), "iforest": IForestDetector(M, seed=sd)}
        if sd == SEEDS[0]:
            meth.update({"knn": KNNFullDetector(M), "pca": PCADetector(M), "mp": MPDiscordDetector(M)})
        for r in _detect_all(skab, M, B, meth, sd):
            rows.append(dict(dataset="SKAB (33 runs)", **r))
        print("E4 skab seed", sd, flush=True)
    ref, tsegs = load_tep()
    mt = CFG["window_tep"]
    for s in tsegs:
        s.ref = ref
    for sd in SEEDS:
        meth = {"coreset": CoresetDetector(mt, B, R, seed=sd), "iforest": IForestDetector(mt, seed=sd)}
        if sd == SEEDS[0]:
            meth.update({"knn": KNNFullDetector(mt), "pca": PCADetector(mt)})
        for r in _detect_all(tsegs, mt, B, meth, sd):
            rows.append(dict(dataset="TEP (21 faults)", **r))
        print("E4 tep seed", sd, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e4_detection_raw.csv", index=False)
    cols = ["auc_pr", "auc_roc", "f1_best", "f1_best_pa", "f1_ref_thr", "f1_ref_thr_pa", "score_seconds"]
    g = df.groupby(["dataset", "method"])[cols].agg(["mean", "std"]).round(4)
    g.columns = ["_".join(c) for c in g.columns]; g = g.reset_index(); g.to_csv(f"{RES}/e4_detection_summary.csv", index=False)
    # root-cause illustration on one SKAB run
    s = skab[0]; det = CoresetDetector(M, B, R).fit([q.ref for q in skab])
    x = np.concatenate([s.ref[-(M - 1):], s.test]); A = det.d.channel_attribution(x)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(det.score(x), lw=0.8); ax[0].fill_between(range(len(s.labels)), 0, s.labels * det.score(x).max(), color="r", alpha=0.15); ax[0].set_ylabel("discord score")
    ax[1].imshow(A.T, aspect="auto", cmap="magma"); ax[1].set_yticks(range(8)); ax[1].set_yticklabels([c[:12] for c in __import__("motifscale.data", fromlist=["x"]).SKAB_SENSORS], fontsize=7)
    ax[1].set_xlabel(f"time ({s.name})"); fig.suptitle("Discord score and per-sensor attribution (red = labelled anomaly)"); fig.tight_layout()
    fig.savefig(f"{FIG}/e4_rootcause_example.png", dpi=130); plt.close()
    return g


def e5_knob():
    skab = load_skab(); rows = []
    for b in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for sd in SEEDS:
            t = time.perf_counter()
            r = _detect_all(skab, M, b, {"coreset": CoresetDetector(M, b, R, seed=sd)}, sd)[0]
            det = CoresetDetector(M, b, R, seed=sd).fit([s.ref for s in skab])
            rows.append(dict(budget=b, seed=sd, k=det.d.stats_["n_motifs"], fit_seconds=det.d.stats_["fit_seconds"],
                             pool_coverage=det.d.stats_["pool_coverage"], total_seconds=time.perf_counter() - t,
                             auc_pr=r["auc_pr"], auc_roc=r["auc_roc"], f1_best=r["f1_best"], f1_best_pa=r["f1_best_pa"]))
        print("E5", b, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/e5_knob_raw.csv", index=False)
    g = df.groupby("budget").agg(["mean", "std"]).drop(columns="seed", level=0).round(4)
    g.columns = ["_".join(c) for c in g.columns]; g = g.reset_index(); g.to_csv(f"{RES}/e5_knob_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].errorbar(g.budget, g.auc_pr_mean, g.auc_pr_std, fmt="o-", label="AUC-PR"); ax[0].errorbar(g.budget, g.auc_roc_mean, g.auc_roc_std, fmt="s-", label="AUC-ROC")
    ax[0].set_xscale("log"); ax[0].set_xlabel("budget"); ax[0].legend()
    ax[1].plot(g.budget, g.fit_seconds_mean, "o-"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("budget"); ax[1].set_ylabel("fit seconds")
    fig.suptitle("E5 sensitivity to the single knob (SKAB)"); fig.tight_layout(); fig.savefig(f"{FIG}/e5_knob.png", dpi=140); plt.close()
    return g


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5"]
    meta = {}
    for w in which:
        t = time.perf_counter(); globals()[{"e1": "e1_scaling", "e2": "e2_pareto", "e3": "e3_stability", "e4": "e4_detection", "e5": "e5_knob"}[w]]()
        meta[w] = time.perf_counter() - t; print(w, "done", round(meta[w], 1), "s", flush=True)
    json.dump(meta, open(f"{RES}/runtime_{'_'.join(which)}.json", "w"), indent=1)
