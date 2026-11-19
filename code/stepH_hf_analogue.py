# -*- coding: utf-8 -*-
"""Step H  public-label analogue (202610靶区优选)

Purpose: show that the target-derived-field failure mode reproduces with a
fully public target set (IHFC heat-flow measurement points), so a reader can
re-run the whole chain without the closed spring/well inventories.

Target labels : heat-flow measurement locations (public, IHFC GHFDB)
t1 = 7 fault/stress features only        (no heat-flow-derived field)
t2 = t1 + hf_bg / hf_res / hf_grad       (target-derived at large sigma; this is
                                          the 'geo' set of the main experiment)
t3 = t2 + h_den / h_q / h_lgq / h_qg     (explicit target-derived fields)

Same grid, sampling, block CV, buffering, model and seed as the main experiment.
"""
import os, sys, json, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree
from shapely import contains_xy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import (kernel_smooth, rasterize_points, china_polygon, load_cache,
                    OUT as BASE_OUT, TR, RES)

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTD, exist_ok=True)

BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
SIGMAS = [0.05, 0.10, 0.20, 0.40, 0.80]

FEATS_T1 = ["f_dist", "f_den", "f_end", "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
FEATS_HF = ["hf_bg", "hf_res", "hf_grad"]
HF_FIELDS = ["h_den", "h_q", "h_lgq", "h_qg"]
FEATS_ALL = FEATS_T1 + FEATS_HF + HF_FIELDS
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t_all = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
cell_km2 = np.asarray(z["cell_km2"], float)
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
pxl, pyl = px[il], py[il]
bx = np.floor(pxl / BLK_KM).astype(np.int64)
by = np.floor(pyl / BLK_KM).astype(np.int64)
blkl = (bx + 1000) * 10000 + (by + 1000)

d = load_cache()
poly = china_polygon(d["border"])
m = contains_xy(poly, np.asarray(d["hf_lon"], float), np.asarray(d["hf_lat"], float))
hlon = np.asarray(d["hf_lon"], float)[m]
hlat = np.asarray(d["hf_lat"], float)[m]
hq = np.asarray(d["hf_q"], float)[m]
print("heat-flow points total %d ; inside China %d" % (len(d["hf_lon"]), m.sum()), flush=True)


def hf_fields(sigma, den_sigma):
    f = {}
    f["h_q"] = kernel_smooth(hlon, hlat, hq, LO, LA, sigma)[0]
    f["h_lgq"] = kernel_smooth(hlon, hlat, np.log10(hq + 1.0), LO, LA, sigma)[0]
    sm = np.where(np.isfinite(f["h_q"]), f["h_q"], np.nanmedian(f["h_q"]))
    gy, gx = np.gradient(sm)
    gy = gy / (RES * 111.32)
    gx = gx / (RES * 111.32 * np.cos(np.radians(LA)))
    f["h_qg"] = np.hypot(gx, gy)
    den = kernel_smooth(hlon, hlat, np.ones_like(hlon), LO, LA, den_sigma)[1]
    f["h_den"] = den / cell_km2 * 1e4
    return {k: np.asarray(f[k], np.float32).ravel()[il] for k in HF_FIELDS}


def build_Xl(hf):
    cols = [np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_T1 + FEATS_HF]
    cols += [np.asarray(hf[k], np.float32) for k in HF_FIELDS]
    return np.column_stack(cols).astype(np.float32)


def build_dataset(pos_land):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
    n_neg = min(len(cand), NEG_RATIO * len(pos_i))
    neg_i = rng.choice(cand, size=n_neg, replace=False)
    return pos_i, neg_i, cand


def fold_split(blk, rng):
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(rng.permutation(ub))}
    return np.array([fo[b] for b in blk])


def block_cv(Xl, feats, pos_i, neg_i, want_perm=True):
    cols = [FEATS_ALL.index(f) for f in feats]
    si = np.r_[pos_i, neg_i]
    X = Xl[np.ix_(si, cols)]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx = np.r_[pxl[pos_i], pxl[neg_i]]
    sy = np.r_[pyl[pos_i], pyl[neg_i]]
    blk = np.r_[blkl[pos_i], blkl[neg_i]]
    fold = fold_split(blk, np.random.default_rng(SEED))
    rows, perm_rows = [], None
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        mo = lgb.LGBMClassifier(**PARAMS)
        mo.fit(X[tr_idx], y[tr_idx])
        p = mo.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        if want_perm and k == 0:
            pi = permutation_importance(mo, X[te], y[te], n_repeats=5,
                                        random_state=SEED, scoring="average_precision")
            perm_rows = sorted(zip(feats, pi.importances_mean.tolist()),
                               key=lambda t: -t[1])
    return rows, perm_rows


def summ(rows):
    a = np.array([r["auc"] for r in rows])
    p = np.array([r["ap"] for r in rows])
    return dict(auc=float(a.mean()), auc_sd=float(a.std(ddof=1)),
                auc_rng=float(a.max() - a.min()), ap=float(p.mean()))


lab_hf = rasterize_points(hlon, hlat, lon_c, lat_c, RES) > 0
pos_land = lab_hf.ravel()[il]
pos_i, neg_i, cand = build_dataset(pos_land)
print("land cells %d ; heat-flow label cells %d ; pseudo-negatives %d"
      % (len(il), pos_i.size, neg_i.size), flush=True)

F40 = hf_fields(0.40, 2.0)
Xl = build_Xl(F40)
TIERS = [("t1_fault_stress_only", FEATS_T1),
         ("t2_plus_large_sigma_hf", FEATS_T1 + FEATS_HF),
         ("t3_plus_explicit_target_derived", FEATS_ALL)]
RES_T, CACHE_P = {}, {}
print("\n================ tiers (target = public heat-flow points) ================", flush=True)
for name, feats in TIERS:
    rows, perm = block_cv(Xl, feats, pos_i, neg_i)
    s = summ(rows)
    RES_T[name] = dict(n_features=len(feats), **s,
                       top=[list(t) for t in (perm or [])[:3]], cv=rows)
    CACHE_P[name] = perm
    print("  %-34s k=%2d  AUC %.4f (sd %.4f, rng %.4f)  AP %.4f  top=%s"
          % (name, len(feats), s["auc"], s["auc_sd"], s["auc_rng"], s["ap"],
             (perm or [["-", 0]])[0][0]), flush=True)

print("\n================ sigma sweep (t3, all four target-derived fields rebuilt at a common sigma) ================", flush=True)
SWEEP = []
for sg in SIGMAS:
    ff = hf_fields(sg, sg)
    Xs = build_Xl(ff)
    rows, perm = block_cv(Xs, FEATS_ALL, pos_i, neg_i)
    s = summ(rows)
    SWEEP.append(dict(sigma=sg, km=round(sg * 111.0, 1), **s,
                      top=[list(t) for t in perm[:3]],
                      perm=[list(t) for t in perm]))
    print("  sigma %.2f deg (%5.1f km)  AUC %.4f  AP %.4f  top1 %-8s %.4f  top2 %-8s %.4f"
          % (sg, sg * 111.0, s["auc"], s["ap"], perm[0][0], perm[0][1],
             perm[1][0], perm[1][1]), flush=True)

t3 = RES_T["t3_plus_explicit_target_derived"]
t2 = RES_T["t2_plus_large_sigma_hf"]
t1 = RES_T["t1_fault_stress_only"]
p2 = CACHE_P["t2_plus_large_sigma_hf"]
p3 = CACHE_P["t3_plus_explicit_target_derived"]
res = dict(
    seed=SEED, target="IHFC heat-flow measurement locations (public)",
    n_hf_points_china=int(m.sum()), n_label_cells=int(pos_i.size), n_negatives=int(neg_i.size),
    block_km=BLK_KM, buf_km=BUF_KM, pos_buf_km=POS_BUF_KM, neg_ratio=NEG_RATIO,
    tiers=RES_T, sigma_sweep=SWEEP, sweep_density_mode="rebuilt_at_sigma",
    headline=dict(auc_t1=round(t1["auc"], 4), auc_t2=round(t2["auc"], 4),
                  auc_t3=round(t3["auc"], 4),
                  delta_t1_to_t3=round(t3["auc"] - t1["auc"], 4),
                  delta_t2_to_t3=round(t3["auc"] - t2["auc"], 4),
                  rng_t1=round(t1["auc_rng"], 4), rng_t3=round(t3["auc_rng"], 4),
                  top_t3=[list(t) for t in p3[:3]], top_t2=[list(t) for t in p2[:3]]),
    elapsed_s=round(time.time() - t_all, 1))
json.dump(res, open(os.path.join(OUTD, "hf_analogue.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nHEADLINE  t1 %.4f -> t2 %.4f -> t3 %.4f   (d t1->t3 %+.4f)"
      % (t1["auc"], t2["auc"], t3["auc"], t3["auc"] - t1["auc"]), flush=True)
print("t3 top3:", [(a, round(b, 4)) for a, b in p3[:3]], flush=True)
print("t2 top3:", [(a, round(b, 4)) for a, b in p2[:3]], flush=True)
print("output %s  (%.0fs)" % (os.path.join(OUTD, "hf_analogue.json"), time.time() - t_all), flush=True)