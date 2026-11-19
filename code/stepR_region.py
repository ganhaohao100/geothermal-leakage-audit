# -*- coding: utf-8 -*-
"""Regional replication of the label-leakage test (South China, H3 test).

Domain cropped to the box; the four target-derived fields are REBUILT from the
springs inside the box only, so the regional run does not inherit national
spring information. Geo-physical fields are kept as compiled from the national
heat-flow and fault databases, which is what a regional study would use.
"""
import io
import json
import os
import sys
import time

import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import load_cache, kernel_smooth, rasterize_points, TR, RES

os.makedirs(OUTD, exist_ok=True)

BBOX = (108.0, 122.0, 20.0, 30.0)
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_BAG, N_FOLD = 3, 5, 5
SEED = 20260927
T0 = 10.0
RHO_W, CP = 990.0, 4180.0
SIGMAS = [0.05, 0.10, 0.20, 0.40, 0.80]
FEATS_GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
FEATS_S = ["s_T", "s_lgQ", "s_heat", "s_den"]
FEATS_ALL = FEATS_GEO + FEATS_S
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE, "out", "features.npz"), allow_pickle=True)
lon_n, lat_n = z["lon_c"], z["lat_c"]
ix = np.nonzero((lon_n >= BBOX[0]) & (lon_n <= BBOX[1]))[0]
iy = np.nonzero((lat_n >= BBOX[2]) & (lat_n <= BBOX[3]))[0]
lon_c, lat_c = lon_n[ix], lat_n[iy]
LO, LA = np.meshgrid(lon_c, lat_c)
mask = np.asarray(z["mask"])[np.ix_(iy, ix)]
cell_km2 = (RES * 111.32) * (RES * 111.32 * np.cos(np.radians(LA)))

GEO = {k: np.asarray(z[k], np.float32)[np.ix_(iy, ix)] for k in FEATS_GEO}
LAB = {k: np.asarray(z[k], np.float32)[np.ix_(iy, ix)]
       for k in ("lab_spring", "lab_well")}
print("box %s -> cells %d, land %d" % (BBOX, mask.size, mask.sum()), flush=True)

d = load_cache()
sp_lon = np.asarray(d["spring_lon"], float)
sp_lat = np.asarray(d["spring_lat"], float)
inb = ((sp_lon >= BBOX[0]) & (sp_lon <= BBOX[1]) &
       (sp_lat >= BBOX[2]) & (sp_lat <= BBOX[3]))
slon, slat = sp_lon[inb], sp_lat[inb]
sT = np.asarray(d["spring_T"], float)[inb]
sQ = np.asarray(d["spring_Q"], float)[inb]
Hrel = RHO_W * CP * (sQ / 1000.0) * np.maximum(sT - T0, 0.0)
print("springs in box: %d (of %d)" % (slon.size, sp_lon.size), flush=True)


def spring_fields(sigma, den_sigma=None):
    """den_sigma=None -> 密度场用两格高斯（与全国 headline 口径一致）；
    den_sigma=sigma -> 四场统一按该 sigma 用核密度重建（与全国 stepS 扫描口径一致）。"""
    out = {}
    out["s_T"] = kernel_smooth(slon, slat, sT, LO, LA, sigma)[0].astype(np.float32)
    out["s_lgQ"] = kernel_smooth(slon, slat, np.log10(sQ + 1.0), LO, LA,
                                 sigma)[0].astype(np.float32)
    out["s_heat"] = kernel_smooth(slon, slat, np.log10(Hrel + 1.0), LO, LA,
                                  sigma)[0].astype(np.float32)
    if den_sigma is None:
        sd = rasterize_points(slon, slat, lon_c, lat_c, RES)
        out["s_den"] = (gaussian_filter(sd, 2.0) / cell_km2 * 1e4).astype(np.float32)
    else:
        den = kernel_smooth(slon, slat, np.ones_like(slon), LO, LA, den_sigma)[1]
        out["s_den"] = (den / cell_km2 * 1e4).astype(np.float32)
    return out


S040 = spring_fields(0.40)

print("spring fields rebuilt (%.0fs)" % (time.time() - t0), flush=True)

px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]


def stack(feats, sfields):
    cols = []
    for k in feats:
        a = GEO[k] if k in GEO else sfields[k]
        cols.append(np.asarray(a, np.float32).ravel()[il])
    return np.column_stack(cols)


def build_dataset(pos_land):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([px[il][pos_i], py[il][pos_i]]))
    dmin, _ = tp.query(np.column_stack([px[il], py[il]]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
    n_neg = min(len(cand), NEG_RATIO * len(pos_i))
    neg_i = rng.choice(cand, size=n_neg, replace=False)
    return pos_i, neg_i, cand


def fold_split(blk, rng):
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(rng.permutation(ub))}
    return np.array([fo[b] for b in blk])


def run_cv(Xl, pos_i, neg_i, ref, buf_km=BUF_KM, perm=False):
    sx_all, sy_all = px[il], py[il]
    bx = np.floor(sx_all / BLK_KM).astype(np.int64)
    by = np.floor(sy_all / BLK_KM).astype(np.int64)
    blkl = (bx + 1000) * 10000 + (by + 1000)
    si = np.r_[pos_i, neg_i]
    X = Xl[si]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx, sy = sx_all[si], sy_all[si]
    blk = blkl[si]
    fold = fold_split(blk, np.random.default_rng(SEED))
    rows, perm_rows = [], None
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > buf_km]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         n_pos_test=int(y[te].sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        if perm and k == 0:
            pi = permutation_importance(m, X[te], y[te], n_repeats=5,
                                        random_state=SEED, scoring="average_precision")
            perm_rows = sorted(zip(ref, pi.importances_mean.tolist()),
                               key=lambda t: -t[1])
    return rows, perm_rows


def summarise(rows):
    auc = np.array([r["auc"] for r in rows])
    ap = np.array([r["ap"] for r in rows])
    return dict(auc=float(auc.mean()), auc_sd=float(auc.std()),
                auc_rng=float(auc.max() - auc.min()), ap=float(ap.mean()),
                folds=auc.tolist())


RESULT = {}
for key, lab_key, feats in [("spring_geo", "lab_spring", FEATS_GEO),
                            ("spring_all", "lab_spring", FEATS_ALL),
                            ("well_geo", "lab_well", FEATS_GEO),
                            ("well_all", "lab_well", FEATS_ALL)]:
    pos_land = LAB[lab_key].ravel()[il] > 0.5
    pos_i, neg_i, cand = build_dataset(pos_land)
    Xl = stack(feats, S040)
    rows, perm_rows = run_cv(Xl, pos_i, neg_i, feats, perm=True)
    s = summarise(rows)
    s.update(tag=key, feats=feats, n_pos=int(len(pos_i)), n_neg=int(len(neg_i)),
             cv=rows, perm=perm_rows)
    RESULT[key] = s
    print("%-11s n_pos=%4d n_neg=%5d  AUC=%.4f (sd %.4f, range %.4f)  AP=%.4f"
          % (key, len(pos_i), len(neg_i), s["auc"], s["auc_sd"], s["auc_rng"], s["ap"]),
          flush=True)
    if perm_rows:
        print("     perm top5: " + ", ".join("%s %+.4f" % t for t in perm_rows[:5]),
              flush=True)

# ---- sigma sweep inside the region (spring-type) ----
pos_land = LAB["lab_spring"].ravel()[il] > 0.5
pos_i, neg_i, cand = build_dataset(pos_land)
sweep = []
for sg in SIGMAS:
    sf = spring_fields(sg, den_sigma=sg)
    Xl = stack(FEATS_ALL, sf)
    rows, perm_rows = run_cv(Xl, pos_i, neg_i, FEATS_ALL, perm=True)
    s = summarise(rows)
    s.update(sigma=sg, km=round(sg * 111.32, 1), top=perm_rows[:3],
             perm=perm_rows)
    sweep.append(s)
    print("sigma=%.2f  AUC=%.4f (range %.4f)  top=%s"
          % (sg, s["auc"], s["auc_rng"], perm_rows[0][0]), flush=True)

OUT = dict(bbox=BBOX, seed=SEED, block_km=BLK_KM, buffer_km=BUF_KM,
           n_fold=N_FOLD, neg_ratio=NEG_RATIO, n_springs_in_box=int(slon.size),
           n_cells=int(mask.sum()), models=RESULT, sweep_den_mode="kernel_density_rebuilt_at_sigma", sweep_spring_all=sweep,
           elapsed_s=round(time.time() - t0, 1))
with io.open(os.path.join(OUTD, "region_sea.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=1)
print("\nwrote region_sea.json  (%.0fs)" % (time.time() - t0))