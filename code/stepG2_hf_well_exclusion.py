# -*- coding: utf-8 -*-
"""Sensitivity: drop well cells within 10 km of an IHFC heat-flow site.
Same negatives, same folds, same 10-feature harness as the well_geo baseline.
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree
from shapely import contains_xy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import load_cache, china_polygon, TR, RES, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
FEATS_GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
LO, LA = np.meshgrid(lon_c, lat_c)
il = np.nonzero(mask.ravel())[0]
lon_l, lat_l = LO.ravel()[il], LA.ravel()[il]
px, py = TR.transform(lon_l, lat_l)
px, py = px / 1000.0, py / 1000.0

X = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_GEO])
POS_W = np.asarray(z["lab_well"], np.float32).ravel()[il] > 0

def blocked(sx, sy):
    return ((np.floor(sx / BLK_KM).astype(np.int64) + 1000) * 10000
            + (np.floor(sy / BLK_KM).astype(np.int64) + 1000))

def folds_of(blk):
    ub = np.unique(blk)
    return {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}

rng = np.random.default_rng(SEED)
pos_i = np.nonzero(POS_W)[0]
tp = cKDTree(np.column_stack([px[pos_i], py[pos_i]]))
dmin, _ = tp.query(np.column_stack([px, py]), k=1)
cand = np.nonzero((~POS_W) & (dmin > POS_BUF_KM))[0]
neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)

blk_all = blocked(px, py)
si_full = np.r_[pos_i, neg_i]
blk_full = blk_all[si_full]
fo = folds_of(blk_full)

# wells within 10 km of a heat-flow site -> drop their cells
d = load_cache()
poly = china_polygon(d["border"])
mh = contains_xy(poly, np.asarray(d["hf_lon"], float), np.asarray(d["hf_lat"], float))
hlon, hlat = np.asarray(d["hf_lon"], float)[mh], np.asarray(d["hf_lat"], float)[mh]
hx, hy = TR.transform(hlon, hlat); hx, hy = hx / 1000.0, hy / 1000.0
htree = cKDTree(np.column_stack([hx, hy]))
WLON, WLAT = np.asarray(z["wlon"], float), np.asarray(z["wlat"], float)
wx, wy = TR.transform(WLON, WLAT); wx, wy = wx / 1000.0, wy / 1000.0
wd, _ = htree.query(np.column_stack([wx, wy]), k=1)
near = wd <= 10.0
def cell_ix(x, y):
    lon, lat = TR.transform(x * 1000.0, y * 1000.0, direction="INVERSE")
    ix = np.clip(np.round((lon - lon_c[0]) / RES).astype(int), 0, len(lon_c) - 1)
    iy = np.clip(np.round((lat - lat_c[0]) / RES).astype(int), 0, len(lat_c) - 1)
    return iy * len(lon_c) + ix
cell_flat = cell_ix(wx, wy)
near_cells = np.unique(cell_flat[near])

# map land-index -> flat cell id to compare
land_flat = il
near_land = np.isin(land_flat, near_cells)
drop_mask = near_land[pos_i]

def cv(pos_keep):
    pos = pos_i[pos_keep]
    si = np.r_[pos, neg_i]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg_i))]
    fold = np.array([fo[b] for b in blk_all[si]])
    sx, sy = px[si], py[si]
    rows = []
    for k in range(N_FOLD):
        te = fold == k; tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[si[tr_idx]], y[tr_idx])
        p = m.predict_proba(X[si[te]])[:, 1]
        rows.append((float(roc_auc_score(y[te], p)), float(average_precision_score(y[te], p)), int(y[te].sum())))
    a = np.array([r[0] for r in rows]); ap = np.array([r[1] for r in rows])
    return dict(n_pos=int(len(pos)), n_neg=int(len(neg_i)),
                auc=round(float(a.mean()), 4), auc_sd=round(float(a.std(ddof=1)), 4),
                auc_rng=round(float(a.max() - a.min()), 4), ap=round(float(ap.mean()), 4),
                folds=[round(float(v), 4) for v in a], pos_per_fold=[r[2] for r in rows])

base = cv(np.ones(len(pos_i), bool))
sens = cv(~drop_mask)
print("baseline", base, flush=True)
print("drop<=10km", sens, "cells dropped", int(drop_mask.sum()),
      "wells dropped", int(near.sum()), flush=True)
print("delta AUC", round(sens["auc"] - base["auc"], 4), flush=True)

out = dict(seed=SEED, block_km=BLK_KM, buffer_km=BUF_KM, neg_ratio=NEG_RATIO,
           n_well_records=int(WLON.size), n_well_cells=int(len(pos_i)),
           n_wells_within_10km=int(near.sum()), frac_wells_within_10km=round(float(near.mean()), 4),
           n_cells_dropped=int(drop_mask.sum()), baseline=base, drop_10km=sens,
           delta_auc=round(sens["auc"] - base["auc"], 4), elapsed_s=round(time.time() - t0, 1))
with io.open(os.path.join(OUTD, "hf_well_exclusion.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("wrote hf_well_exclusion.json", out["elapsed_s"], "s")
