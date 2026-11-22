# -*- coding: utf-8 -*-
"""Step T2  Split-distance sensitivity of the play-type transfer (Section 3.1).

The Table 1 design is repeated with the spring-distance threshold set to 38 km
and 75 km. Wells at or within the threshold form group A; wells beyond it form
group B. Groups, shared 5 deg tiles and fold assignments are rebuilt for each
threshold, so the cell counts change with the split.

Output: results/split_sensitivity.json
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR  # noqa: E402
from common import TR  # noqa: E402

BLK_KM, BUF_KM = 50.0, 10.0
TILE_DEG = 5.0
NEG_RATIO, N_FOLD = 3, 5
MIN_B, MIN_A = 5, 10
SEED = 20260927
SPLITS = (38.0, 75.0)

FEATS = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
         "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(DATA_DIR, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
ny, nx = mask.shape
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
X = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS])
pxl, pyl = px[il], py[il]
bx = np.floor(pxl / BLK_KM).astype(np.int64)
by = np.floor(pyl / BLK_KM).astype(np.int64)
blkl = (bx + 1000) * 10000 + (by + 1000)

W = np.asarray(z["lab_well"], np.float32)
pos = np.nonzero(W.ravel()[il] > 0)[0]
slon, slat = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
spx, spy = TR.transform(slon, slat)
spx, spy = spx / 1000.0, spy / 1000.0
dspring, _ = cKDTree(np.column_stack([spx, spy])).query(
    np.column_stack([pxl[pos], pyl[pos]]), k=1)

rng = np.random.default_rng(SEED)
tp = cKDTree(np.column_stack([pxl[pos], pyl[pos]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((W.ravel()[il] == 0) & (dmin > BUF_KM))[0]
neg = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos)), replace=False)
neg = np.setdiff1d(neg, pos)

lonP = LO.ravel()[il][pos]; latP = LA.ravel()[il][pos]
lonN = LO.ravel()[il][neg]; latN = LA.ravel()[il][neg]


def tile_counter(lo, la):
    return Counter(zip(np.floor(lo / TILE_DEG).astype(int).tolist(),
                       np.floor(la / TILE_DEG).astype(int).tolist()))


def run(train_pos_list, eval_map, fold_all):
    train_pos = np.unique(np.concatenate(train_pos_list))
    out = {}
    for f in range(N_FOLD):
        trp = train_pos[fold_all[train_pos] != f]
        trn = negS[fo["n"] != f]
        tr = np.r_[trp, trn]
        ytr = np.r_[np.ones(len(trp)), np.zeros(len(trn))]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr], ytr)
        for name, cells in eval_map.items():
            tep = cells[fold_all[cells] == f]
            ten = negS[fo["n"] == f]
            if len(tep) == 0:
                continue
            te = np.r_[tep, ten]
            yte = np.r_[np.ones(len(tep)), np.zeros(len(ten))]
            p = m.predict_proba(X[te])[:, 1]
            out.setdefault(name, []).append(
                {"auc": float(roc_auc_score(yte, p)),
                 "ap": float(average_precision_score(yte, p))})
    for name, rows in out.items():
        arr = np.array([r["auc"] for r in rows])
        out[name] = {"folds": rows, "auc_mean": float(arr.mean()),
                     "auc_sd": float(arr.std(ddof=1)),
                     "ap_mean": float(np.mean([r["ap"] for r in rows]))}
    return out


# Fixed shared-tile design: the six 5 deg tiles of the 50 km split define one
# common cell set, and only the spring-distance threshold changes between runs.
is_a50 = dspring <= 50.0
is_b50 = ~is_a50
cA = tile_counter(lonP[is_a50], latP[is_a50])
cB = tile_counter(lonP[is_b50], latP[is_b50])
cN = tile_counter(lonN, latN)
keep = {k for k in cB if cB[k] >= MIN_B and cA.get(k, 0) >= MIN_A}
sel_a = np.array([k in keep for k in zip(np.floor(lonP[is_a50] / TILE_DEG).astype(int).tolist(),
                                         np.floor(latP[is_a50] / TILE_DEG).astype(int).tolist())])
sel_b = np.array([k in keep for k in zip(np.floor(lonP[is_b50] / TILE_DEG).astype(int).tolist(),
                                         np.floor(latP[is_b50] / TILE_DEG).astype(int).tolist())])
sel_n = np.array([k in keep for k in zip(np.floor(lonN / TILE_DEG).astype(int).tolist(),
                                         np.floor(latN / TILE_DEG).astype(int).tolist())])
shared = pos[np.r_[np.nonzero(is_a50)[0][sel_a], np.nonzero(is_b50)[0][sel_b]]]
shared_d = dspring[np.r_[np.nonzero(is_a50)[0][sel_a], np.nonzero(is_b50)[0][sel_b]]]
negS = neg[sel_n]
ub = np.unique(blkl[np.r_[shared, negS]])
fold_of = {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}
fold_all = np.array([fold_of.get(b, -1) for b in blkl])
fo = dict(n=fold_all[negS])

res = {}
for SPLIT in SPLITS:
    is_a = shared_d <= SPLIT
    posA = shared[is_a]
    posB = shared[~is_a]
    r = dict(n_a=int(len(posA)), n_b=int(len(posB)), n_neg=int(len(negS)),
             n_tiles=len(keep))
    r["A_to_A"] = run([posA], {"A": posA}, fold_all)["A"]
    r["A_to_B"] = run([posA], {"B": posB}, fold_all)["B"]
    r["B_to_B"] = run([posB], {"B": posB}, fold_all)["B"]
    r["B_to_A"] = run([posB], {"A": posA}, fold_all)["A"]
    pool = run([posA, posB], {"A": posA, "B": posB}, fold_all)
    r["pooled_A"] = pool["A"]
    r["pooled_B"] = pool["B"]
    res[str(int(SPLIT))] = r
    print(SPLIT, r["n_a"], r["n_b"], r["n_tiles"],
          round(r["A_to_A"]["auc_mean"], 3), round(r["A_to_B"]["auc_mean"], 3),
          round(r["B_to_B"]["auc_mean"], 3), round(r["B_to_A"]["auc_mean"], 3), flush=True)

res["design"] = dict(block_km=BLK_KM, buffer_km=BUF_KM, tile_deg=TILE_DEG,
                     shared_tile_split_km=50.0,
                     n_fold=N_FOLD, neg_ratio=NEG_RATIO, seed=SEED,
                     splits=list(SPLITS))
res["elapsed_s"] = round(time.time() - t0, 1)
outdir = os.path.join(_ROOT, "results")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "split_sensitivity.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("wrote split_sensitivity.json")
