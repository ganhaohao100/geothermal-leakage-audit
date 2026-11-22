# -*- coding: utf-8 -*-
"""Step T  Play-type transfer between the two well populations (Table 1).

Wells are split by their distance to the nearest mapped spring. Wells at or
within 50 km form group A (a proxy for spring-associated, fault-controlled
systems); wells beyond 50 km form group B (a proxy for spring-independent,
basin-hosted systems). A model calibrated on one group is scored on the other.

The comparison is restricted to 5 deg tiles that contain at least five group-B
well cells and ten group-A well cells, so that both populations are evaluated
against pseudo-negatives drawn from the same regional background.

Output: results/group_transfer.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR, OUTD  # noqa: E402
from common import TR, RES  # noqa: E402

BLK_KM, BUF_KM = 50.0, 10.0
SPLIT_KM, TILE_DEG = 50.0, 5.0
NEG_RATIO, N_FOLD, N_BOOT = 3, 5, 1000
MIN_B, MIN_A = 5, 10
SEED = 20260927
DEG_KM = 111.32

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
is_a = dspring <= SPLIT_KM
is_b = ~is_a

# national 1:3 pseudo-negative pool, identical to the headline run
rng = np.random.default_rng(SEED)
tp = cKDTree(np.column_stack([pxl[pos], pyl[pos]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((W.ravel()[il] == 0) & (dmin > BUF_KM))[0]
neg = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos)), replace=False)
neg = np.setdiff1d(neg, pos)

# shared 5 deg tiles
from collections import Counter
lonA = LO.ravel()[il][pos][is_a]; latA = LA.ravel()[il][pos][is_a]
lonB = LO.ravel()[il][pos][is_b]; latB = LA.ravel()[il][pos][is_b]
lonN = LO.ravel()[il][neg]; latN = LA.ravel()[il][neg]

def tile_counter(lo, la):
    return Counter(zip(np.floor(lo / TILE_DEG).astype(int).tolist(),
                       np.floor(la / TILE_DEG).astype(int).tolist()))

cA, cB, cN = tile_counter(lonA, latA), tile_counter(lonB, latB), tile_counter(lonN, latN)
keep = {k for k in cB if cB[k] >= MIN_B and cA.get(k, 0) >= MIN_A}
sel_a = np.array([k in keep for k in zip(np.floor(lonA / TILE_DEG).astype(int).tolist(),
                                         np.floor(latA / TILE_DEG).astype(int).tolist())])
sel_b = np.array([k in keep for k in zip(np.floor(lonB / TILE_DEG).astype(int).tolist(),
                                         np.floor(latB / TILE_DEG).astype(int).tolist())])
sel_n = np.array([k in keep for k in zip(np.floor(lonN / TILE_DEG).astype(int).tolist(),
                                         np.floor(latN / TILE_DEG).astype(int).tolist())])

posA = pos[is_a][sel_a]
posB = pos[is_b][sel_b]
negS = neg[sel_n]

# One fold map for every configuration, built over the blocks of the shared
# tiles so that each fold holds a comparable share of both populations.
ub = np.unique(blkl[np.r_[posA, posB, negS]])
fold_of = {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}
fold_all = np.array([fold_of.get(b, -1) for b in blkl])
fo = dict(a=fold_all[posA], b=fold_all[posB], n=fold_all[negS])


def run(train_pos_list, eval_map):
    """train_pos_list: positive arrays used for calibration; eval_map: name -> cells."""
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
        apr = np.array([r["ap"] for r in rows])
        out[name] = {"folds": rows, "auc_mean": float(arr.mean()),
                     "auc_sd": float(arr.std(ddof=1)),
                     "ap_mean": float(apr.mean())}
    return out


res = {}
res["design"] = dict(split_km=SPLIT_KM, tile_deg=TILE_DEG, block_km=BLK_KM,
                     buffer_km=BUF_KM, n_fold=N_FOLD, neg_ratio=NEG_RATIO,
                     n_tiles=len(keep), tiles=sorted([list(k) for k in keep]),
                     n_a=int(len(posA)), n_b=int(len(posB)), n_neg=int(len(negS)))
res["A_to_A"] = run([posA], {"A": posA})["A"]
res["A_to_B"] = run([posA], {"B": posB})["B"]
res["B_to_B"] = run([posB], {"B": posB})["B"]
res["B_to_A"] = run([posB], {"A": posA})["A"]
pool = run([posA, posB], {"A": posA, "B": posB})
res["pooled_A"] = pool["A"]
res["pooled_B"] = pool["B"]

# bootstrap over 50 km spatial blocks within each held-out fold
def boot(pos_groups, target, n=N_BOOT):
    vals = []
    rng = np.random.default_rng(SEED)
    preds = {}
    for f in range(N_FOLD):
        train_pos = np.unique(np.concatenate(pos_groups))
        tr = np.r_[train_pos[fold_all[train_pos] != f], negS[fo["n"] != f]]
        ytr = np.r_[np.ones((fold_all[train_pos] != f).sum()), np.zeros((fo["n"] != f).sum())]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr], ytr)
        te = np.r_[target[fold_all[target] == f], negS[fo["n"] == f]]
        yte = np.r_[np.ones((fold_all[target] == f).sum()), np.zeros((fo["n"] == f).sum())]
        p = m.predict_proba(X[te])[:, 1]
        blk = np.r_[blkl[target[fold_all[target] == f]], blkl[negS[fo["n"] == f]]]
        preds[f] = (yte, p, blk)
    for _ in range(n):
        ys, ps = [], []
        for f, (yte, p, blk) in preds.items():
            ubk = np.unique(blk)
            pick = rng.choice(ubk, size=len(ubk), replace=True)
            idx = np.concatenate([np.nonzero(blk == b)[0] for b in pick])
            ys.append(yte[idx]); ps.append(p[idx])
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        vals.append(roc_auc_score(ys, ps))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

res["A_to_A"]["ci95"] = boot([posA], posA)
res["A_to_B"]["ci95"] = boot([posA], posB)
res["B_to_B"]["ci95"] = boot([posB], posB)
res["B_to_A"]["ci95"] = boot([posB], posA)
res["pooled_A"]["ci95"] = boot([posA, posB], posA)
res["pooled_B"]["ci95"] = boot([posA, posB], posB)
res["manuscript_table1"] = {
    key: [round(res[key]["auc_mean"], 3), round(res[key]["auc_sd"], 3),
          [round(res[key]["ci95"][0], 3), round(res[key]["ci95"][1], 3)]]
    for key in ("A_to_A", "A_to_B", "B_to_B", "B_to_A", "pooled_A", "pooled_B")
}
res["note"] = ("Table 1 reports the five-fold mean, the sample standard "
               "deviation and the 2.5th-97.5th percentile interval of 1,000 "
               "spatial-block bootstrap resamples; manuscript_table1 lists the "
               "same values in the row order of Table 1.")
res["elapsed_s"] = round(time.time() - t0, 1)
RESULT_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)
with open(os.path.join(RESULT_DIR, "group_transfer.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps({k: (v.get("auc_mean"), v.get("auc_sd"), v.get("ci95")) for k, v in res.items() if isinstance(v, dict) and "auc_mean" in v}, indent=2))
print("design", res["design"])
