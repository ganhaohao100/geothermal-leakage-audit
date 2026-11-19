# -*- coding: utf-8 -*-
"""Permutation importance under both scorings, to fix the metric label.

step2_model.py uses scoring="average_precision", so the numbers in
cv_report.json (e.g. s_den 0.587) are PR-AUC decreases. The figure axis labels
say "dAUC". This script recomputes fold-1 permutation importance under both
scorings so the two can be compared before we rename the axes.
"""
import io
import json
import os
import sys

import numpy as np
import lightgbm as lgb
from scipy.spatial import cKDTree
from sklearn.inspection import permutation_importance

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
CASE_OUT = OUTD
from common import TR

BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
FEATS = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
         "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
S_FIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]
ALL = FEATS + S_FIELDS
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

z = np.load(os.path.join(BASE, "out", "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
F = {k: z[k] for k in z.files if z[k].shape == mask.shape}
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
Xl = np.column_stack([np.asarray(F[k], np.float32).ravel()[il] for k in ALL])
pxl, pyl = px[il], py[il]
blkl = (np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000 + \
       (np.floor(pyl / BLK_KM).astype(np.int64) + 1000)


def build(pos_land):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
    n_neg = min(len(cand), NEG_RATIO * len(pos_i))
    return pos_i, rng.choice(cand, size=n_neg, replace=False)


def fold1(pos_i, neg_i, feats):
    cols = [ALL.index(f) for f in feats]
    si = np.r_[pos_i, neg_i]
    X = Xl[np.ix_(si, cols)]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx = np.r_[pxl[pos_i], pxl[neg_i]]
    sy = np.r_[pyl[pos_i], pyl[neg_i]]
    blk = np.r_[blkl[pos_i], blkl[neg_i]]
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}
    fold = np.array([fo[b] for b in blk])
    te = fold == 0
    tr = ~te
    tv = cKDTree(np.column_stack([sx[te], sy[te]]))
    dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
    m = lgb.LGBMClassifier(**PARAMS)
    tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
    m.fit(X[tr_idx], y[tr_idx])
    out = {}
    for scoring in ("roc_auc", "average_precision"):
        pi = permutation_importance(m, X[te], y[te], n_repeats=5,
                                    random_state=SEED, scoring=scoring)
        out[scoring] = sorted(zip(feats, pi.importances_mean.tolist()),
                              key=lambda t: -t[1])
    return out


RES = {}
for tgt, lk, feats in (("spring_all", "lab_spring", ALL),
                       ("well_geo", "lab_well", FEATS),
                       ("well_all", "lab_well", ALL)):
    pos_i, neg_i = build(np.asarray(F[lk], np.float32).ravel()[il] > 0)
    RES[tgt] = fold1(pos_i, neg_i, feats)
    print("=== %s" % tgt, flush=True)
    for scoring in ("average_precision", "roc_auc"):
        top = RES[tgt][scoring]
        print("   %-18s top1 %-9s %+.4f | top2 %-9s %+.4f | ratio %.1fx"
              % (scoring, top[0][0], top[0][1], top[1][0], top[1][1],
                 top[0][1] / max(top[1][1], 1e-9)), flush=True)

json.dump(RES, io.open(os.path.join(CASE_OUT, "perm_metric_check.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print("done")