# -*- coding: utf-8 -*-
"""Fault-increment validation for the leakage-free reservoir-type baseline.

Two independent diagnostics of the claim that "distance to the nearest fault
segment carries no information beyond the heat-flow fields at screening scale":

  A. feature ablation   - retrain the identical PU + spatial-block CV harness
                          with fault / heat-flow features removed, paired by fold
  B. redundancy check   - univariate per-fold AUC of every geo feature, and the
                          spatial correlation matrix between fault and heat-flow
                          fields (the credit-splitting concern for permutation
                          importance)

The baseline project is read-only here; everything is written to the case out dir.
"""
import io
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

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
FAULT = ["f_dist", "f_den", "f_end"]
HEAT = ["hf_bg", "hf_res", "hf_grad"]
DROP = {f: [g for g in FEATS if g != f] for f in FEATS}
DROP["no_fault"] = [f for f in FEATS if f not in FAULT]
DROP["no_hf"] = [f for f in FEATS if f not in HEAT]

PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE, "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
F = {k: z[k] for k in z.files if z[k].shape == mask.shape}
ny, nx = mask.shape
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0

il = np.nonzero(mask.ravel())[0]
Xl = np.column_stack([np.asarray(F[k], np.float32).ravel()[il] for k in FEATS])
pxl, pyl = px[il], py[il]
blkl = (np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000 + \
       (np.floor(pyl / BLK_KM).astype(np.int64) + 1000)
print("land samples %d ; features %d ; blocks %d"
      % (len(il), Xl.shape[1], len(np.unique(blkl))), flush=True)


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


DS = {}
for name, lk in (("well", "lab_well"), ("spring", "lab_spring")):
    pos_land = np.asarray(F[lk], np.float32).ravel()[il] > 0
    pos_i, neg_i, cand = build_dataset(pos_land)
    si = np.r_[pos_i, neg_i]
    DS[name] = dict(
        pos_i=pos_i, neg_i=neg_i, cand=cand,
        y=np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))],
        sx=np.r_[pxl[pos_i], pxl[neg_i]], sy=np.r_[pyl[pos_i], pyl[neg_i]],
        blk=np.r_[blkl[pos_i], blkl[neg_i]],
        fold=fold_split(np.r_[blkl[pos_i], blkl[neg_i]], np.random.default_rng(SEED)),
        rows=np.r_[pos_i, neg_i])
    print("%s: pos %d neg %d" % (name, len(pos_i), len(neg_i)), flush=True)


def run_cv(ds, feats):
    cols = [FEATS.index(f) for f in feats]
    X = Xl[np.ix_(ds["rows"], cols)]
    y = ds["y"]
    rows, perm_rows = [], None
    for k in range(N_FOLD):
        te = ds["fold"] == k
        tr = ~te
        tv = cKDTree(np.column_stack([ds["sx"][te], ds["sy"][te]]))
        dd, _ = tv.query(np.column_stack([ds["sx"][tr], ds["sy"][tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
    return rows


def univariate(ds):
    """per-fold AUC of each single geo feature used directly as the score"""
    X = Xl[ds["rows"]]
    y = ds["y"]
    out = {f: [] for f in FEATS}
    for k in range(N_FOLD):
        te = ds["fold"] == k
        for j, f in enumerate(FEATS):
            col = X[te, j]
            ok = np.isfinite(col)
            if ok.sum() < 100 or len(np.unique(y[te][ok])) < 2:
                out[f].append(float("nan"))
                continue
            out[f].append(float(roc_auc_score(y[te][ok], col[ok])))
    return {f: [float(v) for v in vs] for f, vs in out.items()}


def summarize(rows):
    a = np.array([r["auc"] for r in rows])
    p = np.array([r["ap"] for r in rows])
    return dict(auc_mean=float(a.mean()), auc_sd=float(a.std()),
                auc_range=float(a.max() - a.min()),
                ap_mean=float(p.mean()), folds=[float(v) for v in a])


RESULT = {"meta": dict(seed=SEED, block_km=BLK_KM, buffer_km=BUF_KM, folds=N_FOLD,
                       feats=FEATS, fault=FAULT, heat=HEAT)}

for name in ("well", "spring"):
    ds = DS[name]
    RESULT[name] = {"ablation": {}, "univariate": univariate(ds)}
    # reference first, so every delta is paired against the full feature set
    order = [("ref10", FEATS), ("no_fdist", DROP["f_dist"]), ("no_fault", DROP["no_fault"]),
             ("no_hf", DROP["no_hf"])]
    if name == "spring":
        order = [("ref10", FEATS), ("no_fault", DROP["no_fault"]), ("no_hf", DROP["no_hf"])]
    for tag, feats in order:
        rows = run_cv(ds, feats)
        RESULT[name]["ablation"][tag] = dict(feats=feats, **summarize(rows), cv=rows)
        print("  %-9s n=%2d  AUC %.4f (range %.4f)  drops %s"
              % (name + "/" + tag, len(feats),
                 RESULT[name]["ablation"][tag]["auc_mean"],
                 RESULT[name]["ablation"][tag]["auc_range"],
                 ",".join(f for f in FEATS if f not in feats)), flush=True)

# paired fold-wise deltas
for name in ("well", "spring"):
    ref = RESULT[name]["ablation"]["ref10"]["folds"]
    for tag, blk in RESULT[name]["ablation"].items():
        if tag == "ref10":
            continue
        d = np.array(ref) - np.array(blk["folds"])
        blk["delta_auc_mean"] = float(d.mean())
        blk["delta_auc_folds"] = [float(v) for v in d]
        print("  %s %s : delta AUC %+.4f  per-fold %s"
              % (name, tag, d.mean(), np.round(d, 4).tolist()), flush=True)

# ---------------------------------------------------------------- redundancy
sub = np.unique(np.random.default_rng(SEED).choice(len(il), size=min(120000, len(il)),
                                                   replace=False))
COL = {f: Xl[sub, j].astype(float) for j, f in enumerate(FEATS)}
rho = np.full((len(FEATS), len(FEATS)), np.nan)
for a in range(len(FEATS)):
    for b in range(len(FEATS)):
        ca, cb = COL[FEATS[a]], COL[FEATS[b]]
        ok = np.isfinite(ca) & np.isfinite(cb)
        if ok.sum() > 500:
            rho[a, b] = float(spearmanr(ca[ok], cb[ok]).statistic)
RESULT["redundancy"] = dict(features=FEATS, spearman=rho.tolist(),
                            pairs={})
for f in FAULT:
    for h in HEAT:
        RESULT["redundancy"]["pairs"]["%s~%s" % (f, h)] = float(
            rho[FEATS.index(f), FEATS.index(h)])
print("\nSpearman rho, fault vs heat-flow fields:", flush=True)
for k, v in RESULT["redundancy"]["pairs"].items():
    print("   %-18s %+.3f" % (k, v), flush=True)

json.dump(RESULT, io.open(os.path.join(CASE_OUT, "fault_increment.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
print("\ndone %.0fs -> fault_increment.json" % (time.time() - t0), flush=True)