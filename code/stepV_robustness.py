# -*- coding: utf-8 -*-
"""Step U  Robustness checks for reviewer 2.4.

Part A  Pseudo-negative sampling variance: 10 independent 1:3 draws per
        configuration, each with a fresh spatial fold map.
Part B  Algorithm heterogeneity: the same four configurations under a random
        forest instead of LightGBM.
Part C  Table 1 sensitivity: (i) min_child_samples = 10 and (ii) group A
        downsampled to n = 110, to separate leaf capacity and sample size
        from a genuine difference in decision surfaces.

Output: results/robustness.json
"""
import os, sys, json, time, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR  # noqa: E402
from common import TR  # noqa: E402

BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
N_REP = 10
SEED = 20260927
SPLIT_KM, TILE_DEG = 50.0, 5.0
MIN_B, MIN_A = 5, 10

FEATS_SPRING = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
                "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
FEATS_ALL = FEATS_SPRING + ["s_T", "s_lgQ", "s_heat", "s_den"]

PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

JOBS = [("spring_geo", "lab_spring", FEATS_SPRING),
        ("spring_all", "lab_spring", FEATS_ALL),
        ("well_geo", "lab_well", FEATS_SPRING),
        ("well_all", "lab_well", FEATS_ALL)]

t0 = time.time()
z = np.load(os.path.join(DATA_DIR, "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
ny, nx = mask.shape
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
Xl = np.column_stack([np.asarray(z[k], np.float32).ravel()[il]
                      for k in FEATS_ALL])
pxl, pyl = px[il], py[il]
blkl = ((np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000
        + (np.floor(pyl / BLK_KM).astype(np.int64) + 1000))
print("land cells %d, features %d, blocks %d" %
      (len(il), Xl.shape[1], len(np.unique(blkl))), flush=True)


def candidates(pos_land):
    """Cells outside the positive set and beyond the 10 km positive buffer."""
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
    return pos_i, cand


def draw_neg(cand, n_pos, seed):
    rng = np.random.default_rng(seed)
    n_neg = min(len(cand), NEG_RATIO * n_pos)
    return rng.choice(cand, size=n_neg, replace=False)


def fold_split(blk, seed):
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(seed).permutation(ub))}
    return np.array([fo[b] for b in blk])


def block_cv(pos_i, neg_i, feats, seed, clf_factory="lgb", extra_params=None):
    cols = [FEATS_ALL.index(f) for f in feats]
    si = np.r_[pos_i, neg_i]
    X = Xl[np.ix_(si, cols)]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx = np.r_[pxl[pos_i], pxl[neg_i]]
    sy = np.r_[pyl[pos_i], pyl[neg_i]]
    blk = np.r_[blkl[pos_i], blkl[neg_i]]
    fold = fold_split(blk, seed)
    rows = []
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        if clf_factory == "rf":
            m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                       max_features="sqrt", n_jobs=-1,
                                       random_state=SEED)
        else:
            p = dict(PARAMS)
            p["random_state"] = seed
            if extra_params:
                p.update(extra_params)
            m = lgb.LGBMClassifier(**p)
        m.fit(X[tr_idx], y[tr_idx])
        pr = m.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, auc=float(roc_auc_score(y[te], pr)),
                         ap=float(average_precision_score(y[te], pr)),
                         n_train=int(len(tr_idx)),
                         n_pos=int(y[te].sum()),
                         n_neg=int((y[te] == 0).sum())))
    auc = np.array([r["auc"] for r in rows])
    ap = np.array([r["ap"] for r in rows])
    return dict(auc_mean=float(auc.mean()), auc_sd=float(auc.std(ddof=1)),
                auc_min=float(auc.min()), auc_max=float(auc.max()),
                ap_mean=float(ap.mean()), ap_sd=float(ap.std(ddof=1)),
                folds=rows,
                n_train_mean=float(np.mean([r["n_train"] for r in rows])))


RES = {}

# ---------------------------------------------------------------- Part A
print("\n=== Part A: 10 independent pseudo-negative draws ===", flush=True)
RES["pseudoneg_repeats"] = dict(n_rep=N_REP, configurations={})
for key, lab_key, feats in JOBS:
    pos_land = np.asarray(z[lab_key], np.float32).ravel()[il] > 0
    pos_i, cand = candidates(pos_land)
    reps = []
    for r in range(N_REP):
        s = SEED + 101 * r
        neg_i = draw_neg(cand, len(pos_i), s)
        rec = block_cv(pos_i, neg_i, feats, s)
        reps.append(rec)
        print("  %-11s rep %2d  AUC %.4f  AP %.4f" %
              (key, r + 1, rec["auc_mean"], rec["ap_mean"]), flush=True)
    a = np.array([r["auc_mean"] for r in reps])
    q = np.array([r["ap_mean"] for r in reps])
    RES["pseudoneg_repeats"]["configurations"][key] = dict(
        auc_mean=float(a.mean()), auc_sd=float(a.std(ddof=1)),
        auc_min=float(a.min()), auc_max=float(a.max()),
        auc_ci95=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        ap_mean=float(q.mean()), ap_sd=float(q.std(ddof=1)),
        ap_min=float(q.min()), ap_max=float(q.max()),
        per_draw=[round(float(v), 4) for v in a])

# ---------------------------------------------------------------- Part B
print("\n=== Part B: random-forest replication ===", flush=True)
RES["random_forest"] = dict(n_estimators=500, configurations={})
for key, lab_key, feats in JOBS:
    pos_land = np.asarray(z[lab_key], np.float32).ravel()[il] > 0
    pos_i, cand = candidates(pos_land)
    neg_i = draw_neg(cand, len(pos_i), SEED)
    rec = block_cv(pos_i, neg_i, feats, SEED, clf_factory="rf")
    RES["random_forest"]["configurations"][key] = rec
    print("  %-11s AUC %.4f ± %.4f  AP %.4f" %
          (key, rec["auc_mean"], rec["auc_sd"], rec["ap_mean"]), flush=True)

# ---------------------------------------------------------------- Part C
print("\n=== Part C: Table 1 sensitivity ===", flush=True)
W = np.asarray(z["lab_well"], np.float32)
pos = np.nonzero(W.ravel()[il] > 0)[0]
slon, slat = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
spx, spy = TR.transform(slon, slat)
spx, spy = spx / 1000.0, spy / 1000.0
dspring, _ = cKDTree(np.column_stack([spx, spy])).query(
    np.column_stack([pxl[pos], pyl[pos]]), k=1)
is_a, is_b = dspring <= SPLIT_KM, dspring > SPLIT_KM

pos_land = W.ravel()[il] > 0
_, cand = candidates(pos_land)
neg = draw_neg(cand, len(pos), SEED)
neg = np.setdiff1d(neg, pos)

lonA = LO.ravel()[il][pos][is_a]; latA = LA.ravel()[il][pos][is_a]
lonB = LO.ravel()[il][pos][is_b]; latB = LA.ravel()[il][pos][is_b]
lonN = LO.ravel()[il][neg]; latN = LA.ravel()[il][neg]
from collections import Counter
def tc(lo, la):
    return Counter(zip(np.floor(lo / TILE_DEG).astype(int).tolist(),
                       np.floor(la / TILE_DEG).astype(int).tolist()))
cA, cB, cN = tc(lonA, latA), tc(lonB, latB), tc(lonN, latN)
keep = {k for k in cB if cB[k] >= MIN_B and cA.get(k, 0) >= MIN_A}
def in_keep(lo, la):
    return np.array([k in keep for k in zip(
        np.floor(lo / TILE_DEG).astype(int).tolist(),
        np.floor(la / TILE_DEG).astype(int).tolist())])
posA = pos[is_a][in_keep(lonA, latA)]
posB = pos[is_b][in_keep(lonB, latB)]
negS = neg[in_keep(lonN, latN)]

ub = np.unique(blkl[np.r_[posA, posB, negS]])
fold_of = {b: i % N_FOLD for i, b in enumerate(
    np.random.default_rng(SEED).permutation(ub))}
fold_all = np.array([fold_of.get(b, -1) for b in blkl])


def table1_run(train_pos_list, eval_map, extra_params=None, cap_a=None,
               folds=5):
    train_pos = np.unique(np.concatenate(train_pos_list))
    out = {}
    rng = np.random.default_rng(SEED + 3)
    for f in range(folds):
        trp = train_pos[fold_all[train_pos] != f]
        if cap_a is not None and len(trp) > cap_a:
            trp = rng.choice(trp, size=cap_a, replace=False)
        trn = negS[fold_all[negS] != f]
        tr = np.r_[trp, trn]
        ytr = np.r_[np.ones(len(trp)), np.zeros(len(trn))]
        p = dict(PARAMS)
        if extra_params:
            p.update(extra_params)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xl[np.ix_(tr, [FEATS_ALL.index(c) for c in FEATS_SPRING])], ytr)
        for name, cells in eval_map.items():
            tep = cells[fold_all[cells] == f]
            ten = negS[fold_all[negS] == f]
            if len(tep) == 0:
                continue
            te = np.r_[tep, ten]
            yte = np.r_[np.ones(len(tep)), np.zeros(len(ten))]
            pr = m.predict_proba(
                Xl[np.ix_(te, [FEATS_ALL.index(c) for c in FEATS_SPRING])])[:, 1]
            out.setdefault(name, []).append(float(roc_auc_score(yte, pr)))
    return {k: dict(auc=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                    folds=[round(x, 4) for x in v]) for k, v in out.items()}


def table1_block(extra_params=None, cap_a=None, label=""):
    res = {}
    res["A_to_A"] = table1_run([posA], {"A": posA}, extra_params, cap_a)["A"]
    res["A_to_B"] = table1_run([posA], {"B": posB}, extra_params, cap_a)["B"]
    res["B_to_B"] = table1_run([posB], {"B": posB}, extra_params, None)["B"]
    res["B_to_A"] = table1_run([posB], {"A": posA}, extra_params, None)["A"]
    pool = table1_run([posA, posB], {"A": posA, "B": posB}, extra_params, None)
    res["pooled_A"], res["pooled_B"] = pool["A"], pool["B"]
    print("  %-28s A->A %.3f  A->B %.3f  B->B %.3f  B->A %.3f  pool %.3f/%.3f" %
          (label, res["A_to_A"]["auc"], res["A_to_B"]["auc"],
           res["B_to_B"]["auc"], res["B_to_A"]["auc"],
           res["pooled_A"]["auc"], res["pooled_B"]["auc"]), flush=True)
    return res

RES["table1_sensitivity"] = dict(
    design=dict(n_a=int(len(posA)), n_b=int(len(posB)), n_neg=int(len(negS)),
                n_tiles=len(keep), split_km=SPLIT_KM, block_km=BLK_KM,
                buffer_km=BUF_KM, neg_ratio=NEG_RATIO),
    base=table1_block(label="baseline (min_child=30)"),
    min_child_10=table1_block({"min_child_samples": 10},
                              label="min_child_samples=10"),
    a_downsampled=np.nan)
RES["table1_sensitivity"]["a_downsampled"] = table1_block(
    None, cap_a=int(len(posB)), label="A downsampled to n=110")

RES["elapsed_s"] = round(time.time() - t0, 1)
_RES = os.path.join(_ROOT, "results")
os.makedirs(_RES, exist_ok=True)
with open(os.path.join(_RES, "robustness.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=1, ensure_ascii=False)
print("\nwritten results/robustness.json  (%.0fs)" % (time.time() - t0))
