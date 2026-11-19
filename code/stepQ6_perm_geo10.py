# -*- coding: utf-8 -*-
"""Step Q6  P2-1：十场配置的跨折置换重要性（Table 1 需要与 14 场同口径）

四种配置（全国泉型/井型、华南泉型/井型）分别在「仅十项地质与地球物理场」
特征集下计算五折置换重要性（AP 与 AUC），报 mean ± SD 与首位/次位比值。
结果并入 out/p2_diagnostics.json 的 perm_importance_geo10。
"""
import io, json, os, sys, time
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
from common import load_cache, kernel_smooth, rasterize_points, TR, RES, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD, N_REP = 3, 5, 5
SEED = 20260927
SIG_BASE, DEN_CELLS, DEG_KM = 0.40, 2.0, 111.32
BBOX = (108.0, 122.0, 20.0, 30.0)
T0C, RHO_W, CP = 10.0, 990.0, 4180.0
GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
       "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)


def run(Xl, pos_i, neg_i, pxl, pyl, names):
    cols = [GEO.index(f) for f in names]
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    X = Xl[np.ix_(si, cols)]
    sx, sy = pxl[si], pyl[si]
    blk = ((np.floor(sx / BLK_KM).astype(np.int64) + 1000) * 10000
           + (np.floor(sy / BLK_KM).astype(np.int64) + 1000))
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(SEED).permutation(ub))}
    fold = np.array([fo[b] for b in blk])
    pap = np.full((N_FOLD, len(names)), np.nan)
    pauc = np.full((N_FOLD, len(names)), np.nan)
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        for scoring, mat in (("average_precision", pap), ("roc_auc", pauc)):
            pi = permutation_importance(m, X[te], y[te], n_repeats=N_REP,
                                        random_state=SEED, scoring=scoring)
            mat[k] = pi.importances_mean
    o_ap = list(np.argsort(-pap.mean(0)))
    o_auc = list(np.argsort(-pauc.mean(0)))
    return dict(perm_ap={f: round(float(v), 4) for f, v in zip(names, pap.mean(0))},
                perm_ap_sd={f: round(float(v), 4) for f, v in zip(names, pap.std(0))},
                perm_auc={f: round(float(v), 4) for f, v in zip(names, pauc.mean(0))},
                rank_ap=[names[i] for i in o_ap],
                cliff_ap=dict(top1=names[o_ap[0]], top2=names[o_ap[1]],
                              ratio=round(float(pap.mean(0)[o_ap[0]]
                                                / max(pap.mean(0)[o_ap[1]], 1e-9)), 1)),
                cliff_auc=dict(top1=names[o_auc[0]], top2=names[o_auc[1]],
                               ratio=round(float(pauc.mean(0)[o_auc[0]]
                                                 / max(pauc.mean(0)[o_auc[1]], 1e-9)), 1)))


t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_n, lat_n, mask_n = z["lon_c"], z["lat_c"], z["mask"]
LO_n, LA_n = np.meshgrid(lon_n, lat_n)
X0, Y0 = TR.transform(LO_n.ravel(), LA_n.ravel())
il_n = np.nonzero(mask_n.ravel())[0]
pxn, pyn = X0[il_n] / 1000.0, Y0[il_n] / 1000.0
GEO_n = np.column_stack([np.asarray(z[k], np.float32).ravel()[il_n] for k in GEO])

ix = np.nonzero((lon_n >= BBOX[0]) & (lon_n <= BBOX[1]))[0]
iy = np.nonzero((lat_n >= BBOX[2]) & (lat_n <= BBOX[3]))[0]
lon_r, lat_r = lon_n[ix], lat_n[iy]
LO_r, LA_r = np.meshgrid(lon_r, lat_r)
mask_r = np.asarray(mask_n)[np.ix_(iy, ix)]
Xr, Yr = TR.transform(LO_r.ravel(), LA_r.ravel())
il_r = np.nonzero(mask_r.ravel())[0]
pxr, pyr = Xr[il_r] / 1000.0, Yr[il_r] / 1000.0
GEO_r = np.column_stack([np.asarray(z[k], np.float32)[np.ix_(iy, ix)].ravel()[il_r]
                         for k in GEO])

CFG = {
    "national_spring": (GEO_n, pxn, pyn, np.asarray(z["lab_spring"], np.float32).ravel()[il_n] > 0),
    "national_well": (GEO_n, pxn, pyn, np.asarray(z["lab_well"], np.float32).ravel()[il_n] > 0),
    "region_spring": (GEO_r, pxr, pyr,
                      np.asarray(z["lab_spring"], np.float32)[np.ix_(iy, ix)].ravel()[il_r] > 0),
    "region_well": (GEO_r, pxr, pyr,
                    np.asarray(z["lab_well"], np.float32)[np.ix_(iy, ix)].ravel()[il_r] > 0),
}
OUT = {}
for tag, (Xl, pxl, pyl, pos) in CFG.items():
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos) & (dmin > POS_BUF_KM))[0]
    neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)
    r = run(Xl, pos_i, neg_i, pxl, pyl, GEO)
    OUT[tag] = r
    print("  %-16s top3(AP): %s | ratio(AP) %.1fx / (AUC) %.1fx"
          % (tag, ", ".join("%s %.4f±%.4f" % (f, r["perm_ap"][f], r["perm_ap_sd"][f])
                            for f in r["rank_ap"][:3]),
             r["cliff_ap"]["ratio"], r["cliff_auc"]["ratio"]), flush=True)

P = os.path.join(OUTD, "p2_diagnostics.json")
R = json.load(io.open(P, encoding="utf-8"))
R["perm_importance_geo10"] = OUT
R["elapsed_s"] = round(R.get("elapsed_s", 0) + time.time() - t0, 1)
json.dump(R, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("merged perm_importance_geo10 (%.0fs)" % (time.time() - t0), flush=True)