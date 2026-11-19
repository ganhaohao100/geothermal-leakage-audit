# -*- coding: utf-8 -*-
"""Step Q3  P2-10：交叉验证缓冲宽度扫描（判据可操作化的直接证据）

Gate 2 现状只说"带缓冲的空间阻断"，未给可操作判据。本脚本实测：
在保持 headline 其余口径不变的情况下，把 CV 缓冲从 10 km 加到 25/50/75/100 km，
看泉型与井型的折均值 AUC 是否回落。若在远超相关长度（~38 km）的缓冲下虚高仍存，
则"加大缓冲"不能替代排除目标派生场（Gate 1）。

结果并入 out/p2_diagnostics.json 的 buffer_sweep 键。
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import TR, RES, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, POS_BUF_KM = 50.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
BUFFERS = [10.0, 25.0, 50.0, 75.0, 100.0]
FEATS_GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
SFIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]
FEATS_ALL = FEATS_GEO + SFIELDS
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
LO, LA = np.meshgrid(lon_c, lat_c)
il = np.nonzero(mask.ravel())[0]
pxx, pyy = TR.transform(LO.ravel(), LA.ravel())
pxl, pyl = pxx[il] / 1000.0, pyy[il] / 1000.0
blkl = ((np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000
        + (np.floor(pyl / BLK_KM).astype(np.int64) + 1000))
Xl = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_ALL])
POS = {"spring": np.asarray(z["lab_spring"], np.float32).ravel()[il] > 0,
       "well": np.asarray(z["lab_well"], np.float32).ravel()[il] > 0}


def build(pos_land):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
    return pos_i, rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)),
                             replace=False)


def run(pos_i, neg_i, buf_km):
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx, sy = pxl[si], pyl[si]
    blk = blkl[si]
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(SEED).permutation(ub))}
    fold = np.array([fo[b] for b in blk])
    X = Xl[si]
    rows = []
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
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
    a = np.array([r["auc"] for r in rows])
    q = np.array([r["ap"] for r in rows])
    return dict(auc=round(float(a.mean()), 4), auc_sd=round(float(a.std()), 4),
                auc_rng=round(float(a.max() - a.min()), 4),
                ap=round(float(q.mean()), 4),
                folds_auc=[round(float(v), 4) for v in a],
                n_train_mean=round(float(np.mean([r["n_train"] for r in rows])), 1))


OUT = {}
print("buffer sweep (spring / well, 14 fields, headline otherwise)", flush=True)
print("  target  buf_km  n_train  AUC     SD      range   AP", flush=True)
for tag, pos in POS.items():
    pos_i, neg_i = build(pos)
    OUT[tag] = []
    for b in BUFFERS:
        r = run(pos_i, neg_i, b)
        r["buf_km"] = b
        OUT[tag].append(r)
        print("  %-7s %5.0f  %7.0f  %.4f  %.4f  %.4f  %.4f"
              % (tag, b, r["n_train_mean"], r["auc"], r["auc_sd"], r["auc_rng"], r["ap"]),
              flush=True)

P = os.path.join(OUTD, "p2_diagnostics.json")
R = json.load(io.open(P, encoding="utf-8"))
R["buffer_sweep"] = dict(buffers_km=BUFFERS, results=OUT,
                         note=("headline buffer is 10 km; the finest target-derived "
                               "field has an effective correlation length of ~38 km "
                               "(3.4 x 11 km), so 25-100 km buffers exceed it"))
R["elapsed_s"] = round(R.get("elapsed_s", 0) + time.time() - t0, 1)
json.dump(R, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nmerged buffer_sweep into p2_diagnostics.json (%.0fs)" % (time.time() - t0), flush=True)