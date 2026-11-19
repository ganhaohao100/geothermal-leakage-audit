# -*- coding: utf-8 -*-
"""Step S  sigma 扫描对照实验（202610靶区优选 案例）

两个目的：
 1) G1 复现性核对：在同一脚本内重跑 step2_model.py 的四套配置，与 cv_report.json 对比。
 2) 解决 D4 口径不一致：把 4 个温泉派生场统一在同一 sigma 下重建，扫描
    sigma in {0.05, 0.10, 0.20, 0.40, 0.80} 度，检验泄漏是否随平滑尺度变化。

只读基线项目的 features.npz 与 cv_report.json，输出写回本案例目录。
"""
import os, sys, json, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import kernel_smooth, rasterize_points, OUT as BASE_OUT, TR, RES

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTD, exist_ok=True)

BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
T0C = 10.0
RHO_W, CP = 990.0, 4180.0
SIGMAS = [0.05, 0.10, 0.20, 0.40, 0.80]

FEATS_SPRING = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
                "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
SPRING_FIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]
FEATS_WELL = FEATS_SPRING + SPRING_FIELDS
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t_all = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
cell_km2 = z["cell_km2"]
ny, nx = mask.shape
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
pxl, pyl = px[il], py[il]
bx = np.floor(pxl / BLK_KM).astype(np.int64)
by = np.floor(pyl / BLK_KM).astype(np.int64)
blkl = (bx + 1000) * 10000 + (by + 1000)
GEO = {k: np.asarray(z[k], np.float32) for k in FEATS_SPRING}
SLON, SLAT = z["slon"], z["slat"]
ST, SQ, H = np.asarray(z["sT"], float), np.asarray(z["sQ"], float), np.asarray(z["H"], float)
LAB = {"lab_spring": z["lab_spring"], "lab_well": z["lab_well"]}
print("land cells %d ; spring points %d" % (len(il), len(SLON)), flush=True)


def spring_fields(sigma=None):
    """sigma=None -> 基线原始场（s_T/s_lgQ/s_heat 为 0.40 度，s_den 为 2 格高斯）。"""
    if sigma is None:
        out = {k: np.asarray(z[k], np.float32)[mask] for k in SPRING_FIELDS}
    else:
        f = {}
        f["s_T"] = kernel_smooth(SLON, SLAT, ST, LO, LA, sigma)[0]
        f["s_lgQ"] = kernel_smooth(SLON, SLAT, np.log10(SQ + 1.0), LO, LA, sigma)[0]
        f["s_heat"] = kernel_smooth(SLON, SLAT, np.log10(H + 1.0), LO, LA, sigma)[0]
        den = kernel_smooth(SLON, SLAT, np.ones_like(SLON), LO, LA, sigma)[1]
        f["s_den"] = den / cell_km2 * 1e4
        out = {k: np.asarray(f[k], np.float32)[mask] for k in SPRING_FIELDS}
    return out


def build_Xl(sp):
    cols = [GEO[k].ravel()[il] for k in FEATS_SPRING]
    cols += [np.asarray(sp[k], np.float32) for k in SPRING_FIELDS]
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
    cols = [FEATS_WELL.index(f) for f in feats]
    si = np.r_[pos_i, neg_i]
    X = Xl[np.ix_(si, cols)]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx = np.r_[pxl[pos_i], pxl[neg_i]]
    sy = np.r_[pyl[pos_i], pyl[neg_i]]
    blk = np.r_[blkl[pos_i], blkl[neg_i]]
    fold = fold_split(blk, np.random.default_rng(SEED))
    rows, perm_folds = [], []
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        if want_perm:
            pi = permutation_importance(m, X[te], y[te], n_repeats=5,
                                        random_state=SEED, scoring="average_precision")
            perm_folds.append(dict(zip(feats, pi.importances_mean.tolist())))
    perm_rows = None
    if perm_folds:
        means = {f: float(np.mean([d[f] for d in perm_folds])) for f in feats}
        sds = {f: float(np.std([d[f] for d in perm_folds], ddof=1)) for f in feats}
        perm_rows = sorted(((f, means[f], sds[f], [d[f] for d in perm_folds])
                            for f in feats), key=lambda t: -t[1])
    return rows, perm_rows


def summ(rows):
    a = np.array([r["auc"] for r in rows])
    p = np.array([r["ap"] for r in rows])
    return dict(auc=float(a.mean()), auc_sd=float(a.std()),
                auc_rng=float(a.max() - a.min()), ap=float(p.mean()))


# ---------- 1. G1 复现性核对 ----------
PUB = json.load(open(os.path.join(BASE_OUT, "cv_report.json"), encoding="utf-8"))
CONFIGS = [("spring_geo", "lab_spring", FEATS_SPRING),
           ("spring_all", "lab_spring", FEATS_WELL),
           ("well_geo", "lab_well", FEATS_SPRING),
           ("well_all", "lab_well", FEATS_WELL)]
SP_PUB = spring_fields(None)
XL_PUB = build_Xl(SP_PUB)
G1, CACHE_DS = {}, {}
print("\n================ G1 复现性核对 ================", flush=True)
for key, lab, feats in CONFIGS:
    ds = CACHE_DS.get(lab)
    if ds is None:
        ds = build_dataset(np.asarray(LAB[lab], np.float32).ravel()[il] > 0)
        CACHE_DS[lab] = ds
    pos_i, neg_i, _ = ds
    rows, perm = block_cv(XL_PUB, feats, pos_i, neg_i)
    s = summ(rows)
    ref = PUB[key]
    ref_a = float(np.mean([r["auc"] for r in ref["cv"]]))
    ref_p = float(np.mean([r["ap"] for r in ref["cv"]]))
    ok = abs(s["auc"] - ref_a) < 1e-9 and abs(s["ap"] - ref_p) < 1e-9
    print("  %-11s AUC %.6f (ref %.6f)  AP %.6f (ref %.6f)  %s"
          % (key, s["auc"], ref_a, s["ap"], ref_p, "PASS" if ok else "MISMATCH"), flush=True)
    G1[key] = dict(auc=round(s["auc"], 6), ap=round(s["ap"], 6),
                   ref_auc=round(ref_a, 6), ref_ap=round(ref_p, 6), match=bool(ok))

# ---------- 2. sigma 扫描（温泉型，含 4 个派生场）----------
pos_s, neg_s, _ = CACHE_DS["lab_spring"]
SWEEP, EXPOS = [], []
print("\n================ sigma 扫描（温泉型 + 4 个派生场）================", flush=True)
print("  sigma 度   等效 km   AUC      AP      首位特征       次位特征", flush=True)
for sg in SIGMAS:
    sp = spring_fields(sg)
    Xl = build_Xl(sp)
    rows, perm = block_cv(Xl, FEATS_WELL, pos_s, neg_s)
    s = summ(rows)
    top = perm[0]
    second = perm[1]
    SWEEP.append(dict(sigma=sg, km=round(sg * 111.0, 1), **s,
                      top=[list(top), list(second)],
                      perm=[list(t) for t in perm]))
    print("  %-9.2f %-8.1f %.4f   %.4f  %-14s %.4f"
          % (sg, sg * 111.0, s["auc"], s["ap"], top[0], second[1]), flush=True)

# 参考线
REF_GEO = summ(PUB["spring_geo"]["cv"])
REF_MIX = summ(PUB["spring_all"]["cv"])
print("\n  参考：仅地质-地球物理 AUC %.4f ; 基线混合配置（s_den 为 2 格）AUC %.4f"
      % (REF_GEO["auc"], REF_MIX["auc"]), flush=True)

res = dict(seed=SEED, sigmas=SIGMAS,
           g1_reproduce=G1,
           reference=dict(geo_only=REF_GEO, published_mixed=REF_MIX,
                          note="published_mixed: s_T/s_lgQ/s_heat sigma=0.40 deg, s_den = gaussian_filter(2 cells)"),
           sweep_spring_all=SWEEP,
           elapsed_s=round(time.time() - t_all, 1))
json.dump(res, open(os.path.join(OUTD, "sigma_sweep.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n输出 %s  (%.0fs)" % (os.path.join(OUTD, "sigma_sweep.json"), time.time() - t_all),
      flush=True)