# -*- coding: utf-8 -*-
"""Step Q5  P2-2：geo10 -> all14 的配对 bootstrap 置信区间（四种配置）

Table 1 的增量是 10 场 -> 14 场的折均值差；本脚本给出同一比较的
95% bootstrap 区间（对折外预测配对重抽 1000 次），使表格与正文口径一致。
结果并入 out/p2_diagnostics.json 的 paired_ci_geo_vs_all。
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon, ttest_rel

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import load_cache, kernel_smooth, rasterize_points, TR, RES, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD, N_BOOT = 3, 5, 1000
SEED = 20260927
SIG_BASE, DEN_CELLS, DEG_KM = 0.40, 2.0, 111.32
BBOX = (108.0, 122.0, 20.0, 30.0)
T0C, RHO_W, CP = 10.0, 990.0, 4180.0
GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
       "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
SF = ["s_T", "s_lgQ", "s_heat", "s_den"]
ALL = GEO + SF
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)
t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_n, lat_n, mask_n = z["lon_c"], z["lat_c"], z["mask"]
LO_n, LA_n = np.meshgrid(lon_n, lat_n)
X0, Y0 = TR.transform(LO_n.ravel(), LA_n.ravel())
il_n = np.nonzero(mask_n.ravel())[0]
pxn, pyn = X0[il_n] / 1000.0, Y0[il_n] / 1000.0


def blocked(sx, sy):
    return ((np.floor(sx / BLK_KM).astype(np.int64) + 1000) * 10000
            + (np.floor(sy / BLK_KM).astype(np.int64) + 1000))


def oof(Xl, xl, nm, pos_i, neg_i, pxl, pyl):
    cols = [ALL.index(f) for f in nm]
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    X = Xl[np.ix_(si, cols)]
    sx, sy = pxl[si], pyl[si]
    blk = blocked(sx, sy)
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(SEED).permutation(ub))}
    fold = np.array([fo[b] for b in blk])
    p_pool = np.zeros(len(y))
    aucs, aps = [], []
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        p_pool[te] = p
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
    return y, p_pool, np.array(aucs), np.array(aps)


def boot_diff(y, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y)
    da = np.full(n, np.nan)
    dp = np.full(n, np.nan)
    for i in range(n):
        idx = rng.integers(0, N, N)
        if y[idx].min() == y[idx].max():
            continue
        da[i] = roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx])
        dp[i] = (average_precision_score(y[idx], pb[idx])
                 - average_precision_score(y[idx], pa[idx]))
    return dict(
        auc_point=round(float(roc_auc_score(y, pb) - roc_auc_score(y, pa)), 4),
        auc_mean=round(float(np.nanmean(da)), 4),
        auc_ci95=[round(float(np.nanpercentile(da, 2.5)), 4),
                  round(float(np.nanpercentile(da, 97.5)), 4)],
        ap_point=round(float(average_precision_score(y, pb)
                             - average_precision_score(y, pa)), 4),
        ap_mean=round(float(np.nanmean(dp)), 4),
        ap_ci95=[round(float(np.nanpercentile(dp, 2.5)), 4),
                 round(float(np.nanpercentile(dp, 97.5)), 4)])


# ---- 全国 ----
GEO_n = {k: np.asarray(z[k], np.float32).ravel()[il_n] for k in GEO}
SF_n = {k: np.asarray(z[k], np.float32).ravel()[il_n] for k in SF}
Xg = np.column_stack([GEO_n[k] for k in GEO])
Xa = np.column_stack([GEO_n[k] for k in GEO] + [SF_n[k] for k in SF])

# ---- 区域 ----
ix = np.nonzero((lon_n >= BBOX[0]) & (lon_n <= BBOX[1]))[0]
iy = np.nonzero((lat_n >= BBOX[2]) & (lat_n <= BBOX[3]))[0]
lon_r, lat_r = lon_n[ix], lat_n[iy]
LO_r, LA_r = np.meshgrid(lon_r, lat_r)
mask_r = np.asarray(mask_n)[np.ix_(iy, ix)]
cell_r = (RES * DEG_KM) * (RES * DEG_KM * np.cos(np.radians(LA_r)))
Xr, Yr = TR.transform(LO_r.ravel(), LA_r.ravel())
il_r = np.nonzero(mask_r.ravel())[0]
pxr, pyr = Xr[il_r] / 1000.0, Yr[il_r] / 1000.0
GEO_r = {k: np.asarray(z[k], np.float32)[np.ix_(iy, ix)].ravel()[il_r] for k in GEO}
d = load_cache()
sp_lon = np.asarray(d["spring_lon"], float)
sp_lat = np.asarray(d["spring_lat"], float)
inb = ((sp_lon >= BBOX[0]) & (sp_lon <= BBOX[1])
       & (sp_lat >= BBOX[2]) & (sp_lat <= BBOX[3]))
slon, slat = sp_lon[inb], sp_lat[inb]
sT = np.asarray(d["spring_T"], float)[inb]
sQ = np.asarray(d["spring_Q"], float)[inb]
Hrel = RHO_W * CP * (sQ / 1000.0) * np.maximum(sT - T0C, 0.0)
SF_r = dict(
    s_T=kernel_smooth(slon, slat, sT, LO_r, LA_r, SIG_BASE)[0],
    s_lgQ=kernel_smooth(slon, slat, np.log10(sQ + 1.0), LO_r, LA_r, SIG_BASE)[0],
    s_heat=kernel_smooth(slon, slat, np.log10(Hrel + 1.0), LO_r, LA_r, SIG_BASE)[0])
SF_r["s_den"] = (gaussian_filter(rasterize_points(slon, slat, lon_r, lat_r, RES),
                                DEN_CELLS) / cell_r * 1e4)
SF_r = {k: np.asarray(SF_r[k], np.float32).ravel()[il_r] for k in SF}
Xgr = np.column_stack([GEO_r[k] for k in GEO])
Xar = np.column_stack([GEO_r[k] for k in GEO] + [SF_r[k] for k in SF])

CFG = {
    "national_spring": (Xg, Xa, pxn, pyn, np.asarray(z["lab_spring"], np.float32).ravel()[il_n] > 0),
    "national_well": (Xg, Xa, pxn, pyn, np.asarray(z["lab_well"], np.float32).ravel()[il_n] > 0),
    "region_spring": (Xgr, Xar, pxr, pyr,
                      np.asarray(z["lab_spring"], np.float32)[np.ix_(iy, ix)].ravel()[il_r] > 0),
    "region_well": (Xgr, Xar, pxr, pyr,
                    np.asarray(z["lab_well"], np.float32)[np.ix_(iy, ix)].ravel()[il_r] > 0),
}
RESULT = {}
for tag, (Xgv, Xav, pxl, pyl, pos) in CFG.items():
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos) & (dmin > POS_BUF_KM))[0]
    neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)
    y, pa, aa, ap_a = oof(Xgv, pxl, GEO, pos_i, neg_i, pxl, pyl)
    _, pb, ab, ap_b = oof(Xav, pxl, ALL, pos_i, neg_i, pxl, pyl)
    rec = boot_diff(y, pa, pb)
    rec.update(geo_auc=round(float(aa.mean()), 4), all_auc=round(float(ab.mean()), 4),
               geo_ap=round(float(ap_a.mean()), 4), all_ap=round(float(ap_b.mean()), 4),
               fold_mean_delta_auc=round(float(ab.mean() - aa.mean()), 4),
               fold_mean_delta_ap=round(float(ap_b.mean() - ap_a.mean()), 4),
               wilcoxon_p_auc=round(float(wilcoxon(ab, aa).pvalue), 4),
               ttest_p_auc=round(float(ttest_rel(ab, aa).pvalue), 5),
               n_pos=int(len(pos_i)), n_neg=int(len(neg_i)))
    RESULT[tag] = rec
    print("  %-16s geo %.4f -> all %.4f  ΔAUC %+.4f [%.4f, %.4f]  ΔAP %+.4f [%.4f, %.4f]  p_t %.4f"
          % (tag, rec["geo_auc"], rec["all_auc"], rec["auc_mean"], rec["auc_ci95"][0],
             rec["auc_ci95"][1], rec["ap_mean"], rec["ap_ci95"][0], rec["ap_ci95"][1],
             rec["ttest_p_auc"]), flush=True)

P = os.path.join(OUTD, "p2_diagnostics.json")
R = json.load(io.open(P, encoding="utf-8"))
R["paired_ci_geo_vs_all"] = RESULT
R["elapsed_s"] = round(R.get("elapsed_s", 0) + time.time() - t0, 1)
json.dump(R, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("merged paired_ci_geo_vs_all (%.0fs)" % (time.time() - t0), flush=True)