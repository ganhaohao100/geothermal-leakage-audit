# -*- coding: utf-8 -*-
"""Step Q  Phase 5 P2 定量诊断（P2-1 / P2-2 / P2-3 / P2-5 / P2-10）

A 置换重要性跨 5 折（mean ± SD）+ 相关族分组置换 + 特征相关矩阵  -> P2-1
B 折外预测合并后的 bootstrap CI、折间配对检验                     -> P2-2
C 相对单一热流背景图的增量对照（hf_bg / 热流三场 / 10 场 / 14 场）-> P2-5
D 峭壁比值（首位/次位，AP 与 AUC 两种打分）之灵敏度/特异度        -> P2-3
E 标签点最近邻距离分布 + 标签指示场变差函数 range                  -> P2-10

输出 out/p2_diagnostics.json
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon, ttest_rel
from shapely import contains_xy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import (load_cache, china_polygon, kernel_smooth, rasterize_points,
                    TR, RES, OUT as BASE_OUT)

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTD, exist_ok=True)

BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD, N_BOOT, N_REP = 3, 5, 1000, 5
SEED = 20260927
SIG_BASE, DEN_CELLS = 0.40, 2.0
DEG_KM = 111.32
BBOX = (108.0, 122.0, 20.0, 30.0)
T0C, RHO_W, CP = 10.0, 990.0, 4180.0

FEATS_GEO = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
SFIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]
FEATS_ALL = FEATS_GEO + SFIELDS
SUBSETS = {"hf_bg_only": ["hf_bg"],
           "heat_flow_3": ["hf_bg", "hf_res", "hf_grad"],
           "geo_10": list(FEATS_GEO),
           "all_14": list(FEATS_ALL)}
GROUPS = [("heat_flow", ["hf_bg", "hf_res", "hf_grad"]),
          ("faults", ["f_dist", "f_den", "f_end"]),
          ("tendency", ["Ts_mean", "Ts_max", "Td_mean", "Td_max"]),
          ("spring_derived", list(SFIELDS))]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_n, lat_n, mask_n = z["lon_c"], z["lat_c"], z["mask"]


def project(lon, lat):
    x, y = TR.transform(lon, lat)
    return x / 1000.0, y / 1000.0


def blocked(sx, sy, blk_km=BLK_KM):
    return ((np.floor(sx / blk_km).astype(np.int64) + 1000) * 10000
            + (np.floor(sy / blk_km).astype(np.int64) + 1000))


def folds_of(blk):
    ub = np.unique(blk)
    return {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(SEED).permutation(ub))}


def build_dataset(pos_land, pxl, pyl, buf=POS_BUF_KM):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > buf))[0]
    n_neg = min(len(cand), NEG_RATIO * len(pos_i))
    return pos_i, rng.choice(cand, size=n_neg, replace=False)


def run_cv(Xl, pos_i, neg_i, pxl, pyl, names, groups=None):
    """返回 rows（每折指标）、perm（各折 mean/SD）、gperm（分组置换）、
    pooled（折外 y/p）、fold 向量。"""
    cols = [FEATS_ALL.index(f) for f in names]
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    blk = np.r_[blocked(pxl, pyl)[pos_i], blocked(pxl, pyl)[neg_i]]
    sx, sy = pxl[si], pyl[si]
    X = Xl[np.ix_(si, cols)]
    fo = folds_of(blk)
    fold = np.array([fo[b] for b in blk])
    rows, p_pool = [], np.zeros(len(y))
    perm_ap = np.full((N_FOLD, len(names)), np.nan)
    perm_auc = np.full((N_FOLD, len(names)), np.nan)
    gperm_ap = np.full((N_FOLD, len(groups or [])), np.nan)
    gperm_auc = np.full((N_FOLD, len(groups or [])), np.nan)
    rng = np.random.default_rng(SEED + 7)
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
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         n_pos_test=int(y[te].sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        for scoring, mat in (("average_precision", perm_ap), ("roc_auc", perm_auc)):
            pi = permutation_importance(m, X[te], y[te], n_repeats=N_REP,
                                        random_state=SEED, scoring=scoring)
            mat[k] = pi.importances_mean
        if groups:
            Xte = X[te]
            for gi, (gname, gcols) in enumerate(groups):
                idxs = [names.index(c) for c in gcols if c in names]
                if not idxs:
                    continue
                base_p = p
                base_ap = average_precision_score(y[te], base_p)
                base_auc = roc_auc_score(y[te], base_p)
                da, du = [], []
                for _ in range(N_REP):
                    Xp = Xte.copy()
                    Xp[:, idxs] = Xp[rng.permutation(Xte.shape[0])][:, idxs]
                    pp = m.predict_proba(Xp)[:, 1]
                    da.append(base_ap - average_precision_score(y[te], pp))
                    du.append(base_auc - roc_auc_score(y[te], pp))
                gperm_ap[k, gi] = float(np.mean(da))
                gperm_auc[k, gi] = float(np.mean(du))
    return rows, perm_ap, perm_auc, gperm_ap, gperm_auc, y, p_pool, fold


def fold_summary(rows):
    a = np.array([r["auc"] for r in rows])
    p = np.array([r["ap"] for r in rows])
    return dict(auc=round(float(a.mean()), 4), auc_sd=round(float(a.std()), 4),
                auc_rng=round(float(a.max() - a.min()), 4),
                ap=round(float(p.mean()), 4), ap_sd=round(float(p.std()), 4),
                folds_auc=[round(float(v), 4) for v in a],
                folds_ap=[round(float(v), 4) for v in p])


def boot_ci(y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y)
    A = np.full(n, np.nan)
    P = np.full(n, np.nan)
    for i in range(n):
        idx = rng.integers(0, N, N)
        if y[idx].min() == y[idx].max():
            continue
        A[i] = roc_auc_score(y[idx], p[idx])
        P[i] = average_precision_score(y[idx], p[idx])
    return (round(float(np.nanpercentile(A, 2.5)), 4),
            round(float(np.nanpercentile(A, 97.5)), 4),
            round(float(np.nanpercentile(P, 2.5)), 4),
            round(float(np.nanpercentile(P, 97.5)), 4))


def boot_paired(y, p_lo, p_hi, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y)
    Da = np.full(n, np.nan)
    Dp = np.full(n, np.nan)
    for i in range(n):
        idx = rng.integers(0, N, N)
        if y[idx].min() == y[idx].max():
            continue
        Da[i] = roc_auc_score(y[idx], p_hi[idx]) - roc_auc_score(y[idx], p_lo[idx])
        Dp[i] = (average_precision_score(y[idx], p_hi[idx])
                 - average_precision_score(y[idx], p_lo[idx]))
    return (round(float(np.nanmean(Da)), 4),
            round(float(np.nanpercentile(Da, 2.5)), 4),
            round(float(np.nanpercentile(Da, 97.5)), 4),
            round(float(np.nanmean(Dp)), 4),
            round(float(np.nanpercentile(Dp, 2.5)), 4),
            round(float(np.nanpercentile(Dp, 97.5)), 4))


def paired_test(a_lo, a_hi, p_lo, p_hi):
    out = {}
    for tag, lo, hi in (("auc", a_lo, a_hi), ("ap", p_lo, p_hi)):
        d = np.array(hi) - np.array(lo)
        out[tag] = dict(mean_diff=round(float(d.mean()), 4),
                        sd=round(float(d.std(ddof=1)), 4),
                        folds=[round(float(v), 4) for v in d])
        try:
            out[tag]["wilcoxon_p"] = round(float(wilcoxon(d).pvalue), 4)
        except Exception:
            out[tag]["wilcoxon_p"] = None
        try:
            out[tag]["ttest_p"] = round(float(ttest_rel(hi, lo).pvalue), 4)
        except Exception:
            out[tag]["ttest_p"] = None
    return out


# ---------------- 全国 / 区域 两套格网 ----------------
il_n = np.nonzero(mask_n.ravel())[0]
LO_n, LA_n = np.meshgrid(lon_n, lat_n)
pxn_all, pyn_all = project(LO_n.ravel(), LA_n.ravel())
pxn, pyn = pxn_all[il_n], pyn_all[il_n]
GEO_n = {k: np.asarray(z[k], np.float32).ravel()[il_n] for k in FEATS_GEO}
SF_n = {k: np.asarray(z[k], np.float32).ravel()[il_n] for k in SFIELDS}
POS_S = np.asarray(z["lab_spring"], np.float32).ravel()[il_n] > 0
POS_W = np.asarray(z["lab_well"], np.float32).ravel()[il_n] > 0
print("national land cells %d ; springs %d ; wells %d" %
      (il_n.size, POS_S.sum(), POS_W.sum()), flush=True)

ix = np.nonzero((lon_n >= BBOX[0]) & (lon_n <= BBOX[1]))[0]
iy = np.nonzero((lat_n >= BBOX[2]) & (lat_n <= BBOX[3]))[0]
lon_r, lat_r = lon_n[ix], lat_n[iy]
LO_r, LA_r = np.meshgrid(lon_r, lat_r)
mask_r = np.asarray(mask_n)[np.ix_(iy, ix)]
cell_km2_r = (RES * DEG_KM) * (RES * DEG_KM * np.cos(np.radians(LA_r)))
GEO_r = {k: np.asarray(z[k], np.float32)[np.ix_(iy, ix)] for k in FEATS_GEO}
il_r = np.nonzero(mask_r.ravel())[0]
PXR, PYR = project(LO_r.ravel(), LA_r.ravel())
pxr, pyr = PXR[il_r], PYR[il_r]

d = load_cache()
sp_lon = np.asarray(d["spring_lon"], float)
sp_lat = np.asarray(d["spring_lat"], float)
sp_T = np.asarray(d["spring_T"], float)
sp_Q = np.asarray(d["spring_Q"], float)
inb = ((sp_lon >= BBOX[0]) & (sp_lon <= BBOX[1]) &
       (sp_lat >= BBOX[2]) & (sp_lat <= BBOX[3]))
slon, slat = sp_lon[inb], sp_lat[inb]
sT, sQ = sp_T[inb], sp_Q[inb]
Hrel = RHO_W * CP * (sQ / 1000.0) * np.maximum(sT - T0C, 0.0)


def region_spring_fields(sigma=SIG_BASE):
    o = dict(s_T=kernel_smooth(slon, slat, sT, LO_r, LA_r, sigma)[0],
             s_lgQ=kernel_smooth(slon, slat, np.log10(sQ + 1.0), LO_r, LA_r, sigma)[0],
             s_heat=kernel_smooth(slon, slat, np.log10(Hrel + 1.0), LO_r, LA_r, sigma)[0])
    sd = rasterize_points(slon, slat, lon_r, lat_r, RES)
    o["s_den"] = gaussian_filter(sd, DEN_CELLS) / cell_km2_r * 1e4
    return {k: np.asarray(o[k], np.float32).ravel()[il_r] for k in SFIELDS}


SF_r = region_spring_fields()

def region_pos(label):
    lab = np.asarray(z[label], np.float32)[np.ix_(iy, ix)]
    return lab.ravel()[il_r] > 0


POS_S_R = region_pos("lab_spring")
POS_W_R = region_pos("lab_well")
print("region cells %d ; springs %d ; wells %d ; springs in box %d"
      % (il_r.size, POS_S_R.sum(), POS_W_R.sum(), slon.size), flush=True)


def stack_national(names):
    return np.column_stack([GEO_n[k] if k in GEO_n else SF_n[k] for k in names])


def stack_region(names, sf):
    return np.column_stack([GEO_r[k].ravel()[il_r] if k in GEO_r else sf[k]
                            for k in names])


REPORT = dict(seed=SEED, block_km=BLK_KM, buffer_km=BUF_KM, n_fold=N_FOLD,
              neg_ratio=NEG_RATIO, n_boot=N_BOOT, n_repeat_perm=N_REP,
              groups={g: c for g, c in GROUPS})
CONFIGS = {
    "national_spring": dict(pos=POS_S, pxl=pxn, pyl=pyn, stack=lambda nm: stack_national(nm)),
    "national_well": dict(pos=POS_W, pxl=pxn, pyl=pyn, stack=lambda nm: stack_national(nm)),
    "region_spring": dict(pos=POS_S_R, pxl=pxr, pyl=pyr, stack=lambda nm: stack_region(nm, SF_r)),
    "region_well": dict(pos=POS_W_R, pxl=pxr, pyl=pyr, stack=lambda nm: stack_region(nm, SF_r)),
}

# ---------------- A 跨折置换重要性 + 分组置换 + 相关矩阵 ----------------
print("\n================ A. 跨折置换重要性（14 场）与分组置换 ================", flush=True)
PERM = {}
for tag, cfg in CONFIGS.items():
    pos_i, neg_i = build_dataset(cfg["pos"], cfg["pxl"], cfg["pyl"])
    Xl = cfg["stack"](FEATS_ALL)
    rows, pap, pauc, gap, gauc, y, p_pool, fold = run_cv(
        Xl, pos_i, neg_i, cfg["pxl"], cfg["pyl"], FEATS_ALL, groups=GROUPS)
    mean_ap = pap.mean(0)
    mean_auc = pauc.mean(0)
    order_ap = np.argsort(-mean_ap)
    order_auc = np.argsort(-mean_auc)
    top1_ap, top2_ap = mean_ap[order_ap[0]], mean_ap[order_ap[1]]
    top1_auc, top2_auc = mean_auc[order_auc[0]], mean_auc[order_auc[1]]
    PERM[tag] = dict(
        n_pos=int(len(pos_i)), n_neg=int(len(neg_i)),
        perm_ap_mean={f: round(float(v), 4) for f, v in zip(FEATS_ALL, mean_ap)},
        perm_ap_sd={f: round(float(v), 4) for f, v in zip(FEATS_ALL, pap.std(0))},
        perm_auc_mean={f: round(float(v), 4) for f, v in zip(FEATS_ALL, mean_auc)},
        perm_auc_sd={f: round(float(v), 4) for f, v in zip(FEATS_ALL, pauc.std(0))},
        rank_ap=[FEATS_ALL[i] for i in order_ap],
        rank_auc=[FEATS_ALL[i] for i in order_auc],
        cliff_ap=dict(top1=FEATS_ALL[order_ap[0]], top2=FEATS_ALL[order_ap[1]],
                      ratio=round(float(top1_ap / max(top2_ap, 1e-9)), 1)),
        cliff_auc=dict(top1=FEATS_ALL[order_auc[0]], top2=FEATS_ALL[order_auc[1]],
                       ratio=round(float(top1_auc / max(top2_auc, 1e-9)), 1)),
        grouped_ap={GROUPS[i][0]: round(float(gap[:, i].mean()), 4)
                    for i in range(len(GROUPS))},
        grouped_ap_sd={GROUPS[i][0]: round(float(gap[:, i].std()), 4)
                       for i in range(len(GROUPS))},
        grouped_auc={GROUPS[i][0]: round(float(gauc[:, i].mean()), 4)
                     for i in range(len(GROUPS))},
    )
    corr = np.corrcoef(Xl.T)
    PERM[tag]["corr_pearson"] = {f: {g: round(float(corr[i, j]), 2)
                                     for j, g in enumerate(FEATS_ALL)}
                                 for i, f in enumerate(FEATS_ALL)}
    print("  %-16s top1(AP)=%s %+.4f  top2=%s %+.4f  ratio=%.1fx | ratio(AUC)=%.1fx"
          % (tag, FEATS_ALL[order_ap[0]], top1_ap, FEATS_ALL[order_ap[1]], top2_ap,
             PERM[tag]["cliff_ap"]["ratio"], PERM[tag]["cliff_auc"]["ratio"]), flush=True)
    print("      grouping(AP): " + ", ".join(
        "%s %+.4f" % (g, v) for g, v in PERM[tag]["grouped_ap"].items()), flush=True)
REPORT["perm_importance"] = PERM

# ---------------- B/C 增量、bootstrap CI、配对检验 ----------------
print("\n================ B/C. 增量对照 + bootstrap CI + 折间配对检验 ================", flush=True)
INC = {}
for tag, cfg in CONFIGS.items():
    pos_i, neg_i = build_dataset(cfg["pos"], cfg["pxl"], cfg["pyl"])
    rec = dict(n_pos=int(len(pos_i)), n_neg=int(len(neg_i)), subsets={})
    pooled = {}
    for sname, feats in SUBSETS.items():
        if tag.startswith("region") and sname in ("hf_bg_only", "heat_flow_3"):
            pass
        Xl = cfg["stack"](feats)
        rows, _, _, _, _, y, p_pool, fold = run_cv(
            Xl, pos_i, neg_i, cfg["pxl"], cfg["pyl"], feats)
        s = fold_summary(rows)
        lo, hi, plo, phi = boot_ci(y, p_pool)
        pooled[sname] = (y, p_pool)
        rec["subsets"][sname] = dict(**s, pooled_auc=round(float(roc_auc_score(y, p_pool)), 4),
                                     pooled_ap=round(float(average_precision_score(y, p_pool)), 4),
                                     auc_ci95=[lo, hi], ap_ci95=[plo, phi])
    lo, hi = pooled["heat_flow_3"], pooled["all_14"]
    d = boot_paired(lo[0], lo[1], hi[1])
    rec["delta_hf3_to_all14"] = dict(auc_mean=d[0], auc_ci95=[d[1], d[2]],
                                     ap_mean=d[3], ap_ci95=[d[4], d[5]])
    g = pooled["geo_10"]
    rec["delta_geo_to_all14"] = dict(
        auc_mean=round(float(roc_auc_score(hi[0], hi[1]) - roc_auc_score(g[0], g[1])), 4))
    rec["paired_fold_test_geo_to_all"] = paired_test(
        rec["subsets"]["geo_10"]["folds_auc"], rec["subsets"]["all_14"]["folds_auc"],
        rec["subsets"]["geo_10"]["folds_ap"], rec["subsets"]["all_14"]["folds_ap"])
    INC[tag] = rec
    print("  %-16s hf_bg A=%.4f | hf3 A=%.4f | geo10 A=%.4f | all14 A=%.4f  Δ(geo->all)=%+.4f"
          % (tag, rec["subsets"]["hf_bg_only"]["auc"], rec["subsets"]["heat_flow_3"]["auc"],
             rec["subsets"]["geo_10"]["auc"], rec["subsets"]["all_14"]["auc"],
             rec["delta_geo_to_all14"]["auc_mean"]), flush=True)
    print("      ΔAP(hf3->all14)=%+.4f [%.4f, %.4f] ; ΔAUC=%+.4f [%.4f, %.4f]"
          % (rec["delta_hf3_to_all14"]["ap_mean"],
             rec["delta_hf3_to_all14"]["ap_ci95"][0], rec["delta_hf3_to_all14"]["ap_ci95"][1],
             rec["delta_hf3_to_all14"]["auc_mean"],
             rec["delta_hf3_to_all14"]["auc_ci95"][0], rec["delta_hf3_to_all14"]["auc_ci95"][1]),
          flush=True)
    print("      paired fold Wilcoxon p(AP)=%s ; t p(AP)=%s"
          % (rec["paired_fold_test_geo_to_all"]["ap"]["wilcoxon_p"],
             rec["paired_fold_test_geo_to_all"]["ap"]["ttest_p"]), flush=True)
REPORT["increment"] = INC

# ---------------- E 最近邻距离 + 变差函数 ----------------
print("\n================ E. 标签点最近邻距离与变差函数 range ================", flush=True)


def nn_stats(x, y):
    t = cKDTree(np.column_stack([x, y]))
    dd, _ = t.query(np.column_stack([x, y]), k=2)
    d1 = dd[:, 1]
    return dict(n=int(len(x)), median_km=round(float(np.median(d1)), 1),
                p10_km=round(float(np.percentile(d1, 10)), 1),
                p90_km=round(float(np.percentile(d1, 90)), 1),
                p100_km=round(float(d1.max()), 1))


def empirical_variogram(x, y, v, edges, npairs=1500000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    i = rng.integers(0, n, npairs)
    j = rng.integers(0, n, npairs)
    m0 = i != j
    i, j = i[m0], j[m0]
    dist = np.hypot(x[i] - x[j], y[i] - y[j])
    gamma = 0.5 * (v[i] - v[j]) ** 2
    mid, gam, cnt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dist >= a) & (dist < b)
        if m.sum() < 500:
            continue
        mid.append(0.5 * (a + b))
        gam.append(float(gamma[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(mid), np.array(gam), np.array(cnt)


def exp_model(h, nug, sill, rng_km):
    return nug + sill * (1.0 - np.exp(-h / rng_km))


def range_of(x, y, v, tag):
    edges = np.arange(0.0, 305.0, 5.0)
    h, g, c = empirical_variogram(x, y, v, edges)
    out = dict(lags=h.tolist(), gamma=[round(float(t), 5) for t in g],
               n_pairs=c.tolist())
    try:
        p0 = [float(np.percentile(g, 5)), float(np.percentile(g, 95)), 40.0]
        popt, _ = curve_fit(exp_model, h, g, p0=p0, maxfev=20000,
                            bounds=([0.0, 0.0, 1.0], [1.0, 1.0, 400.0]))
        out["range_km"] = round(float(popt[2]), 1)
        out["practical_range_km"] = round(float(3.0 * popt[2]), 1)
        out["nugget"] = round(float(popt[0]), 5)
        out["sill"] = round(float(popt[1]), 5)
    except Exception as e:
        out["fit_error"] = str(e)
    thr = 0.95 * float(np.max(g))
    k = np.nonzero(g >= thr)[0]
    out["empirical_95pct_lag_km"] = float(h[k[0]]) if k.size else None
    return out


rng = np.random.default_rng(SEED)
sub = rng.choice(il_n.size, size=20000, replace=False)
gx, gy = pxn[sub], pyn[sub]
d = load_cache()
poly = china_polygon(d["border"])
mh = contains_xy(poly, np.asarray(d["hf_lon"], float), np.asarray(d["hf_lat"], float))
hlon = np.asarray(d["hf_lon"], float)[mh]
hlat = np.asarray(d["hf_lat"], float)[mh]
hx, hy = project(hlon, hlat)
wl_x, wl_y = project(np.asarray(z["wlon"], float), np.asarray(z["wlat"], float))
sl_x, sl_y = project(np.asarray(z["slon"], float), np.asarray(z["slat"], float))

LBL = dict(spring=dict(nn=nn_stats(sl_x, sl_y)),
           well=dict(nn=nn_stats(wl_x, wl_y)),
           hf=dict(nn=nn_stats(hx, hy)))
# 指示场（陆域格点，20000 抽样）
LBL["spring"]["variogram"] = range_of(gx, gy, POS_S[sub].astype(float), "spring")
LBL["well"]["variogram"] = range_of(gx, gy, POS_W[sub].astype(float), "well")
hfcnt = np.asarray(z["hf_n"], np.float32).ravel()[il_n]
LBL["hf"]["variogram"] = range_of(gx, gy, (hfcnt[sub] > 0).astype(float), "hf")
for k, v in LBL.items():
    print("  %-7s n=%5d  NN中位 %.1f km (P10 %.1f / P90 %.1f)  range %.1f km (实用 %.1f km)"
          % (k, v["nn"]["n"], v["nn"]["median_km"], v["nn"]["p10_km"],
             v["nn"]["p90_km"], v["variogram"].get("range_km", float("nan")),
             v["variogram"].get("practical_range_km", float("nan"))), flush=True)
REPORT["label_geometry"] = LBL
REPORT["hf_sites_on_land"] = int(hlon.size)
REPORT["elapsed_s"] = round(time.time() - t0, 1)
with io.open(os.path.join(OUTD, "p2_diagnostics.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT, fh, ensure_ascii=False, indent=1)
print("\nwrote p2_diagnostics.json  (%.0fs)" % (time.time() - t0), flush=True)