# -*- coding: utf-8 -*-
"""Step C  泄漏归因分解（Phase 5 审稿意见 DA C1 / 路线图 P1-1）

三项对照，全部只读基线 features.npz：
  A 单变量 AUC   : 14 个特征各自在五折测试集上的判别能力（无需训练）
  B 排除半径扫描 : 伪负样本排除半径 POS_BUF in {0,3,5,10,20,35,50} km
                   其余口径与 headline(spring_all) 完全一致
  C 留折重建     : 派生场只用「训练折的温泉」重建后再跑同一 CV
                   LFO-all = 四场全部按折重建；LFO-den = 仅 s_den 按折重建
  D 井型对照     : 同一组派生场用于井型标签（s_den 非井派生），看排除半径是否仍有效

输出 out/leak_decomposition.json
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
from common import kernel_smooth, rasterize_points, OUT as BASE_OUT, TR, RES

HERE = os.path.dirname(os.path.abspath(__file__))

BLK_KM, BUF_KM = 50.0, 10.0
NEG_RATIO, N_FOLD = 3, 5
SEED = 20260927
SIG_BASE, DEN_CELLS = 0.40, 2.0
BUFS = [0.0, 3.0, 5.0, 10.0, 20.0, 35.0, 50.0]
WELL_BUFS = [3.0, 10.0, 20.0]
T0C, RHO_W, CP = 10.0, 990.0, 4180.0
DEG_KM = 111.32

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

GEO = {k: np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_GEO}
SLON, SLAT = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
ST = np.asarray(z["sT"], float)
SQ = np.asarray(z["sQ"], float)
H = np.asarray(z["H"], float)
spx, spy = TR.transform(SLON, SLAT)
spx, spy = spx / 1000.0, spy / 1000.0
sp_blk = ((np.floor(spx / BLK_KM).astype(np.int64) + 1000) * 10000
          + (np.floor(spy / BLK_KM).astype(np.int64) + 1000))
SPR = np.asarray(z["lab_spring"], np.float32)
WEL = np.asarray(z["lab_well"], np.float32)
POS_S = SPR.ravel()[il] > 0
POS_W = WEL.ravel()[il] > 0
MPOS = dict(spring_grid=int((SPR > 0).sum()), spring_land=int(POS_S.sum()),
            well_grid=int((WEL > 0).sum()), well_land=int(POS_W.sum()))
print("land %d ; springs %d ; den kernel %.1f km" % (len(il), SLON.size,
      DEN_CELLS * RES * DEG_KM), flush=True)
print("positive cells (grid / on-land): spring %d / %d ; well %d / %d"
      % (MPOS["spring_grid"], MPOS["spring_land"],
         MPOS["well_grid"], MPOS["well_land"]), flush=True)


def fields_published(idx=None):
    """headline 口径：三场 0.40 度核平滑；s_den = 点栅格 + 两格高斯。"""
    if idx is None:
        return {k: np.asarray(z[k], np.float32).ravel()[il] for k in SFIELDS}
    sl, sa = SLON[idx], SLAT[idx]
    o = dict(
        s_T=kernel_smooth(sl, sa, ST[idx], LO, LA, SIG_BASE)[0],
        s_lgQ=kernel_smooth(sl, sa, np.log10(SQ[idx] + 1.0), LO, LA, SIG_BASE)[0],
        s_heat=kernel_smooth(sl, sa, np.log10(H[idx] + 1.0), LO, LA, SIG_BASE)[0])
    sd = rasterize_points(sl, sa, lon_c, lat_c, RES)
    o["s_den"] = gaussian_filter(sd, DEN_CELLS) / cell_km2 * 1e4
    return {k: np.asarray(o[k], np.float32).ravel()[il] for k in SFIELDS}


def stack(sf, feats):
    return np.column_stack([GEO[k] if k in GEO else np.asarray(sf[k], np.float32)
                            for k in feats]).astype(np.float32)


def build_dataset(pos_land, pos_buf_km):
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos_land)[0]
    tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero((~pos_land) & (dmin > pos_buf_km))[0]
    n_neg = min(len(cand), NEG_RATIO * len(pos_i))
    neg_i = rng.choice(cand, size=n_neg, replace=False)
    return pos_i, neg_i, cand


def fold_map(blk):
    ub = np.unique(blk)
    return {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}


def pack(pos_i, neg_i, fo):
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx = np.r_[pxl[pos_i], pxl[neg_i]]
    sy = np.r_[pyl[pos_i], pyl[neg_i]]
    blk = np.r_[blkl[pos_i], blkl[neg_i]]
    fold = np.array([fo[b] for b in blk])
    return si, y, sx, sy, fold


def run_cv(pos_i, neg_i, feats, Xl=None, Xl_of=None, want_perm=False, want_uni=True):
    fo = fold_map(np.r_[blkl[pos_i], blkl[neg_i]])
    si, y, sx, sy, fold = pack(pos_i, neg_i, fo)
    cols = [FEATS_ALL.index(f) for f in feats]
    rows, perm_rows, uni_rows = [], None, []
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        Xk = Xl if Xl_of is None else Xl_of(k)
        X = Xk[np.ix_(si, cols)]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        if want_uni:
            u = {}
            for j, f in enumerate(feats):
                v = X[te][:, j].astype(float)
                nan = int(np.isnan(v).sum())
                if nan:
                    u[f + "_nan"] = nan
                fill = np.nanmedian(v) if np.isfinite(v).any() else 0.0
                u[f] = round(float(roc_auc_score(y[te], np.where(np.isfinite(v), v, fill))), 4)
            uni_rows.append(u)
        if want_perm and k == 0:
            pi = permutation_importance(m, X[te], y[te], n_repeats=5,
                                        random_state=SEED, scoring="average_precision")
            perm_rows = sorted(zip(feats, pi.importances_mean.tolist()),
                               key=lambda t: -t[1])
    return rows, perm_rows, uni_rows, fo


def summ(rows):
    a = np.array([r["auc"] for r in rows])
    p = np.array([r["ap"] for r in rows])
    return dict(auc=round(float(a.mean()), 4), ap=round(float(p.mean()), 4),
                auc_sd=round(float(a.std()), 4),
                auc_rng=round(float(a.max() - a.min()), 4),
                folds=[round(float(v), 4) for v in a])


def mean_uni(uni_rows, key):
    v = [u[key] for u in uni_rows if key in u]
    return round(float(np.mean(v)), 4) if v else None


REPORT = dict(seed=SEED, block_km=BLK_KM, buffer_km=BUF_KM, den_cells=DEN_CELLS,
              den_kernel_km=round(DEN_CELLS * RES * DEG_KM, 1), sig_base=SIG_BASE,
              bufs=BUFS, n_pos=MPOS)

print("\n================ B. 伪负样本排除半径扫描（温泉型）================", flush=True)
print("  buf_km  n_neg  n_train   AUC     AP     极差   s_den单变量  首位特征", flush=True)
XL_PUB = stack(fields_published(None), FEATS_ALL)
SWEEP = []
for b in BUFS:
    pos_i, neg_i, cand = build_dataset(POS_S, b)
    rows, perm, uni, fo = run_cv(pos_i, neg_i, FEATS_ALL, Xl=XL_PUB,
                                 want_perm=True, want_uni=True)
    s = summ(rows)
    uden = mean_uni(uni, "s_den")
    top = perm[0] if perm else None
    SWEEP.append(dict(buf_km=b, n_neg=int(len(neg_i)), n_cand=int(len(cand)), **s,
                      n_train_mean=round(float(np.mean([r["n_train"] for r in rows])), 1),
                      uni_auc={k: mean_uni(uni, k) for k in FEATS_ALL},
                      perm=[list(t) for t in perm[:6]] if perm else None))
    print("  %5.1f  %5d  %7.0f   %.4f  %.4f  %.4f   %.4f       %s"
          % (b, len(neg_i), SWEEP[-1]["n_train_mean"], s["auc"], s["ap"],
             s["auc_rng"], uden, top[0] if top else "-"), flush=True)
REPORT["exclusion_sweep_spring"] = SWEEP

print("\n================ C. 留折重建（四场只用训练折温泉）================", flush=True)
pos_i, neg_i, cand = build_dataset(POS_S, BUF_KM)
fo = fold_map(np.r_[blkl[pos_i], blkl[neg_i]])
si, y, sx, sy, fold = pack(pos_i, neg_i, fo)
sp_fold = np.array([fo.get(int(b), -1) for b in sp_blk])
print("  springs per fold: " + ", ".join("f%d=%d" % (k, int((sp_fold == k).sum()))
                                         for k in range(N_FOLD)), flush=True)
LFO = {}
for name, mode in [("LFO-all", "all"), ("LFO-den", "den")]:
    per_fold, per_fold_ap, n_keep = [], [], []
    for k in range(N_FOLD):
        te = fold == k
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dsp, _ = tv.query(np.column_stack([spx, spy]), k=1)
        keep = (sp_fold != k) & (dsp > BUF_KM)
        n_keep.append(int(keep.sum()))
        sfk = fields_published(np.nonzero(keep)[0])
        if mode == "den":
            sf = dict(fields_published(None))
            sf["s_den"] = sfk["s_den"]
        else:
            sf = sfk

        def Xf(kk, sf=sf):
            return stack(sf, FEATS_ALL)
        rows, _, _, _ = run_cv(pos_i, neg_i, FEATS_ALL, Xl_of=Xf, want_uni=False)
        per_fold.append(rows[k]["auc"])
        per_fold_ap.append(rows[k]["ap"])
    LFO[name] = dict(springs_kept=n_keep, folds=[round(float(v), 4) for v in per_fold],
                     auc=round(float(np.mean(per_fold)), 4),
                     auc_sd=round(float(np.std(per_fold, ddof=1)), 4),
                     ap=round(float(np.mean(per_fold_ap)), 4),
                     ap_sd=round(float(np.std(per_fold_ap, ddof=1)), 4))
    print("  %-8s 留折重建 AUC = %.4f (sd %.4f)  folds=%s"
          % (name, LFO[name]["auc"], LFO[name]["auc_sd"],
             " / ".join("%.3f" % v for v in per_fold)), flush=True)
REPORT["leave_fold_out"] = LFO

print("\n================ D. 井型对照（同一组派生场；s_den 非井派生）================", flush=True)
WSWEEP = []
for b in WELL_BUFS:
    pos_i, neg_i, cand = build_dataset(POS_W, b)
    rows, perm, uni, fo = run_cv(pos_i, neg_i, FEATS_ALL, Xl=XL_PUB,
                                 want_perm=True, want_uni=True)
    s = summ(rows)
    WSWEEP.append(dict(buf_km=b, n_neg=int(len(neg_i)), **s,
                       uni_auc={k: mean_uni(uni, k) for k in FEATS_ALL},
                       perm=[list(t) for t in perm[:4]] if perm else None))
    print("  buf %5.1f  n_neg=%5d  AUC=%.4f (range %.4f)  s_den单变量=%.4f  首位=%s"
          % (b, len(neg_i), s["auc"], s["auc_rng"], mean_uni(uni, "s_den"),
             perm[0][0]), flush=True)
REPORT["exclusion_sweep_well"] = WSWEEP

row10 = [r for r in SWEEP if r["buf_km"] == BUF_KM][0]
REPORT["headline_univariate"] = row10["uni_auc"]
REPORT["headline_model"] = dict(auc=row10["auc"], ap=row10["ap"], auc_rng=row10["auc_rng"])
print("\n  headline(buf=10km) 模型 AUC %.4f ; 单变量 s_den AUC %.4f"
      % (row10["auc"], row10["uni_auc"]["s_den"]), flush=True)
REPORT["elapsed_s"] = round(time.time() - t0, 1)
with io.open(os.path.join(OUTD, "leak_decomposition.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT, fh, ensure_ascii=False, indent=1)
print("\nwrote leak_decomposition.json  (%.0fs)" % (time.time() - t0), flush=True)