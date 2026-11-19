# -*- coding: utf-8 -*-
"""Step C2  LFO x near-field-exclusion sweep

目的：检验"第二级效应"——LFO 重建后残余 (LFO AUC - 地质基线 AUC) 是否随
"建场时的近场排除半径 R" 增大而收窄。

设计（与 stepC 保持完全一致的建场/分折/拟合口径）：
  - 验证缓冲区（训练行过滤）固定 10 km = headline
  - 建场排除条件 = (sp_fold != k) & (dist(spring -> test cells) > R)
  - R 扫 {10, 25, 50, 75, 100} km
  - 同时算：published 14 场 (expect 0.9955)、geological 10 场 (expect 0.8470)
输出：out/lfo_buffer_sweep.json
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import kernel_smooth, rasterize_points, OUT as BASE_OUT, TR, RES

HERE = os.path.dirname(os.path.abspath(__file__))

BLK_KM, BUF_KM = 50.0, 10.0
NEG_RATIO, N_FOLD, SEED = 3, 5, 20260927
SIG_BASE, DEN_CELLS = 0.40, 2.0
DEG_KM = 111.32
RBUF = [10.0, 25.0, 50.0, 75.0, 100.0]

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
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]
pxl, pyl = px[il], py[il]
blkl = ((np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000
        + (np.floor(pyl / BLK_KM).astype(np.int64) + 1000))

GEO = {k: np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_GEO}
SLON, SLAT = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
ST, SQ, H = np.asarray(z["sT"], float), np.asarray(z["sQ"], float), np.asarray(z["H"], float)
spx, spy = TR.transform(SLON, SLAT)
spx, spy = spx / 1000.0, spy / 1000.0
sp_blk = ((np.floor(spx / BLK_KM).astype(np.int64) + 1000) * 10000
          + (np.floor(spy / BLK_KM).astype(np.int64) + 1000))
SPR = np.asarray(z["lab_spring"], np.float32)
POS_S = SPR.ravel()[il] > 0
print("land %d ; springs %d" % (len(il), SLON.size), flush=True)


def fields_published(idx=None):
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


# ---- dataset (identical to stepC block C) ----
rng = np.random.default_rng(SEED)
pos_i = np.nonzero(POS_S)[0]
tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((~POS_S) & (dmin > BUF_KM))[0]
neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)


def fold_map(blk):
    ub = np.unique(blk)
    return {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}


fo = fold_map(np.r_[blkl[pos_i], blkl[neg_i]])
si = np.r_[pos_i, neg_i]
y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
sx = np.r_[pxl[pos_i], pxl[neg_i]]
sy = np.r_[pyl[pos_i], pyl[neg_i]]
fold = np.array([fo[b] for b in np.r_[blkl[pos_i], blkl[neg_i]]])
sp_fold = np.array([fo.get(int(b), -1) for b in sp_blk])
print("pos=%d neg=%d ; springs per fold %s"
      % (len(pos_i), len(neg_i), [int((sp_fold == k).sum()) for k in range(N_FOLD)]), flush=True)


def fit_fold(k, Xfull, cols):
    te = fold == k
    tr = ~te
    tv = cKDTree(np.column_stack([sx[te], sy[te]]))
    dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
    tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
    X = Xfull[np.ix_(si, cols)]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X[tr_idx], y[tr_idx])
    p = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y[te], p)), int(len(tr_idx))


def folds_for(Xfull, cols):
    return [round(fit_fold(k, Xfull, cols)[0], 4) for k in range(N_FOLD)]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = dict(mean_a=round(float(a.mean()), 4), mean_b=round(float(b.mean()), 4),
               diff=round(float((a - b).mean()), 4), sd_a=round(float(a.std()), 4))
    try:
        out["t_p"] = round(float(stats.ttest_rel(a, b).pvalue), 4)
    except Exception:
        out["t_p"] = None
    try:
        out["wilcoxon_p"] = round(float(stats.wilcoxon(a, b).pvalue), 4)
    except Exception as e:
        out["wilcoxon_p"] = str(e)[:40]
    return out


XL_PUB = stack(fields_published(None), FEATS_ALL)
COLS_GEO = [FEATS_ALL.index(f) for f in FEATS_GEO]
COLS_ALL = list(range(len(FEATS_ALL)))
CI = [FEATS_ALL.index("f_dist"), FEATS_ALL.index("hf_bg")]

print("\n[A] published 14 fields (headline)", flush=True)
pub = folds_for(XL_PUB, COLS_ALL)
print("    folds=%s mean=%.4f" % (pub, np.mean(pub)), flush=True)
print("[B] geological 10 fields (reference)", flush=True)
ref = folds_for(XL_PUB, COLS_GEO)
print("    folds=%s mean=%.4f" % (ref, np.mean(ref)), flush=True)

SW = []
print("\n[C] LFO x near-field exclusion radius", flush=True)
print("    R_km  springs_kept(first fold)  folds  mean   sd    range   diff_vs_ref  t_p    wilcox", flush=True)
for R in RBUF:
    per_fold, kept = [], []
    for k in range(N_FOLD):
        te = fold == k
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dsp, _ = tv.query(np.column_stack([spx, spy]), k=1)
        keep = (sp_fold != k) & (dsp > R)
        kept.append(int(keep.sum()))
        if keep.sum() == 0:
            per_fold.append(float("nan"))
            continue
        X = stack(fields_published(np.nonzero(keep)[0]), FEATS_ALL)
        per_fold.append(round(fit_fold(k, X, COLS_ALL)[0], 4))
    st = paired(per_fold, ref)
    row = dict(R_km=R, springs_kept=kept, folds=per_fold,
               mean=round(float(np.nanmean(per_fold)), 4),
               sd=round(float(np.nanstd(per_fold)), 4),
               rng=round(float(np.nanmax(per_fold) - np.nanmin(per_fold)), 4),
               vs_ref=st)
    SW.append(row)
    print("    %5.0f  %6d  %s  %.4f %.4f %.4f   %+.4f     %s  %s"
          % (R, kept[0], row["folds"], row["mean"], row["sd"], row["rng"],
             st["diff"], st["t_p"], st["wilcoxon_p"]), flush=True)

TOT = float(np.mean(pub) - np.mean(ref))
L1 = float(np.mean(pub) - SW[0]["mean"])
L2 = float(SW[0]["mean"] - np.mean(ref))
print("\n[D] decomposition at R=10 km", flush=True)
print("    total increment            = %+.4f" % TOT, flush=True)
print("    L1 full-inventory self-enc = %+.4f  (%.0f%%)" % (L1, 100 * L1 / TOT), flush=True)
print("    L2 LFO residual            = %+.4f  (%.0f%%)" % (L2, 100 * L2 / TOT), flush=True)

REPORT = dict(block_km=BLK_KM, val_buffer_km=BUF_KM, seed=SEED, n_pos=int(len(pos_i)),
              n_neg=int(len(neg_i)), near_field_radii_km=RBUF,
              published_14_folds=pub, published_14_mean=round(float(np.mean(pub)), 4),
              reference_10_folds=ref, reference_10_mean=round(float(np.mean(ref)), 4),
              lfo_sweep=SW, decomposition=dict(total=round(TOT, 4),
                                               level1=round(L1, 4),
                                               level2=round(L2, 4),
                                               level1_pct=round(100 * L1 / TOT, 1),
                                               level2_pct=round(100 * L2 / TOT, 1)),
              elapsed_s=round(time.time() - t0, 1))
with io.open(os.path.join(OUTD, "lfo_buffer_sweep.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT, fh, ensure_ascii=False, indent=1)
print("\nwrote lfo_buffer_sweep.json (%.0fs)" % (time.time() - t0), flush=True)
