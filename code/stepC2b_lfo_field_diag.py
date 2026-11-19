# -*- coding: utf-8 -*-
"""Step C2b  LFO x R 的场诊断：残余由哪些场承载？测试折上的场是否已退化为区域场？
输出 out/lfo_field_diag.json
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import kernel_smooth, rasterize_points, OUT as BASE_OUT, TR, RES

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, NEG_RATIO, N_FOLD, SEED = 50.0, 10.0, 3, 5, 20260927
SIG_BASE, DEN_CELLS = 0.40, 2.0
RBUF = [10.0, 25.0, 50.0, 75.0, 100.0]
SFIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask, cell_km2 = z["lon_c"], z["lat_c"], z["mask"], z["cell_km2"]
LO, LA = np.meshgrid(lon_c, lat_c)
px, py = TR.transform(LO.ravel(), LA.ravel()); px, py = px / 1000.0, py / 1000.0
il = np.nonzero(mask.ravel())[0]; pxl, pyl = px[il], py[il]
blkl = ((np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000
        + (np.floor(pyl / BLK_KM).astype(np.int64) + 1000))
SLON, SLAT = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
ST, SQ, H = np.asarray(z["sT"], float), np.asarray(z["sQ"], float), np.asarray(z["H"], float)
spx, spy = TR.transform(SLON, SLAT); spx, spy = spx / 1000.0, spy / 1000.0
sp_blk = ((np.floor(spx / BLK_KM).astype(np.int64) + 1000) * 10000
          + (np.floor(spy / BLK_KM).astype(np.int64) + 1000))
POS_S = np.asarray(z["lab_spring"], np.float32).ravel()[il] > 0
rng = np.random.default_rng(SEED)
pos_i = np.nonzero(POS_S)[0]
tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((~POS_S) & (dmin > BUF_KM))[0]
neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)
fo = {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(
    np.unique(np.r_[blkl[pos_i], blkl[neg_i]])))}
si = np.r_[pos_i, neg_i]
y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
fold = np.array([fo[b] for b in np.r_[blkl[pos_i], blkl[neg_i]]])
sp_fold = np.array([fo.get(int(b), -1) for b in sp_blk])


def fields(idx):
    sl, sa = SLON[idx], SLAT[idx]
    o = dict(s_T=kernel_smooth(sl, sa, ST[idx], LO, LA, SIG_BASE)[0],
             s_lgQ=kernel_smooth(sl, sa, np.log10(SQ[idx] + 1.0), LO, LA, SIG_BASE)[0],
             s_heat=kernel_smooth(sl, sa, np.log10(H[idx] + 1.0), LO, LA, SIG_BASE)[0])
    sd = rasterize_points(sl, sa, lon_c, lat_c, RES)
    o["s_den"] = gaussian_filter(sd, DEN_CELLS) / cell_km2 * 1e4
    return {k: np.asarray(o[k], np.float32).ravel()[il] for k in SFIELDS}


REPORT = dict(note="LFO rebuilt fields, diagnostics on the held-out fold", radii=RBUF)
print("  R_km  n_kept  nonzero%_test(s_den/s_T)  univariate AUC on test folds (s_T/s_lgQ/s_heat/s_den)")
for R in RBUF:
    acc = {k: [] for k in SFIELDS}
    nz = {"s_den": [], "s_T": []}
    kept = []
    for k in range(N_FOLD):
        te = fold == k
        tv = cKDTree(np.column_stack([pxl[si[te]], pyl[si[te]]]))
        dsp, _ = tv.query(np.column_stack([spx, spy]), k=1)
        keep = (sp_fold != k) & (dsp > R)
        kept.append(int(keep.sum()))
        f = fields(np.nonzero(keep)[0])
        v = {kk: f[kk][si[te]] for kk in SFIELDS}
        for kk in SFIELDS:
            vv = np.asarray(v[kk], float)
            fin = np.isfinite(vv)
            fill = float(np.median(vv[fin])) if fin.any() else 0.0
            acc[kk].append(float(roc_auc_score(y[te], np.where(fin, vv, fill))))
        for kk in ("s_den", "s_T"):
            nz[kk].append(float((np.isfinite(v[kk]) & (v[kk] > 0)).mean()))
    row = dict(R_km=R, n_kept=kept,
               nonzero_test={kk: round(float(np.mean(nz[kk])), 4) for kk in nz},
               uni_auc_test={kk: round(float(np.mean(acc[kk])), 4) for kk in SFIELDS})
    REPORT[str(R)] = row
    print("  %5.0f  %6d   %.3f / %.3f            %.3f  %.3f  %.3f  %.3f"
          % (R, kept[0], row["nonzero_test"]["s_den"], row["nonzero_test"]["s_T"],
             row["uni_auc_test"]["s_T"], row["uni_auc_test"]["s_lgQ"],
             row["uni_auc_test"]["s_heat"], row["uni_auc_test"]["s_den"]), flush=True)
REPORT["elapsed_s"] = round(time.time() - t0, 1)
with io.open(os.path.join(OUTD, "lfo_field_diag.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT, fh, ensure_ascii=False, indent=1)
print("\nwrote lfo_field_diag.json (%.0fs)" % (time.time() - t0), flush=True)
