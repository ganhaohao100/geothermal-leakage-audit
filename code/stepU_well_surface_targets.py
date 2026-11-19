# -*- coding: utf-8 -*-
"""Step U  Well-type screening surface and the nine 99th-percentile areas (Fig. 6, Table S2).

The surface is built from the ten geological and geophysical predictors with the
four spring-derived fields excluded. Cells within 10 km of a spring and 5 km of
a well are masked as known thermal displays. Candidate areas are connected
components of the remaining surface above a percentile, retained when their
area is at least 1,500 km2.

Output: results/targets_well_geo.csv and results/well_surface_summary.json
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import label, gaussian_filter
from scipy.spatial import cKDTree
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR  # noqa: E402
from common import TR, RES  # noqa: E402

BLK_KM, BUF_KM = 50.0, 10.0
NEG_RATIO, N_FOLD, N_BAG = 3, 5, 5
SEED = 20260927
R_SPRING, R_WELL = 10.0, 5.0
MIN_AREA, LEVELS = 1500.0, (99, 98, 95)

FEATS = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
         "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)


def main():
    t0 = time.time()
    z = np.load(os.path.join(DATA_DIR, "features.npz"), allow_pickle=True)
    lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
    cell_km2 = np.asarray(z["cell_km2"], float)
    ny, nx = mask.shape
    LO, LA = np.meshgrid(lon_c, lat_c)
    px, py = TR.transform(LO.ravel(), LA.ravel())
    px, py = px / 1000.0, py / 1000.0
    il = np.nonzero(mask.ravel())[0]
    Xl = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS])
    pxl, pyl = px[il], py[il]

    pos = np.nonzero(np.asarray(z["lab_well"], np.float32).ravel()[il] > 0)[0]
    rng = np.random.default_rng(SEED)
    tp = cKDTree(np.column_stack([pxl[pos], pyl[pos]]))
    dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
    cand = np.nonzero(
        (np.asarray(z["lab_well"], np.float32).ravel()[il] == 0) & (dmin > BUF_KM))[0]
    neg = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos)), replace=False)

    # PU bagging: five models on resampled pseudo-negatives
    yb = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    prob = np.zeros(len(il))
    rb = np.random.default_rng(SEED + 7)
    for b in range(N_BAG):
        sel = neg if b == 0 else rb.choice(cand, size=len(neg), replace=False)
        ii = np.r_[pos, sel]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xl[ii], yb)
        prob += m.predict_proba(Xl)[:, 1] / N_BAG
    pred = np.full(ny * nx, np.nan, np.float32)
    pred[il] = prob.astype(np.float32)
    pred = pred.reshape(ny, nx)

    def buffer_mask(lo, la, r_km):
        x, y = TR.transform(np.asarray(lo, float), np.asarray(la, float))
        t = cKDTree(np.column_stack([x / 1000.0, y / 1000.0]))
        dd, _ = t.query(np.column_stack([px[mask.ravel()], py[mask.ravel()]]), k=1)
        out = mask.copy()
        out[mask] = dd <= r_km
        return out

    known = (buffer_mask(z["slon"], z["slat"], R_SPRING)
             | buffer_mask(z["wlon"], z["wlat"], R_WELL))
    fav = np.where(mask & ~known, pred, np.nan)
    v = fav[np.isfinite(fav)]
    q = {p: float(np.percentile(v, p)) for p in LEVELS}

    rows = []
    for p in LEVELS:
        m = np.isfinite(fav) & (fav >= q[p])
        lab, n = label(m)
        comps = []
        for i in range(1, n + 1):
            sel = lab == i
            area = float(cell_km2[sel].sum())
            if area < MIN_AREA:
                continue
            j = int(np.nanargmax(np.where(sel, pred, -np.inf)))
            iy, ix = divmod(j, pred.shape[1])
            comps.append(dict(area_km2=area, fav_mean=float(np.nanmean(pred[sel])),
                              fav_max=float(np.nanmax(pred[sel])), ncells=int(sel.sum()),
                              lonc=float(lon_c[ix]), latc=float(lat_c[iy])))
        comps.sort(key=lambda d: -d["fav_max"])
        for k, c in enumerate(comps, 1):
            c.update(rank=k, tier="P%d" % p)
            rows.append(c)

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "targets_well_geo.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tier", "lonc", "latc", "area_km2", "fav_mean",
                    "fav_max", "ncells"])
        for c in rows:
            w.writerow([c["rank"], c["tier"], "%.3f" % c["lonc"], "%.3f" % c["latc"],
                        "%.10f" % c["area_km2"], "%.10f" % c["fav_mean"],
                        "%.10f" % c["fav_max"], c["ncells"]])
    summary = dict(n_cells_land=int(mask.sum()), n_eligible=int(v.size),
                   masked_fraction=float(1 - v.size / mask.sum()),
                   percentiles=q,
                   counts={str(p): sum(1 for c in rows if c["tier"] == "P%d" % p)
                           for p in LEVELS},
                   areas={str(p): float(sum(c["area_km2"] for c in rows
                                            if c["tier"] == "P%d" % p))
                          for p in LEVELS},
                   elapsed_s=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, "well_surface_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
