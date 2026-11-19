# -*- coding: utf-8 -*-
"""Step Q2  P2 补充：NaN 安全相关矩阵 + 场尺度变差函数（s_den / hf_bg / f_dist / s_T）

P2-1 需要「共线性证据」；P2-10 需要「可操作的相关长度」。指标点二值指示场的
变差函数在稀疏标签下不可识别（sill≈0 / 拟合触界），故改为直接对**建模所用的
连续场**估计 range，这才是缓冲宽度应当对齐的量。
结果并入 out/p2_diagnostics.json。
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import TR, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260927
FEATS_ALL = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max", "s_T", "s_lgQ", "s_heat", "s_den"]
FIELDS = ["s_den", "s_T", "hf_bg", "hf_res", "f_dist", "f_den"]

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
LO, LA = np.meshgrid(lon_c, lat_c)
il = np.nonzero(mask.ravel())[0]
x, y = TR.transform(LO.ravel(), LA.ravel())
x, y = x[il] / 1000.0, y[il] / 1000.0

X = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_ALL]).astype(float)
finite = np.isfinite(X)
print("land %d ; finite per feature: %s" % (il.size,
      {k: int(finite[:, i].sum()) for i, k in enumerate(FEATS_ALL)}), flush=True)

# ---- NaN 安全 Pearson 相关（成对完整观测）----
C = np.full((len(FEATS_ALL), len(FEATS_ALL)), np.nan)
for i in range(len(FEATS_ALL)):
    for j in range(i, len(FEATS_ALL)):
        m = finite[:, i] & finite[:, j]
        if m.sum() < 50:
            continue
        a, b = X[m, i], X[m, j]
        if a.std() == 0 or b.std() == 0:
            continue
        C[i, j] = C[j, i] = float(np.corrcoef(a, b)[0, 1])
CORR = {f: {g: (None if not np.isfinite(C[i, j]) else round(float(C[i, j]), 2))
            for j, g in enumerate(FEATS_ALL)} for i, f in enumerate(FEATS_ALL)}
print("corr matrix done (%.0fs)" % (time.time() - t0), flush=True)
for a, b in (("hf_bg", "hf_res"), ("hf_bg", "hf_grad"), ("hf_res", "hf_grad"),
             ("s_den", "hf_bg"), ("s_den", "s_T"), ("s_den", "s_heat"),
             ("f_dist", "f_den"), ("f_dist", "f_end"), ("f_den", "f_end"),
             ("Ts_mean", "Ts_max"), ("Td_mean", "Td_max")):
    print("   r(%s,%s) = %s" % (a, b, CORR[a][b]), flush=True)

# ---- 场尺度变差函数 ----
rng = np.random.default_rng(SEED)
sub = rng.choice(il.size, size=25000, replace=False)
xs, ys = x[sub], y[sub]
edges = np.arange(0.0, 405.0, 5.0)


def empirical_variogram(v, npairs=2000000):
    i = rng.integers(0, len(v), npairs)
    j = rng.integers(0, len(v), npairs)
    m0 = i != j
    i, j = i[m0], j[m0]
    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
    g = 0.5 * (v[i] - v[j]) ** 2
    mid, gam, cnt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() < 500:
            continue
        mid.append(0.5 * (a + b)); gam.append(float(g[m].mean())); cnt.append(int(m.sum()))
    return np.array(mid), np.array(gam), np.array(cnt)


def fit_exp(h, g):
    f = lambda hh, nug, sill, r: nug + sill * (1.0 - np.exp(-hh / r))
    p0 = [float(np.percentile(g, 5)), float(np.percentile(g, 95)), 40.0]
    popt, _ = curve_fit(f, h, g, p0=p0, maxfev=40000,
                        bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, 400.0]))
    return popt


VAR = {}
for k in FIELDS:
    v = X[sub, FEATS_ALL.index(k)]
    ok = np.isfinite(v)
    if ok.sum() < 0.5 * v.size:
        VAR[k] = dict(note="too many non-finite cells; variogram not estimated",
                      finite_fraction=round(float(ok.mean()), 3))
        print("  %-7s skipped (finite %.2f)" % (k, ok.mean()), flush=True)
        continue
    vv = np.where(ok, v, np.nanmedian(v))
    h, g, c = empirical_variogram(vv)
    rec = dict(lags_km=h.tolist(), gamma=[round(float(t), 6) for t in g],
               n_pairs=c.tolist(), finite_fraction=round(float(ok.mean()), 3))
    try:
        nug, sill, r = fit_exp(h, g)
        rec.update(nugget=round(float(nug), 6), sill=round(float(sill), 6),
                   range_km=round(float(r), 1), practical_range_km=round(float(3 * r), 1))
    except Exception as e:
        rec["fit_error"] = str(e)
    thr = 0.95 * float(np.nanmax(g))
    kk = np.nonzero(g >= thr)[0]
    rec["empirical_95pct_lag_km"] = float(h[kk[0]]) if kk.size else None
    VAR[k] = rec
    print("  %-7s range %.1f km (practical %.1f km, emp95 %.1f km)  finite %.2f"
          % (k, rec.get("range_km", float("nan")), rec.get("practical_range_km", float("nan")),
             rec.get("empirical_95pct_lag_km", float("nan")), ok.mean()), flush=True)

# ---- 标签点最近邻（加入热流，统一口径）----
P = os.path.join(OUTD, "p2_diagnostics.json")
R = json.load(io.open(P, encoding="utf-8"))
R["feature_corr_pearson_nan_safe"] = CORR
R["field_variogram"] = VAR
# 指标指示场的拟合不可识别者，标注并降级为经验分位
for k in ("well", "hf"):
    vg = R["label_geometry"][k]["variogram"]
    if vg.get("sill", 1) is not None and (vg.get("sill") or 0) < 1e-4:
        vg["fit_reliable"] = False
        vg["note"] = ("indicator variogram not identifiable at these lags "
                      "(sill ~ 0); use nearest-neighbour quantiles instead")
    elif vg.get("range_km") in (400.0,):
        vg["fit_reliable"] = False
        vg["note"] = "fit hit the 400 km upper bound; range not identifiable"
    else:
        vg["fit_reliable"] = True
for k in ("spring",):
    R["label_geometry"][k]["variogram"]["fit_reliable"] = True
R["elapsed_s"] = round(R.get("elapsed_s", 0) + time.time() - t0, 1)
json.dump(R, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nmerged into p2_diagnostics.json  (%.0fs)" % (time.time() - t0), flush=True)