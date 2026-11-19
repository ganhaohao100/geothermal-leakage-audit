# -*- coding: utf-8 -*-
"""Step G  IHFC 热流测点与地热井的空间共位诊断（Phase 5 意见 P1-5）

问题：若「井」在空间上系统性落在热流测点附近，则 hf_bg/hf_res 在井位上
不只是插值场，而部分携带「此处做过测量」这一调查设计信息——这是一种
与目标无关、但仍是标签代理的混淆。本脚本量化：

 1 数量与距离分布：1118 井 / 2068 泉 / 随机陆区格点 → 最近热流测点距离
 2 同格共位：井位格是否含热流测点，相对随机陆区的富集倍数
 3 单变量判别：hf_bg / hf_n（测点存在性）/ s_den / f_dist 对井标签的 AUC
    （与主实验同一空间分块五折，测试集平均）

只读基线 features.npz 与 fig1_cache.pkl；输出 out/hf_well_colocation.json
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb  # noqa: F401  (保持与其它 step 一致的运行环境)
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
from shapely import contains_xy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import load_cache, china_polygon, TR, RES, OUT as BASE_OUT

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, N_FOLD = 50.0, 10.0, 5
SEED = 20260927
DIST_EDGES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
ny, nx = mask.shape
LO, LA = np.meshgrid(lon_c, lat_c)
il = np.nonzero(mask.ravel())[0]
px, py = TR.transform(LO.ravel(), LA.ravel())
px, py = px / 1000.0, py / 1000.0
pxl, pyl = px[il], py[il]
blkl = ((np.floor(pxl / BLK_KM).astype(np.int64) + 1000) * 10000
        + (np.floor(pyl / BLK_KM).astype(np.int64) + 1000))

d = load_cache()
poly = china_polygon(d["border"])
mh = contains_xy(poly, np.asarray(d["hf_lon"], float), np.asarray(d["hf_lat"], float))
hlon, hlat = np.asarray(d["hf_lon"], float)[mh], np.asarray(d["hf_lat"], float)[mh]
hx, hy = TR.transform(hlon, hlat)
hx, hy = hx / 1000.0, hy / 1000.0
tree = cKDTree(np.column_stack([hx, hy]))

WLON, WLAT = np.asarray(z["wlon"], float), np.asarray(z["wlat"], float)
SLON, SLAT = np.asarray(z["slon"], float), np.asarray(z["slat"], float)
wx, wy = TR.transform(WLON, WLAT); wx, wy = wx / 1000.0, wy / 1000.0
sx, sy = TR.transform(SLON, SLAT); sx, sy = sx / 1000.0, sy / 1000.0
rng = np.random.default_rng(SEED)
ridx = rng.choice(len(il), size=5000, replace=False)
print("HF sites on land: %d ; wells: %d ; springs: %d ; random land: %d"
      % (hlon.size, WLON.size, SLON.size, ridx.size), flush=True)


def dist_stats(x, y, name):
    dd, _ = tree.query(np.column_stack([x, y]), k=1)
    q = np.percentile(dd, [10, 25, 50, 75, 90])
    frac = {("%g_km" % e): round(float((dd <= e).mean()), 4) for e in DIST_EDGES}
    print("  %-13s median=%6.1f km  q10-q90=%5.1f-%5.1f  <=1/5/10/20km: %.3f / %.3f / %.3f / %.3f"
          % (name, q[2], q[0], q[4], frac["1_km"], frac["5_km"], frac["10_km"], frac["20_km"]),
          flush=True)
    return dict(n=int(len(dd)), median_km=round(float(q[2]), 2),
                q10_km=round(float(q[0]), 2), q25_km=round(float(q[1]), 2),
                q75_km=round(float(q[3]), 2), q90_km=round(float(q[4]), 2),
                mean_km=round(float(dd.mean()), 2), frac_within=frac)


print("\n[1] 到最近 IHFC 热流测点的距离", flush=True)
ST = dict(well=dist_stats(wx, wy, "wells"),
          spring=dist_stats(sx, sy, "springs"),
          random_land=dist_stats(pxl[ridx], pyl[ridx], "random land"))

# ---- 2 同格 / 邻格共位 ----
hf_n = np.asarray(z["hf_n"], np.float32)
lab_w = np.asarray(z["lab_well"], np.float32)


def cell_ix(x, y):
    lon, lat = TR.transform(x * 1000.0, y * 1000.0, direction="INVERSE")
    ix = np.clip(np.round((lon - lon_c[0]) / RES).astype(int), 0, len(lon_c) - 1)
    iy = np.clip(np.round((lat - lat_c[0]) / RES).astype(int), 0, len(lat_c) - 1)
    return iy, ix


wy_i, wx_i = cell_ix(wx, wy)
ry_i, rx_i = cell_ix(pxl[ridx], pyl[ridx])
on_hf_well = float((hf_n[wy_i, wx_i] > 0).mean())
on_hf_rand = float((hf_n[ry_i, rx_i] > 0).mean())
enr = {k: round(float(ST["well"]["frac_within"][k])
                / max(float(ST["random_land"]["frac_within"][k]), 1e-9), 2)
       for k in ST["well"]["frac_within"]}
print("\n[2] 井位格与热流测点共位", flush=True)
print("  热流场支撑内(同格 hf_n>0): 井 %.4f  随机陆区 %.4f  富集 %.1fx"
      % (on_hf_well, on_hf_rand, on_hf_well / max(on_hf_rand, 1e-9)), flush=True)
print("  距离富集倍数（井 / 随机陆区）: " + ", ".join("%s %.1fx" % (k, v) for k, v in enr.items()), flush=True)

# ---- 3 单变量判别（与主实验同一分块五折）----
def fold_map(blk):
    ub = np.unique(blk)
    return {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}


pos = np.nonzero(np.asarray(z["lab_well"], np.float32).ravel()[il] > 0)[0]
tp = cKDTree(np.column_stack([pxl[pos], pyl[pos]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((np.asarray(z["lab_well"], np.float32).ravel()[il] <= 0) & (dmin > BUF_KM))[0]
neg = rng.choice(cand, size=min(len(cand), 3 * len(pos)), replace=False)
si = np.r_[pos, neg]
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
blk = blkl[si]
fo = fold_map(blk)
fold = np.array([fo[b] for b in blk])
sxc, syc = pxl[si], pyl[si]
UNI = {}
for f in ["hf_bg", "hf_n", "hf_res", "hf_grad", "s_den", "f_dist"]:
    v = np.asarray(z[f], np.float32).ravel()[il][si].astype(float)
    fill = np.nanmedian(v) if np.isfinite(v).any() else 0.0
    v = np.where(np.isfinite(v), v, fill)
    a = []
    for k in range(N_FOLD):
        te = fold == k
        a.append(roc_auc_score(y[te], v[te]))
    UNI[f] = dict(mean=round(float(np.mean(a)), 4), folds=[round(float(t), 4) for t in a])
    print("\n[3] %-8s 井标签单变量 AUC = %.4f" % (f, UNI[f]["mean"]), end="")
print("", flush=True)

OUT = dict(seed=SEED, n_hf_land=int(hlon.size), n_well=int(WLON.size),
           n_spring=int(SLON.size), dist=ST,
           colocation=dict(well_same_cell_hf=round(on_hf_well, 4),
                           random_same_cell_hf=round(on_hf_rand, 4),
                           enrichment_same_cell=round(on_hf_well / max(on_hf_rand, 1e-9), 2),
                           enrichment_by_distance=enr),
           univariate_well=UNI, elapsed_s=round(time.time() - t0, 1))
with io.open(os.path.join(OUTD, "hf_well_colocation.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=1)
print("\nwrote hf_well_colocation.json  (%.0fs)" % (time.time() - t0), flush=True)