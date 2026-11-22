# -*- coding: utf-8 -*-
"""基线流程公共模块：数据加载、国界掩膜、规则格网、核平滑、断裂力学参数。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from shapely import contains_xy
from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union
from pyproj import Transformer

from config import DATA_DIR, WORK_DIR
FD = DATA_DIR
BASE = DATA_DIR
OUT = DATA_DIR
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(WORK_DIR, "fig1_cache.pkl")

LON_RNG = (73.0, 136.0)
LAT_RNG = (17.0, 54.5)
RES = 0.05                      # 约 5.5 km x 4.2 km
PROJ = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
TR = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)

from nine_region import (STRESS_DOMAINS, REGIME, AXES, MAGS,
                        stress_regime)  # noqa: F401

Z_REF = 3000.0          # 参考储层埋深 m
RHO_ROCK, RHO_W, G = 2650.0, 990.0, 9.81


def load_cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def china_polygon(border, min_pts=50, tol=0.008):
    polys = []
    for sg in border:
        if len(sg) >= min_pts and np.allclose(sg[0], sg[-1], atol=1e-6):
            try:
                q = SPoly(np.asarray(sg[:, :2], float))
                if not q.is_valid:
                    q = q.buffer(0)
                q = q.simplify(tol, preserve_topology=True)
                if (not q.is_empty) and q.area > 0:
                    polys.append(q)
            except Exception:
                pass
    return unary_union(polys)


def make_grid(poly):
    lon_c = np.arange(LON_RNG[0] + RES / 2, LON_RNG[1], RES)
    lat_c = np.arange(LAT_RNG[0] + RES / 2, LAT_RNG[1], RES)
    LO, LA = np.meshgrid(lon_c, lat_c)
    m = contains_xy(poly, LO.ravel(), LA.ravel()).reshape(LO.shape)
    return lon_c, lat_c, LO, LA, m


def to_cell(lon, lat, lon_c, lat_c):
    ix = np.clip(np.round((np.asarray(lon) - lon_c[0]) / RES).astype(int), 0, len(lon_c) - 1)
    iy = np.clip(np.round((np.asarray(lat) - lat_c[0]) / RES).astype(int), 0, len(lat_c) - 1)
    return iy, ix


def kernel_smooth(plon, plat, val, LO, LA, sigma, w=None, radius=3.4):
    """归一化核平滑（等效低通滤波）。返回 (场, 权重和/测点密度)。"""
    ny, nx = LO.shape
    num = np.zeros((ny, nx)); den = np.zeros((ny, nx))
    if w is None:
        w = np.ones(len(plon))
    dx = LO[0, 1] - LO[0, 0]; dy = LA[1, 0] - LA[0, 0]
    r = radius * sigma
    for x, y, v, ww in zip(np.asarray(plon), np.asarray(plat), np.asarray(val), np.asarray(w)):
        i0 = max(int(np.floor((y - r - LA[0, 0]) / dy)), 0)
        i1 = min(int(np.ceil((y + r - LA[0, 0]) / dy)) + 1, ny)
        j0 = max(int(np.floor((x - r - LO[0, 0]) / dx)), 0)
        j1 = min(int(np.ceil((x + r - LO[0, 0]) / dx)) + 1, nx)
        if i1 <= i0 or j1 <= j0:
            continue
        gy = (LA[i0:i1, 0] - y) / sigma
        gx = (LO[0, j0:j1] - x) / sigma
        k = np.exp(-0.5 * (gy[:, None] ** 2 + gx[None, :] ** 2)) * ww
        num[i0:i1, j0:j1] += k * v
        den[i0:i1, j0:j1] += k
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
    return field, den



def stress_tensor(lon, lat):
    reg, az = stress_regime(lon, lat)
    sv = RHO_ROCK * G * Z_REF / 1e6
    a = np.radians(az)
    e1 = np.array([np.sin(a), np.cos(a), 0.0])
    e2 = np.array([np.cos(a), -np.sin(a), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    basis = {"e1": e1, "e2": e2, "z": ez}
    S = np.zeros((3, 3))
    for axis, mag in zip(AXES[reg], MAGS[reg]):
        n = basis[axis]
        S += mag * sv * np.outer(n, n)
    return S


def ts_td(S, strike_deg, dip_deg, P):
    th = np.radians(float(strike_deg)); de = np.radians(float(dip_deg))
    phi = th + np.pi / 2.0
    n = np.array([-np.sin(de) * np.sin(phi), -np.sin(de) * np.cos(phi), np.cos(de)])
    u = np.array([np.cos(th), -np.sin(th), 0.0])
    v = np.cross(n, u); v = v / max(np.linalg.norm(v), 1e-12)
    Se = S - P * np.eye(3)
    t = Se @ n
    sn = float(t @ n)
    tau = float(np.linalg.norm(t - sn * n))
    Ts = tau / max(sn, 1e-6)
    ev = np.sort(np.linalg.eigvalsh(Se))
    s1, s3 = ev[2], ev[0]
    Td = (s1 - sn) / max(s1 - s3, 1e-9)
    return float(Ts), float(np.clip(Td, 0.0, 1.0))


def resample_segments(segs, max_m=2000.0):
    out = []
    for sg in segs:
        x, y = TR.transform(sg[:, 0], sg[:, 1])
        d = np.hypot(np.diff(x), np.diff(y))
        tot = float(d.sum())
        if tot < 1.0:
            continue
        s = np.concatenate([[0.0], np.cumsum(d)])
        n = max(2, int(tot // max_m) + 1)
        si = np.linspace(0.0, tot, n)
        out.append(np.column_stack([np.interp(si, s, sg[:, 0]),
                                    np.interp(si, s, sg[:, 1])]))
    return out


def chunk_slices(n, k):
    """[0,n) 上步长 k、窗口 k+1 的连续切片（相邻切片共享端点）。"""
    return [slice(i, min(i + k + 1, n)) for i in range(0, n - 1, k)]


def fault_table(segs, dip=70.0, max_m=2000.0, chunk_m=25000.0, min_km=1.5):
    """把断裂按 <=chunk_m 切段，逐段算中点、走向、长度与 Ts/Td。"""
    P = RHO_W * G * Z_REF / 1e6
    k = max(2, int(round(chunk_m / max_m)))
    rows = []
    for sg in resample_segments(segs, max_m):
        if len(sg) < 2:
            continue
        for sl in chunk_slices(len(sg), k):
            c = sg[sl]
            if len(c) < 2:
                continue
            x, y = TR.transform(c[:, 0], c[:, 1])
            L = float(np.hypot(np.diff(x), np.diff(y)).sum()) / 1000.0
            if L < min_km:
                continue
            cx = x - x.mean(); cy = y - y.mean()
            cov = np.cov(np.vstack([cx, cy]))
            if not np.all(np.isfinite(cov)):
                continue
            _, vec = np.linalg.eigh(cov)
            dx, dy = vec[:, -1]
            strike = float(np.degrees(np.arctan2(dx, dy)) % 180.0)
            mlon, mlat = float(c[len(c) // 2, 0]), float(c[len(c) // 2, 1])
            Ts, Td = ts_td(stress_tensor(mlon, mlat), strike, dip, P)
            rows.append((mlon, mlat, strike, L, Ts, Td,
                         float(x.mean()), float(y.mean()), float(y.max() - y.min())))
    return np.array(rows, float)


def rasterize_line_len(segs, lon_c, lat_c, res):
    """折线按端点各摊一半到格网，返回每格累积线长 (km)。"""
    ny, nx = len(lat_c), len(lon_c)
    A = np.zeros((ny, nx), np.float64)
    for sg in segs:
        lon = np.asarray(sg[:, 0], float); lat = np.asarray(sg[:, 1], float)
        if lon.size < 2:
            continue
        ix = np.clip(np.round((lon - lon_c[0]) / res).astype(int), 0, nx - 1)
        iy = np.clip(np.round((lat - lat_c[0]) / res).astype(int), 0, ny - 1)
        dl = np.hypot(np.diff(lon) * 111.32 * np.cos(np.radians(lat[:-1])),
                      np.diff(lat) * 111.32)
        np.add.at(A, (iy[:-1], ix[:-1]), dl * 0.5)
        np.add.at(A, (iy[1:], ix[1:]), dl * 0.5)
    return A


def rasterize_points(lon, lat, lon_c, lat_c, res, w=None):
    """点（或带权点）落格累加。"""
    ny, nx = len(lat_c), len(lon_c)
    A = np.zeros((ny, nx), np.float64)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    if lon.size == 0:
        return A
    ix = np.clip(np.round((lon - lon_c[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(np.round((lat - lat_c[0]) / res).astype(int), 0, ny - 1)
    w = np.ones(lon.size) if w is None else np.asarray(w, float)
    np.add.at(A, (iy, ix), w)
    return A


def nearest_fault_stats(cell_xy, ftab, ks=12, max_km=15.0, decay_km=5.0):
    """按最近的断裂段投影 Ts/Td：返回 (最近距离km, 加权均值, 加权最大)。"""
    from scipy.spatial import cKDTree
    xy = ftab[:, 6:8]
    tree = cKDTree(xy)
    d, idx = tree.query(cell_xy, k=min(ks, len(xy)))
    d = np.atleast_2d(d); idx = np.atleast_2d(idx)
    near = d <= max_km * 1000.0
    w = np.exp(-d / (decay_km * 1000.0)) * near
    out = []
    for col in (4, 5):
        v = ftab[idx, col]
        wsum = w.sum(axis=1)
        mean = np.where(wsum > 0, (v * w).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)
        vmax = np.where(near.any(axis=1),
                        np.max(np.where(near, v, -np.inf), axis=1), np.nan)
        out.append((mean, vmax))
    return d[:, 0] / 1000.0, out