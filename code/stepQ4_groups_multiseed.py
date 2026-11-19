# -*- coding: utf-8 -*-
"""Step Q4  P2 补充二：IHFC 同型演示的分组置换 + 全国多种子稳健性

Part 1（P2-3）
  DA A3 要求：在附件同型演示里报「目标派生族」与「热流族」的分组置换。
  若族级统计量在单场峭壁失灵处仍能点火，则 Gate 3 应从"单场峭壁"改为"族级峭壁"。
  t1/t2/t3 三档 + sigma=0.10 / 0.40 两档：
    单场置换（AP 与 AUC，跨折 mean±SD）+ 四族分组置换。
Part 2（P2-2）
  单种子 → 三种折划分随机种子（模型随机种子同步），报 geo10 与 all14 的
  折均值 AUC 与 Δ 在种子间的离散度。

输出 out/hf_analogue_groups.json 与 out/multiseed.json
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree
from shapely import contains_xy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import (kernel_smooth, rasterize_points, china_polygon, load_cache,
                    OUT as BASE_OUT, TR, RES)

HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD, N_REP = 3, 5, 5
SEED = 20260927
SEEDS = [20260927, 12345, 777]
FEATS_T1 = ["f_dist", "f_den", "f_end", "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
FEATS_HF = ["hf_bg", "hf_res", "hf_grad"]
HF_FIELDS = ["h_den", "h_q", "h_lgq", "h_qg"]
FEATS_ALL_H = FEATS_T1 + FEATS_HF + HF_FIELDS
GEO_FEATS = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
             "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
SFIELDS = ["s_T", "s_lgQ", "s_heat", "s_den"]
FEATS_ALL_S = GEO_FEATS + SFIELDS
GROUPS_H = [("fault_geometry", ["f_dist", "f_den", "f_end"]),
            ("tendency", ["Ts_mean", "Ts_max", "Td_mean", "Td_max"]),
            ("heat_flow", ["hf_bg", "hf_res", "hf_grad"]),
            ("explicit_target_derived", ["h_den", "h_q", "h_lgq", "h_qg"])]
GROUPS_S = [("heat_flow", ["hf_bg", "hf_res", "hf_grad"]),
            ("faults", ["f_dist", "f_den", "f_end"]),
            ("tendency", ["Ts_mean", "Ts_max", "Td_mean", "Td_max"]),
            ("spring_derived", SFIELDS)]


def params(seed):
    return dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                min_child_samples=30, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                n_jobs=-1, verbose=-1)


def cv_full(X, y, sx, sy, names, groups, seed=SEED):
    """跨折：单场置换（AP + AUC）、分组置换（AP + AUC）、折外预测合并。"""
    blk = ((np.floor(sx / BLK_KM).astype(np.int64) + 1000) * 10000
           + (np.floor(sy / BLK_KM).astype(np.int64) + 1000))
    ub = np.unique(blk)
    fo = {b: i % N_FOLD for i, b in enumerate(
        np.random.default_rng(seed).permutation(ub))}
    fold = np.array([fo[b] for b in blk])
    pap = np.full((N_FOLD, len(names)), np.nan)
    pauc = np.full((N_FOLD, len(names)), np.nan)
    gap = np.full((N_FOLD, len(groups)), np.nan)
    gauc = np.full((N_FOLD, len(groups)), np.nan)
    rows = []
    p_pool = np.zeros(len(y))
    rng = np.random.default_rng(seed + 7)
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**params(seed))
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        p_pool[te] = p
        rows.append(dict(fold=k + 1, n_train=int(len(tr_idx)), n_test=int(te.sum()),
                         auc=float(roc_auc_score(y[te], p)),
                         ap=float(average_precision_score(y[te], p))))
        for scoring, mat in (("average_precision", pap), ("roc_auc", pauc)):
            pi = permutation_importance(m, X[te], y[te], n_repeats=N_REP,
                                        random_state=seed, scoring=scoring)
            mat[k] = pi.importances_mean
        Xte = X[te]
        b_ap = average_precision_score(y[te], p)
        b_auc = roc_auc_score(y[te], p)
        for gi, (gname, gcols) in enumerate(groups):
            idxs = [names.index(c) for c in gcols if c in names]
            if not idxs:
                continue
            da, du = [], []
            for _ in range(N_REP):
                Xp = Xte.copy()
                Xp[:, idxs] = Xp[rng.permutation(Xte.shape[0])][:, idxs]
                pp = m.predict_proba(Xp)[:, 1]
                da.append(b_ap - average_precision_score(y[te], pp))
                du.append(b_auc - roc_auc_score(y[te], pp))
            gap[k, gi] = float(np.mean(da))
            gauc[k, gi] = float(np.mean(du))
    a = np.array([r["auc"] for r in rows])
    rec = dict(auc=round(float(a.mean()), 4), auc_sd=round(float(a.std()), 4),
               auc_rng=round(float(a.max() - a.min()), 4),
               ap=round(float(np.mean([r["ap"] for r in rows])), 4),
               folds_auc=[round(float(v), 4) for v in a],
               perm_ap={f: round(float(v), 4) for f, v in zip(names, pap.mean(0))},
               perm_ap_sd={f: round(float(v), 4) for f, v in zip(names, pap.std(0))},
               perm_auc={f: round(float(v), 4) for f, v in zip(names, pauc.mean(0))},
               grouped_ap={groups[i][0]: round(float(gap[:, i].mean()), 4)
                           for i in range(len(groups))},
               grouped_ap_sd={groups[i][0]: round(float(gap[:, i].std()), 4)
                              for i in range(len(groups))},
               grouped_auc={groups[i][0]: round(float(gauc[:, i].mean()), 4)
                            for i in range(len(groups))})
    o_ap = list(np.argsort(-pap.mean(0)))
    o_auc = list(np.argsort(-pauc.mean(0)))
    rec["cliff_ap"] = dict(top1=names[o_ap[0]], top2=names[o_ap[1]],
                           ratio=round(float(pap.mean(0)[o_ap[0]]
                                             / max(pap.mean(0)[o_ap[1]], 1e-9)), 1))
    rec["cliff_auc"] = dict(top1=names[o_auc[0]], top2=names[o_auc[1]],
                            ratio=round(float(pauc.mean(0)[o_auc[0]]
                                              / max(pauc.mean(0)[o_auc[1]], 1e-9)), 1))
    return rec, y, p_pool, rows

# ================= Part 1: IHFC 同型演示 =================
t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"))
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
cell_km2 = np.asarray(z["cell_km2"], float)
LO, LA = np.meshgrid(lon_c, lat_c)
pxx, pyy = TR.transform(LO.ravel(), LA.ravel())
il = np.nonzero(mask.ravel())[0]
pxl, pyl = pxx[il] / 1000.0, pyy[il] / 1000.0

d = load_cache()
poly = china_polygon(d["border"])
m = contains_xy(poly, np.asarray(d["hf_lon"], float), np.asarray(d["hf_lat"], float))
hlon = np.asarray(d["hf_lon"], float)[m]
hlat = np.asarray(d["hf_lat"], float)[m]
hq = np.asarray(d["hf_q"], float)[m]


def hf_fields(sigma, den_sigma):
    f = {}
    f["h_q"] = kernel_smooth(hlon, hlat, hq, LO, LA, sigma)[0]
    f["h_lgq"] = kernel_smooth(hlon, hlat, np.log10(hq + 1.0), LO, LA, sigma)[0]
    sm = np.where(np.isfinite(f["h_q"]), f["h_q"], np.nanmedian(f["h_q"]))
    gy, gx = np.gradient(sm)
    gy = gy / (RES * 111.32)
    gx = gx / (RES * 111.32 * np.cos(np.radians(LA)))
    f["h_qg"] = np.hypot(gx, gy)
    den = kernel_smooth(hlon, hlat, np.ones_like(hlon), LO, LA, den_sigma)[1]
    f["h_den"] = den / cell_km2 * 1e4
    return {k: np.asarray(f[k], np.float32).ravel()[il] for k in HF_FIELDS}


def build_Xl(hf):
    cols = [np.asarray(z[k], np.float32).ravel()[il] for k in FEATS_T1 + FEATS_HF]
    cols += [np.asarray(hf[k], np.float32) for k in HF_FIELDS]
    return np.column_stack(cols).astype(np.float32)


pos_land = (rasterize_points(hlon, hlat, lon_c, lat_c, RES) > 0).ravel()[il]
rng = np.random.default_rng(SEED)
pos_i = np.nonzero(pos_land)[0]
tp = cKDTree(np.column_stack([pxl[pos_i], pyl[pos_i]]))
dmin, _ = tp.query(np.column_stack([pxl, pyl]), k=1)
cand = np.nonzero((~pos_land) & (dmin > POS_BUF_KM))[0]
neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO * len(pos_i)), replace=False)
si = np.r_[pos_i, neg_i]
y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
sx, sy = pxl[si], pyl[si]
print("analogue: hf label cells %d ; negatives %d" % (pos_i.size, neg_i.size), flush=True)

F40 = hf_fields(0.40, 2.0)
Xl40 = build_Xl(F40)
RES_H = {}
for name, feats in [("t1_fault_stress_only", FEATS_T1),
                    ("t2_plus_large_sigma_hf", FEATS_T1 + FEATS_HF),
                    ("t3_plus_explicit_target_derived", FEATS_ALL_H)]:
    rec, _, _, _ = cv_full(Xl40[np.ix_(si, [FEATS_ALL_H.index(f) for f in feats])],
                           y, sx, sy, feats, GROUPS_H)
    RES_H[name] = rec
    print("  %-34s AUC %.4f  cliff(AP) %.1fx [%s/%s]  cliff(AUC) %.1fx"
          % (name, rec["auc"], rec["cliff_ap"]["ratio"], rec["cliff_ap"]["top1"],
             rec["cliff_ap"]["top2"], rec["cliff_auc"]["ratio"]), flush=True)
    print("      grouped AP: " + ", ".join("%s %+.4f" % (k, v)
                                           for k, v in rec["grouped_ap"].items()), flush=True)
    print("      grouped AUC: " + ", ".join("%s %+.4f" % (k, v)
                                            for k, v in rec["grouped_auc"].items()), flush=True)

SWEEP_H = {}
for sg in (0.10, 0.40):
    Xs = build_Xl(hf_fields(sg, sg))
    rec, _, _, _ = cv_full(Xs[np.ix_(si, list(range(len(FEATS_ALL_H))))],
                           y, sx, sy, FEATS_ALL_H, GROUPS_H)
    SWEEP_H["sigma_%.2f" % sg] = rec
    print("  sigma %.2f  AUC %.4f  cliff(AP) %.1fx  grouped AP: %s"
          % (sg, rec["auc"], rec["cliff_ap"]["ratio"],
             {k: v for k, v in rec["grouped_ap"].items()}), flush=True)

json.dump(dict(tiers=RES_H, sweep=SWEEP_H, groups={g: c for g, c in GROUPS_H},
               n_label_cells=int(pos_i.size), n_negatives=int(neg_i.size), seed=SEED),
          io.open(os.path.join(OUTD, "hf_analogue_groups.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote hf_analogue_groups.json (%.0fs)" % (time.time() - t0), flush=True)

# ================= Part 2: 全国多种子 =================
print("\n================ Part 2 多种子（全国泉型/井型，geo10 vs all14）================", flush=True)
il_s = np.nonzero(mask.ravel())[0]
pxn, pyn = pxx[il_s] / 1000.0, pyy[il_s] / 1000.0
Xgeo = np.column_stack([np.asarray(z[k], np.float32).ravel()[il_s] for k in GEO_FEATS])
Xall = np.column_stack([np.asarray(z[k], np.float32).ravel()[il_s] for k in FEATS_ALL_S])
POS = {"spring": np.asarray(z["lab_spring"], np.float32).ravel()[il_s] > 0,
       "well": np.asarray(z["lab_well"], np.float32).ravel()[il_s] > 0}
MS = {}
for tag, pos in POS.items():
    rng = np.random.default_rng(SEED)
    pos_i = np.nonzero(pos)[0]
    tp = cKDTree(np.column_stack([pxn[pos_i], pyn[pos_i]]))
    dmin, _ = tp.query(np.column_stack([pxn, pyn]), k=1)
    c2 = np.nonzero((~pos) & (dmin > POS_BUF_KM))[0]
    neg_i = rng.choice(c2, size=min(len(c2), NEG_RATIO * len(pos_i)), replace=False)
    si = np.r_[pos_i, neg_i]
    y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
    sx, sy = pxn[si], pyn[si]
    MS[tag] = dict(n_pos=int(len(pos_i)), n_neg=int(len(neg_i)), seeds={})
    for sd in SEEDS:
        r10, _, _, _ = cv_full(Xgeo[si], y, sx, sy, GEO_FEATS, GROUPS_S, seed=sd)
        r14, _, _, _ = cv_full(Xall[si], y, sx, sy, FEATS_ALL_S, GROUPS_S, seed=sd)
        MS[tag]["seeds"][str(sd)] = dict(
            geo10=dict(auc=r10["auc"], auc_sd=r10["auc_sd"], ap=r10["ap"]),
            all14=dict(auc=r14["auc"], auc_sd=r14["auc_sd"], ap=r14["ap"]),
            delta_auc=round(r14["auc"] - r10["auc"], 4),
            delta_ap=round(r14["ap"] - r10["ap"], 4),
            cliff_ap_all14=r14["cliff_ap"]["ratio"])
        print("  %-7s seed %-9d geo10 %.4f  all14 %.4f  Δ %+.4f (ΔAP %+.4f)  cliff %.1fx"
              % (tag, sd, r10["auc"], r14["auc"], r14["auc"] - r10["auc"],
                 r14["ap"] - r10["ap"], r14["cliff_ap"]["ratio"]), flush=True)
    da = np.array([v["delta_auc"] for v in MS[tag]["seeds"].values()])
    dp = np.array([v["delta_ap"] for v in MS[tag]["seeds"].values()])
    MS[tag]["delta_auc_across_seeds"] = dict(mean=round(float(da.mean()), 4),
                                             sd=round(float(da.std()), 4),
                                             min=round(float(da.min()), 4),
                                             max=round(float(da.max()), 4))
    MS[tag]["delta_ap_across_seeds"] = dict(mean=round(float(dp.mean()), 4),
                                            sd=round(float(dp.std()), 4))
json.dump(MS, io.open(os.path.join(OUTD, "multiseed.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote multiseed.json  total %.0fs" % (time.time() - t0), flush=True)