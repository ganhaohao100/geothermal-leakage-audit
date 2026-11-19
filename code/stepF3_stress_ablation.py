# -*- coding: utf-8 -*-
"""Group ablation: drop the four stress attributes from the well-type ten-field baseline.
Same pseudo-negatives, same fold map, same LightGBM settings as stepF.
"""
import io, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import DATA_DIR as BASE, OUTD  # noqa: E402
from common import TR, OUT as BASE_OUT
HERE = os.path.dirname(os.path.abspath(__file__))
BLK_KM, BUF_KM, POS_BUF_KM = 50.0, 10.0, 10.0
NEG_RATIO, N_FOLD, SEED = 3, 5, 20260927
FEATS = ["hf_bg", "hf_res", "hf_grad", "f_dist", "f_den", "f_end",
         "Ts_mean", "Ts_max", "Td_mean", "Td_max"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
              n_jobs=-1, verbose=-1)

t0 = time.time()
z = np.load(os.path.join(BASE_OUT, "features.npz"), allow_pickle=True)
lon_c, lat_c, mask = z["lon_c"], z["lat_c"], z["mask"]
LO, LA = np.meshgrid(lon_c, lat_c)
il = np.nonzero(mask.ravel())[0]
px, py = TR.transform(LO.ravel(), LA.ravel()); px, py = px/1000.0, py/1000.0
px, py = px[il], py[il]
X_all = np.column_stack([np.asarray(z[k], np.float32).ravel()[il] for k in FEATS])
POS = np.asarray(z["lab_well"], np.float32).ravel()[il] > 0

def blocked(sx, sy):
    return ((np.floor(sx/BLK_KM).astype(np.int64)+1000)*10000
            + (np.floor(sy/BLK_KM).astype(np.int64)+1000))

rng = np.random.default_rng(SEED)
pos_i = np.nonzero(POS)[0]
tp = cKDTree(np.column_stack([px[pos_i], py[pos_i]]))
dmin, _ = tp.query(np.column_stack([px, py]), k=1)
cand = np.nonzero((~POS) & (dmin > POS_BUF_KM))[0]
neg_i = rng.choice(cand, size=min(len(cand), NEG_RATIO*len(pos_i)), replace=False)
si = np.r_[pos_i, neg_i]
y = np.r_[np.ones(len(pos_i)), np.zeros(len(neg_i))]
blk = blocked(px, py)[si]
ub = np.unique(blk)
fo = {b: i % N_FOLD for i, b in enumerate(np.random.default_rng(SEED).permutation(ub))}
fold = np.array([fo[b] for b in blk])
sx, sy = px[si], py[si]

def run(cols):
    X = X_all[np.ix_(si, cols)]
    a, p = [], []
    for k in range(N_FOLD):
        te = fold == k; tr = ~te
        tv = cKDTree(np.column_stack([sx[te], sy[te]]))
        dd, _ = tv.query(np.column_stack([sx[tr], sy[tr]]), k=1)
        tr_idx = np.nonzero(tr)[0][dd > BUF_KM]
        m = lgb.LGBMClassifier(**PARAMS); m.fit(X[tr_idx], y[tr_idx])
        pr = m.predict_proba(X[te])[:, 1]
        a.append(float(roc_auc_score(y[te], pr)))
        p.append(float(average_precision_score(y[te], pr)))
    return np.array(a), np.array(p)

ref_a, ref_p = run([FEATS.index(f) for f in FEATS])
DROP = ["Ts_mean", "Ts_max", "Td_mean", "Td_max"]
no_a, no_p = run([FEATS.index(f) for f in FEATS if f not in DROP])
d = ref_a - no_a

def s(v): return round(float(v), 4)
out = dict(seed=SEED, fc="well-type", dropped="stress(4)",
           ref=dict(n=10, auc=s(ref_a.mean()), auc_sd_pop=s(ref_a.std()),
                    ap=s(ref_p.mean()), folds=[s(v) for v in ref_a]),
           no_stress=dict(n=6, auc=s(no_a.mean()), auc_sd_pop=s(no_a.std()),
                           ap=s(no_p.mean()), folds=[s(v) for v in no_a]),
           delta_mean=s(d.mean()), delta_sd_pop=s(d.std()),
           delta_sd_sample=s(d.std(ddof=1)), delta_folds=[s(v) for v in d],
           elapsed_s=round(time.time()-t0, 1))
print(json.dumps(out, indent=1), flush=True)
io.open(os.path.join(OUTD, "stress_ablation.json"), "w", encoding="utf-8").write(
    json.dumps(out, ensure_ascii=False, indent=1))
