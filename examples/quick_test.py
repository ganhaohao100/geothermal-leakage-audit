"""Synthetic quick test for the audit package.

This script verifies that the Python environment can run the modelling stack
without any restricted spring or well data. It generates a small synthetic
label set, fits a random forest across five stratified folds, and prints the
mean held-out AUC.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 20261122

rng = np.random.default_rng(SEED)
n_samples, n_features = 600, 8
X = rng.normal(size=(n_samples, n_features))
logit = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.4 * rng.normal(size=n_samples)
y = (logit > np.quantile(logit, 0.75)).astype(int)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for train, test in folds.split(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=0,
    )
    model.fit(X[train], y[train])
    scores.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))

auc = float(np.mean(scores))
print(f"quick test AUC = {auc:.3f}")
assert auc > 0.70, "quick test failed: synthetic signal was not recovered"
print("quick test passed")
