"""Synthetic quick test for the geothermal leakage audit repository.

This test deliberately uses no project data. It exercises the grid, Gaussian
smoothing, spatial-block assignment, buffer exclusion and AUC routines that the
analysis scripts rely on. Expected runtime is under 10 seconds.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score


def make_grid(n=80, seed=20261118):
    rng = np.random.default_rng(seed)
    points = rng.integers(0, n, size=(240, 2))
    counts = np.zeros((n, n), dtype=float)
    for i, j in points:
        counts[i, j] += 1.0
    return counts


def assign_blocks(n, block=10):
    rows, cols = np.indices((n, n))
    return (rows // block) * (n // block) + (cols // block)


def buffered_split(blocks, test_blocks, buffer_cells=1):
    test_mask = np.isin(blocks, test_blocks)
    train_mask = ~test_mask
    for di in range(-buffer_cells, buffer_cells + 1):
        for dj in range(-buffer_cells, buffer_cells + 1):
            shifted = np.zeros_like(test_mask)
            src = test_mask[max(0, di):test_mask.shape[0] + min(0, di),
                            max(0, dj):test_mask.shape[1] + min(0, dj)]
            dst = shifted[max(0, -di):test_mask.shape[0] + min(0, -di),
                          max(0, -dj):test_mask.shape[1] + min(0, -dj)]
            dst[:] = src
            train_mask &= ~shifted
    return train_mask, test_mask


def main():
    counts = make_grid()
    smoothed = gaussian_filter(counts, sigma=2.0, mode="wrap")
    assert np.isclose(smoothed.sum(), counts.sum(), rtol=1e-6)
    assert smoothed.shape == counts.shape

    blocks = assign_blocks(counts.shape[0], block=10)
    unique = np.unique(blocks)
    assert unique.size == 64
    train, test = buffered_split(blocks, test_blocks=(27, 28, 35, 36), buffer_cells=1)
    assert test.sum() > 0 and train.sum() > 0
    assert not np.any(train & test)

    rng = np.random.default_rng(7)
    scores = smoothed.ravel()
    labels = (scores > np.quantile(scores, 0.70)).astype(int)
    flip = rng.random(labels.size) < 0.15
    labels[flip] = 1 - labels[flip]
    auc = roc_auc_score(labels, scores)
    assert 0.0 <= auc <= 1.0
    print(f"quick-test passed: grid={counts.shape[0]}x{counts.shape[1]}, "
          f"blocks={unique.size}, withheld_cells={test.sum()}, "
          f"training_cells={train.sum()}, synthetic_AUC={auc:.3f}")


if __name__ == "__main__":
    main()

