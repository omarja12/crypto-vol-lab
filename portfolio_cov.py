"""
Covariance estimation for the vol-portfolio problem.

THREE ESTIMATORS
----------------
sample        - the naive one. Near-singular when assets are few and
                correlated, which is exactly this problem. Included as the
                baseline that the others must beat.
ledoit_wolf   - shrinks the sample matrix toward a scaled identity by an
                analytically optimal amount. The minimum defensible choice.
pca_factor    - reconstructs covariance from k principal components plus
                idiosyncratic variance. Most stable and most interpretable
                here, because the vol surface genuinely lives on three factors
                (level, slope, curvature), so a 3-factor model is not an
                approximation imposed for tractability but a description of the
                actual structure.

WHAT GOES IN
------------
Rows are dates, columns are (underlying, tenor) buckets. The RETURNS are
variance-risk-premium returns or delta-hedged P&L, NEVER raw option returns.
Raw option returns make sample variance understate the risk of short options,
which sends any optimiser straight into short gamma. This module cannot enforce
that; it is on the caller. It is the single most important rule in the file.
"""

from __future__ import annotations

import numpy as np


def sample_cov(returns):
    """Plain sample covariance. The baseline, not a recommendation."""
    R = np.asarray(returns, float)
    return np.cov(R, rowvar=False)


def ledoit_wolf(returns):
    """
    Ledoit-Wolf shrinkage toward a scaled identity.

    Returns (shrunk_cov, shrinkage_intensity). An intensity near 1 means the
    sample matrix carried almost no usable information and you are leaning
    entirely on the structured target - itself a useful warning about how
    little your data supports.

    Implements the 2004 estimator directly rather than depending on sklearn,
    to keep the module standalone.
    """
    R = np.asarray(returns, float)
    n, p = R.shape
    if n < 2:
        raise ValueError("need at least two observations")

    X = R - R.mean(axis=0)
    S = (X.T @ X) / n

    mu = np.trace(S) / p                      # target is mu * I
    target = mu * np.eye(p)

    d2 = np.sum((S - target) ** 2)            # ||S - target||^2

    # Expected estimation error of S, from the fourth moments.
    b2 = 0.0
    for t in range(n):
        xt = X[t][:, None]
        b2 += np.sum((xt @ xt.T - S) ** 2)
    b2 /= n * n
    b2 = min(b2, d2)                          # clamp so intensity stays in [0, 1]

    intensity = b2 / d2 if d2 > 0 else 1.0
    shrunk = intensity * target + (1.0 - intensity) * S
    return shrunk, float(intensity)


def pca_factor_cov(returns, k=3, return_components=False):
    """
    Factor-model covariance from the top k principal components.

        cov = B @ cov_factors @ B.T + diag(idiosyncratic)

    The systematic part is rank k; the diagonal restores per-asset variance the
    factors do not capture, which keeps the matrix positive definite and
    invertible even when k is small.

    k = 3 is the natural choice for a vol surface: level, slope, curvature.
    Inspect the explained-variance ratio to confirm three factors actually
    capture the surface before trusting the decomposition.
    """
    R = np.asarray(returns, float)
    n, p = R.shape
    k = min(k, p, n - 1)

    X = R - R.mean(axis=0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    eigvals = s ** 2 / n
    explained = eigvals / eigvals.sum()

    B = Vt[:k].T                               # loadings, p x k
    factor_scores = X @ B                       # n x k
    cov_factors = np.cov(factor_scores, rowvar=False)
    if k == 1:
        cov_factors = np.array([[float(cov_factors)]])

    systematic = B @ cov_factors @ B.T
    total_var = np.var(X, axis=0)
    idio = np.maximum(total_var - np.diag(systematic), 1e-12)

    cov = systematic + np.diag(idio)

    if return_components:
        return cov, {
            "loadings": B,
            "explained_variance_ratio": explained[:k],
            "cumulative_explained": float(explained[:k].sum()),
            "idiosyncratic": idio,
        }
    return cov


def correlation_from_cov(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def condition_number(cov):
    """
    Ratio of largest to smallest eigenvalue.

    A large value means the matrix is close to singular and its inverse - which
    mean-variance needs - will amplify estimation error violently. Watching this
    across estimators shows directly what shrinkage buys you.
    """
    ev = np.linalg.eigvalsh(cov)
    return float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf


def nearest_psd(cov, epsilon=1e-10):
    """Clip negative eigenvalues to restore positive semidefiniteness."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    vals = np.maximum(vals, epsilon)
    return vecs @ np.diag(vals) @ vecs.T
