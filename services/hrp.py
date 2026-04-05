"""
Hierarchical Risk Parity (Lopez de Prado, 2016).
Steps:
  1. Build distance matrix from correlation.
  2. Single-linkage hierarchical clustering.
  3. Quasi-diagonalise the covariance matrix.
  4. Recursive bisection to assign weights.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ivp(cov: pd.DataFrame) -> np.ndarray:
    """Inverse-Variance Portfolio weights for a sub-cluster."""
    v = np.diag(cov.values)
    v = np.where(v <= 0, 1e-12, v)
    w = 1.0 / v
    return w / w.sum()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    w = _ivp(sub)
    return float(w @ sub.values @ w)


def _quasi_diag(link: np.ndarray) -> list:
    """Return leaf order that quasi-diagonalises the matrix."""
    link = link.astype(int)
    n = link[-1, 3]  # total original items
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    while sort_ix.max() >= n:
        sort_ix.index = range(0, len(sort_ix) * 2, 2)
        mask = sort_ix >= n
        idx = sort_ix.index[mask]
        jdx = sort_ix.values[mask] - n
        sort_ix[idx] = link[jdx, 0]
        new = pd.Series(link[jdx, 1], index=idx + 1)
        sort_ix = pd.concat([sort_ix, new]).sort_index()
        sort_ix.index = range(len(sort_ix))

    return sort_ix.tolist()


def _rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Recursive bisection to assign final HRP weights."""
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while clusters:
        clusters = [
            sub[j:k]
            for sub in clusters
            for j, k in ((0, len(sub) // 2), (len(sub) // 2, len(sub)))
            if len(sub) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha

    return w / w.sum()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hrp(returns: pd.DataFrame) -> dict:
    cov = returns.cov()
    corr = returns.corr()

    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values)

    link = linkage(condensed, method="single")

    sort_idx = _quasi_diag(link)
    sort_labels = corr.index[sort_idx].tolist()

    weights = _rec_bipart(cov, sort_labels)

    # Dendrogram coordinates for Plotly
    d = dendrogram(link, labels=list(corr.index), no_plot=True)

    port_var = float(weights.values @ cov.values @ weights.values)
    ann_vol = float(np.sqrt(252 * port_var))
    ann_ret = float(252 * returns.mean() @ weights)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return {
        "weights": weights.to_dict(),
        "correlation": corr.to_dict(),
        "tickers": list(corr.index),
        "dendrogram": {
            "icoord": d["icoord"],
            "dcoord": d["dcoord"],
            "ivl": d["ivl"],
        },
        "metrics": {
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "n_assets": len(weights),
        },
    }
