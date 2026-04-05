"""
Monte Carlo simulation via Geometric Brownian Motion with full covariance
(Cholesky decomposition for correlated assets).
"""

import numpy as np
import pandas as pd


def run_monte_carlo(
    returns: pd.DataFrame,
    weights: dict,
    n_sims: int = 500,
    n_days: int = 252,
    initial_value: float = 100_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    ret = returns[tickers]
    mu = ret.mean().values          # daily mean vector
    cov = ret.cov().values          # daily covariance matrix

    # Regularise covariance matrix
    cov += np.eye(len(tickers)) * 1e-10
    L = np.linalg.cholesky(cov)

    # Simulate paths
    paths = np.empty((n_sims, n_days + 1))
    paths[:, 0] = initial_value

    for t in range(1, n_days + 1):
        z = rng.standard_normal((len(tickers), n_sims))
        corr_shocks = L @ z                        # (n_assets, n_sims)
        asset_rets = mu[:, None] + corr_shocks     # (n_assets, n_sims)
        port_rets = w @ asset_rets                 # (n_sims,)
        paths[:, t] = paths[:, t - 1] * (1.0 + port_rets)

    final = paths[:, -1]
    gains = final - initial_value

    var_95 = float(np.percentile(gains, 5))
    cvar_95 = float(gains[gains <= var_95].mean())

    port_daily_vol = float(np.sqrt(w @ cov @ w))
    port_daily_mu = float(w @ mu)
    ann_ret = port_daily_mu * 252
    ann_vol = port_daily_vol * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Return only 200 sampled paths for visualisation, full stats
    sample_idx = rng.choice(n_sims, size=min(200, n_sims), replace=False)
    sampled_paths = paths[sample_idx].tolist()

    return {
        "paths": sampled_paths,
        "final_values": final.tolist(),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "prob_profit": float((final > initial_value).mean()),
        "median_final": float(np.median(final)),
        "p5": float(np.percentile(final, 5)),
        "p25": float(np.percentile(final, 25)),
        "p75": float(np.percentile(final, 75)),
        "p95": float(np.percentile(final, 95)),
        "initial_value": initial_value,
        "metrics": {
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
        },
        "n_sims": n_sims,
        "n_days": n_days,
    }
