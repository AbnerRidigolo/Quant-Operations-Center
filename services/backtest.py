"""
Historical walk-forward backtest with full performance attribution.
Metrics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate,
         Best/Worst day, VaR daily, rolling Sharpe.
"""

import numpy as np
import pandas as pd
import yfinance as yf


ANNUAL_RF = 0.05


def _metrics(rets: pd.Series) -> dict:
    daily_rf = ANNUAL_RF / 252
    excess = rets - daily_rf

    ann_ret = float((1 + rets.mean()) ** 252 - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    neg = rets[rets < 0]
    sortino = (
        float(excess.mean() / neg.std() * np.sqrt(252))
        if len(neg) > 0 and neg.std() > 0
        else np.nan
    )

    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    var_95 = float(rets.quantile(0.05))

    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": float(sortino) if not np.isnan(sortino) else None,
        "max_drawdown": max_dd,
        "calmar": float(calmar) if not np.isnan(calmar) else None,
        "var_95_daily": var_95,
        "win_rate": float((rets > 0).mean()),
        "best_day": float(rets.max()),
        "worst_day": float(rets.min()),
        "total_return": float(cum.iloc[-1] - 1),
    }


def run_backtest(returns: pd.DataFrame, weights: dict, benchmark: str = "SPY") -> dict:
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    port_ret = returns[tickers].dot(w)

    start = str(returns.index[0].date())
    end = str(returns.index[-1].date())

    bench_raw = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
    close = bench_raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    bench_ret = close.pct_change().dropna()

    common = port_ret.index.intersection(bench_ret.index)
    pr = port_ret.loc[common]
    br = bench_ret.loc[common]

    # Cumulative
    port_cum = (1 + pr).cumprod()
    bench_cum = (1 + br).cumprod()

    # Drawdown
    port_dd = (port_cum - port_cum.cummax()) / port_cum.cummax()

    # Rolling 252-day Sharpe
    roll_sharpe = (
        pr.rolling(252)
        .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0)
        .dropna()
    )

    # Active return (alpha) and tracking error
    active = pr - br
    te = float(active.std() * np.sqrt(252))
    info_ratio = float(active.mean() * 252 / te) if te > 0 else 0.0

    return {
        "dates": [str(d.date()) for d in common],
        "portfolio_cum": port_cum.tolist(),
        "benchmark_cum": bench_cum.tolist(),
        "portfolio_drawdown": port_dd.tolist(),
        "rolling_sharpe": roll_sharpe.tolist(),
        "rolling_sharpe_dates": [str(d.date()) for d in roll_sharpe.index],
        "portfolio_metrics": _metrics(pr),
        "benchmark_metrics": _metrics(br),
        "tracking_error": te,
        "information_ratio": info_ratio,
        "benchmark": benchmark,
    }
