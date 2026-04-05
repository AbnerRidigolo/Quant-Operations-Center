"""
4-Factor exposure (Mkt-RF, SMB, HML, MOM) using ETF proxies:
  Mkt-RF : SPY - daily RF
  SMB    : IWM - SPY          (small vs large cap)
  HML    : IVE - IVW          (value vs growth)
  MOM    : MTUM - SPY         (momentum vs market)
OLS regression of portfolio excess returns on factor returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression


FACTOR_PROXIES = ["SPY", "IWM", "IVE", "IVW", "MTUM"]
ANNUAL_RF = 0.05


def _fetch_factor_returns(start: str, end: str) -> pd.DataFrame:
    data = yf.download(FACTOR_PROXIES, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    rets = prices.pct_change().dropna()

    daily_rf = ANNUAL_RF / 252
    factors = pd.DataFrame(
        {
            "Mkt-RF": rets["SPY"] - daily_rf,
            "SMB": rets["IWM"] - rets["SPY"],
            "HML": rets["IVE"] - rets["IVW"],
            "MOM": rets["MTUM"] - rets["SPY"],
        },
        index=rets.index,
    )
    return factors.dropna()


def compute_factor_exposure(returns: pd.DataFrame, weights: dict) -> dict:
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    port_ret = returns[tickers].dot(w)

    start = str(returns.index[0].date())
    end = str(returns.index[-1].date())
    factors = _fetch_factor_returns(start, end)

    common = port_ret.index.intersection(factors.index)
    daily_rf = ANNUAL_RF / 252
    y = (port_ret.loc[common] - daily_rf).values
    X = factors.loc[common].values
    factor_names = list(factors.columns)

    reg = LinearRegression().fit(X, y)
    betas = reg.coef_
    alpha_daily = float(reg.intercept_)
    r2 = float(reg.score(X, y))

    # Residuals
    y_hat = reg.predict(X)
    residuals = y - y_hat
    idio_vol = float(residuals.std() * np.sqrt(252))

    # Rolling 63-day betas (Mkt-RF only for chart)
    port_excess = port_ret.loc[common] - daily_rf
    mkt_rf = factors.loc[common, "Mkt-RF"]
    roll_beta = (
        port_excess.rolling(63)
        .cov(mkt_rf)
        .div(mkt_rf.rolling(63).var())
        .dropna()
    )

    return {
        "alpha_annual": alpha_daily * 252,
        "betas": dict(zip(factor_names, betas.tolist())),
        "r_squared": r2,
        "idiosyncratic_vol": idio_vol,
        "factor_names": factor_names,
        "portfolio_returns": port_ret.loc[common].tolist(),
        "market_returns": factors.loc[common, "Mkt-RF"].tolist(),
        "dates": [str(d.date()) for d in common],
        "rolling_beta": roll_beta.tolist(),
        "rolling_beta_dates": [str(d.date()) for d in roll_beta.index],
    }
