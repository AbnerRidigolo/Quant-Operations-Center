import yfinance as yf
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=64)
def fetch_prices(tickers_tuple: tuple, start: str, end: str) -> pd.DataFrame:
    tickers = list(tickers_tuple)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single ticker — single-level columns
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    # Drop tickers with too few observations
    prices = prices.loc[:, prices.count() >= 60]
    return prices


def fetch_returns(tickers_tuple: tuple, start: str, end: str) -> pd.DataFrame:
    prices = fetch_prices(tickers_tuple, start, end)
    returns = prices.pct_change().dropna()
    return returns
