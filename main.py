from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.data import fetch_returns
from services.hrp import compute_hrp
from services.montecarlo import run_monte_carlo
from services.factors import compute_factor_exposure
from services.backtest import run_backtest

app = FastAPI(title="Quant Portfolio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TickerRequest(BaseModel):
    tickers: list[str]
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


class WeightedRequest(BaseModel):
    tickers: list[str]
    weights: dict[str, float]
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


class MonteCarloRequest(WeightedRequest):
    n_sims: int = Field(default=500, ge=100, le=5000)
    n_days: int = Field(default=252, ge=21, le=1260)
    initial_value: float = Field(default=100_000.0, ge=1_000.0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/hrp")
def hrp_endpoint(req: TickerRequest):
    try:
        returns = fetch_returns(tuple(req.tickers), req.start_date, req.end_date)
        return compute_hrp(returns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/montecarlo")
def montecarlo_endpoint(req: MonteCarloRequest):
    try:
        returns = fetch_returns(tuple(req.tickers), req.start_date, req.end_date)
        return run_monte_carlo(
            returns,
            req.weights,
            req.n_sims,
            req.n_days,
            req.initial_value,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/factors")
def factors_endpoint(req: WeightedRequest):
    try:
        returns = fetch_returns(tuple(req.tickers), req.start_date, req.end_date)
        return compute_factor_exposure(returns, req.weights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
def backtest_endpoint(req: WeightedRequest):
    try:
        returns = fetch_returns(tuple(req.tickers), req.start_date, req.end_date)
        return run_backtest(returns, req.weights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
