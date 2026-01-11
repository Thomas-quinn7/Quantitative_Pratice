import argparse
import sys
from typing import List, Optional

import os
from datetime import datetime
import numpy as np
import pandas as pd

# Support both package execution (python -m Stock_pred.garch_main)
# and direct script execution (python Stock_pred/garch_main.py)
try:
    from .garch_model import (
        fetch_prices,
        log_returns,
        fit_garch,
        forecast_volatility,
        plot_conditional_volatility,
    )
    from .arima_models import (
        fit_arima_grid,
        forecast_returns,
        forecast_prices_from_returns,
    )
except Exception:  # pragma: no cover - fallback for direct script run
    import os

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from Stock_pred.garch_model import (  # type: ignore
        fetch_prices,
        log_returns,
        fit_garch,
        forecast_volatility,
        plot_conditional_volatility,
    )
    from Stock_pred.arima_models import (  # type: ignore
        fit_arima_grid,
        forecast_returns,
        forecast_prices_from_returns,
    )


def _append_records(out_csv: str, records: list[dict]) -> None:
    if not records:
        return
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=header, index=False)


def run_garch_for_tickers(
    tickers: List[str],
    period: str = "3y",
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    horizon: int = 20,
    plot: bool = False,
    out_csv: Optional[str] = None,
) -> None:
    run_dt = datetime.now()
    run_date = run_dt.date().isoformat()
    for t in tickers:
        try:
            px = fetch_prices(t, period=period)
            rets = log_returns(px)
            fitted = fit_garch(rets, p=p, q=q, dist=dist, show_output=False)
            var, vol = forecast_volatility(fitted, horizon=horizon)

            ann_factor = np.sqrt(252.0)
            ann_vol = vol * ann_factor
            print(f"Ticker: {t}")
            print(f"  Next {horizon} days daily vol forecast:")
            print(vol.to_string())
            print(f"  Annualized vol (sqrt(252)) forecast:")
            print(ann_vol.to_string())
            if plot:
                plot_conditional_volatility(
                    fitted, title=f"{t} GARCH({p},{q}) Conditional Volatility"
                )

            if out_csv:
                records = []
                for step_idx, (d_vol, a_vol) in enumerate(zip(vol.values, ann_vol.values), start=1):
                    records.append(
                        {
                            "run_datetime": run_dt.isoformat(timespec="seconds"),
                            "run_date": run_date,
                            "ticker": t,
                            "model": "GARCH",
                            "p": p,
                            "q": q,
                            "dist": dist,
                            "horizon": horizon,
                            "step": step_idx,
                            "daily_vol": float(d_vol),
                            "annualized_vol": float(a_vol),
                            "best_order_p": None,
                            "best_order_d": None,
                            "best_order_q": None,
                            "aic": None,
                            "ret_forecast": None,
                            "price_forecast": None,
                        }
                    )
                _append_records(out_csv, records)
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")


def run_arima_for_tickers(
    tickers: List[str],
    period: str = "3y",
    steps: int = 5,
    verbose: bool = False,
    out_csv: Optional[str] = None,
) -> None:
    run_dt = datetime.now()
    run_date = run_dt.date().isoformat()
    for t in tickers:
        try:
            px = fetch_prices(t, period=period)
            rets = log_returns(px)
            best = fit_arima_grid(rets, verbose=verbose)
            r_fc = forecast_returns(best, steps=steps)
            p_fc = forecast_prices_from_returns(float(px.iloc[-1]), r_fc)
            print(f"Ticker: {t}")
            print(f"  Best ARIMA order: {best.order}  (AIC={best.aic:.2f})")
            print("  Return forecast:")
            print(r_fc.to_string())
            print("  Price forecast:")
            print(p_fc.to_string())

            if out_csv:
                order_p, order_d, order_q = best.order
                records = []
                for step_idx in range(1, steps + 1):
                    ret_val = float(r_fc.iloc[step_idx - 1])
                    price_val = float(p_fc.iloc[step_idx - 1])
                    records.append(
                        {
                            "run_datetime": run_dt.isoformat(timespec="seconds"),
                            "run_date": run_date,
                            "ticker": t,
                            "model": "ARIMA",
                            "p": None,
                            "q": None,
                            "dist": None,
                            "horizon": steps,
                            "step": step_idx,
                            "daily_vol": None,
                            "annualized_vol": None,
                            "best_order_p": int(order_p),
                            "best_order_d": int(order_d),
                            "best_order_q": int(order_q),
                            "aic": float(best.aic),
                            "ret_forecast": ret_val,
                            "price_forecast": price_val,
                        }
                    )
                _append_records(out_csv, records)
        except Exception as exc:
            print(f"[WARN] {t}: {exc}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run GARCH fits for tickers")
    ap.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL"], help="Tickers list") # Default tickers
    ap.add_argument("--period", default="3y", help="History period, e.g. 1y, 3y, 5y")
    ap.add_argument("--p", type=int, default=1, help="ARCH order")
    ap.add_argument("--q", type=int, default=1, help="GARCH order")
    ap.add_argument("--dist", default="t", choices=["normal", "t", "skewt", "ged"], help="Innovation distribution")
    ap.add_argument("--horizon", type=int, default=20, help="Forecast horizon (days)")
    ap.add_argument("--plot", action="store_true", help="Plot conditional volatility")
    ap.add_argument(
        "--models",
        choices=["garch", "arima", "both"],
        default="both",
        help="Which models to run",
    )
    ap.add_argument("--steps", type=int, default=5, help="ARIMA forecast steps")
    ap.add_argument("--arima-verbose", action="store_true", help="Verbose ARIMA grid search logs")
    default_csv = os.path.join(os.path.dirname(__file__), "model_results.csv")
    ap.add_argument("--out-csv", default=default_csv, help="Path to append results CSV")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv or [])

    if ns.models in ("garch", "both"):
        run_garch_for_tickers(
            tickers=ns.tickers,
            period=ns.period,
            p=ns.p,
            q=ns.q,
            dist=ns.dist,
            horizon=ns.horizon,
            plot=ns.plot,
            out_csv=ns.out_csv,
        )

    if ns.models in ("arima", "both"):
        run_arima_for_tickers(
            tickers=ns.tickers,
            period=ns.period,
            steps=ns.steps,
            verbose=ns.arima_verbose,
            out_csv=ns.out_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
