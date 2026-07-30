import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TARGET_COL = "new_cases"

# The OWID master file reports weekly totals (rows are 7 days apart),
# so one ARIMA "step" = one week. FORECAST_STEPS=14 -> a 14-week horizon.
FORECAST_STEPS = 14

MAX_P = 4
MAX_Q = 4
MAX_D = 2

# Palette (dataviz reference): ink for history, blue for model output
COLOR_HISTORY = "#52514e"
COLOR_MODEL = "#2a78d6"
COLOR_ACTUAL = "#e34948"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"


def load_series(csv_path: str) -> pd.Series:
    """Steps 1: date -> datetime, set as index, enforce weekly freq, no missing values."""
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target col: {TARGET_COL}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    s = pd.Series(
        pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(df["date"]),
        name=TARGET_COL,
    )

    # Snap to a regular weekly grid; interpolate any weeks lost to row-cleaning
    freq = pd.infer_freq(s.index) or "W-SUN"
    s = s.asfreq(freq)
    n_missing = int(s.isna().sum())
    if n_missing:
        print(f"Filled {n_missing} missing week(s) by linear interpolation.")
        s = s.interpolate(method="linear")

    return s


def choose_d(s: pd.Series, alpha: float = 0.05) -> int:
    """Step 2: difference until the ADF test says stationary (up to MAX_D)."""
    for d in range(MAX_D + 1):
        test = s if d == 0 else s.diff(d and 1).dropna() if d == 1 else s.diff().diff().dropna()
        pval = adfuller(test.dropna(), autolag="AIC")[1]
        print(f"ADF test, d={d}: p-value = {pval:.4f}")
        if pval < alpha:
            return d
    return MAX_D


def select_order(s: pd.Series, d: int):
    """Step 3: automated (p, d, q) selection by AIC grid search."""
    best_aic, best_order = np.inf, (0, d, 0)
    for p in range(MAX_P + 1):
        for q in range(MAX_Q + 1):
            try:
                fit = ARIMA(s, order=(p, d, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, (p, d, q)
            except Exception:
                continue
    print(f"Selected order {best_order} (AIC = {best_aic:.1f})")
    return best_order


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)) * 100)


def backtest(s_log: pd.Series, order, steps: int):
    """Hold out the last `steps` weeks, forecast them, and score vs a naive baseline."""
    train, test = s_log.iloc[:-steps], s_log.iloc[-steps:]

    fit = ARIMA(train, order=order).fit()
    pred_log = fit.forecast(steps=steps)

    y_true = np.expm1(test.to_numpy())
    y_pred = np.expm1(pred_log.to_numpy())
    naive = np.full(steps, np.expm1(train.iloc[-1]))  # "next week = this week"

    metrics = {
        "arima_mae": float(np.mean(np.abs(y_true - y_pred))),
        "arima_smape": smape(y_true, y_pred),
        "naive_mae": float(np.mean(np.abs(y_true - naive))),
        "naive_smape": smape(y_true, naive),
    }

    print("\nBacktest on the last "
          f"{steps} weeks ({test.index[0].date()} -> {test.index[-1].date()}):")
    print(f"  ARIMA  MAE = {metrics['arima_mae']:,.0f}   sMAPE = {metrics['arima_smape']:.1f}%")
    print(f"  Naive  MAE = {metrics['naive_mae']:,.0f}   sMAPE = {metrics['naive_smape']:.1f}%")

    return pd.Series(y_pred, index=test.index, name="backtest_pred"), metrics


def forecast_future(s_log: pd.Series, order, steps: int):
    """Steps 4-5: fit on the full series and forecast the next `steps` weeks."""
    fit = ARIMA(s_log, order=order).fit()
    res = fit.get_forecast(steps=steps)

    mean = np.expm1(res.predicted_mean)
    ci = np.expm1(res.conf_int(alpha=0.05))
    mean.name = "forecast_new_cases"
    ci.columns = ["ci_lower_95", "ci_upper_95"]

    out = pd.concat([mean, ci], axis=1).clip(lower=0)
    return out, fit


def plot_results(s: pd.Series, backtest_pred: pd.Series, forecast: pd.DataFrame,
                 country: str, order, out_png: str):
    """Step 6: full history on top, recent window + forecast below. One y-axis each."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.3},
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.01)

    # -- top: full history for context
    ax1.plot(s.index, s.values, color=COLOR_HISTORY, linewidth=1.6, label="Weekly new cases")
    ax1.set_title(f"{country} — weekly new COVID-19 cases (full history)",
                  loc="left", fontsize=11, color="#0b0b0b")
    ax1.legend(frameon=False, fontsize=9)

    # -- bottom: last ~60 weeks + backtest + 14-week forecast
    recent = s.iloc[-60:]
    ax2.plot(recent.index, recent.values, color=COLOR_HISTORY, linewidth=1.8,
             label="Actual")
    ax2.plot(backtest_pred.index, backtest_pred.values, color=COLOR_ACTUAL,
             linewidth=1.8, linestyle="--", label=f"Backtest forecast (last {len(backtest_pred)} wks)")
    ax2.plot(forecast.index, forecast["forecast_new_cases"], color=COLOR_MODEL,
             linewidth=2.0, label=f"Forecast (next {len(forecast)} wks)")
    ax2.fill_between(forecast.index, forecast["ci_lower_95"], forecast["ci_upper_95"],
                     color=COLOR_MODEL, alpha=0.15, linewidth=0, label="95% interval")
    ax2.axvline(s.index[-1], color=GRID, linewidth=1)
    # weekly cases span several orders of magnitude across the window
    ax2.set_yscale("log")
    ax2.set_title(f"ARIMA{order} — backtest and {len(forecast)}-week forecast (log scale)",
                  loc="left", fontsize=11, color="#0b0b0b")
    ax2.legend(frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(colors="#52514e", labelsize=9)

    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


# Each region is modelled independently: ARIMA trains on that region's own
# weekly new_cases history and forecasts its next FORECAST_STEPS weeks.
# Antarctica is excluded (no OWID case data).
REGIONS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

# Shortlist of orders to compare (see grid analysis). d is fixed at 1: the
# series is stationary after one differencing and d=2 overdifferences.
CANDIDATE_ORDERS = [(0, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2)]


def run_region(region: str, data_folder: str, results_root: str):
    csv_path = os.path.join(data_folder, f"{region}.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {region}: CSV not found ({csv_path})")
        return None

    output_dir = os.path.join(results_root, region)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare data (weekly cases, datetime index, no missing values)
    s = load_series(csv_path)
    print(f"Series: {len(s)} weekly points, {s.index[0].date()} -> {s.index[-1].date()}")

    # log1p stabilises the huge wave spikes so they don't distort the fit
    s_log = np.log1p(s)

    summary_rows = []
    for order in CANDIDATE_ORDERS:
        p, d_, q = order
        print(f"\n{'='*60}\n{region} — ARIMA{order}")

        try:
            # 4-5a. Honesty check: how well would this predict the last N weeks?
            bt, metrics = backtest(s_log, order, FORECAST_STEPS)

            # 4-5b. Refit on everything, forecast the next N weeks
            forecast, fit = forecast_future(s_log, order, FORECAST_STEPS)
        except Exception as e:
            print(f"  Skipped ARIMA{order}: {e}")
            summary_rows.append({"region": region, "order": f"({p},{d_},{q})", "error": str(e)})
            continue

        print(f"\n{FORECAST_STEPS}-week forecast for {region}:")
        print(forecast.round(0).to_string())

        tag = f"{region}_{FORECAST_STEPS}_weeks_arima_p{p}_d{d_}_q{q}"

        forecast_csv = os.path.join(output_dir, f"{tag}_forecast.csv")
        forecast.to_csv(forecast_csv, index_label="date")
        print(f"Saved forecast: {forecast_csv}")

        with open(os.path.join(output_dir, f"{tag}_model_summary.txt"), "w") as fp:
            fp.write(str(fit.summary()))

        # 6. Visualise
        out_png = os.path.join(output_dir, f"{tag}_forecast.png")
        plot_results(s, bt, forecast, region, order, out_png)

        summary_rows.append({
            "region": region,
            "order": f"({p},{d_},{q})",
            "aic": round(fit.aic, 1),
            "arima_smape": round(metrics["arima_smape"], 1),
            "naive_smape": round(metrics["naive_smape"], 1),
        })

    # Per-region comparison table across the shortlist
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"{region}_{FORECAST_STEPS}_weeks_arima_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n{'='*60}\n{region} — shortlist comparison ({FORECAST_STEPS}-week backtest):")
    print(summary_df.to_string(index=False))
    print(f"Saved comparison: {summary_csv}")

    return summary_df


def main():
    data_folder = r"/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new4"
    results_root = r"/Users/tarushshankar/COVID-19-1/data/ARIMA_Results"
    os.makedirs(results_root, exist_ok=True)

    all_summaries = []
    for region in REGIONS:
        print(f"\n{'#'*70}\n# {region}\n{'#'*70}")
        df = run_region(region, data_folder, results_root)
        if df is not None:
            all_summaries.append(df)

    # Combined comparison across all regions
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_csv = os.path.join(results_root, f"ALL_REGIONS_{FORECAST_STEPS}_weeks_arima_comparison.csv")
        combined.to_csv(combined_csv, index=False)
        print(f"\n{'#'*70}\nAll-regions comparison ({FORECAST_STEPS}-week backtest):")
        print(combined.to_string(index=False))
        print(f"\nSaved combined comparison: {combined_csv}")


if __name__ == "__main__":
    main()
