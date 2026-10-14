"""
Phase 5: Forecasting models and out-of-sample evaluation
Inputs : data/processed/factors_all.csv, targets.csv, target_lags.csv,
         X_all_scaled_trainfit.csv, preprocess_config.json, factor_diagnostics.json
Outputs: data/processed/forecasts_h{1,3,6,12}.csv    (actual + every model's forecast)
         data/processed/results_metrics.csv, results_dm_tests.csv, results_crisis.csv
         data/processed/latest_forecast.csv          (live forecast from the newest data)
         reports/figures/15_*.png ... 19_*.png

Design (no look-ahead):
  * Rolling-origin (expanding window) evaluation over the TEST period only.
  * At forecast date t, a model may train only on rows whose target was already
    realised: rows s <= t - h.
  * Models are refitted every REFIT_EVERY test months (speed vs realism trade-off).
  * Target = cumulative % change in the commodity index over the next h months.

Run:  python 05_forecasting.py      (a few minutes; reduce REFIT_EVERY cost by raising it)
"""

# %% 0. Setup
import json
import os
import warnings
from pathlib import Path

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:                                   # Jupyter / interactive
    os.chdir(r"E:\Shlok.py\CommodityInflationPrediction")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROC, FIG = Path("data/processed"), Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

REFIT_EVERY = 6          # refit models every 6 test months
FAVAR_LAGS = 2
RANDOM_STATE = 42
EVENTS = [("2008-09-01", "2009-06-01", "GFC"),
          ("2020-03-01", "2020-06-01", "COVID"),
          ("2022-02-01", "2022-12-01", "Energy shock")]

try:
    from xgboost import XGBRegressor
    GB_NAME = "FAR-XGBoost"
    GB = lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0)
except Exception:
    GB_NAME = "FAR-GradBoost"
    GB = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           subsample=0.8, random_state=RANDOM_STATE)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / name}")


# %% 1. Load data
print("=" * 60)
print("1. LOAD")
print("=" * 60)
cfg = json.load(open(PROC / "preprocess_config.json"))
diag = json.load(open(PROC / "factor_diagnostics.json"))
HORIZONS = cfg["horizons"]
TEST_START = pd.Timestamp(cfg["test_period"][0])

F = pd.read_csv(PROC / "factors_all.csv", index_col=0, parse_dates=True)
targets = pd.read_csv(PROC / "targets.csv", index_col=0, parse_dates=True)
lagf = pd.read_csv(PROC / "target_lags.csv", index_col=0, parse_dates=True)
Xall = pd.read_csv(PROC / "X_all_scaled_trainfit.csv", index_col=0, parse_dates=True)

F_lags = pd.concat([F.shift(1).add_suffix("_l1"), F.shift(2).add_suffix("_l2")], axis=1)
feat = {
    "AR": lagf.reindex(F.index),
    "F": F,
    "F+AR": pd.concat([F, lagf.reindex(F.index)], axis=1),
    "F+lags+AR": pd.concat([F, F_lags, lagf.reindex(F.index)], axis=1),
    "ALL+AR": pd.concat([Xall.reindex(F.index).add_prefix("x_"), lagf.reindex(F.index)], axis=1),
}
idx = pd.concat(feat.values(), axis=1).dropna().index
feat = {k: v.loc[idx] for k, v in feat.items()}
print(f"Usable sample: {idx.min().date()} -> {idx.max().date()} ({len(idx)} months)")
print(f"Test starts  : {TEST_START.date()}   Horizons: {HORIZONS}   Factors: {F.shape[1]}")

mom = targets["y_mom"].dropna().asfreq("MS")
fvar = pd.concat([F, targets["y_mom"].rename("mom")], axis=1).loc[idx].dropna()

# %% 2. Model definitions
RIDGE = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30),
                                                        cv=TimeSeriesSplit(4)))
SK_MODELS = {      # name: (feature set, factory)
    "Naive (zero)":          ("AR", lambda: DummyRegressor(strategy="constant", constant=0.0)),
    "Hist. mean":            ("AR", lambda: DummyRegressor(strategy="mean")),
    "AR (OLS)":              ("AR", lambda: make_pipeline(StandardScaler(), LinearRegression())),
    "FAR-OLS (factors)":     ("F", lambda: make_pipeline(StandardScaler(), LinearRegression())),
    "FAR-OLS (factors+AR)":  ("F+AR", lambda: make_pipeline(StandardScaler(), LinearRegression())),
    "FAR-Ridge":             ("F+lags+AR", RIDGE),
    "FAR-Lasso":             ("F+lags+AR", lambda: make_pipeline(
                                 StandardScaler(), LassoCV(cv=TimeSeriesSplit(4), max_iter=20000))),
    "FAR-RandomForest":      ("F+AR", lambda: RandomForestRegressor(
                                 n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                 random_state=RANDOM_STATE, n_jobs=-1)),
    GB_NAME:                 ("F+AR", GB),
    "Ridge (all indicators)": ("ALL+AR", RIDGE),     # tests whether factor compression helps
}
SPECIAL = ["ARIMA(1,0,1)", "FAVAR"]
ALL_MODELS = list(SK_MODELS) + SPECIAL


# %% 3. Rolling-origin forecasting loop
def arima_fit(endog):
    return ARIMA(endog, order=(1, 0, 1), trend="c").fit()


print("\n2. ROLLING-ORIGIN FORECASTS (this takes a while)")
forecasts = {}
for h in HORIZONS:
    y = targets[f"y_h{h}"].reindex(idx)
    test_pos = [i for i, d in enumerate(idx) if d >= TEST_START and pd.notna(y.iloc[i])]
    preds = {m: [] for m in ALL_MODELS}
    fitted, arima_res, var_res = {}, None, None

    for j, pos in enumerate(test_pos):
        refit = (j % REFIT_EVERY == 0)
        t = idx[pos]
        tr = np.arange(0, pos - h + 1)                       # only rows with realised targets
        tr = tr[y.iloc[tr].notna().values]

        for name, (fs, factory) in SK_MODELS.items():
            if refit or name not in fitted:
                fitted[name] = factory().fit(feat[fs].iloc[tr], y.iloc[tr])
            preds[name].append(float(fitted[name].predict(feat[fs].iloc[[pos]])[0]))

        # ARIMA on monthly changes -> sum of h-step forecasts
        try:
            endog = mom.loc[:t]
            if refit or arima_res is None:
                arima_res = arima_fit(endog)
            res_t = arima_res.apply(endog, refit=False)
            preds["ARIMA(1,0,1)"].append(float(res_t.forecast(h).sum()))
        except Exception:
            preds["ARIMA(1,0,1)"].append(float(h * mom.loc[:t].mean()))

        # FAVAR: VAR on [factors, monthly change] -> sum of h-step forecasts of monthly change
        try:
            data_t = fvar.loc[:t].values
            if refit or var_res is None:
                var_res = VAR(data_t).fit(FAVAR_LAGS)
            fc = var_res.forecast(data_t[-FAVAR_LAGS:], steps=h)[:, -1].sum()
            preds["FAVAR"].append(float(fc))
        except Exception:
            preds["FAVAR"].append(float(h * mom.loc[:t].mean()))

    fc_df = pd.DataFrame(preds, index=idx[test_pos])
    fc_df.insert(0, "actual", y.iloc[test_pos].values)
    fc_df.to_csv(PROC / f"forecasts_h{h}.csv")
    forecasts[h] = fc_df
    print(f"  h={h:>2}m: {len(fc_df)} out-of-sample forecasts "
          f"({fc_df.index.min().date()} -> {fc_df.index.max().date()})")


# %% 4. Metrics
def dm_test(e_model, e_bench, h):
    """Diebold-Mariano (squared loss) with Harvey-Leybourne-Newbold correction.
    Negative statistic = model more accurate than benchmark."""
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    n = len(d)
    s = np.var(d)
    for k in range(1, h):
        s += 2 * np.cov(d[k:], d[:-k], bias=True)[0, 1]
    var = s / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


print("\n3. METRICS")
rows, dm_rows = [], []
for h, fc in forecasts.items():
    a = fc["actual"].values
    sse_mean = ((a - fc["Hist. mean"].values) ** 2).sum()
    rmse_ar = np.sqrt(np.mean((a - fc["AR (OLS)"].values) ** 2))
    for m in ALL_MODELS:
        e = a - fc[m].values
        rmse = np.sqrt(np.mean(e ** 2))
        dacc = np.nan if m == "Naive (zero)" else np.mean(np.sign(fc[m].values) == np.sign(a))
        rows.append({"horizon": h, "model": m, "RMSE": rmse, "MAE": np.mean(np.abs(e)),
                     "rel_RMSE_vs_AR": rmse / rmse_ar,
                     "R2_oos_vs_mean": 1 - (e ** 2).sum() / sse_mean,
                     "dir_accuracy": dacc, "n": len(a)})
        if m != "AR (OLS)":
            stat, p = dm_test(e, a - fc["AR (OLS)"].values, h)
            dm_rows.append({"horizon": h, "model": m, "DM_stat_vs_AR": stat, "p_value": p,
                            "significantly_better": bool(stat < 0 and p < 0.10)})
metrics = pd.DataFrame(rows)
dm_tbl = pd.DataFrame(dm_rows)
metrics.to_csv(PROC / "results_metrics.csv", index=False)
dm_tbl.to_csv(PROC / "results_dm_tests.csv", index=False)

rmse_tbl = metrics.pivot(index="model", columns="horizon", values="RMSE").loc[ALL_MODELS]
rel_tbl = metrics.pivot(index="model", columns="horizon", values="rel_RMSE_vs_AR").loc[ALL_MODELS]
r2_tbl = metrics.pivot(index="model", columns="horizon", values="R2_oos_vs_mean").loc[ALL_MODELS]
print("\nRMSE (lower is better):")
print(rmse_tbl.round(2).to_string())
print("\nRelative RMSE vs AR benchmark (<1 = beats AR):")
print(rel_tbl.round(3).to_string())
print("\nOut-of-sample R2 vs historical mean (>0 = beats the mean):")
print(r2_tbl.round(3).to_string())
sig = dm_tbl[dm_tbl["significantly_better"]]
print("\nModels significantly better than AR (Diebold-Mariano, 10% level):")
print(sig[["horizon", "model", "DM_stat_vs_AR", "p_value"]].round(3).to_string(index=False)
      if len(sig) else "  none")

# best factor-based model per horizon
factor_models = [m for m in ALL_MODELS if "FAR" in m]
best = {h: rmse_tbl.loc[factor_models, h].idxmin() for h in HORIZONS}
print("\nBest factor-based model per horizon:", best)

# %% 5. Plots
print("\n4. PLOTS")
# 15: actual vs forecast
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
for ax, h in zip(axes.ravel(), HORIZONS):
    fc = forecasts[h]
    ax.plot(fc.index, fc["actual"], "k", linewidth=1.5, label="Actual")
    ax.plot(fc.index, fc["AR (OLS)"], "--", color="gray", label="AR benchmark")
    ax.plot(fc.index, fc[best[h]], color="crimson", label=best[h])
    ax.set_title(f"{h}-month-ahead commodity inflation (%)")
    ax.legend(fontsize=8)
save(fig, "15_forecast_vs_actual.png")

# 16: RMSE grouped bars
fig, ax = plt.subplots(figsize=(13, 5))
rmse_tbl.T.plot(kind="bar", ax=ax, width=0.85, colormap="tab20")
ax.set_ylabel("RMSE (percentage points)"); ax.set_xlabel("Forecast horizon (months)")
ax.set_title("Out-of-sample RMSE by model and horizon")
ax.legend(fontsize=7, ncol=3, loc="upper left")
save(fig, "16_rmse_by_horizon.png")

# 17: relative RMSE heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(rel_tbl, annot=True, fmt=".2f", cmap="RdYlGn_r", center=1, ax=ax,
            cbar_kws={"label": "RMSE / RMSE(AR)"})
ax.set_title("Relative RMSE vs AR benchmark (green < 1 is better)")
save(fig, "17_relative_rmse_heatmap.png")

# 19: cumulative squared forecast error difference
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for ax, h in zip(axes.ravel(), HORIZONS):
    fc = forecasts[h]
    diff = ((fc["actual"] - fc["AR (OLS)"]) ** 2 - (fc["actual"] - fc[best[h]]) ** 2).cumsum()
    ax.plot(diff.index, diff, color="navy")
    ax.axhline(0, color="k", linewidth=0.8)
    for (s, e, _), c in zip(EVENTS, ["#d62728", "#ff7f0e", "#9467bd"]):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.15)
    ax.set_title(f"h={h}: cumulative gain of {best[h]} over AR\n(rising = factor model better)",
                 fontsize=9)
save(fig, "19_cumulative_gain_vs_AR.png")

# %% 6. Crisis vs calm robustness
print("\n5. ROBUSTNESS: CRISIS vs CALM PERIODS")
rob = []
for h, fc in forecasts.items():
    crisis = pd.Series(False, index=fc.index)
    for s, e, _ in EVENTS:
        crisis |= (fc.index >= s) & (fc.index <= e)
    for label, mask in [("crisis", crisis), ("calm", ~crisis)]:
        if mask.sum() < 5:
            continue
        a = fc.loc[mask, "actual"].values
        rm_ar = np.sqrt(np.mean((a - fc.loc[mask, "AR (OLS)"].values) ** 2))
        for m in factor_models + ["FAVAR"]:
            rm = np.sqrt(np.mean((a - fc.loc[mask, m].values) ** 2))
            rob.append({"horizon": h, "period": label, "model": m, "n": int(mask.sum()),
                        "rel_RMSE_vs_AR": rm / rm_ar})
rob = pd.DataFrame(rob)
rob.to_csv(PROC / "results_crisis.csv", index=False)
if len(rob):
    print(rob.pivot_table(index=["model", "horizon"], columns="period",
                          values="rel_RMSE_vs_AR").round(3).to_string())

# %% 7. FAVAR impulse responses (training window)
print("\n6. FAVAR IMPULSE RESPONSES")
train_end = pd.Timestamp(cfg["train_period"][1])
dtr = fvar.loc[:train_end]
vres = VAR(dtr.values).fit(FAVAR_LAGS)
irf = vres.irf(12)
cum = irf.orth_irfs.cumsum(axis=0)                 # [step, response, impulse]
mom_i = dtr.shape[1] - 1
labels = diag.get("factor_labels", {})
fig, ax = plt.subplots(figsize=(10, 5))
for j, c in enumerate(F.columns):
    ax.plot(range(13), cum[:, mom_i, j], marker="o", label=labels.get(c, c))
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("Months after a 1-s.d. factor shock")
ax.set_ylabel("Cumulative change in commodity prices (%)")
ax.set_title("FAVAR: cumulative response of commodity inflation to factor shocks\n"
             "(Cholesky ordering: factors first, commodity inflation last)")
ax.legend(fontsize=8)
save(fig, "18_favar_irf.png")

# %% 8. Live forecast from the newest data
print("\n7. LIVE FORECAST (trained on all realised data)")
live = []
last_t = idx[-1]
for h in HORIZONS:
    y = targets[f"y_h{h}"].reindex(idx)
    tr = np.where(y.notna().values)[0]
    for name in ["Hist. mean", "AR (OLS)", "FAR-Ridge", best[h]]:
        if name in SK_MODELS:
            fs, factory = SK_MODELS[name]
            mdl = factory().fit(feat[fs].iloc[tr], y.iloc[tr])
            live.append({"horizon_months": h, "model": name,
                         "base_month": last_t.date(),
                         "forecast_pct_change": float(mdl.predict(feat[fs].iloc[[-1]])[0])})
live = pd.DataFrame(live).drop_duplicates(["horizon_months", "model"])
live.to_csv(PROC / "latest_forecast.csv", index=False)
print(live.round(2).to_string(index=False))
print("\nPhase 5 complete. Next: Phase 6/7 (evaluation robustness, SHAP, interpretation).")
