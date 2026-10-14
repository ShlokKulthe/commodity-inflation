"""
Phase 7: Interpretation and reporting
Inputs : factors_all.csv, factor_loadings.csv, factor_variance.csv, factor_diagnostics.json,
         targets.csv, target_lags.csv, preprocess_config.json     (Phases 3-4)
         results_eval_all.csv, results_crisis.csv, latest_forecast.csv   (Phases 5-6, optional)
Outputs: data/processed/perm_importance.csv, shap_importance.csv, far_regression_hac.csv,
         incremental_r2.csv, fevd_inflation.csv, current_forecast_contributions.csv
         reports/figures/24_*.png ... 29_*.png
         reports/results_summary.md      <- auto-filled draft of your results section

Questions answered
  Q1. Which factors matter for forecasting, and at which horizon?   (permutation + SHAP)
  Q2. Are the factor effects statistically significant?             (HAC regressions)
  Q3. How much of commodity-inflation variance does each factor explain?  (FEVD)
  Q4. What is driving the CURRENT forecast?                         (contributions)
  Q5. Draft write-up with the hypotheses H1-H4 checked automatically.

Run:  python 07_interpretation.py
"""

# %% 0. Setup
import json
import os
import warnings
from pathlib import Path

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    os.chdir(r"E:\Shlok.py\CommodityInflationPrediction")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROC, FIG, REP = Path("data/processed"), Path("reports/figures"), Path("reports")
FIG.mkdir(parents=True, exist_ok=True)
RS = 42
N_REP = 100               # permutation repeats
SHAP_H = 6                # horizon used for the SHAP beeswarm plot
FAVAR_LAGS = 2

try:
    from xgboost import XGBRegressor
    GB = lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=RS, verbosity=0)
except Exception:
    GB = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                           subsample=0.8, random_state=RS)
RIDGE = lambda: make_pipeline(StandardScaler(),
                              RidgeCV(alphas=np.logspace(-2, 3, 30), cv=TimeSeriesSplit(4)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / name}")


def rd(name, **kw):
    p = PROC / name
    return pd.read_csv(p, **kw) if p.exists() else None


# %% 1. Load
print("=" * 60)
print("1. LOAD")
print("=" * 60)
cfg = json.load(open(PROC / "preprocess_config.json"))
diag = json.load(open(PROC / "factor_diagnostics.json"))
HORIZONS = cfg["horizons"]
TRAIN_END = pd.Timestamp(cfg["train_period"][1])
TEST_START = pd.Timestamp(cfg["test_period"][0])

F = pd.read_csv(PROC / "factors_all.csv", index_col=0, parse_dates=True)
targets = pd.read_csv(PROC / "targets.csv", index_col=0, parse_dates=True)
lagf = pd.read_csv(PROC / "target_lags.csv", index_col=0, parse_dates=True)
idx = F.index.intersection(lagf.dropna().index)
F, lagf = F.loc[idx], lagf.loc[idx]
labels = diag["factor_labels"]                       # {"F1": "F1: Energy", ...}
FCOLS, LAGCOLS = list(F.columns), list(lagf.columns)
Xfa = pd.concat([F, lagf], axis=1)
print(f"Sample {idx.min().date()} -> {idx.max().date()} | factors: {[labels[c] for c in FCOLS]}")


def split(h):
    y = targets[f"y_h{h}"].reindex(idx)
    tr = (idx <= TRAIN_END) & y.notna().values
    te = (idx >= TEST_START) & y.notna().values
    return y, tr, te


rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# %% 2. Permutation importance (out-of-sample, test period)
print("\n2. PERMUTATION IMPORTANCE (fit on train, evaluated on test)")
groups = {labels[c]: [c] for c in FCOLS}
groups["AR lags"] = LAGCOLS
rng = np.random.default_rng(RS)
perm_rows = []
fitted_gb = {}
for h in HORIZONS:
    y, tr, te = split(h)
    if te.sum() < 10:
        print(f"  h={h}: too few test rows, skipped")
        continue
    for mname, factory in [("Ridge", RIDGE), ("Boosting", GB)]:
        model = factory().fit(Xfa[tr], y[tr])
        if mname == "Boosting":
            fitted_gb[h] = model
        Xte, yte = Xfa[te], y[te]
        base = rmse(yte, model.predict(Xte))
        for g, cols in groups.items():
            incs = []
            for _ in range(N_REP):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].values[rng.permutation(len(Xte))]
                incs.append(rmse(yte, model.predict(Xp)) - base)
            perm_rows.append({"model": mname, "horizon": h, "feature": g,
                              "delta_rmse": float(np.mean(incs)), "base_rmse": base})
perm = pd.DataFrame(perm_rows)
perm.to_csv(PROC / "perm_importance.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4 + 0.3 * len(groups)))
for ax, mname in zip(axes, ["Ridge", "Boosting"]):
    t = perm[perm["model"] == mname].pivot(index="feature", columns="horizon", values="delta_rmse")
    t = t.loc[[g for g in groups if g in t.index]]
    sns.heatmap(t, annot=True, fmt=".3f", cmap="Reds", ax=ax, cbar_kws={"label": "RMSE increase"})
    ax.set_title(f"Permutation importance - {mname}\n(RMSE increase when feature is shuffled)")
    ax.set_xlabel("Horizon (months)"); ax.set_ylabel("")
    print(f"\n{mname}:\n{t.round(3).to_string()}")
save(fig, "24_permutation_importance.png")

# %% 3. SHAP (tree model)
print("\n3. SHAP")
shap_rows = []
try:
    import shap
    nice = [labels.get(c, c) for c in Xfa.columns]
    for h, model in fitted_gb.items():
        y, tr, te = split(h)
        sv = shap.TreeExplainer(model).shap_values(Xfa[te].values)
        mean_abs = np.abs(sv).mean(axis=0)
        for name, v in zip(nice, mean_abs):
            shap_rows.append({"horizon": h, "feature": name, "mean_abs_shap": float(v)})
        if h == SHAP_H or (SHAP_H not in fitted_gb and h == max(fitted_gb)):
            plt.figure(figsize=(9, 6))
            shap.summary_plot(sv, Xfa[te].values, feature_names=nice, show=False, max_display=15)
            plt.title(f"SHAP summary - boosting model, {h}-month horizon (test period)")
            save(plt.gcf(), "25_shap_summary.png")
    sh = pd.DataFrame(shap_rows)
    sh.to_csv(PROC / "shap_importance.csv", index=False)
    piv = sh.pivot(index="feature", columns="horizon", values="mean_abs_shap")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index].head(12)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(piv) + 2))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Mean |SHAP| by horizon (top features)")
    save(fig, "26_shap_importance_heatmap.png")
except Exception as e:
    print(f"  SHAP skipped ({type(e).__name__}: {e}). Install with `pip install shap`; "
          "permutation importance above already covers the same question.")

# %% 4. Factor-augmented regressions with HAC standard errors (in-sample explanation)
print("\n4. FACTOR-AUGMENTED REGRESSIONS (Newey-West / HAC errors)")
reg_rows, r2_rows = [], []
for h in HORIZONS:
    y, _, _ = split(h)
    ok = y.notna().values
    yy = y[ok]
    Z = (Xfa[ok] - Xfa[ok].mean()) / Xfa[ok].std()
    cols = FCOLS + ["mom_lag1"]
    fit = sm.OLS(yy, sm.add_constant(Z[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)})
    for c in fit.params.index:
        reg_rows.append({"horizon": h, "variable": labels.get(c, c), "coef": fit.params[c],
                         "t_stat": fit.tvalues[c], "p_value": fit.pvalues[c]})
    ar_fit = sm.OLS(yy, sm.add_constant(Z[LAGCOLS])).fit()
    far_fit = sm.OLS(yy, sm.add_constant(Z[FCOLS + LAGCOLS])).fit()
    r2_rows.append({"horizon": h, "adjR2_AR_only": ar_fit.rsquared_adj,
                    "adjR2_AR_plus_factors": far_fit.rsquared_adj,
                    "gain": far_fit.rsquared_adj - ar_fit.rsquared_adj, "n": int(ok.sum())})
reg = pd.DataFrame(reg_rows)
r2t = pd.DataFrame(r2_rows)
reg.to_csv(PROC / "far_regression_hac.csv", index=False)
r2t.to_csv(PROC / "incremental_r2.csv", index=False)
print("\nIn-sample adjusted R2: AR only vs AR + factors")
print(r2t.round(3).to_string(index=False))

coef = reg[reg["variable"] != "const"].pivot(index="variable", columns="horizon", values="coef")
pv = reg[reg["variable"] != "const"].pivot(index="variable", columns="horizon", values="p_value")
annot = coef.round(2).astype(str) + np.where(pv < 0.05, "*", "")
fig, ax = plt.subplots(figsize=(8, 0.6 * len(coef) + 2))
sns.heatmap(coef, annot=annot, fmt="", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "% change per +1 s.d."})
ax.set_title("Effect of a 1-s.d. rise in each factor on future commodity inflation\n"
             "(* = significant at 5%, HAC errors; in-sample)")
ax.set_xlabel("Horizon (months)"); ax.set_ylabel("")
save(fig, "27_factor_coefficients.png")

# %% 5. Forecast-error variance decomposition (FAVAR)
print("\n5. FEVD OF COMMODITY INFLATION")
fvar = pd.concat([F, targets["y_mom"].rename("mom")], axis=1).loc[idx].dropna()
vres = VAR(fvar.values).fit(FAVAR_LAGS)
dec = vres.fevd(12).decomp                       # [variable, step, shock]
mom_i = fvar.shape[1] - 1
names = [labels.get(c, c) for c in FCOLS] + ["Own shocks (inflation)"]
steps = [1, 3, 6, 12]
fe = pd.DataFrame(dec[mom_i][[s - 1 for s in steps]], index=[f"{s}m" for s in steps],
                  columns=names) * 100
fe.to_csv(PROC / "fevd_inflation.csv")
print("Share of commodity-inflation forecast-error variance (%):")
print(fe.round(1).to_string())
fig, ax = plt.subplots(figsize=(10, 4.5))
fe.plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
ax.set_xlabel("% of forecast-error variance"); ax.set_title("Who explains commodity inflation? (FAVAR FEVD)")
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "28_fevd.png")

# %% 6. What drives the current forecast?
print("\n6. CONTRIBUTIONS TO THE LATEST FORECAST")
contrib_rows = []
for h in HORIZONS:
    y, _, _ = split(h)
    ok = y.notna().values
    model = RIDGE().fit(Xfa[ok], y[ok])
    sc, rg = model[0], model[-1]
    z = sc.transform(Xfa.iloc[[-1]])[0]
    parts = dict(zip(Xfa.columns, rg.coef_ * z))
    row = {"horizon": h, "intercept": float(rg.intercept_)}
    for c in FCOLS:
        row[labels[c]] = parts[c]
    row["AR lags"] = sum(parts[c] for c in LAGCOLS)
    row["forecast"] = float(rg.intercept_ + sum(parts.values()))
    contrib_rows.append(row)
contrib = pd.DataFrame(contrib_rows).set_index("horizon")
contrib.to_csv(PROC / "current_forecast_contributions.csv")
print(f"Base month: {idx[-1].date()}  (values = percentage points of cumulative change)")
print(contrib.round(2).to_string())
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4 * len(HORIZONS), 4), sharey=True)
for ax, h in zip(np.atleast_1d(axes), HORIZONS):
    s = contrib.loc[h].drop(["forecast"])
    ax.barh(s.index, s.values, color=np.where(s.values >= 0, "#c0392b", "#2980b9"))
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title(f"h={h}m  (forecast {contrib.loc[h, 'forecast']:+.1f}%)", fontsize=10)
save(fig, "29_current_forecast_drivers.png")

# %% 7. Auto-drafted results section
print("\n7. DRAFT REPORT")
ev = rd("results_eval_all.csv")
crisis = rd("results_crisis.csv")
fv = rd("factor_variance.csv", index_col=0)


def top_factor(h, model="Ridge"):
    d = perm[(perm["model"] == model) & (perm["horizon"] == h) & (perm["feature"] != "AR lags")]
    return d.sort_values("delta_rmse", ascending=False).iloc[0]["feature"] if len(d) else "n/a"


# hypothesis checks
K, cumv = diag["n_factors"], diag["cum_variance_pct"]
h1 = ("Supported" if (K <= 5 and cumv >= 50) else "Partly supported" if cumv >= 40 else "Not supported")
hs, hl = min(HORIZONS), max(HORIZONS)
ts, tl = top_factor(hs), top_factor(hl)
h2 = (f"Partly supported - dominant factor shifts from {ts} (h={hs}m) to {tl} (h={hl}m)"
      if ts != tl else f"Not supported - {ts} dominates at all horizons")
h3_lines, h3_ok = [], []
if ev is not None:
    cvr = ev[(ev["sample"].str.startswith("CV")) & (ev["model"] == "FAR-Ridge")]
    for _, r in cvr.iterrows():
        good = r["ClarkWest_p"] < 0.10 and r["CI95_high"] < 1
        h3_ok.append(good)
        h3_lines.append(f"h={int(r['horizon'])}m: rel. RMSE {r['rel_RMSE_vs_AR']:.3f} "
                        f"[{r['CI95_low']:.3f}, {r['CI95_high']:.3f}], Clark-West p={r['ClarkWest_p']:.3f}")
    h3 = ("Supported at all horizons" if all(h3_ok) else "Supported at some horizons" if any(h3_ok)
          else "Not supported (no robust gain over AR)")
else:
    h3 = "Not evaluated (run Phase 6)"
h4 = "Not evaluated (run Phase 5)"
if crisis is not None and len(crisis):
    c = crisis[crisis["model"] == "FAR-Ridge"].groupby("period")["rel_RMSE_vs_AR"].mean()
    if {"crisis", "calm"} <= set(c.index):
        diff = c["crisis"] - c["calm"]
        h4 = (f"Supported - relative RMSE is {c['crisis']:.2f} in crises vs {c['calm']:.2f} in calm periods"
              if abs(diff) > 0.05 else
              f"Not clearly supported - relative RMSE {c['crisis']:.2f} (crisis) vs {c['calm']:.2f} (calm)")

lines = []
A = lines.append
A("# Results summary (auto-generated draft - review and edit every statement)\n")
A("## 1. Data and method")
A(f"- Sample: {idx.min().date()} to {idx.max().date()} ({len(idx)} monthly observations); "
  f"{len(Xfa.columns) - len(LAGCOLS)} factors extracted from {cfg.get('indicators') and len(cfg['indicators'])} stationary indicators.")
A(f"- Target: cumulative % change of `{cfg['target']}` over the next {', '.join(map(str, HORIZONS))} months.")
A(f"- Train: {cfg['train_period'][0]} to {cfg['train_period'][1]}; test: {cfg['test_period'][0]} to {cfg['test_period'][1]}; "
  f"{cfg['gap_months']}-month embargo between them.")
A(f"- Suitability: KMO = {diag['kmo_overall']}, Bartlett p = {diag['bartlett_p']:.2g}.")
A(f"- Number of factors: K = {K} (Kaiser {diag['k_kaiser']}, parallel analysis {diag['k_parallel']}, "
  f"Bai-Ng {diag['k_baing']}); cumulative variance explained {cumv}%.\n")
A("## 2. Factor interpretation")
if fv is not None:
    for f_, r in fv.iterrows():
        A(f"- **{labels.get(f_, f_)}** - top loadings: {r.get('top_variables', '')}; "
          f"explains {r.get('pct_variance', float('nan')):.1f}% of total variance.")
A("")
A("## 3. Forecast performance (leakage-free rolling-origin CV, FAR-Ridge vs AR)")
A("\n".join(f"- {l}" for l in h3_lines) if h3_lines else "- Run Phase 6 to populate.")
A("")
A("## 4. What matters, and when")
for h in HORIZONS:
    A(f"- h={h}m: most important factor (permutation, Ridge) = {top_factor(h)}; "
      f"(boosting) = {top_factor(h, 'Boosting')}")
A(f"- FEVD at 12 months: " + ", ".join(f"{n} {v:.0f}%" for n, v in fe.loc['12m'].items()))
A("")
A("## 5. Hypotheses (automatic first pass - verify against figures)")
A(f"- **H1** (3-5 factors explain most co-movement): {h1}")
A(f"- **H2** (energy/demand dominate short horizons, dollar/monetary long): {h2}")
A(f"- **H3** (factor models beat AR benchmarks): {h3}")
A(f"- **H4** (instability in crises): {h4}\n")
A("## 6. Latest forecast")
lf = rd("latest_forecast.csv")
if lf is not None:
    sub = lf[lf["model"] == "FAR-Ridge"]
    for _, r in sub.iterrows():
        A(f"- Over the next {int(r['horizon_months'])} months from {r['base_month']}: "
          f"{r['forecast_pct_change']:+.1f}% (FAR-Ridge)")
A("- Current drivers: see `reports/figures/29_current_forecast_drivers.png`.\n")
A("## 7. Limitations")
A("- Short test window and overlapping multi-month targets: treat small accuracy gains cautiously.")
A("- Factor labels are interpretive; loadings can change across regimes.")
A("- IMF/FRED monthly commodity data are revised and published with delay; real-time performance may differ.")
A("- Macro inputs are US-centric. Emerging-market effects (e.g. India's rupee and subsidies) are not modelled.")
A("- Forecasts are statistical estimates for research use, not investment advice.")
(REP / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("  saved reports/results_summary.md")
print("\nPhase 7 complete. Next: Phase 8 (packaging: README, requirements, optional dashboard).")