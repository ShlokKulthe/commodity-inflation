"""
Phase 6: Evaluation and robustness
Inputs : data/processed/X_stationary.csv, targets.csv, target_lags.csv,
         factor_diagnostics.json, preprocess_config.json,
         forecasts_h{h}.csv (Phase 5, optional but recommended)
Outputs: data/processed/cv_forecasts_h{h}.csv, results_eval_all.csv, results_K_sensitivity.csv
         reports/figures/20_*.png ... 23_*.png

What this adds on top of Phase 5
  1. LONG-SAMPLE rolling-origin CV in which the scaler AND the factor model are re-estimated
     at every origin (strictly leakage-free; also covers the 2008-09 crisis).
  2. Forecast combination (AR + factor model), which often beats both.
  3. Clark-West test (correct test for nested models) and Diebold-Mariano test.
  4. Block-bootstrap confidence intervals for relative RMSE.
  5. Rolling relative RMSE (does the factor edge persist over time?).
  6. Sensitivity to the number of factors K.

Run:  python 06_evaluation_robustness.py
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
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROC, FIG = Path("data/processed"), Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

MIN_TRAIN = 96            # months of data before the first origin (8 years)
STEP = 12                 # re-estimate everything once a year
RS = 42
N_BOOT = 1000
EVENTS = [("2008-09-01", "2009-06-01", "GFC"),
          ("2020-03-01", "2020-06-01", "COVID"),
          ("2022-02-01", "2022-12-01", "Energy shock")]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / name}")


def shade(ax):
    for (s, e, _), c in zip(EVENTS, ["#d62728", "#ff7f0e", "#9467bd"]):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.15)


class Prep(BaseEstimator, TransformerMixin):
    """Winsorize (1%, 99%) then z-score, using statistics of the fitting sample only."""
    def __init__(self, ql=0.01, qh=0.99):
        self.ql, self.qh = ql, qh

    def fit(self, X, y=None):
        self.cols_ = X.columns
        self.lo_, self.hi_ = X.quantile(self.ql), X.quantile(self.qh)
        self.sc_ = StandardScaler().fit(X.clip(self.lo_, self.hi_, axis=1))
        return self

    def transform(self, X):
        Z = self.sc_.transform(X[self.cols_].clip(self.lo_, self.hi_, axis=1))
        return pd.DataFrame(Z, index=X.index, columns=self.cols_)


RIDGE = lambda: make_pipeline(StandardScaler(),
                              RidgeCV(alphas=np.logspace(-2, 3, 30), cv=TimeSeriesSplit(4)))
OLS = lambda: make_pipeline(StandardScaler(), LinearRegression())

# %% 1. Load
print("=" * 60)
print("1. LOAD")
print("=" * 60)
cfg = json.load(open(PROC / "preprocess_config.json"))
diag = json.load(open(PROC / "factor_diagnostics.json"))
HORIZONS, K0 = cfg["horizons"], int(diag["n_factors"])

X = pd.read_csv(PROC / "X_stationary.csv", index_col=0, parse_dates=True)
targets = pd.read_csv(PROC / "targets.csv", index_col=0, parse_dates=True)
lagf = pd.read_csv(PROC / "target_lags.csv", index_col=0, parse_dates=True)
idx = X.index.intersection(lagf.dropna().index)
X, lagf = X.loc[idx], lagf.loc[idx]
n = len(idx)
print(f"Sample: {idx.min().date()} -> {idx.max().date()} ({n} months), "
      f"{X.shape[1]} indicators, K = {K0}")
print(f"First origin after {MIN_TRAIN} months -> {idx[MIN_TRAIN].date()}")


# %% 2. Leakage-free rolling-origin CV (factors re-estimated at every origin)
def cv_forecasts(K, light=False):
    """light=True computes only AR and FAR-Ridge (used for the K sensitivity)."""
    recs = {h: [] for h in HORIZONS}
    for o in range(MIN_TRAIN, n, STEP):
        prep = Prep().fit(X.iloc[:o + 1])
        Xs = prep.transform(X)
        fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=RS,
                            max_iter=2000).fit(Xs.iloc[:o + 1])
        F = pd.DataFrame(fa.transform(Xs), index=idx, columns=[f"F{i + 1}" for i in range(K)])
        feats = {"AR": lagf, "F+AR": pd.concat([F, lagf], axis=1)}
        if not light:
            feats["ALL+AR"] = pd.concat([Xs.add_prefix("x_"), lagf], axis=1)

        for h in HORIZONS:
            y = targets[f"y_h{h}"].reindex(idx)
            tr = np.arange(0, o - h + 1)
            tr = tr[y.iloc[tr].notna().values]
            fpos = [p for p in range(o, min(o + STEP, n)) if pd.notna(y.iloc[p])]
            if len(tr) < 40 or not fpos:
                continue
            ytr = y.iloc[tr]
            m_ar = OLS().fit(feats["AR"].iloc[tr], ytr)
            m_far = RIDGE().fit(feats["F+AR"].iloc[tr], ytr)
            p_ar = m_ar.predict(feats["AR"].iloc[fpos])
            p_far = m_far.predict(feats["F+AR"].iloc[fpos])
            cols = {"actual": y.iloc[fpos].values, "AR (OLS)": p_ar, "FAR-Ridge": p_far}
            if not light:
                cols["Hist. mean"] = np.repeat(ytr.mean(), len(fpos))
                m_all = RIDGE().fit(feats["ALL+AR"].iloc[tr], ytr)
                cols["Ridge (all indicators)"] = m_all.predict(feats["ALL+AR"].iloc[fpos])
            cols["Combo (AR+FAR)"] = 0.5 * (p_ar + p_far)
            recs[h].append(pd.DataFrame(cols, index=idx[fpos]))
    return {h: pd.concat(v) for h, v in recs.items() if v}


print("\n2. LEAKAGE-FREE ROLLING-ORIGIN CV (scaler + factors refit each year)")
cv = cv_forecasts(K0)
for h, df in cv.items():
    df.to_csv(PROC / f"cv_forecasts_h{h}.csv")
    print(f"  h={h:>2}m: {len(df)} forecasts ({df.index.min().date()} -> {df.index.max().date()})")


# %% 3. Statistical tests
def dm_test(e_m, e_b, h):
    d = e_m ** 2 - e_b ** 2
    m = len(d)
    s = np.var(d)
    for k in range(1, h):
        s += 2 * np.cov(d[k:], d[:-k], bias=True)[0, 1]
    v = s / m
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v) * np.sqrt((m + 1 - 2 * h + h * (h - 1) / m) / m)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=m - 1))


def clark_west(y, f_b, f_m, h):
    """One-sided test that the larger (nesting) model has lower MSE than the benchmark."""
    adj = (y - f_b) ** 2 - ((y - f_m) ** 2 - (f_b - f_m) ** 2)
    res = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
    t = float(res.tvalues[0])
    return t, 1 - stats.norm.cdf(t)


def block_boot_ci(e_m, e_b, L, B=N_BOOT, seed=RS):
    rng = np.random.default_rng(seed)
    m, L = len(e_m), min(L, len(e_m))
    nb = int(np.ceil(m / L))
    out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, m - L + 1, nb)
        ii = (st[:, None] + np.arange(L)).ravel()[:m]
        out[b] = np.sqrt(np.mean(e_m[ii] ** 2)) / np.sqrt(np.mean(e_b[ii] ** 2))
    return np.percentile(out, [2.5, 97.5])


def evaluate(fcs, sample):
    rows = []
    for h, fc in fcs.items():
        fc = fc.copy()
        if "Combo (AR+FAR)" not in fc and "FAR-Ridge" in fc:
            fc["Combo (AR+FAR)"] = 0.5 * (fc["AR (OLS)"] + fc["FAR-Ridge"])
        y, ar = fc["actual"].values, fc["AR (OLS)"].values
        e_b = y - ar
        for m in [c for c in fc.columns if c not in ("actual", "AR (OLS)", "Naive (zero)")]:
            e_m = y - fc[m].values
            cw_t, cw_p = clark_west(y, ar, fc[m].values, h)
            dm_s, dm_p = dm_test(e_m, e_b, h)
            lo, hi = block_boot_ci(e_m, e_b, L=max(h, 3))
            rows.append({"sample": sample, "horizon": h, "model": m, "n": len(y),
                         "RMSE": np.sqrt(np.mean(e_m ** 2)),
                         "rel_RMSE_vs_AR": np.sqrt(np.mean(e_m ** 2)) / np.sqrt(np.mean(e_b ** 2)),
                         "CI95_low": lo, "CI95_high": hi,
                         "ClarkWest_t": cw_t, "ClarkWest_p": cw_p,
                         "DM_stat": dm_s, "DM_p": dm_p})
    return pd.DataFrame(rows)


print("\n3. TESTS (Clark-West, Diebold-Mariano, block-bootstrap CIs)")
ev = [evaluate(cv, "CV 2009+ (refit factors)")]
p5 = {}
for h in HORIZONS:
    f = PROC / f"forecasts_h{h}.csv"
    if f.exists():
        p5[h] = pd.read_csv(f, index_col=0, parse_dates=True)
if p5:
    ev.append(evaluate(p5, "Phase 5 test set"))
ev = pd.concat(ev, ignore_index=True)
ev.to_csv(PROC / "results_eval_all.csv", index=False)

show = ev[ev["model"].isin(["FAR-Ridge", "Combo (AR+FAR)", "Ridge (all indicators)"])]
for smp, g in show.groupby("sample"):
    print(f"\n--- {smp} ---")
    t = g.pivot_table(index="model", columns="horizon", values="rel_RMSE_vs_AR")
    print("Relative RMSE vs AR (<1 = better):")
    print(t.round(3).to_string())
    cw = g.pivot_table(index="model", columns="horizon", values="ClarkWest_p")
    print("Clark-West one-sided p-values (<0.10 = significantly better than AR):")
    print(cw.round(3).to_string())
    ci = g[g["model"] == "FAR-Ridge"][["horizon", "rel_RMSE_vs_AR", "CI95_low", "CI95_high"]]
    print("FAR-Ridge relative RMSE with 95% block-bootstrap CI (CI excluding 1 = robust):")
    print(ci.round(3).to_string(index=False))

# %% 4. Plots
print("\n4. PLOTS")
# 20: long-sample forecasts
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, h in zip(axes.ravel(), HORIZONS):
    if h not in cv:
        continue
    d = cv[h]
    ax.plot(d.index, d["actual"], "k", linewidth=1.4, label="Actual")
    ax.plot(d.index, d["AR (OLS)"], "--", color="gray", label="AR")
    ax.plot(d.index, d["FAR-Ridge"], color="crimson", label="FAR-Ridge")
    shade(ax)
    ax.set_title(f"{h}-month-ahead (leakage-free CV)")
    ax.legend(fontsize=8)
save(fig, "20_cv_forecasts.png")

# 21: rolling relative RMSE
fig, ax = plt.subplots(figsize=(13, 5))
for h in HORIZONS:
    if h not in cv:
        continue
    d = cv[h]
    r = np.sqrt(((d["actual"] - d["FAR-Ridge"]) ** 2).rolling(36).mean()) / \
        np.sqrt(((d["actual"] - d["AR (OLS)"]) ** 2).rolling(36).mean())
    ax.plot(d.index, r, label=f"h={h}m")
ax.axhline(1, color="k", linewidth=1)
shade(ax)
ax.set_ylabel("RMSE(FAR-Ridge) / RMSE(AR)")
ax.set_title("36-month rolling relative RMSE (below 1 = factor model better)")
ax.legend(ncol=4)
save(fig, "21_rolling_relative_rmse.png")

# 23: bootstrap CI chart
fig, ax = plt.subplots(figsize=(10, 5))
sub = ev[ev["model"].isin(["FAR-Ridge", "Combo (AR+FAR)"])].copy()
combos = list(sub.groupby(["sample", "model"]).groups.keys())
for k, (smp, mdl) in enumerate(combos):
    g = sub[(sub["sample"] == smp) & (sub["model"] == mdl)].sort_values("horizon")
    xs = np.arange(len(g)) + (k - len(combos) / 2) * 0.12
    ax.errorbar(xs, g["rel_RMSE_vs_AR"],
                yerr=[g["rel_RMSE_vs_AR"] - g["CI95_low"], g["CI95_high"] - g["rel_RMSE_vs_AR"]],
                fmt="o", capsize=3, label=f"{mdl} | {smp}")
ax.set_xticks(np.arange(len(HORIZONS)))
ax.set_xticklabels([f"h={h}m" for h in HORIZONS])
ax.axhline(1, color="k", linewidth=1)
ax.set_ylabel("Relative RMSE vs AR (95% block-bootstrap CI)")
ax.set_title("Is the improvement over AR statistically distinguishable from zero?")
ax.legend(fontsize=7)
save(fig, "23_bootstrap_ci.png")

# %% 5. Sensitivity to the number of factors
print("\n5. SENSITIVITY TO NUMBER OF FACTORS")
K_RANGE = [k for k in range(1, 8) if k < X.shape[1]]
sens = []
for K in K_RANGE:
    out = cv_forecasts(K, light=True)
    for h, d in out.items():
        e_m, e_b = d["actual"] - d["FAR-Ridge"], d["actual"] - d["AR (OLS)"]
        sens.append({"K": K, "horizon": h,
                     "rel_RMSE_vs_AR": np.sqrt((e_m ** 2).mean()) / np.sqrt((e_b ** 2).mean())})
    print(f"  K={K} done")
sens = pd.DataFrame(sens)
sens.to_csv(PROC / "results_K_sensitivity.csv", index=False)
piv = sens.pivot(index="K", columns="horizon", values="rel_RMSE_vs_AR")
print("\nRelative RMSE vs AR by number of factors (<1 = better):")
print(piv.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
for h in piv.columns:
    ax.plot(piv.index, piv[h], marker="o", label=f"h={h}m")
ax.axhline(1, color="k", linewidth=1)
ax.axvline(K0, color="green", linestyle="--", label=f"chosen K={K0}")
ax.set_xlabel("Number of factors K"); ax.set_ylabel("Relative RMSE vs AR")
ax.set_title("Robustness to the number of factors")
ax.legend(ncol=3, fontsize=8)
save(fig, "22_K_sensitivity.png")

# %% 6. Summary
print("\n" + "=" * 60)
print("SUMMARY: factor model (FAR-Ridge) vs AR in the leakage-free CV sample")
print("=" * 60)
s = ev[(ev["sample"].str.startswith("CV")) & (ev["model"] == "FAR-Ridge")]
for _, r in s.iterrows():
    verdict = ("significantly better" if r["ClarkWest_p"] < 0.10 and r["CI95_high"] < 1.0 else
               "better, but not robust" if r["rel_RMSE_vs_AR"] < 1 else "no improvement")
    print(f"  h={int(r['horizon']):>2}m: rel RMSE {r['rel_RMSE_vs_AR']:.3f} "
          f"[{r['CI95_low']:.3f}, {r['CI95_high']:.3f}]  CW p={r['ClarkWest_p']:.3f}  -> {verdict}")
print("\nPhase 6 complete. Next: Phase 7 (interpretation: permutation importance, SHAP, write-up).")