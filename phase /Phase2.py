"""
Phase 2: Exploratory Data Analysis + Stationarity Testing
Input : data/processed/monthly_merged.csv   (from Phase 1)
Output: reports/figures/*.png, data/processed/stationarity_report.csv

Run:  python 02_eda.py
(or paste into Jupyter cells, split at the '# %%' markers)
"""

# %% 0. Imports and setup
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

DATA = Path("data/processed/monthly_merged.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "all_commodity_index"       # change to energy_index, food_index, etc.

# Episodes shaded on plots (start, end, label)
EVENTS = [
    ("2008-09-01", "2009-06-01", "GFC"),
    ("2020-03-01", "2020-06-01", "COVID"),
    ("2022-02-01", "2022-12-01", "Energy shock"),
]


def shade_events(ax, label=True):
    colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for (s, e, name), c in zip(EVENTS, colors):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.15,
                   label=name if label else None)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / name}")


# %% 1. Load and overview
df = pd.read_csv(DATA, index_col="date", parse_dates=True)
print("=" * 60)
print("1. OVERVIEW")
print("=" * 60)
print(f"Shape : {df.shape}")
print(f"Period: {df.index.min().date()} -> {df.index.max().date()}")
print("\nDescriptive statistics:")
print(df.describe().T.round(2).to_string())

# %% 2. Missing-value map
print("\n2. MISSING VALUES")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isna().T, cbar=False, cmap="Reds", ax=ax,
            xticklabels=False, yticklabels=True)
ax.set_title("Missing data map (red = missing)")
ax.set_xlabel("Time (1996 -> today)")
save(fig, "01_missing_map.png")

# %% 3. Trend plots by group
print("\n3. TREND PLOTS")
GROUPS = {
    "Commodity indices": ["all_commodity_index", "energy_index", "food_index", "metals_index"],
    "Energy": ["brent_oil", "wti_oil", "nat_gas_eu"],
    "Metals": ["copper", "aluminum"],
    "Agriculture": ["wheat", "maize", "sugar", "cotton"],
    "Macro": ["us_cpi", "us_ppi", "us_ind_production", "us_m2"],
    "Rates and risk": ["fed_funds_rate", "yield_spread_10y2y", "vix", "consumer_sentiment"],
    "Currency": ["usd_index", "usd_inr", "usd_cny", "usd_per_eur"],
}

fig, axes = plt.subplots(len(GROUPS), 1, figsize=(13, 3 * len(GROUPS)), sharex=True)
for ax, (title, cols) in zip(axes, GROUPS.items()):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        ax.set_visible(False)
        continue
    # rebase to 100 at first valid obs so variables with different scales are comparable
    rebased = df[cols].apply(lambda s: s / s.dropna().iloc[0] * 100)
    rebased.plot(ax=ax, linewidth=1.2)
    shade_events(ax, label=(title == "Commodity indices"))
    ax.set_title(title + " (rebased, first obs = 100)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_xlabel("")
save(fig, "02_trends_by_group.png")

# %% 4. Target variable: level and YoY inflation
print("\n4. TARGET VARIABLE")
if TARGET not in df.columns:
    raise SystemExit(f"{TARGET} not in dataset. Available: {list(df.columns)}")

y = df[TARGET].dropna()
yoy = y.pct_change(12) * 100
mom = y.pct_change(1) * 100

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
y.plot(ax=axes[0], color="navy"); axes[0].set_title(f"{TARGET}: level")
mom.plot(ax=axes[1], color="darkgreen"); axes[1].set_title("Month-on-month change (%)")
yoy.plot(ax=axes[2], color="crimson"); axes[2].set_title("Year-on-year inflation (%)")
axes[2].axhline(0, color="k", linewidth=0.8)
for ax in axes:
    shade_events(ax, label=False)
save(fig, "03_target_level_mom_yoy.png")

print(f"YoY inflation  mean={yoy.mean():.2f}%  std={yoy.std():.2f}%  "
      f"min={yoy.min():.1f}% ({yoy.idxmin().date()})  max={yoy.max():.1f}% ({yoy.idxmax().date()})")

# %% 5. Seasonality and trend decomposition (target)
print("\n5. SEASONAL DECOMPOSITION")
dec = seasonal_decompose(np.log(y), model="additive", period=12, extrapolate_trend="freq")
fig = dec.plot()
fig.set_size_inches(12, 8)
fig.suptitle(f"Decomposition of log({TARGET})", y=1.02)
save(fig, "04_decomposition.png")

month_profile = mom.groupby(mom.index.month).mean()
fig, ax = plt.subplots(figsize=(8, 4))
month_profile.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average MoM change by calendar month (%)")
ax.set_xlabel("Month")
save(fig, "05_monthly_seasonality.png")

# %% 6. Distribution of target changes
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(mom.dropna(), kde=True, ax=axes[0], color="teal")
axes[0].set_title(f"MoM change distribution\nskew={mom.skew():.2f}  kurt={mom.kurt():.2f}")
sns.histplot(yoy.dropna(), kde=True, ax=axes[1], color="salmon")
axes[1].set_title(f"YoY change distribution\nskew={yoy.skew():.2f}  kurt={yoy.kurt():.2f}")
save(fig, "06_target_distribution.png")

# %% 7. Stationarity tests
print("\n7. STATIONARITY TESTS (ADF + KPSS)")
print("   Stationary if ADF p<0.05 AND KPSS p>0.05\n")

# Variables that are rates/spreads can be zero or negative -> use simple differences.
# Strictly positive prices/indices -> use log differences.
RATE_LIKE = {"fed_funds_rate", "us_unemployment", "yield_spread_10y2y", "vix",
             "consumer_sentiment"}


def adf_kpss(s: pd.Series):
    s = s.dropna()
    if len(s) < 40:
        return np.nan, np.nan
    try:
        adf_p = adfuller(s, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    return adf_p, kpss_p


def is_stationary(adf_p, kpss_p):
    return (adf_p < 0.05) and (kpss_p > 0.05)


rows = []
for col in df.columns:
    s = df[col].dropna()
    if len(s) < 40:
        rows.append({"variable": col, "n_obs": len(s), "recommended": "too few obs"})
        continue

    adf_l, kpss_l = adf_kpss(s)

    # first-difference version
    if col in RATE_LIKE or (s <= 0).any():
        d = s.diff(); tname = "diff"
    else:
        d = np.log(s).diff(); tname = "log_diff"
    adf_d, kpss_d = adf_kpss(d)

    if is_stationary(adf_l, kpss_l):
        rec = "level (already stationary)"
    elif is_stationary(adf_d, kpss_d):
        rec = tname
    else:
        rec = f"{tname} (borderline; check manually)"

    rows.append({
        "variable": col, "n_obs": len(s),
        "ADF_p_level": round(adf_l, 4), "KPSS_p_level": round(kpss_l, 4),
        "stationary_level": is_stationary(adf_l, kpss_l),
        f"ADF_p_{tname}": round(adf_d, 4), f"KPSS_p_{tname}": round(kpss_d, 4),
        "stationary_after_diff": is_stationary(adf_d, kpss_d),
        "recommended": rec,
    })

report = pd.DataFrame(rows).set_index("variable")
report.to_csv("data/processed/stationarity_report.csv")
cols_show = ["n_obs", "ADF_p_level", "KPSS_p_level", "stationary_level", "recommended"]
print(report[cols_show].to_string())
print(f"\n  Level-stationary variables: {int(report['stationary_level'].sum())} / {len(report)}")
print("  Saved: data/processed/stationarity_report.csv")

# %% 8. Correlations: levels (spurious!) vs stationary changes
print("\n8. CORRELATION HEATMAPS")


def apply_transform(s, rec):
    s = s.dropna()
    if rec.startswith("level"):
        return s
    if rec.startswith("log_diff"):
        return np.log(s).diff()
    if rec.startswith("diff"):
        return s.diff()
    return None


stationary = {}
for col, rec in report["recommended"].items():
    if rec == "too few obs":
        continue
    out = apply_transform(df[col], rec)
    if out is not None:
        stationary[col] = out
stat_df = pd.DataFrame(stationary).dropna(how="all")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
corr_lvl = df.corr()
corr_chg = stat_df.corr()
mask = lambda c: np.triu(np.ones_like(c, dtype=bool))
sns.heatmap(corr_lvl, mask=mask(corr_lvl), cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=axes[0], cbar_kws={"shrink": 0.6})
axes[0].set_title("Correlation of LEVELS (inflated by common trends)")
sns.heatmap(corr_chg, mask=mask(corr_chg), cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=axes[1], cbar_kws={"shrink": 0.6})
axes[1].set_title("Correlation of STATIONARY changes (use this for factor analysis)")
save(fig, "07_correlation_levels_vs_changes.png")

# Top correlations with target changes
if TARGET in corr_chg.columns:
    top = corr_chg[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False).head(10)
    print(f"\nTop 10 correlations with {TARGET} (stationary form):")
    print(top.round(3).to_string())

# Clustered view: which variables move together?
clean_corr = corr_chg.dropna(axis=0, how="all").dropna(axis=1, how="all")
cg = sns.clustermap(clean_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    figsize=(11, 11), linewidths=0.2)
cg.fig.suptitle("Clustered correlation (hints at latent factors)", y=1.01)
cg.savefig(FIG_DIR / "08_clustermap.png", dpi=150, bbox_inches="tight")
plt.close(cg.fig)
print(f"  saved {FIG_DIR / '08_clustermap.png'}")

# %% 9. Structural breaks: rolling correlation with target
print("\n9. ROLLING CORRELATIONS (structural stability)")
pairs = [c for c in ["brent_oil", "copper", "wheat", "usd_index", "vix"] if c in stat_df.columns]
tgt_chg = stat_df[TARGET] if TARGET in stat_df.columns else np.log(y).diff()
fig, ax = plt.subplots(figsize=(13, 5))
for c in pairs:
    stat_df[c].rolling(36).corr(tgt_chg).plot(ax=ax, label=c, linewidth=1.3)
shade_events(ax, label=False)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_title(f"36-month rolling correlation with {TARGET} changes")
ax.legend(ncol=5)
save(fig, "09_rolling_correlation.png")

# Simple split-sample comparison of correlations
mid = stat_df.index[len(stat_df) // 2]
early, late = stat_df.loc[:mid].corr()[TARGET], stat_df.loc[mid:].corr()[TARGET]
shift = (late - early).drop(TARGET).sort_values(key=np.abs, ascending=False).head(8)
print(f"\nBiggest correlation shifts with {TARGET} (before vs after {mid.date()}):")
print(pd.DataFrame({"early": early, "late": late, "change": late - early})
      .loc[shift.index].round(2).to_string())

# %% 10. Factor-analysis readiness preview
print("\n10. FACTOR-ANALYSIS READINESS PREVIEW")
try:
    from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    X = stat_df.dropna(axis=1, thresh=int(0.8 * len(stat_df))).dropna()
    chi2, p = calculate_bartlett_sphericity(X)
    _, kmo = calculate_kmo(X)
    print(f"  Sample used: {X.shape[0]} months x {X.shape[1]} variables")
    print(f"  Bartlett's test: chi2={chi2:.1f}, p={p:.4g}  (want p < 0.05)")
    print(f"  KMO overall    : {kmo:.3f}  (want > 0.6; > 0.8 is good)")
except Exception as e:
    print(f"  Skipped ({e})")

print("\nPhase 2 complete. Review reports/figures/ and data/processed/stationarity_report.csv")
