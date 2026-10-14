"""
Phase 1: Data collection (last 30 years, monthly)
Commodity inflation prediction with factor analysis

Sources:
  - FRED  (commodity prices, macro, currency, risk)   -> needs a free API key
  - Yahoo Finance (futures, dollar index, VIX)        -> no key needed
  - World Bank Pink Sheet (optional, manual download) -> Excel file

Setup:
  pip install pandas numpy fredapi yfinance openpyxl
  Set your key:  export FRED_API_KEY="your_key"   (Windows: setx FRED_API_KEY "your_key")
Run:
  python 01_data_collection.py
"""

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
END = pd.Timestamp(date.today())
START = END - pd.DateOffset(years=30)

from fredapi import Fred
fred = Fred(api_key=os.getenv("FRED_API_KEY"))
print(fred.get_series("CPIAUCSL").tail())

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# name -> FRED series id
FRED_SERIES = {
    # Commodity prices and indices (IMF series hosted on FRED, monthly)
    "all_commodity_index": "PALLFNFINDEXM",
    "energy_index":        "PNRGINDEXM",
    "food_index":          "PFOODINDEXM",
    "metals_index":        "PMETAINDEXM",
    "brent_oil":           "POILBREUSDM",
    "wti_oil":             "POILWTIUSDM",
    "nat_gas_eu":          "PNGASEUUSDM",
    "copper":              "PCOPPUSDM",
    "aluminum":            "PALUMUSDM",
    "wheat":               "PWHEAMTUSDM",
    "maize":               "PMAIZMTUSDM",
    "sugar":               "PSUGAISAUSDM",
    "cotton":              "PCOTTINDUSDM",
    # Macro (US)
    "us_cpi":              "CPIAUCSL",
    "us_ppi":              "PPIACO",
    "us_ind_production":   "INDPRO",
    "fed_funds_rate":      "FEDFUNDS",
    "us_m2":               "M2SL",
    "us_unemployment":     "UNRATE",
    "consumer_sentiment":  "UMCSENT",
    "yield_spread_10y2y":  "T10Y2Y",
    # Currency (daily, resampled to monthly)
    "usd_inr":             "DEXINUS",
    "usd_cny":             "DEXCHUS",
    "usd_per_eur":         "DEXUSEU",   # starts 1999
    # Risk
    "vix":                 "VIXCLS",
}

# name -> Yahoo ticker (futures history usually starts around 2000)
YAHOO_TICKERS = {
    "wti_futures":    "CL=F",
    "brent_futures":  "BZ=F",
    "gold_futures":   "GC=F",
    "copper_futures": "HG=F",
    "wheat_futures":  "ZW=F",
    "corn_futures":   "ZC=F",
    "usd_index":      "DX-Y.NYB",
    "sp500":          "^GSPC",
}


# ----------------------------------------------------------------------
# 2. Downloaders
# ----------------------------------------------------------------------
def fetch_fred(series_map: dict) -> pd.DataFrame:
    """Download FRED series and return a daily/monthly mixed DataFrame."""
    fred = Fred(api_key=FRED_API_KEY)
    frames, failed = {}, []
    for name, sid in series_map.items():
        try:
            s = fred.get_series(sid, observation_start=START, observation_end=END)
            frames[name] = s.rename(name)
            print(f"  [OK]   FRED {sid:<15} -> {name} ({s.notna().sum()} obs)")
        except Exception as e:
            failed.append(name)
            print(f"  [FAIL] FRED {sid:<15} -> {name}: {e}")
    if failed:
        print(f"  FRED failures: {failed}")
    return pd.concat(frames.values(), axis=1) if frames else pd.DataFrame()


def fetch_yahoo(ticker_map: dict) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    frames, failed = {}, []
    for name, tk in ticker_map.items():
        try:
            d = yf.download(tk, start=START, end=END, progress=False, auto_adjust=True)
            if d.empty:
                raise ValueError("no data returned")
            close = d["Close"]
            if isinstance(close, pd.DataFrame):      # newer yfinance returns 1-col frame
                close = close.iloc[:, 0]
            frames[name] = close.rename(name)
            print(f"  [OK]   Yahoo {tk:<10} -> {name} ({close.notna().sum()} obs)")
        except Exception as e:
            failed.append(name)
            print(f"  [FAIL] Yahoo {tk:<10} -> {name}: {e}")
    if failed:
        print(f"  Yahoo failures: {failed}")
    return pd.concat(frames.values(), axis=1) if frames else pd.DataFrame()


def load_pink_sheet(path: str = "data/raw/CMO-Historical-Data-Monthly.xlsx") -> pd.DataFrame:
    """
    OPTIONAL: World Bank Pink Sheet.
    Download 'Monthly prices' Excel from the World Bank Commodity Markets page,
    save it in data/raw/, and adjust sheet/header rows below if the layout changes.
    """
    p = Path(path)
    if not p.exists():
        print("  [SKIP] Pink Sheet file not found (optional).")
        return pd.DataFrame()
    df = pd.read_excel(p, sheet_name="Monthly Prices", header=4)
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)].copy()
    df["date"] = pd.to_datetime(df["date"].str.replace("M", "-") + "-01")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df.columns = ["pink_" + str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  [OK]   Pink Sheet -> {df.shape[1]} series")
    return df.loc[START:END]


# ----------------------------------------------------------------------
# 3. Frequency alignment
# ----------------------------------------------------------------------
def to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Resample any-frequency data to month-start frequency."""
    if df.empty:
        return df
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    m = df.resample("MS").mean() if how == "mean" else df.resample("MS").last()
    return m


# ----------------------------------------------------------------------
# 4. Main pipeline
# ----------------------------------------------------------------------
def main():
    print(f"Collecting data from {START.date()} to {END.date()}\n")

    print("Downloading FRED ...")
    fred_raw = fetch_fred(FRED_SERIES)
    print("\nDownloading Yahoo Finance ...")
    yahoo_raw = fetch_yahoo(YAHOO_TICKERS)
    print("\nLoading Pink Sheet ...")
    pink_raw = load_pink_sheet()

    # Save raw downloads (untouched) for reproducibility
    if not fred_raw.empty:
        fred_raw.to_csv(RAW_DIR / "fred_raw.csv")
    if not yahoo_raw.empty:
        yahoo_raw.to_csv(RAW_DIR / "yahoo_raw.csv")
    if not pink_raw.empty:
        pink_raw.to_csv(RAW_DIR / "pink_sheet_raw.csv")

    # Align everything to monthly frequency
    parts = [to_monthly(fred_raw), to_monthly(yahoo_raw), pink_raw]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit("No data downloaded. Check your internet connection and FRED key.")

    monthly = pd.concat(parts, axis=1)
    monthly = monthly.loc[:, ~monthly.columns.duplicated()]
    monthly = monthly.loc[START:END]
    monthly.index.name = "date"

    # Light cleaning (no future information is used):
    #  - forward fill only short gaps (max 2 months), e.g. publication delays
    #  - drop columns that are mostly empty
    monthly = monthly.ffill(limit=2)
    missing_share = monthly.isna().mean()
    too_empty = missing_share[missing_share > 0.5].index.tolist()
    if too_empty:
        print(f"\nDropping columns with >50% missing: {too_empty}")
        monthly = monthly.drop(columns=too_empty)

    # Save
    monthly.to_csv(PROC_DIR / "monthly_merged.csv")

    # Summary report
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Shape            : {monthly.shape[0]} months x {monthly.shape[1]} variables")
    print(f"Period           : {monthly.index.min().date()} -> {monthly.index.max().date()}")
    print(f"Total missing    : {monthly.isna().sum().sum()} cells "
          f"({monthly.isna().mean().mean():.1%})")
    print("\nMissing share per variable (top 10):")
    print(monthly.isna().mean().sort_values(ascending=False).head(10).round(3))

    # Data dictionary
    dd = pd.DataFrame({
        "first_valid": monthly.apply(lambda s: s.first_valid_index()),
        "last_valid":  monthly.apply(lambda s: s.last_valid_index()),
        "n_obs":       monthly.notna().sum(),
        "missing_%":   (monthly.isna().mean() * 100).round(1),
        "mean":        monthly.mean().round(3),
        "std":         monthly.std().round(3),
    })
    dd.to_csv(PROC_DIR / "data_dictionary.csv")
    print(f"\nSaved: {PROC_DIR / 'monthly_merged.csv'}")
    print(f"Saved: {PROC_DIR / 'data_dictionary.csv'}")
    return monthly


if __name__ == "__main__":
    df = main()
    print("\nPreview:")
    print(df.tail())

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


"""
Phase 3: Preprocessing (leakage-safe)
Inputs : data/processed/monthly_merged.csv        (Phase 1)
         data/processed/stationarity_report.csv   (Phase 2)
Outputs: data/processed/X_stationary.csv          unscaled stationary indicators (all dates)
         data/processed/X_train_scaled.csv / X_test_scaled.csv
         data/processed/targets.csv               h-month-ahead commodity inflation targets
         data/processed/target_lags.csv           past-only lag / rolling features of the target
         data/processed/preprocess_config.json
         models/scaler.joblib

Phase 4 can also import LeakageSafePreprocessor from this file and refit it
inside every cross-validation fold.

Run:  python 03_preprocessing.py
"""

# %% 0. Imports and configuration
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

PROC = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

TARGET = "all_commodity_index"            # must match Phase 2
HORIZONS = [1, 3, 6, 12]                  # forecast horizons in months
TEST_FRACTION = 0.25                      # last 25% of time = hold-out test
GAP = max(HORIZONS)                       # embargo between train and test (months)

# Drop indicators that start too late (keeps ~30-year coverage). Set None to keep all.
MAX_FIRST_VALID = "2001-01-01"
MAX_MISSING_SHARE = 0.20                  # drop columns with more missing than this
WINSOR_Q = (0.01, 0.99)                   # winsorization quantiles (train-only)

# Indicators that mechanically contain the target (components of the all-commodity index).
# They are excluded from the factor inputs to avoid target leakage. Edit to taste.
DROP_FROM_X = {TARGET, "energy_index", "food_index", "metals_index"}

# Publication lags in months (how late the data become known in real time).
PUB_LAG = {
    "us_cpi": 1, "us_ppi": 1, "us_ind_production": 1, "us_m2": 1,
    "us_unemployment": 1, "consumer_sentiment": 0, "yield_spread_10y2y": 0,
    "fed_funds_rate": 0, "vix": 0,
    # IMF commodity series on FRED are released with ~1 month delay
    "brent_oil": 1, "wti_oil": 1, "nat_gas_eu": 1, "copper": 1, "aluminum": 1,
    "wheat": 1, "maize": 1, "sugar": 1, "cotton": 1,
}


# %% 1. Leakage-safe preprocessor (reusable in CV folds)
class LeakageSafePreprocessor(BaseEstimator, TransformerMixin):
    """Winsorize with train quantiles, then z-score with train mean/std."""

    def __init__(self, q_low=0.01, q_high=0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.lower_ = X.quantile(self.q_low)
        self.upper_ = X.quantile(self.q_high)
        self.scaler_ = StandardScaler().fit(X.clip(self.lower_, self.upper_, axis=1))
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_) if not isinstance(X, pd.DataFrame) else X
        X = X[self.columns_]
        clipped = X.clip(self.lower_, self.upper_, axis=1)
        out = self.scaler_.transform(clipped)
        return pd.DataFrame(out, index=X.index, columns=self.columns_)


def expanding_window_splits(index, n_splits=5, min_train=120, gap=GAP):
    """Yield (train_idx, test_idx) positional arrays with an embargo gap."""
    n = len(index)
    fold = (n - min_train) // n_splits
    for k in range(n_splits):
        train_end = min_train + k * fold
        test_start = train_end + gap
        test_end = min(test_start + fold, n)
        if test_start >= n:
            break
        yield np.arange(0, train_end), np.arange(test_start, test_end)


if __name__ == "__main__":
    # %% 2. Load data
    print("=" * 60)
    print("1. LOAD")
    print("=" * 60)
    raw = pd.read_csv(PROC / "monthly_merged.csv", index_col="date", parse_dates=True)
    rep = pd.read_csv(PROC / "stationarity_report.csv", index_col="variable")
    print(f"Raw data : {raw.shape}  ({raw.index.min().date()} -> {raw.index.max().date()})")

    # %% 3. Build targets (from raw target levels, before anything else)
    print("\n2. TARGETS")
    logp = np.log(raw[TARGET].dropna())
    targets = pd.DataFrame(index=raw.index)
    for h in HORIZONS:
        # cumulative % change over the NEXT h months, known only at t+h
        targets[f"y_h{h}"] = 100 * (logp.shift(-h) - logp)
    targets["y_mom"] = 100 * logp.diff()                  # current 1-month change
    targets.to_csv(PROC / "targets.csv")
    print(targets.describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())

    # %% 4. Apply recommended stationarity transforms
    print("\n3. STATIONARITY TRANSFORMS")
    transformed, skipped = {}, []
    for col in raw.columns:
        if col in DROP_FROM_X:
            continue
        rec = str(rep["recommended"].get(col, "log_diff"))
        s = raw[col]
        if rec == "too few obs":
            skipped.append((col, "too few obs"))
            continue
        if rec.startswith("level"):
            out = s
        elif rec.startswith("log_diff") and (s.dropna() > 0).all():
            out = 100 * np.log(s).diff()
        elif rec.startswith("log_diff") or rec.startswith("diff"):
            out = s.diff()
        else:
            out = s.diff()
        transformed[col] = out
        print(f"  {col:<22} -> {rec}")
    X = pd.DataFrame(transformed)
    print(f"Dropped as target-related: {sorted(DROP_FROM_X & set(raw.columns))}")
    print(f"Skipped: {skipped}")

    # %% 5. Publication lags (real-time realism)
    print("\n4. PUBLICATION LAGS")
    for col, lag in PUB_LAG.items():
        if col in X.columns and lag > 0:
            X[col] = X[col].shift(lag)
            print(f"  {col:<22} shifted by {lag} month(s)")

    # %% 6. Column filtering (coverage and missingness)
    print("\n5. COLUMN FILTERING")
    dropped = []
    if MAX_FIRST_VALID:
        cutoff = pd.Timestamp(MAX_FIRST_VALID)
        for col in list(X.columns):
            first = X[col].first_valid_index()
            if first is None or first > cutoff:
                dropped.append((col, f"starts {first.date() if first is not None else 'never'}"))
                X = X.drop(columns=col)
    X = X.loc[X.apply(lambda s: s.first_valid_index()).max():]   # common start date
    for col in list(X.columns):
        share = X[col].isna().mean()
        if share > MAX_MISSING_SHARE:
            dropped.append((col, f"{share:.0%} missing"))
            X = X.drop(columns=col)
    # fill short gaps using PAST values only (no backward fill, no interpolation)
    X = X.ffill(limit=2)
    # drop leading/trailing rows that still have NaNs in indicators
    X = X.dropna()
    for col, why in dropped:
        print(f"  dropped {col:<22} ({why})")
    print(f"Final indicators: {X.shape[1]} variables, {X.shape[0]} months "
          f"({X.index.min().date()} -> {X.index.max().date()})")
    X.to_csv(PROC / "X_stationary.csv")

    # %% 7. Chronological train/test split with embargo
    print("\n6. TRAIN / TEST SPLIT")
    n = len(X)
    test_start_pos = int(n * (1 - TEST_FRACTION))
    train_end_pos = test_start_pos - GAP            # embargo so targets don't overlap test
    X_train = X.iloc[:train_end_pos]
    X_test = X.iloc[test_start_pos:]
    print(f"Train: {X_train.index.min().date()} -> {X_train.index.max().date()} ({len(X_train)} months)")
    print(f"Gap  : {GAP} months embargo")
    print(f"Test : {X_test.index.min().date()} -> {X_test.index.max().date()} ({len(X_test)} months)")

    # %% 8. Winsorize + scale (fit on TRAIN ONLY)
    print("\n7. WINSORIZE + SCALE (train-only statistics)")
    prep = LeakageSafePreprocessor(*WINSOR_Q).fit(X_train)
    X_train_s = prep.transform(X_train)
    X_test_s = prep.transform(X_test)
    X_all_s = prep.transform(X)                      # for later rolling-origin use
    joblib.dump(prep, MODELS / "scaler.joblib")
    X_train_s.to_csv(PROC / "X_train_scaled.csv")
    X_test_s.to_csv(PROC / "X_test_scaled.csv")
    X_all_s.to_csv(PROC / "X_all_scaled_trainfit.csv")
    print(f"Train mean (should be ~0): {X_train_s.mean().abs().max():.4f}")
    print(f"Train std  (should be ~1): {X_train_s.std().mean():.3f}")
    print(f"Test  mean (NOT exactly 0, which is correct): {X_test_s.mean().abs().mean():.3f}")

    # %% 9. Past-only features of the target (for AR / benchmark models)
    print("\n8. TARGET LAG AND ROLLING FEATURES (past information only)")
    mom = targets["y_mom"]
    lagfeat = pd.DataFrame(index=raw.index)
    for L in [1, 2, 3, 6, 12]:
        lagfeat[f"mom_lag{L}"] = mom.shift(L - 1)     # lag1 = current known month change
    for w in [3, 6, 12]:
        lagfeat[f"mom_roll_mean{w}"] = mom.rolling(w).mean()
        lagfeat[f"mom_roll_std{w}"] = mom.rolling(w).std()
    lagfeat = lagfeat.reindex(X.index)
    lagfeat.to_csv(PROC / "target_lags.csv")
    print(f"Created {lagfeat.shape[1]} features: {list(lagfeat.columns)}")

    # %% 10. Validation checks
    print("\n9. VALIDATION CHECKS")
    # (a) target must not be in X
    assert TARGET not in X.columns, "Target leaked into X!"
    print("  [OK] target not in indicator set")
    # (b) all X indexes align with targets
    assert X.index.isin(targets.index).all()
    print("  [OK] index alignment")
    # (c) no NaNs in scaled inputs
    assert not X_train_s.isna().any().any() and not X_test_s.isna().any().any()
    print("  [OK] no NaNs in scaled train/test")
    # (d) re-check stationarity on training indicators
    bad = []
    for col in X_train.columns:
        try:
            if adfuller(X_train[col], autolag="AIC")[1] > 0.05:
                bad.append(col)
        except Exception:
            pass
    print(f"  Non-stationary after transform (ADF p>0.05): {bad if bad else 'none'}")
    # (e) rows usable per horizon
    for h in HORIZONS:
        ok = targets[f"y_h{h}"].reindex(X.index).notna()
        print(f"  Horizon {h:>2}m: {int(ok.sum())} usable rows "
              f"(last {h} months have no realised target yet)")

    # %% 11. Save configuration for reproducibility
    cfg = {
        "target": TARGET, "horizons": HORIZONS, "test_fraction": TEST_FRACTION,
        "gap_months": GAP, "winsor_quantiles": list(WINSOR_Q),
        "max_first_valid": MAX_FIRST_VALID, "max_missing_share": MAX_MISSING_SHARE,
        "indicators": list(X.columns),
        "train_period": [str(X_train.index.min().date()), str(X_train.index.max().date())],
        "test_period": [str(X_test.index.min().date()), str(X_test.index.max().date())],
        "publication_lags": {k: v for k, v in PUB_LAG.items() if k in X.columns},
    }
    with open(PROC / "preprocess_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("\nSaved config -> data/processed/preprocess_config.json")
    print("Saved scaler -> models/scaler.joblib")
    print("\nPhase 3 complete. Next: Phase 4 (factor extraction).")


"""
Phase 4: Factor extraction (the core of the project)
Inputs : data/processed/X_train_scaled.csv, X_test_scaled.csv, X_all_scaled_trainfit.csv,
         targets.csv, preprocess_config.json            (from Phase 3)
Outputs: data/processed/factors_all.csv (train-fitted factor scores for every month)
         data/processed/factor_loadings.csv, factor_variance.csv, factor_diagnostics.json
         models/factor_model.joblib
         reports/figures/10_*.png ... 14_*.png

Needs only: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, joblib
(KMO, Bartlett and parallel analysis are implemented here, so factor_analyzer is NOT required.)

Run:  python 04_factor_extraction.py
"""

# %% 0. Setup
import json
import os
import warnings
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)       # always work relative to this script's folder

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROC = Path("data/processed")
FIG = Path("reports/figures")
MODELS = Path("models")
for d in (FIG, MODELS):
    d.mkdir(parents=True, exist_ok=True)

K_OVERRIDE = None          # set an integer (e.g. 4) to force the number of factors
K_MIN, K_MAX = 2, 6        # bounds for the automatic choice
N_SIM = 500                # parallel analysis simulations
RANDOM_STATE = 42

EVENTS = [("2008-09-01", "2009-06-01", "GFC"),
          ("2020-03-01", "2020-06-01", "COVID"),
          ("2022-02-01", "2022-12-01", "Energy shock")]

# Variable -> economic group (used to auto-name the factors)
GROUPS = {
    "Energy": ["brent_oil", "wti_oil", "nat_gas_eu", "wti_futures", "brent_futures"],
    "Metals / industrial demand": ["copper", "aluminum", "copper_futures", "gold_futures",
                                   "us_ind_production"],
    "Agriculture": ["wheat", "maize", "sugar", "cotton", "wheat_futures", "corn_futures"],
    "US dollar / currency": ["usd_index", "usd_inr", "usd_cny", "usd_per_eur"],
    "Financial risk / sentiment": ["vix", "sp500", "consumer_sentiment", "yield_spread_10y2y"],
    "Monetary / inflation": ["fed_funds_rate", "us_m2", "us_cpi", "us_ppi", "us_unemployment"],
}
VAR2GROUP = {v: g for g, vs in GROUPS.items() for v in vs}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG / name}")


def shade_events(ax):
    for (s, e, _), c in zip(EVENTS, ["#d62728", "#ff7f0e", "#9467bd"]):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.15)


# %% 1. Load
print("=" * 60)
print("1. LOAD")
print("=" * 60)
X_tr = pd.read_csv(PROC / "X_train_scaled.csv", index_col=0, parse_dates=True)
X_te = pd.read_csv(PROC / "X_test_scaled.csv", index_col=0, parse_dates=True)
X_all = pd.read_csv(PROC / "X_all_scaled_trainfit.csv", index_col=0, parse_dates=True)
targets = pd.read_csv(PROC / "targets.csv", index_col=0, parse_dates=True)
cfg = json.load(open(PROC / "preprocess_config.json"))
HORIZONS = cfg["horizons"]
T, N = X_tr.shape
print(f"Train: {T} months x {N} indicators | Test: {len(X_te)} months")
print(f"Indicators: {list(X_tr.columns)}")
if N < 6:
    print("WARNING: very few indicators. Loosen the column filters in Phase 3.")


# %% 2. Suitability tests (Bartlett, KMO)
def bartlett(R, n_obs):
    p = R.shape[0]
    sign, logdet = np.linalg.slogdet(R)
    chi2 = -(n_obs - 1 - (2 * p + 5) / 6) * logdet
    dof = p * (p - 1) / 2
    return chi2, dof, 1 - stats.chi2.cdf(chi2, dof)


def kmo(R):
    Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    partial = -Rinv / d
    R = R.copy()
    np.fill_diagonal(R, 0)
    np.fill_diagonal(partial, 0)
    r2, p2 = R ** 2, partial ** 2
    per_var = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    overall = r2.sum() / (r2.sum() + p2.sum())
    return overall, per_var


print("\n2. SUITABILITY TESTS")
R = np.corrcoef(X_tr.values, rowvar=False)
chi2, dof, p_val = bartlett(R, T)
kmo_all, kmo_var = kmo(R.copy())
print(f"  Bartlett: chi2={chi2:.1f}, df={dof:.0f}, p={p_val:.3g}  (want p < 0.05)")
print(f"  KMO overall = {kmo_all:.3f}  (>0.6 acceptable, >0.8 good)")
kmo_series = pd.Series(kmo_var, index=X_tr.columns).sort_values()
print("  Lowest per-variable KMO (consider dropping if < 0.5):")
print(kmo_series.head(5).round(3).to_string())

# %% 3. How many factors?
print("\n3. NUMBER OF FACTORS")
eig = np.sort(np.linalg.eigvalsh(R))[::-1]
explained = eig / eig.sum()

# (a) Kaiser criterion
k_kaiser = int((eig > 1).sum())

# (b) Parallel analysis (95th percentile of random-data eigenvalues)
rng = np.random.default_rng(RANDOM_STATE)
sim = np.empty((N_SIM, N))
for i in range(N_SIM):
    Z = rng.standard_normal((T, N))
    sim[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
pa_threshold = np.percentile(sim, 95, axis=0)
k_pa = 0
for a, b in zip(eig, pa_threshold):
    if a > b:
        k_pa += 1
    else:
        break

# (c) Bai-Ng IC_p2
kmax = min(10, N - 1)
ic = []
for k in range(1, kmax + 1):
    V = eig[k:].sum() / N
    pen = k * ((N + T) / (N * T)) * np.log(min(N, T))
    ic.append(np.log(V) + pen)
k_bn = int(np.argmin(ic)) + 1

# (d) variance threshold: smallest k explaining >= 70%
cum = np.cumsum(explained)
k_70 = int(np.searchsorted(cum, 0.70) + 1)

print(f"  Kaiser (eig>1)          : {k_kaiser}")
print(f"  Parallel analysis       : {k_pa}")
print(f"  Bai-Ng IC_p2            : {k_bn}")
print(f"  Variance >= 70%         : {k_70}")
K = K_OVERRIDE if K_OVERRIDE else int(np.clip(k_pa if k_pa > 0 else k_kaiser, K_MIN, K_MAX))
K = min(K, N - 1)
print(f"  --> CHOSEN K = {K}  (set K_OVERRIDE at the top to change)")
print(f"  Cumulative variance at K: {cum[K - 1]:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(1, N + 1)
axes[0].plot(x, eig, "o-", label="Actual eigenvalues")
axes[0].plot(x, pa_threshold, "s--", color="gray", label="Parallel analysis (95%)")
axes[0].axhline(1, color="red", linestyle=":", label="Kaiser (=1)")
axes[0].axvline(K + 0.5, color="green", linestyle="--", label=f"Chosen K={K}")
axes[0].set_title("Scree plot"); axes[0].set_xlabel("Component"); axes[0].legend()
axes[1].bar(x, explained * 100, alpha=0.6, label="Individual")
axes[1].plot(x, cum * 100, "ro-", label="Cumulative")
axes[1].set_title("Variance explained (%)"); axes[1].set_xlabel("Component"); axes[1].legend()
save(fig, "10_scree_parallel.png")

# %% 4. Extract factors: PCA (unrotated) vs Factor Analysis (varimax)
print("\n4. FACTOR EXTRACTION")
pca = PCA(n_components=K, random_state=RANDOM_STATE).fit(X_tr)
fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=RANDOM_STATE,
                    max_iter=2000).fit(X_tr)

L = pd.DataFrame(fa.components_.T, index=X_tr.columns,
                 columns=[f"F{i + 1}" for i in range(K)])

# order factors by variance explained and fix signs (largest |loading| positive)
ss = (L ** 2).sum()
L = L[ss.sort_values(ascending=False).index]
L.columns = [f"F{i + 1}" for i in range(K)]
scores_cols = list(ss.sort_values(ascending=False).index)      # original factor order
order_idx = [int(c[1:]) - 1 for c in scores_cols]
signs = np.array([np.sign(L[c].iloc[np.argmax(np.abs(L[c].values))]) for c in L.columns])
L = L * signs


def scores(model, X):
    S = model.transform(X)[:, order_idx] * signs
    return pd.DataFrame(S, index=X.index, columns=L.columns)


F_tr, F_te, F_all = scores(fa, X_tr), scores(fa, X_te), scores(fa, X_all)

# Factor naming from group loadings
names, rows = {}, []
for c in L.columns:
    grp_strength = {}
    for var, load in L[c].items():
        g = VAR2GROUP.get(var, "Other")
        grp_strength[g] = grp_strength.get(g, 0) + load ** 2
    best = max(grp_strength, key=grp_strength.get)
    top = L[c].abs().sort_values(ascending=False).head(4).index.tolist()
    names[c] = f"{c}: {best}"
    rows.append({"factor": c, "suggested_label": best, "top_variables": ", ".join(top)})
label_table = pd.DataFrame(rows).set_index("factor")

var_tbl = pd.DataFrame({
    "SS_loadings": (L ** 2).sum(),
    "pct_variance": (L ** 2).sum() / N * 100,
})
var_tbl["cum_pct"] = var_tbl["pct_variance"].cumsum()
comm = pd.Series((L ** 2).sum(axis=1), index=L.index, name="communality")

print("\nRotated loadings (Factor Analysis, varimax):")
print(L.round(2).to_string())
print("\nVariance explained by rotated factors:")
print(var_tbl.round(2).to_string())
print("\nSuggested factor labels (CHECK them against the loadings yourself):")
print(label_table.to_string())
print("\nLowest communalities (variables poorly explained by the factors):")
print(comm.sort_values().head(5).round(2).to_string())

# loadings heatmap
fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * N)))
sns.heatmap(L.round(2), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=ax, cbar_kws={"shrink": 0.7})
ax.set_xticklabels([names[c] for c in L.columns], rotation=30, ha="right")
ax.set_title("Factor loadings (varimax-rotated)")
save(fig, "11_loadings_heatmap.png")

# %% 5. Factor scores over time
print("\n5. FACTOR SCORES")
fig, axes = plt.subplots(K, 1, figsize=(13, 2.4 * K), sharex=True)
axes = np.atleast_1d(axes)
for ax, c in zip(axes, L.columns):
    F_all[c].plot(ax=ax, linewidth=1.1)
    shade_events(ax)
    ax.axvline(X_te.index.min(), color="k", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title(names[c], fontsize=10)
axes[-1].set_xlabel("Dashed line = start of test period; shaded = GFC / COVID / 2022 energy shock")
save(fig, "12_factor_scores.png")

# %% 6. Do factors relate to the target? (train sample only)
print("\n6. FACTOR vs TARGET CORRELATIONS (train only)")
rel = pd.DataFrame(index=L.columns)
for h in HORIZONS:
    y = targets[f"y_h{h}"].reindex(F_tr.index)
    ok = y.notna()
    rel[f"h={h}m"] = [np.corrcoef(F_tr.loc[ok, c], y[ok])[0, 1] for c in L.columns]
rel["current_mom"] = [np.corrcoef(F_tr[c], targets["y_mom"].reindex(F_tr.index).fillna(0))[0, 1]
                      for c in L.columns]
print(rel.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 3 + 0.4 * K))
sns.heatmap(rel, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation of factors with future commodity inflation (train)")
save(fig, "13_factor_target_corr.png")

# %% 7. Stability check (supports hypothesis H4)
print("\n7. STABILITY ACROSS TIME (Tucker congruence, early vs late half)")


def pca_loadings(X, k):
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X)
    return p.components_.T * np.sqrt(p.explained_variance_)


mid = len(X_all) // 2
La = pca_loadings(X_all.iloc[:mid], K)
Lb = pca_loadings(X_all.iloc[mid:], K)
cong = np.abs((La.T @ Lb) / np.outer(np.linalg.norm(La, axis=0), np.linalg.norm(Lb, axis=0)))
best = cong.max(axis=1)
for i, v in enumerate(best, 1):
    flag = "stable" if v > 0.95 else ("fair" if v > 0.85 else "UNSTABLE")
    print(f"  Component {i}: congruence = {v:.2f}  ({flag})")
print("  (>0.95 = same factor, 0.85-0.95 = fairly similar, <0.85 = different structure)")

# Is the first-factor share of variance different in crisis vs calm?
crisis = pd.Series(False, index=X_all.index)
for s, e, _ in EVENTS:
    crisis |= (X_all.index >= s) & (X_all.index <= e)
if crisis.sum() > 10:
    v_c = PCA(n_components=1).fit(X_all[crisis]).explained_variance_ratio_[0]
    v_n = PCA(n_components=1).fit(X_all[~crisis]).explained_variance_ratio_[0]
    print(f"  Variance share of 1st component: crisis months = {v_c:.1%}, normal months = {v_n:.1%}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cong, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
ax.set_title("Congruence: early-half vs late-half components")
ax.set_xlabel("Late half"); ax.set_ylabel("Early half")
save(fig, "14_stability_congruence.png")

# %% 8. Compare with PCA (robustness)
print("\n8. ROBUSTNESS: FA scores vs PCA scores")
P_tr = pca.transform(X_tr)
corr_fa_pca = np.abs(np.corrcoef(F_tr.values.T, P_tr.T)[:K, K:])
print("  Max |correlation| of each FA factor with any PCA component:")
for c, v in zip(L.columns, corr_fa_pca.max(axis=1)):
    print(f"    {c}: {v:.2f}")

# %% 9. Save everything
print("\n9. SAVING")
F_all.to_csv(PROC / "factors_all.csv")
F_tr.to_csv(PROC / "factors_train.csv")
F_te.to_csv(PROC / "factors_test.csv")
L.to_csv(PROC / "factor_loadings.csv")
pd.concat([var_tbl, label_table], axis=1).to_csv(PROC / "factor_variance.csv")
rel.to_csv(PROC / "factor_target_corr.csv")
joblib.dump({"fa": fa, "order_idx": order_idx, "signs": signs, "columns": list(X_tr.columns)},
            MODELS / "factor_model.joblib")
diag = {
    "n_factors": K, "kmo_overall": round(float(kmo_all), 3), "bartlett_p": float(p_val),
    "k_kaiser": k_kaiser, "k_parallel": k_pa, "k_baing": k_bn, "k_70pct": k_70,
    "cum_variance_pct": round(float(var_tbl["cum_pct"].iloc[-1]), 1),
    "factor_labels": {c: names[c] for c in L.columns},
}
json.dump(diag, open(PROC / "factor_diagnostics.json", "w"), indent=2)
print("  data/processed/: factors_all.csv, factors_train.csv, factors_test.csv,")
print("                   factor_loadings.csv, factor_variance.csv, factor_target_corr.csv,")
print("                   factor_diagnostics.json")
print("  models/factor_model.joblib")
print("\nPhase 4 complete. Next: Phase 5 (forecasting models).")


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




"""
run_all.py - run the whole commodity-inflation pipeline (Phases 1-7) in order.

Usage (from the project folder):
    python run_all.py                  # run phases 1-7
    python run_all.py --skip-download  # reuse existing data, run phases 2-7
    python run_all.py --from 4         # run phases 4-7
    python run_all.py --only 5         # run a single phase
    python run_all.py --from 3 --to 5  # run phases 3-5

Script discovery: for phase N the runner looks for 0N_*.py, then PhaseN.py / phaseN.py,
so both the original file names and renamed ones work.

FRED key: set FRED_API_KEY as an environment variable, or put it in a file named .env
next to this script:   FRED_API_KEY=your32characterkey
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

PHASES = {
    1: "Data collection",
    2: "EDA and stationarity tests",
    3: "Preprocessing",
    4: "Factor extraction",
    5: "Forecasting models",
    6: "Evaluation and robustness",
    7: "Interpretation and report",
}

# Files that must exist after a phase finishes successfully
EXPECTED = {
    1: ["data/processed/monthly_merged.csv"],
    2: ["data/processed/stationarity_report.csv"],
    3: ["data/processed/X_train_scaled.csv", "data/processed/targets.csv"],
    4: ["data/processed/factors_all.csv", "data/processed/factor_diagnostics.json"],
    5: ["data/processed/results_metrics.csv", "data/processed/forecasts_h1.csv"],
    6: ["data/processed/results_eval_all.csv"],
    7: ["reports/results_summary.md"],
}


def find_script(n: int):
    for pattern in (f"0{n}_*.py", f"Phase{n}.py", f"phase{n}.py", f"Phase_{n}.py"):
        hits = [p for p in sorted(ROOT.glob(pattern)) if p.name != Path(__file__).name]
        if hits:
            return hits[0]
    return None


def load_env_file():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fred_key_ok(script: Path) -> bool:
    key = os.environ.get("FRED_API_KEY", "")
    if re.fullmatch(r"[a-z0-9]{32}", key):
        return True
    # maybe the key is pasted directly inside the script
    text = script.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"FRED_API_KEY\s*=.*[\"']([a-z0-9]{32})[\"']", text))


def run_phase(n: int, script: Path) -> bool:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"phase{n}.log"
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
    start = time.time()
    print(f"\n{'=' * 70}\nPHASE {n}: {PHASES[n]}   ({script.name})\n{'=' * 70}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    secs = time.time() - start
    if proc.returncode != 0:
        print(f"\n[FAILED] Phase {n} exited with code {proc.returncode}. See {log_path}")
        return False
    missing = [f for f in EXPECTED[n] if not (ROOT / f).exists()]
    if missing:
        print(f"\n[FAILED] Phase {n} finished but did not create: {missing}")
        return False
    print(f"\n[OK] Phase {n} finished in {secs:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description="Run the commodity inflation pipeline")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first phase (1-7)")
    ap.add_argument("--to", dest="end", type=int, default=7, help="last phase (1-7)")
    ap.add_argument("--only", type=int, help="run only this phase")
    ap.add_argument("--skip-download", action="store_true", help="skip phase 1 (reuse data)")
    args = ap.parse_args()

    start, end = (args.only, args.only) if args.only else (args.start, args.end)
    if args.skip_download and start == 1:
        start = 2
    phases = [p for p in range(start, end + 1) if p in PHASES]
    if not phases:
        sys.exit("Nothing to run. Phases are numbered 1-7.")

    load_env_file()
    scripts = {}
    for n in phases:
        s = find_script(n)
        if s is None:
            sys.exit(f"Could not find a script for phase {n} (looked for 0{n}_*.py / Phase{n}.py) in {ROOT}")
        scripts[n] = s

    if 1 in phases and not fred_key_ok(scripts[1]):
        sys.exit("No valid FRED API key found.\n"
                 "  Get a free key at https://fred.stlouisfed.org (My Account -> API Keys), then either\n"
                 "  - create a file named .env next to run_all.py containing: FRED_API_KEY=your32characterkey\n"
                 "  - or set the FRED_API_KEY environment variable,\n"
                 "  - or run with --skip-download if data/processed/monthly_merged.csv already exists.")

    results, t0 = {}, time.time()
    for n in phases:
        ok = run_phase(n, scripts[n])
        results[n] = ok
        if not ok:
            break

    print(f"\n{'=' * 70}\nSUMMARY (total {time.time() - t0:.0f}s)\n{'=' * 70}")
    for n in phases:
        status = {True: "OK", False: "FAILED"}.get(results.get(n), "not run")
        print(f"  Phase {n}: {PHASES[n]:<32} {status}")
    if all(results.get(n) for n in phases):
        print("\nAll requested phases completed.")
        if 7 in phases:
            print("  Report   : reports/results_summary.md")
            print("  Figures  : reports/figures/")
            print("  Dashboard: streamlit run app.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()



"""
Streamlit dashboard for the commodity-inflation factor project.

Run (after the pipeline has produced its outputs):
    pip install streamlit plotly
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
FIG = BASE / "reports" / "figures"
REPORT = BASE / "reports" / "results_summary.md"

st.set_page_config(page_title="Commodity Inflation Factors", page_icon="📈", layout="wide")


@st.cache_data
def load(name, index_col=None):
    p = PROC / name
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=index_col, parse_dates=bool(index_col is not None and index_col == 0))


def need(df, hint):
    if df is None:
        st.info(f"Data not found. {hint}")
        return False
    return True


def show_fig(name, caption=None):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


diag_path = PROC / "factor_diagnostics.json"
diag = json.load(open(diag_path)) if diag_path.exists() else {}
cfg_path = PROC / "preprocess_config.json"
cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
HORIZONS = cfg.get("horizons", [1, 3, 6, 12])

st.sidebar.title("Commodity inflation")
page = st.sidebar.radio("Page", ["Overview", "Factors", "Forecast performance",
                                 "What drives inflation", "Written summary"])
st.sidebar.markdown("---")
if diag:
    st.sidebar.metric("Factors (K)", diag.get("n_factors"))
    st.sidebar.metric("Variance explained", f"{diag.get('cum_variance_pct', '?')}%")
    st.sidebar.metric("KMO", diag.get("kmo_overall"))
st.sidebar.caption("Statistical research output, not investment advice.")

# ----------------------------------------------------------------------
if page == "Overview":
    st.title("Commodity inflation: factor-based forecasts")
    st.write("Latest forecast of cumulative commodity price change, and what is driving it.")

    lf = load("latest_forecast.csv")
    contrib = load("current_forecast_contributions.csv", index_col=0)
    if need(lf, "Run Phase 5 (`python run_all.py --only 5`)."):
        sub = lf[lf["model"] == "FAR-Ridge"].sort_values("horizon_months")
        if len(sub):
            base = sub["base_month"].iloc[0]
            st.caption(f"Forecasts made from data up to **{base}**")
            cols = st.columns(len(sub))
            for c, (_, r) in zip(cols, sub.iterrows()):
                c.metric(f"Next {int(r['horizon_months'])} months",
                         f"{r['forecast_pct_change']:+.1f}%")
        st.subheader("All models")
        st.dataframe(lf.pivot_table(index="model", columns="horizon_months",
                                    values="forecast_pct_change").round(2), use_container_width=True)

    if need(contrib, "Run Phase 7."):
        st.subheader("What is driving the current forecast?")
        h = st.selectbox("Horizon (months)", [int(i) for i in contrib.index], key="h_over")
        s = contrib.loc[h].drop(["forecast"])
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                               marker_color=np.where(s.values >= 0, "#c0392b", "#2980b9")))
        fig.update_layout(height=380, xaxis_title="Contribution (percentage points)",
                          title=f"Forecast = {contrib.loc[h, 'forecast']:+.1f}% over {h} months")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red pushes prices up, blue pushes them down. 'AR lags' is momentum in recent price changes.")

# ----------------------------------------------------------------------
elif page == "Factors":
    st.title("Latent factors")
    scores = load("factors_all.csv", index_col=0)
    load_ = load("factor_loadings.csv", index_col=0)
    var = load("factor_variance.csv", index_col=0)
    labels = diag.get("factor_labels", {})
    if need(var, "Run Phase 4."):
        st.subheader("Factor summary")
        st.dataframe(var.round(2), use_container_width=True)
    if need(load_, "Run Phase 4."):
        st.subheader("Loadings (which indicators define each factor)")
        fig = px.imshow(load_.round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=True, aspect="auto")
        fig.update_xaxes(ticktext=[labels.get(c, c) for c in load_.columns],
                         tickvals=list(range(len(load_.columns))))
        fig.update_layout(height=max(400, 22 * len(load_)))
        st.plotly_chart(fig, use_container_width=True)
    if need(scores, "Run Phase 4."):
        st.subheader("Factor scores over time")
        pick = st.multiselect("Factors", list(scores.columns), default=list(scores.columns)[:3],
                              format_func=lambda c: labels.get(c, c))
        if pick:
            d = scores[pick].rename(columns=lambda c: labels.get(c, c))
            fig = px.line(d)
            for s, e, name in [("2008-09-01", "2009-06-01", "GFC"), ("2020-03-01", "2020-06-01", "COVID"),
                               ("2022-02-01", "2022-12-01", "Energy shock")]:
                fig.add_vrect(x0=s, x1=e, opacity=0.15, line_width=0, annotation_text=name)
            fig.update_layout(height=450, legend_title="")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
elif page == "Forecast performance":
    st.title("Forecast performance")
    h = st.selectbox("Horizon (months)", HORIZONS)
    fc = load(f"forecasts_h{h}.csv", index_col=0)
    cv = load(f"cv_forecasts_h{h}.csv", index_col=0)
    src = st.radio("Sample", ["Phase 5 test set", "Leakage-free CV (Phase 6)"], horizontal=True)
    df = fc if src.startswith("Phase 5") else cv
    if need(df, "Run Phases 5 and 6."):
        models = [c for c in df.columns if c != "actual"]
        default = [m for m in ["AR (OLS)", "FAR-Ridge"] if m in models]
        pick = st.multiselect("Models to plot", models, default=default)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["actual"], name="Actual",
                                 line=dict(color="black", width=2)))
        for m in pick:
            fig.add_trace(go.Scatter(x=df.index, y=df[m], name=m))
        fig.update_layout(height=450, yaxis_title=f"% change over next {h} months")
        st.plotly_chart(fig, use_container_width=True)

    ev = load("results_eval_all.csv")
    if need(ev, "Run Phase 6."):
        st.subheader("Is the factor model better than the AR benchmark?")
        e = ev[ev["horizon"] == h][["sample", "model", "n", "RMSE", "rel_RMSE_vs_AR", "CI95_low",
                                    "CI95_high", "ClarkWest_p", "DM_p"]]
        st.dataframe(e.round(3), use_container_width=True, hide_index=True)
        st.caption("Relative RMSE below 1 means better than AR. A 95% interval that excludes 1 and "
                   "Clark-West p < 0.10 indicate a robust improvement.")
    c1, c2 = st.columns(2)
    with c1:
        show_fig("17_relative_rmse_heatmap.png")
    with c2:
        show_fig("21_rolling_relative_rmse.png")
    show_fig("22_K_sensitivity.png")

# ----------------------------------------------------------------------
elif page == "What drives inflation":
    st.title("What drives commodity inflation?")
    perm = load("perm_importance.csv")
    fevd = load("fevd_inflation.csv", index_col=0)
    if need(perm, "Run Phase 7."):
        st.subheader("Permutation importance (out-of-sample)")
        m = st.radio("Model", sorted(perm["model"].unique()), horizontal=True)
        t = perm[perm["model"] == m].pivot(index="feature", columns="horizon", values="delta_rmse")
        fig = px.imshow(t.round(3), color_continuous_scale="Reds", text_auto=True, aspect="auto",
                        labels=dict(color="RMSE increase"))
        st.plotly_chart(fig, use_container_width=True)
    if need(fevd, "Run Phase 7."):
        st.subheader("Share of forecast-error variance (FAVAR)")
        fig = px.bar(fevd, barmode="stack", orientation="v")
        fig.update_layout(yaxis_title="%", xaxis_title="Horizon", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        show_fig("27_factor_coefficients.png")
    with c2:
        show_fig("25_shap_summary.png")
    show_fig("18_favar_irf.png")

# ----------------------------------------------------------------------
else:
    st.title("Written summary")
    if REPORT.exists():
        st.markdown(REPORT.read_text(encoding="utf-8"))
        st.download_button("Download summary (.md)", REPORT.read_text(encoding="utf-8"),
                           file_name="results_summary.md")
    else:
        st.info("Run Phase 7 to generate reports/results_summary.md")