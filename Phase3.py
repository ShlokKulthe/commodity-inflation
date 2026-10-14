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