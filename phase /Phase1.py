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
