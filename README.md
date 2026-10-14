# Decoding Commodity Inflation: A Factor-Based Approach

Identify the latent drivers of global commodity prices with factor analysis, then test whether
those factors improve forecasts of commodity inflation at 1, 3, 6 and 12 months.

## Research questions
1. How many latent factors explain most co-movement in commodity prices?
2. Which factor matters most for energy, metals and agricultural inflation?
3. Do factor-augmented models beat univariate benchmarks (naive, AR, ARIMA), and at which horizons?
4. Is the factor structure stable, or did it change in crises (2008, COVID, 2022 energy shock)?

## Pipeline

| Phase | Script | What it does | Key outputs |
|---|---|---|---|
| 1 | `01_data_collection.py` | Downloads 30 years of monthly data (FRED, Yahoo Finance, optional World Bank Pink Sheet) | `data/processed/monthly_merged.csv` |
| 2 | `02_eda.py` | EDA, seasonality, ADF/KPSS stationarity tests, correlation structure | `stationarity_report.csv`, figures 01-09 |
| 3 | `03_preprocessing.py` | Stationary transforms, publication lags, train/test split with embargo, train-only scaling | `X_*_scaled.csv`, `targets.csv` |
| 4 | `04_factor_extraction.py` | KMO/Bartlett, number of factors, varimax Factor Analysis, stability checks | `factors_all.csv`, `factor_loadings.csv` |
| 5 | `05_forecasting.py` | Rolling-origin forecasts: naive, mean, AR, ARIMA, FAR (OLS/Ridge/Lasso/RF/XGB), FAVAR | `forecasts_h*.csv`, `results_metrics.csv` |
| 6 | `06_evaluation_robustness.py` | Leakage-free long-sample CV, Clark-West and Diebold-Mariano tests, bootstrap CIs, sensitivity to K | `results_eval_all.csv` |
| 7 | `07_interpretation.py` | Permutation importance, SHAP, HAC regressions, FEVD, drivers of the latest forecast, draft report | `reports/results_summary.md` |
| 8 | `run_all.py`, `app.py` | One-command pipeline and Streamlit dashboard | |

## Quick start

```bash
# 1. Environment (Python 3.10-3.12 recommended)
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. FRED API key (free: https://fred.stlouisfed.org -> My Account -> API Keys)
#    Create a file named .env next to run_all.py containing:
#    FRED_API_KEY=your32characterkey

# 3. Run everything
python run_all.py

# 4. Explore
streamlit run app.py
```

Useful options: `python run_all.py --skip-download`, `--from 4`, `--only 5`, `--from 3 --to 5`.
Each phase writes a log to `logs/`.

## Project structure

```
commodity-inflation/
├── 01_data_collection.py ... 07_interpretation.py
├── run_all.py
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/            untouched downloads
│   └── processed/      cleaned data, factors, forecasts, result tables
├── models/             fitted scaler and factor model
├── reports/
│   ├── figures/        all charts
│   └── results_summary.md
└── logs/
```

## Methodology in brief
- **Data:** monthly commodity indices and prices, US macro, currency, financial-risk series.
- **Stationarity:** ADF + KPSS decide between levels, differences and log differences.
- **Leakage control:** scaling and factor models are fitted on training data only; a 12-month
  embargo separates train and test; publication lags are applied to macro series; target-related
  indices are excluded from the factor inputs.
- **Factors:** Factor Analysis with varimax rotation; number of factors from parallel analysis,
  Kaiser and Bai-Ng criteria.
- **Forecasting target:** cumulative % change of the commodity index over the next h months.
- **Evaluation:** rolling-origin (expanding window); RMSE, MAE, out-of-sample R2, directional
  accuracy; Diebold-Mariano and Clark-West tests; block-bootstrap confidence intervals;
  crisis vs calm comparison.

## Results
> Fill in from `reports/results_summary.md` after running the pipeline. Suggested table:

| Horizon | Rel. RMSE vs AR | 95% CI | Clark-West p | Verdict |
|---|---|---|---|---|
| 1m | | | | |
| 3m | | | | |
| 6m | | | | |
| 12m | | | | |

## Limitations
- Short test window; overlapping multi-month targets make errors autocorrelated.
- Factor labels are interpretive and loadings can drift across regimes.
- Commodity data are revised and published with a delay, so real-time accuracy may be lower.
- Macro inputs are US-centric; emerging-market channels (e.g. India's rupee, subsidies) are not modelled.
- For research and education only; not investment advice.

## Data sources
FRED (St. Louis Fed), including IMF commodity series; Yahoo Finance; World Bank Commodity Price
Data (Pink Sheet, optional).

## License
MIT (add a LICENSE file if you publish the repository).
