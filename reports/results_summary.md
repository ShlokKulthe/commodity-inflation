# Results summary (auto-generated draft - review and edit every statement)

## 1. Data and method
- Sample: 2000-09-01 to 2026-09-01 (313 monthly observations); 5 factors extracted from 28 stationary indicators.
- Target: cumulative % change of `all_commodity_index` over the next 1, 3, 6, 12 months.
- Train: 2000-09-01 to 2019-03-01; test: 2020-04-01 to 2026-10-01; 12-month embargo between them.
- Suitability: KMO = 0.67, Bartlett p = 0.
- Number of factors: K = 5 (Kaiser 10, parallel analysis 5, Bai-Ng 2); cumulative variance explained 38.4%.

## 2. Factor interpretation
- **F1: Energy** - top loadings: brent_oil, wti_oil, us_ppi, us_cpi; explains 11.7% of total variance.
- **F2: US dollar / currency** - top loadings: usd_index, usd_per_eur, gold_futures, usd_inr; explains 9.5% of total variance.
- **F3: Financial risk / sentiment** - top loadings: vix, fed_funds_rate, sp500, us_ind_production; explains 7.5% of total variance.
- **F4: Agriculture** - top loadings: maize, wheat, cotton, corn_futures; explains 5.4% of total variance.
- **F5: Monetary / inflation** - top loadings: us_ppi, us_cpi, nat_gas_eu, aluminum; explains 4.3% of total variance.

## 3. Forecast performance (leakage-free rolling-origin CV, FAR-Ridge vs AR)
- h=1m: rel. RMSE 0.984 [0.943, 1.025], Clark-West p=0.014
- h=3m: rel. RMSE 0.927 [0.872, 0.961], Clark-West p=0.000
- h=6m: rel. RMSE 0.882 [0.810, 0.967], Clark-West p=0.000
- h=12m: rel. RMSE 0.842 [0.696, 1.034], Clark-West p=0.021

## 4. What matters, and when
- h=1m: most important factor (permutation, Ridge) = F4: Agriculture; (boosting) = F1: Energy
- h=3m: most important factor (permutation, Ridge) = F4: Agriculture; (boosting) = F4: Agriculture
- h=6m: most important factor (permutation, Ridge) = F4: Agriculture; (boosting) = F4: Agriculture
- h=12m: most important factor (permutation, Ridge) = F4: Agriculture; (boosting) = F3: Financial risk / sentiment
- FEVD at 12 months: F1: Energy 1%, F2: US dollar / currency 8%, F3: Financial risk / sentiment 14%, F4: Agriculture 16%, F5: Monetary / inflation 2%, Own shocks (inflation) 60%

## 5. Hypotheses (automatic first pass - verify against figures)
- **H1** (3-5 factors explain most co-movement): Not supported
- **H2** (energy/demand dominate short horizons, dollar/monetary long): Not supported - F4: Agriculture dominates at all horizons
- **H3** (factor models beat AR benchmarks): Supported at some horizons
- **H4** (instability in crises): Supported - relative RMSE is 0.79 in crises vs 1.01 in calm periods

## 6. Latest forecast
- Over the next 1 months from 2026-09-01: +1.2% (FAR-Ridge)
- Over the next 3 months from 2026-09-01: +2.3% (FAR-Ridge)
- Over the next 6 months from 2026-09-01: +4.0% (FAR-Ridge)
- Over the next 12 months from 2026-09-01: +6.7% (FAR-Ridge)
- Current drivers: see `reports/figures/29_current_forecast_drivers.png`.

## 7. Limitations
- Short test window and overlapping multi-month targets: treat small accuracy gains cautiously.
- Factor labels are interpretive; loadings can change across regimes.
- IMF/FRED monthly commodity data are revised and published with delay; real-time performance may differ.
- Macro inputs are US-centric. Emerging-market effects (e.g. India's rupee and subsidies) are not modelled.
- Forecasts are statistical estimates for research use, not investment advice.