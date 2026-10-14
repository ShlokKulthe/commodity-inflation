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
