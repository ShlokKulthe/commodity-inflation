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