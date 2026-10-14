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