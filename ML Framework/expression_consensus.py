# -*- coding: utf-8 -*-
"""
Step 04b: Primary cohort expression-only ranking (no FBA)
=============================================================
Runs the SAME multi-method consensus framework as Step 04
(PCA, LDA, Lasso, RF+XGBoost+SHAP) but on raw gene expression
instead of GEM-derived flux. This is a fully independent,
FBA-free check: if the same SLC genes emerge as important
from raw expression alone, that strongly supports the flux-
based conclusions are not artifacts of GEM/FBA modeling
choices.

Also runs a quick Wilcoxon DE test per condition vs reference
(Normal) restricted to SLC genes, as a simple sanity check
before the full consensus analysis.

Input: bootstrap expression CSVs from Step 00
  (same *_boot_means.csv files used to drive Step 01 GPR
   scaling -- here used directly as features, no GEM/FBA)

Outputs -> outputs/moduleA_expression/:
  de_wilcoxon_slc.csv
  importance_consensus_expression.csv
  pca_variance_expression.pdf
  pca_scatter_expression.pdf
  lda_scatter_expression.pdf

Usage:
  python scripts/step03_moduleA/04b_expression_consensus.py
HPC:
  sbatch slurm/04b_expression_consensus.sh
"""

import os
import re
import csv
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from scipy.stats import mannwhitneyu
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# =============================================================
#  CONFIGURATION
# =============================================================

# Bootstrap expression files from Step 00, one per condition
# Columns = bootstrap iterations, rows = genes (gene symbols)
BOOT_DIR     = "data/bootstrap_expression"
GPR_CSV      = "GPR2.csv"
REFERENCE_COND = "Normal"   # DE comparison baseline

CONDITION_MAP = {
    "Normal": "Normal",
    "HSCT":   "HSCT",
    "GCSF":   "GCSF",
    "PDAC":   "PDAC",
}

OUTPUT_DIR   = "outputs/moduleA_expression"
N_SPLITS     = 4
N_TREES      = 500
RANDOM_STATE = 42
TOP_N        = 100

# =============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
#  Load bootstrap expression
# =============================================================

def load_bootstrap_expression():
    """
    Load *_boot_means.csv for each condition.
    Returns:
      X: rows = bootstrap samples (100 x n_conditions),
         cols = genes (symbol)
      y: condition label per row
    """
    records = []
    labels  = []

    for cond in CONDITION_MAP:
        pattern = os.path.join(
            BOOT_DIR, f"*{cond}*boot_means.csv"
        )
        files = glob.glob(pattern)
        if not files:
            print(f"  WARNING: no boot file for {cond}")
            continue

        df = pd.read_csv(files[0], index_col="gene")
        # rows=genes, cols=bootstrap iters -> transpose
        df_t = df.T
        print(
            f"  {cond}: {df_t.shape[0]} bootstrap samples, "
            f"{df_t.shape[1]} genes"
        )

        for _, row in df_t.iterrows():
            records.append(row.to_dict())
            labels.append(cond)

    X = pd.DataFrame(records).fillna(0)
    y = pd.Series(labels, name="condition")
    print(f"\nExpression matrix: {X.shape}")
    print(f"Conditions: {y.value_counts().to_dict()}")
    return X, y


def get_slc_genes_from_gpr(gpr_csv_path):
    """Return set of gene symbols starting with SLC."""
    slc_genes = set()
    with open(gpr_csv_path, "r", encoding="utf-8-sig",
              newline="") as f:
        sample = f.read(1024); f.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            gpr = row.get("GPR", "").strip()
            if not gpr:
                continue
            genes = re.split(
                r"\band\b|\bor\b", gpr, flags=re.IGNORECASE
            )
            for g in genes:
                g = re.sub(r"[^A-Za-z0-9]", "", g.strip())
                if g.upper().startswith("SLC"):
                    slc_genes.add(g)
    print(f"  SLC genes in GPR: {len(slc_genes)}")
    return slc_genes


# =============================================================
#  Quick DE check (Wilcoxon, SLC genes only)
# =============================================================

def run_de_wilcoxon(X, y, slc_genes):
    print("\n--- Quick DE check (Wilcoxon, SLC genes) ---")
    expr_genes_upper = {
        c.upper(): c for c in X.columns
    }
    matched_slc = [
        expr_genes_upper[g.upper()]
        for g in slc_genes
        if g.upper() in expr_genes_upper
    ]
    print(
        f"  SLC genes matched in expression data: "
        f"{len(matched_slc)} / {len(slc_genes)}"
    )

    rows = []
    other_conds = [
        c for c in CONDITION_MAP if c != REFERENCE_COND
    ]
    ref_mask = (y == REFERENCE_COND)

    for cond in other_conds:
        cond_mask = (y == cond)
        for gene in matched_slc:
            ref_vals  = X.loc[ref_mask, gene].values
            cond_vals = X.loc[cond_mask, gene].values
            if len(ref_vals) < 3 or len(cond_vals) < 3:
                continue
            try:
                stat, pval = mannwhitneyu(
                    cond_vals, ref_vals,
                    alternative="two-sided"
                )
            except ValueError:
                continue
            log2fc = np.log2(
                (cond_vals.mean() + 1e-9)
                / (ref_vals.mean() + 1e-9)
            )
            rows.append({
                "gene":       gene,
                "comparison": f"{cond}_vs_{REFERENCE_COND}",
                "log2fc":     log2fc,
                "p_value":    pval,
            })

    de_df = pd.DataFrame(rows)
    if len(de_df) > 0:
        de_df["abs_log2fc"] = de_df["log2fc"].abs()
        de_df = de_df.sort_values(
            "p_value", ascending=True
        ).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, "de_wilcoxon_slc.csv")
    de_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    if len(de_df) > 0:
        print("  Top 10 DE SLC genes (by p-value):")
        print(de_df.head(10)[
            ["gene", "comparison", "log2fc", "p_value"]
        ].to_string(index=False))
    return de_df


# =============================================================
#  Full consensus framework on expression
#  (mirrors Step 04 exactly, X = expression not flux)
# =============================================================

def annotate(df, slc_genes, feature_col="feature"):
    df["is_slc"] = (
        df[feature_col].str.upper().isin(
            {g.upper() for g in slc_genes}
        )
    )
    return df


def run_pca(X_scaled, y, feature_names):
    print("\n--- PCA (expression) ---")
    n_components = min(10, X_scaled.shape[0],
                       X_scaled.shape[1])
    pca = PCA(n_components=n_components,
              random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    ev     = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(ev)+1), ev * 100)
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA (expression): variance explained")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pca_variance_expression.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    conditions = y.unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for cond, col in zip(conditions, colors):
        mask = y == cond
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cond, alpha=0.6, s=20, color=col)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA (expression): bootstrap samples")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pca_scatter_expression.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%")

    loadings = pd.DataFrame({
        "feature":     feature_names,
        "pc1_loading": np.abs(pca.components_[0]),
        "pc2_loading": np.abs(pca.components_[1]),
    })
    loadings["pca_score"] = (
        loadings["pc1_loading"] * ev[0]
        + loadings["pc2_loading"] * ev[1]
    )
    loadings = loadings.sort_values(
        "pca_score", ascending=False
    ).reset_index(drop=True)
    return loadings


def run_lda(X_scaled, y, feature_names):
    print("\n--- LDA (expression) ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_comp_requested = min(
        len(le.classes_) - 1, X_scaled.shape[1]
    )

    lda    = LinearDiscriminantAnalysis(
        n_components=n_comp_requested
    )
    coords = lda.fit_transform(X_scaled, y_enc)

    # IMPORTANT: sklearn can silently return FEWER columns
    # than n_components requested if the between-class
    # scatter matrix is rank-deficient (e.g. class means
    # are collinear). 
    n_comp_actual = coords.shape[1]
    print(
        f"  Requested n_components={n_comp_requested}, "
        f"actual rank={n_comp_actual} "
        f"(explained_variance_ratio_={lda.explained_variance_ratio_})"
    )
    if n_comp_actual < n_comp_requested:
        print(
            f"  NOTE: LDA found only {n_comp_actual} "
            f"non-trivial discriminant direction(s) -- "
            f"the between-class scatter is rank-deficient. "
            f"This typically means class means are nearly "
            f"collinear (e.g. disease conditions differ from "
            f"the reference along a shared axis rather than "
            f"independent directions)."
        )

    conditions = le.classes_
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for cond, col in zip(conditions, colors):
        mask = y == cond
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1] if n_comp_actual > 1
            else np.zeros(mask.sum()),
            label=cond, alpha=0.6, s=20, color=col
        )
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2" if n_comp_actual > 1 else "(rank-deficient: 1D only)")
    ax.set_title("LDA (expression): bootstrap samples")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "lda_scatter_expression.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    coef_abs = np.abs(lda.coef_).mean(axis=0)
    importance = pd.DataFrame({
        "feature":   feature_names,
        "lda_score": coef_abs,
    }).sort_values("lda_score", ascending=False
    ).reset_index(drop=True)
    return importance


def run_lasso(X_scaled, y, feature_names):
    print("\n--- Lasso (expression) ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    lasso = LogisticRegressionCV(
        Cs=10, cv=skf, penalty="l1", solver="saga",
        multi_class="multinomial", max_iter=2000,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lasso.fit(X_scaled, y_enc)
    coef_abs = np.abs(lasso.coef_).mean(axis=0)
    importance = pd.DataFrame({
        "feature":     feature_names,
        "lasso_score": coef_abs,
    }).sort_values("lasso_score", ascending=False
    ).reset_index(drop=True)
    n_nonzero = (coef_abs > 0).sum()
    print(f"  Non-zero Lasso coefficients: {n_nonzero}")
    return importance


def run_shap(X_scaled, y, feature_names):
    print("\n--- RF + XGBoost + SHAP (expression) ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators=N_TREES, random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=N_TREES, random_state=RANDOM_STATE,
        eval_metric="mlogloss", use_label_encoder=False,
        n_jobs=-1,
    )

    for name, clf in [("RF", rf), ("XGBoost", xgb_clf)]:
        scores = cross_val_score(
            clf, X_scaled, y_enc, cv=skf, scoring="accuracy"
        )
        print(f"  {name} CV accuracy: "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")

    rf.fit(X_scaled, y_enc)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_scaled)

    # Handle multiclass SHAP shape robustly (list of arrays,
    # 3D array, or 2D array depending on SHAP/sklearn version)
    if isinstance(shap_vals, list):
        mean_abs = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_vals],
            axis=0
        )
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_abs = np.asarray(mean_abs).ravel()
    print(
        f"  mean_abs shape: {mean_abs.shape}, "
        f"n_features: {len(feature_names)}"
    )

    importance = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    return importance


def consensus_ranking(pca_imp, lda_imp, lasso_imp,
                      shap_imp, slc_genes):
    print("\n--- Consensus Ranking (expression) ---")

    def rank_series(df, score_col):
        df = df.copy()
        df["rank"] = df[score_col].rank(ascending=False)
        return df.set_index("feature")["rank"]

    r_pca   = rank_series(pca_imp,   "pca_score")
    r_lda   = rank_series(lda_imp,   "lda_score")
    r_lasso = rank_series(lasso_imp, "lasso_score")
    r_shap  = rank_series(shap_imp,  "mean_abs_shap")

    all_feats = (
        set(r_pca.index) | set(r_lda.index)
        | set(r_lasso.index) | set(r_shap.index)
    )
    n = len(all_feats)

    consensus = pd.DataFrame({
        "feature":    list(all_feats),
        "rank_pca":   [r_pca.get(f, n)   for f in all_feats],
        "rank_lda":   [r_lda.get(f, n)   for f in all_feats],
        "rank_lasso": [r_lasso.get(f, n) for f in all_feats],
        "rank_shap":  [r_shap.get(f, n)  for f in all_feats],
    })
    consensus["mean_rank"] = consensus[
        ["rank_pca", "rank_lda", "rank_lasso", "rank_shap"]
    ].mean(axis=1)
    consensus = consensus.sort_values(
        "mean_rank"
    ).reset_index(drop=True)
    consensus = annotate(consensus, slc_genes)

    out = os.path.join(
        OUTPUT_DIR, "importance_consensus_expression.csv"
    )
    consensus.to_csv(out, index=False)

    top_slc = consensus.head(TOP_N).query("is_slc == True")
    print(f"\n  Top 20 consensus genes (expression-only):")
    print(consensus.head(20)[
        ["feature", "mean_rank", "is_slc"]
    ].to_string(index=False))
    print(
        f"\n  SLC genes in top {TOP_N}: {len(top_slc)}"
    )
    print(f"  Saved: {out}")
    return consensus


# =============================================================
#  MAIN
# =============================================================

def main():
    print("=" * 60)
    print("Step 04b: Primary cohort expression-only check (no FBA)")
    print("=" * 60)

    X, y = load_bootstrap_expression()
    slc_genes = get_slc_genes_from_gpr(GPR_CSV)

    # 1. Quick DE check
    run_de_wilcoxon(X, y, slc_genes)

    # 2. Full consensus framework on expression
    feature_names = list(X.columns)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_imp   = run_pca(X_scaled, y, feature_names)
    lda_imp   = run_lda(X_scaled, y, feature_names)
    lasso_imp = run_lasso(X_scaled, y, feature_names)
    shap_imp  = run_shap(X_scaled, y, feature_names)

    consensus_ranking(
        pca_imp, lda_imp, lasso_imp, shap_imp, slc_genes
    )

    print("\nStep 04b complete.")
    print(
        "Compare outputs/moduleA_expression/"
        "importance_consensus_expression.csv against "
        "outputsFVA/moduleA/importance_consensus.csv "
        "(flux-based) for triangulation."
    )


if __name__ == "__main__":
    main()