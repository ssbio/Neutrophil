# -*- coding: utf-8 -*-
"""
Step 04: Multi-method flux importance analysis
==============================================
Runs four complementary methods on bootstrap flux matrix:
  1. PCA           - variance decomposition + condition separation
  2. LDA           - linear discriminant analysis, loadings as importance
  3. Lasso LR      - sparse logistic regression coefficients
  4. RF + XGBoost  - nonlinear ensemble + SHAP values

Consensus ranking: reactions scoring high across all methods
are the strongest candidates. All features annotated with
SLC status from slc_annotation.csv.

Outputs -> outputs/moduleA/:
  pca_variance.pdf
  pca_scatter.pdf
  lda_scatter.pdf
  importance_lda.csv
  importance_lasso.csv
  importance_shap.csv
  importance_consensus.csv
  top_slc_candidates_for_moduleB.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================================
#  CONFIGURATION
# =============================================================

INPUT_DIR    = ".../outputs/moduleA"
OUTPUT_DIR   = ".../outputs/moduleA2"
N_SPLITS     = 4   # StratifiedKFold (one per condition)
N_TREES      = 500
RANDOM_STATE = 42
TOP_N        = 100  # reactions to include in consensus

# =============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    x_path = os.path.join(INPUT_DIR, "X_features.csv")
    if not os.path.exists(x_path):
        x_path = os.path.join(INPUT_DIR, "X_slc_features.csv")
    X = pd.read_csv(x_path, index_col=0)

    y_df = pd.read_csv(os.path.join(INPUT_DIR, "y_labels.csv"))
    if "condition" in y_df.columns:
        y = y_df["condition"]
    else:
        y = y_df.iloc[:, -1]
    y = y.reset_index(drop=True)

    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"Conditions: {y.value_counts().to_dict()}")
    return X, y


def load_slc_annotation():
    path = os.path.join(INPUT_DIR, "slc_annotation.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)["is_slc"]
    print("  WARNING: slc_annotation.csv not found.")
    return None


def annotate(df, slc_annot, feature_col="feature"):
    """Add is_slc column to an importance DataFrame."""
    if slc_annot is not None:
        df["rxn_id"] = df[feature_col].str.split("__").str[0]
        df["is_slc"] = (
            df["rxn_id"].map(slc_annot).fillna(False).astype(bool)
        )
    else:
        df["is_slc"] = False
    return df


# =============================================================
#  1. PCA
# =============================================================

def run_pca(X_scaled, y, feature_names):
    print("\n--- PCA ---")
    n_components = min(10, X_scaled.shape[0],
                       X_scaled.shape[1])
    pca = PCA(n_components=n_components,
              random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    ev     = pca.explained_variance_ratio_

    # Variance plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(ev)+1), ev * 100)
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA: variance explained")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter PC1 vs PC2
    conditions = y.unique()
    colors     = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    fig, ax    = plt.subplots(figsize=(7, 6))
    for cond, col in zip(conditions, colors):
        mask = y == cond
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cond, alpha=0.6, s=20, color=col)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA: bootstrap flux samples")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%")

    # PC1 loadings as feature importance
    loadings = pd.DataFrame({
        "feature":    feature_names,
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
    print(f"  Top PCA feature: {loadings.iloc[0]['feature']}")
    return loadings, pca, coords


# =============================================================
#  2. LDA
# =============================================================

def run_lda(X_scaled, y, feature_names):
    print("\n--- LDA ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_comp = min(len(le.classes_) - 1, X_scaled.shape[1])

    lda    = LinearDiscriminantAnalysis(n_components=n_comp)
    coords = lda.fit_transform(X_scaled, y_enc)

    # Scatter
    conditions = le.classes_
    colors     = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    fig, ax    = plt.subplots(figsize=(7, 6))
    for cond, col in zip(conditions, colors):
        mask = y == cond
        ax.scatter(coords[mask, 0],
                   coords[mask, 1] if n_comp > 1
                   else np.zeros(mask.sum()),
                   label=cond, alpha=0.6, s=20, color=col)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2" if n_comp > 1 else "")
    ax.set_title("LDA: bootstrap flux samples")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lda_scatter.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # LDA coefficients as importance
    coef_abs = np.abs(lda.coef_).mean(axis=0)
    importance = pd.DataFrame({
        "feature":    feature_names,
        "lda_score":  coef_abs,
    }).sort_values("lda_score", ascending=False
    ).reset_index(drop=True)

    importance.to_csv(
        os.path.join(OUTPUT_DIR, "importance_lda.csv"),
        index=False
    )
    print(f"  Top LDA feature: {importance.iloc[0]['feature']}")
    return importance


# =============================================================
#  3. Lasso Logistic Regression
# =============================================================

def run_lasso(X_scaled, y, feature_names):
    print("\n--- Lasso Logistic Regression ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                           random_state=RANDOM_STATE)
    lasso = LogisticRegressionCV(
        Cs=10,
        cv=skf,
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lasso.fit(X_scaled, y_enc)

    # Mean abs coefficient across classes
    coef_abs = np.abs(lasso.coef_).mean(axis=0)
    importance = pd.DataFrame({
        "feature":     feature_names,
        "lasso_score": coef_abs,
    }).sort_values("lasso_score", ascending=False
    ).reset_index(drop=True)

    n_nonzero = (coef_abs > 0).sum()
    print(f"  Non-zero Lasso coefficients: {n_nonzero}")
    print(f"  Top Lasso feature: {importance.iloc[0]['feature']}")
    importance.to_csv(
        os.path.join(OUTPUT_DIR, "importance_lasso.csv"),
        index=False
    )
    return importance


# =============================================================
#  4. RF + XGBoost + SHAP
# =============================================================

def run_shap(X_scaled, y, feature_names):
    print("\n--- RF + XGBoost + SHAP ---")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
    )

    for name, clf in [("RF", rf), ("XGBoost", xgb_clf)]:
        scores = cross_val_score(
            clf, X_scaled, y_enc, cv=skf,
            scoring="accuracy"
        )
        print(f"  {name} CV accuracy: "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")

    # Fit RF on full data for SHAP
    rf.fit(X_scaled, y_enc)
    explainer  = shap.TreeExplainer(rf)
    shap_vals  = explainer.shap_values(X_scaled)

    # Mean abs SHAP across classes and samples.
    # Handles three possible shapes depending on SHAP version:
    #   - list of (n_samples, n_features) arrays, one per class
    #   - 3D array (n_samples, n_features, n_classes)
    #   - 2D array (n_samples, n_features) for binary case
    if isinstance(shap_vals, list):
        # List of per-class arrays
        per_class_mean = [
            np.abs(sv).mean(axis=0) for sv in shap_vals
        ]
        mean_abs = np.mean(per_class_mean, axis=0)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        # 2D (n_samples, n_features)
        mean_abs = np.abs(shap_vals).mean(axis=0)

    mean_abs = np.asarray(mean_abs).ravel()
    print(f"  mean_abs shape: {mean_abs.shape}, "
          f"n_features: {len(feature_names)}")

    importance = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    importance.to_csv(
        os.path.join(OUTPUT_DIR, "importance_shap.csv"),
        index=False
    )
    print(f"  Top SHAP feature: {importance.iloc[0]['feature']}")
    return importance


# =============================================================
#  5. Consensus ranking
# =============================================================

def consensus_ranking(pca_imp, lda_imp, lasso_imp,
                      shap_imp, slc_annot):
    print("\n--- Consensus Ranking ---")

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
        ["rank_pca", "rank_lda",
         "rank_lasso", "rank_shap"]
    ].mean(axis=1)
    consensus = consensus.sort_values(
        "mean_rank"
    ).reset_index(drop=True)

    # Annotate SLC
    consensus = annotate(consensus, slc_annot)

    consensus.to_csv(
        os.path.join(OUTPUT_DIR, "importance_consensus.csv"),
        index=False
    )

    # Top SLC candidates from consensus top N
    top_slc = (
        consensus.head(TOP_N).query("is_slc == True")
    )
    top_slc.to_csv(
        os.path.join(OUTPUT_DIR,
                     "top_slc_candidates_for_moduleB.csv"),
        index=False
    )

    print(f"  Top 20 consensus features:")
    print(consensus.head(20)[
        ["feature", "mean_rank", "is_slc"]
    ].to_string(index=False))
    print(
        f"\n  SLC reactions in top {TOP_N}: "
        f"{len(top_slc)}"
    )
    return consensus


# =============================================================
#  MAIN
# =============================================================

def main():
    X, y        = load_data()
    slc_annot   = load_slc_annotation()
    feature_names = list(X.columns)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_imp, _, _  = run_pca(X_scaled, y, feature_names)
    lda_imp        = run_lda(X_scaled, y, feature_names)
    lasso_imp      = run_lasso(X_scaled, y, feature_names)
    shap_imp       = run_shap(X_scaled, y, feature_names)

    consensus = consensus_ranking(
        pca_imp, lda_imp, lasso_imp, shap_imp, slc_annot
    )

    print("\nStep 04 complete.")
    print("Next: sbatch slurm/06_perturbation_screen.sh")


if __name__ == "__main__":
    main()