# -*- coding: utf-8 -*-
"""
Step 03b: Build validation cohort feature matrix
==================================================
Same logic as Step 03, applied to the GSE167363 validation
cohort's bootstrap flux output (Step 01b), using the same
fixed Normal GEM topology across Healthy/Survivor/Fatal.

Final X: rows = bootstrap samples (300 total = 100 x 3 conds)
         cols = reactions (raw flux) + reaction CI stats
Final y: condition label per row (Healthy/Survivor/Fatal)

Also saves slc_annotation.csv marking SLC-linked reactions
(same GPR2.csv used for the primary cohort -- GPR mapping
is cohort-independent, it's a property of the GEM/genes,
not the expression data).

Outputs -> outputs/validation/GSE167363/moduleA/:
  X_features.csv
  y_labels.csv
  slc_annotation.csv
"""

import os
import re
import csv
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
#  CONFIGURATION
# =============================================================

DIST_DIR   = "..data/validation/GSE167363/flux_uncertainty"
CI_DIR     = "..data/validation/GSE167363/flux_uncertainty"
OUTPUT_DIR = "..outputs/validation/GSE167363/moduleA"
GPR_CSV    = "GPR2.csv"

CONDITION_MAP = {
    "Healthy":  "Healthy",
    "Survivor": "Survivor",
    "Fatal":    "Fatal",
}

# =============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_slc_ids_from_gpr(gpr_csv_path, all_rxn_ids):
    """Return set of reaction IDs with SLC genes in GPR."""
    slc_rxns = set()
    with open(gpr_csv_path, "r", encoding="utf-8-sig",
              newline="") as f:
        sample = f.read(1024); f.seek(0)
        delim  = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            rxn = row["Rxn"].strip()
            gpr = row.get("GPR", "").strip()
            if not gpr:
                continue
            genes = re.split(
                r"\band\b|\bor\b", gpr,
                flags=re.IGNORECASE
            )
            for g in genes:
                g = re.sub(r"[^A-Za-z0-9]", "", g.strip())
                if g.upper().startswith("SLC"):
                    slc_rxns.add(rxn)
                    break
    ids = slc_rxns.intersection(set(all_rxn_ids))
    print(
        f"  SLC: {len(slc_rxns)} in GPR, "
        f"{len(ids)} in flux data"
    )
    return ids


def load_distribution(cond):
    """
    Load flux_distribution CSV for a condition.
    Rows = reactions, cols = bootstrap iterations.
    Returns DataFrame transposed so rows=samples, cols=reactions.
    """
    pattern = os.path.join(
        DIST_DIR, f"{cond}_flux_distribution.csv"
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No distribution file: {pattern}"
        )
    df = pd.read_csv(files[0], index_col=0)
    # Strip embedded quotes from reaction IDs
    df.index = df.index.str.strip("'")
    # Transpose: rows=bootstrap samples, cols=reactions
    return df.T


def load_ci(cond):
    """
    Load flux_ci CSV for a condition.
    Rows = reactions, cols = mean_flux, sd_flux,
           ci_lower, ci_upper, ci_width, cv.
    Returns DataFrame with reaction as index.
    """
    pattern = os.path.join(
        CI_DIR, f"{cond}_flux_ci.csv"
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No CI file: {pattern}"
        )
    df = pd.read_csv(files[0], index_col=0)
    df.index = df.index.str.strip("'")
    # Keep only numeric CI columns
    keep = ["mean_flux", "sd_flux",
            "ci_lower", "ci_upper", "cv"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def build_feature_matrix():
    records = []
    labels  = []
    all_rxn_ids = None

    for cond in CONDITION_MAP:
        print(f"\n-- {cond} --")

        # Load raw bootstrap solutions (rows=samples)
        dist_df = load_distribution(cond)
        print(
            f"  Bootstrap samples: {dist_df.shape[0]}, "
            f"Reactions: {dist_df.shape[1]}"
        )

        # Load CI summary
        ci_df = load_ci(cond)
        print(f"  CI features: {ci_df.shape[1]} per reaction")

        # Track all reaction IDs from first condition
        if all_rxn_ids is None:
            all_rxn_ids = list(dist_df.columns)

        # For each bootstrap sample, build feature row:
        # raw flux + CI summary stats for that condition
        for idx, row in dist_df.iterrows():
            feat = {}

            # 1. Raw flux values
            for rxn, val in row.items():
                feat[rxn] = val

            # 2. CI summary stats (same for all samples
            #    in this condition -- condition-level stats)
            for rxn in dist_df.columns:
                if rxn in ci_df.index:
                    for stat in ci_df.columns:
                        feat[f"{rxn}__{stat}"] = (
                            ci_df.loc[rxn, stat]
                        )

            records.append(feat)
            labels.append(cond)

    X = pd.DataFrame(records).fillna(0)
    y = pd.Series(labels, name="condition")

    print(f"\nFeature matrix: {X.shape}")
    print(f"  Rows (samples):  {X.shape[0]}")
    print(f"  Cols (features): {X.shape[1]}")
    print(f"Condition counts:")
    print(y.value_counts().to_string())

    # SLC annotation
    if all_rxn_ids and os.path.exists(GPR_CSV):
        slc_ids = get_slc_ids_from_gpr(
            GPR_CSV, all_rxn_ids
        )
        # Mark any feature column derived from SLC reaction
        slc_annot = {}
        for col in X.columns:
            rxn = col.split("__")[0]
            slc_annot[col] = rxn in slc_ids
        slc_series = pd.Series(
            slc_annot, name="is_slc"
        )
        slc_series.to_csv(
            os.path.join(OUTPUT_DIR, "slc_annotation.csv"),
            header=True,
        )
        n_slc = slc_series.sum()
        print(
            f"\nSLC annotation: {n_slc}/{len(slc_series)} "
            f"feature columns are SLC-linked"
        )

    X.to_csv(
        os.path.join(OUTPUT_DIR, "X_features.csv")
    )
    y.to_csv(
        os.path.join(OUTPUT_DIR, "y_labels.csv"),
        index=False,
    )
    print(f"\nSaved to {OUTPUT_DIR}/")
    return X, y


if __name__ == "__main__":
    X, y = build_feature_matrix()