# -*- coding: utf-8 -*-
"""
Step 01: Bootstrap expression uncertainty -> flux CI
=====================================================
Replicates exact GPR-to-reaction-value logic
across 100 bootstrap expression profiles, runs pFBA
on the fixed consensus GEM for each, and computes
flux mean +/- 95% CI per reaction.

GPR scaling rules (from original reasoning):
  No GPR          : use expression directly
                    (1000 if gene not found)
  AND only        : min(expression values)
  OR only         : sum(expression values)
  AND + OR mixed  : resolve inner parentheses first
                    then apply outer operator

Outputs -> data/flux_uncertainty/:
  {condition}_flux_distribution.csv
  {condition}_flux_ci.csv
  all_conditions_flux_ci_mean.csv
  uncertainty_report.csv
  {condition}_uncertainty_plot.pdf

Usage:
  python scripts/step02_gem_fluxes/01_uncertainty_fba.py

"""

import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cobra
from cobra.flux_analysis import pfba

warnings.filterwarnings("ignore")


# =============================================================
#  CONFIGURATION
# =============================================================

# Fixed consensus GEMs (structure from 99% ftINIT rule)
CONSENSUS_GEM_PATHS = {
    "Normal": "Normal.xml",
    "HSCT":   "HSCT.xml",
    "GCSF":   "GCSF.xml",
    "PDAC":   "PDAC.xml",
}

# GPR rules CSV (from original Excel GPR sheet)
# Columns: Rxn (no quotes), GPR (Ensembl IDs)
# Reaction IDs here are bare (MAR04762),
# but in XML they have embedded quotes ('MAR04762').
# We strip XML quotes when looking up in this dict.
GPR_CSV_PATH    = "GPR2.csv"
# Gene symbol -> Ensembl ID mapping (from get_gene_mapping.R)
#GENE_MAP_PATH   = "data/gene_symbol_to_ensembl.csv"

# Bootstrap expression means from Step 00
BOOT_DIR    = "/bootstrap_expression"
BOOT_PREFIX = "GSE167363_"

OUTPUT_DIR = "data/Validation/flux_uncertainty"
CI_LEVEL   = 0.95

# Bound scaling approach:
#   scaled = (raw_value / max_raw_value) * BOUND_MAX
#   new_lb = sign(original_lb) * scaled
#   new_ub = sign(original_ub) * scaled
#
# Keeps reaction directionality intact (reversible
# reactions stay reversible) while normalizing all
# magnitudes to 0 - BOUND_MAX range.
# Reactions with no GPR get 1000 (unconstrained)
# which maps to BOUND_MAX after scaling.

BOUND_MAX = 1000.0

# Biomass reaction ID in the Human1 GEMs.
# Set to None to auto-detect (finds reaction whose
# name/id contains 'biomass', case-insensitive).
# Common Human1 biomass IDs: 'MAR13082', 'HMR_biomass'
BIOMASS_REACTION_ID = "'biomass_mac'"

# =============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Load GPR rules from CSV ----------------------------------

def load_gpr_rules(csv_path):
    """
    Load GPR rules from CSV file.
    CSV columns: Rxn, GPR (tab-separated)
    Handles BOM, trailing whitespace, flexible column names.
    Returns dict: {rxn_id_bare: gpr_string}
    """
    import csv
    gpr_dict = {}
    with open(csv_path, "r", encoding="utf-8-sig",
              newline="") as f:
        # Auto-detect delimiter (tab or comma)
        sample = f.read(1024)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize header names (strip whitespace/BOM)
        reader.fieldnames = [
            h.strip() for h in reader.fieldnames
        ]
        print(f"  GPR CSV columns: {reader.fieldnames}")
        # Find Rxn and GPR columns flexibly
        rxn_col = next(
            (h for h in reader.fieldnames
             if h.strip().lower() in ("rxn", "reaction",
                                      "reaction_id")),
            None
        )
        gpr_col = next(
            (h for h in reader.fieldnames
             if h.strip().lower() in ("gpr", "gene_reaction_rule",
                                      "gpr_rule")),
            None
        )
        if rxn_col is None or gpr_col is None:
            raise ValueError(
                f"Could not find Rxn/GPR columns in "
                f"{csv_path}. Found: {reader.fieldnames}"
            )
        for row in reader:
            rxn = row[rxn_col].strip()
            gpr = row[gpr_col].strip() if row[gpr_col] else ""
            if rxn:
                gpr_dict[rxn] = gpr
    print(
        f"  Loaded {len(gpr_dict)} GPR rules "
        f"from {csv_path}"
    )
    return gpr_dict


def load_gene_mapping(map_path):
    """
    Load gene symbol -> Ensembl ID mapping CSV.
    Columns: hgnc_symbol, ensembl_gene_id
    Returns dict: {symbol: ensembl_id}
    One-to-one; if multiple Ensembl per symbol, keeps first.
    """
    df = pd.read_csv(map_path)
    df = df.dropna()
    df = df[df["hgnc_symbol"] != ""]
    df = df[df["ensembl_gene_id"] != ""]
    # Keep first Ensembl per symbol
    df = df.drop_duplicates(subset="hgnc_symbol", keep="first")
    mapping = dict(zip(df["hgnc_symbol"], df["ensembl_gene_id"]))
    print(f"  Loaded {len(mapping)} symbol->Ensembl mappings")
    return mapping


def translate_expr_to_ensembl(expr_series, gene_mapping):
    """
    Translate expression Series from gene symbols to Ensembl IDs.
    Drops genes with no mapping.
    If multiple symbols map to same Ensembl, keeps max expression.
    """
    translated = {}
    for symbol, val in expr_series.items():
        ensg = gene_mapping.get(symbol)
        if ensg:
            # Keep max if collision
            if ensg not in translated or val > translated[ensg]:
                translated[ensg] = val
    result = pd.Series(translated)
    print(
        f"  Expression: {len(expr_series)} symbols -> "
        f"{len(result)} Ensembl IDs mapped"
    )
    return result


def strip_rxn_quotes(rxn_id):
    """
    Strip embedded single quotes from XML reaction IDs.
    "'MAR04762'" -> "MAR04762"
    "MAR04762"   -> "MAR04762" (unchanged)
    """
    return rxn_id.strip("'")


# -- GPR value computation ------------------------------------
# Replicates the exact original Python script logic

def get_expr_value(gene, expr_dict):
    """
    Look up expression value for a gene.
    Case-insensitive: GPR may use mouse casing (Ldha),
    expression uses human casing (LDHA).
    Returns 0.0 if not found.
    """
    g = gene.strip()
    # Try exact match first
    val = expr_dict.get(g)
    if val is not None:
        return float(val)
    # Try uppercase
    val = expr_dict.get(g.upper())
    if val is not None:
        return float(val)
    # Try title case (Ldha -> Ldha already, but LDHA -> Ldha)
    val = expr_dict.get(g.capitalize())
    if val is not None:
        return float(val)
    return 0.0


def resolve_parentheses(gpr, expr_dict):
    """
    Resolve innermost parentheses in a mixed AND+OR GPR.
    Replicates the calculate_value regex substitution.
    OR inside parens  -> sum of values
    AND inside parens -> min of values
    """
    def calculate_value(match):
        items = match.group(0)
        items = items.replace("(", "").replace(")", "")
        items = items.strip()

        if "and" in items.lower():
            parts = re.split(r"\band\b", items,
                             flags=re.IGNORECASE)
            vals = [
                get_expr_value(p.strip(), expr_dict)
                for p in parts
            ]
            return str(min(vals))

        elif "or" in items.lower():
            parts = re.split(r"\bor\b", items,
                             flags=re.IGNORECASE)
            vals = [
                get_expr_value(p.strip(), expr_dict)
                for p in parts
            ]
            return str(sum(vals))

        else:
            return str(
                get_expr_value(items.strip(), expr_dict)
            )

    return re.sub(r"\([^()]*\)", calculate_value, gpr)


def parse_final_value(gpr_items_str, rxn_id, expr_dict):
    """
    After resolving parentheses, evaluate the remaining
    AND or OR expression to get final scalar value.
    Replicates  the calculate_final_value logic.
    """
    gpr_items_str = gpr_items_str.strip()

    def to_float(item):
        item = item.strip()
        try:
            return float(item)
        except ValueError:
            return get_expr_value(item, expr_dict)

    if re.search(r"\band\b", gpr_items_str,
                 re.IGNORECASE):
        parts = re.split(r"\band\b", gpr_items_str,
                         flags=re.IGNORECASE)
        vals = [to_float(p) for p in parts]
        return min(vals)

    elif re.search(r"\bor\b", gpr_items_str,
                   re.IGNORECASE):
        parts = re.split(r"\bor\b", gpr_items_str,
                         flags=re.IGNORECASE)
        vals = [to_float(p) for p in parts]
        return sum(vals)

    else:
        return to_float(gpr_items_str)


def compute_reaction_value(gpr_str, expr_dict):
    """
    Compute scalar reaction value from GPR string and
    expression dict. Exactly replicates the original
    GPR scoring logic:

      No GPR        : 1000 (unconstrained)
      AND only      : min(expression values)
      OR only       : sum(expression values)
      AND + OR mix  : resolve inner parens first,
                      then apply outer operator
    """
    gpr = gpr_str.strip() if gpr_str else ""

    # No GPR
    if not gpr:
        return 1000.0

    has_and = bool(re.search(r"\band\b", gpr,
                              re.IGNORECASE))
    has_or  = bool(re.search(r"\bor\b",  gpr,
                              re.IGNORECASE))

    # Single gene (no operator)
    if not has_and and not has_or:
        val = expr_dict.get(gpr.strip(), None)
        return float(val) if val is not None else 1000.0

    # AND only
    if has_and and not has_or:
        gpr_clean = gpr.replace("(", "").replace(")", "")
        parts = re.split(r"\band\b", gpr_clean,
                         flags=re.IGNORECASE)
        vals = [
            get_expr_value(p.strip(), expr_dict)
            for p in parts
        ]
        return min(vals)

    # OR only
    if has_or and not has_and:
        gpr_clean = gpr.replace("(", "").replace(")", "")
        parts = re.split(r"\bor\b", gpr_clean,
                         flags=re.IGNORECASE)
        vals = [
            get_expr_value(p.strip(), expr_dict)
            for p in parts
        ]
        return sum(vals)

    # Mixed AND + OR
    resolved = gpr
    # Keep resolving until no more parentheses
    while "(" in resolved:
        resolved = resolve_parentheses(
            resolved, expr_dict
        )
    return parse_final_value(resolved, gpr, expr_dict)


# -- Bound setting --------------------------------------------

def scale_rxn_values(rxn_values):
    """
    Scale raw GPR-computed reaction values to 0-BOUND_MAX.
    scaled = (raw / max_raw) * BOUND_MAX
    Reactions with no GPR have value 1000 which maps to
    BOUND_MAX (fully unconstrained) after scaling.
    """
    if not rxn_values:
        return rxn_values
    max_val = max(rxn_values.values())
    if max_val < 1e-9:
        return {k: 0.0 for k in rxn_values}
    return {
        k: (v / max_val) * BOUND_MAX
        for k, v in rxn_values.items()
    }


def apply_bounds(model, rxn_values):
    """
    Apply scaled reaction values as flux bounds.
    Keeps original lb/ub sign, scales magnitude by value.

    new_lb = sign(original_lb) * scaled_value
    new_ub = sign(original_ub) * scaled_value

    This preserves reaction directionality:
    - Reversible reactions (lb < 0) stay reversible
    - Irreversible reactions (lb = 0) stay irreversible
    - Exchange reactions keep their direction
    """
    scaled = scale_rxn_values(rxn_values)

    for rxn in model.reactions:
        val = scaled.get(rxn.id, None)
        if val is None:
            continue

        lb_orig = rxn.lower_bound
        ub_orig = rxn.upper_bound

        # Scale magnitude, keep sign
        rxn.lower_bound = (
            -val if lb_orig < 0
            else 0.0 if lb_orig == 0
            else val
        )
        rxn.upper_bound = (
            -val if ub_orig < 0
            else 0.0 if ub_orig == 0
            else val
        )


def compute_all_rxn_values(model, expr_dict,
                           gpr_rules):
    """
    Compute reaction values for all reactions in model.
    Uses GPR rules from external CSV (not from XML,
    since XML models have no GPR associations stored).
    Strips embedded quotes from XML reaction IDs to
    match bare IDs in the GPR CSV.
    """
    rxn_vals = {}
    for rxn in model.reactions:
        bare_id = strip_rxn_quotes(rxn.id)
        gpr_str = gpr_rules.get(bare_id, "")
        rxn_vals[rxn.id] = compute_reaction_value(
            gpr_str, expr_dict
        )
    return rxn_vals


# -- pFBA loop ------------------------------------------------

def get_biomass_reaction(model):
    """
    Find and return the biomass reaction.
    Tries BIOMASS_REACTION_ID variants first (with/without
    embedded quotes), then auto-detects by scanning all
    reaction IDs and names for 'biomass'.
    Also sets it as the model objective.
    """
    rxn_ids = {r.id for r in model.reactions}

    if BIOMASS_REACTION_ID is not None:
        # Try as-is, then without quotes, then with quotes
        candidates_to_try = [
            BIOMASS_REACTION_ID,
            BIOMASS_REACTION_ID.strip("'"),
            f"'{BIOMASS_REACTION_ID.strip()}'",
        ]
        for bid in candidates_to_try:
            if bid in rxn_ids:
                rxn = model.reactions.get_by_id(bid)
                model.objective = rxn
                print(f"  Biomass set: {rxn.id}")
                return rxn

        # Still not found -- print available IDs to help debug
        biomass_like = [
            r.id for r in model.reactions
            if "biomass" in r.id.lower()
            or "biomass" in (r.name or "").lower()
        ]
        print(
            f"  WARNING: BIOMASS_REACTION_ID "
            f"'{BIOMASS_REACTION_ID}' not found. "
            f"Biomass-like reactions: {biomass_like}"
        )

    # Auto-detect
    candidates = [
        r for r in model.reactions
        if "biomass" in r.id.lower()
        or "biomass" in (r.name or "").lower()
    ]
    if candidates:
        rxn = candidates[0]
        model.objective = rxn
        print(f"  Biomass auto-detected: {rxn.id}")
        return rxn

    raise ValueError(
        "No biomass reaction found. "
        "Set BIOMASS_REACTION_ID manually. "
        f"Available reactions (first 10): "
        f"{[r.id for r in model.reactions[:10]]}"
    )


def run_pfba_safe(model, label=""):
    """Run pFBA; return flux Series or None on failure."""
    try:
        sol = pfba(model)
        if sol.status == "optimal":
            return sol.fluxes
        print(f"  Non-optimal [{label}]: {sol.status}")
        return None
    except Exception as e:
        print(f"  pFBA failed [{label}]: {e}")
        return None


def run_bootstrap_fba(condition, base_model,
                      boot_df, obs_mean, gpr_rules):
    """
    For one condition: apply each bootstrap expression
    profile as GPR-scaled bounds on the fixed consensus
    GEM, run pFBA, collect flux distribution.
    GPR rules are loaded from external CSV since XML
    models have no gene associations stored.
    """
    boot_cols = list(boot_df.columns)
    n         = len(boot_cols)
    flux_records = {}

    for i, col in enumerate(boot_cols):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  ... pFBA {i+1}/{n}: {col}")

        expr_dict = boot_df[col].to_dict()
        rxn_vals  = compute_all_rxn_values(
            base_model, expr_dict, gpr_rules
        )

        model = base_model.copy()
        apply_bounds(model, rxn_vals)

        # Explicitly set biomass as objective
        # (bounds change does not affect objective
        # but we set it explicitly for safety)
        model.objective = get_biomass_reaction(model)

        fluxes = run_pfba_safe(model, label=col)
        if fluxes is not None:
            flux_records[col] = fluxes

    flux_dist = pd.DataFrame(flux_records)
    print(
        f"  {flux_dist.shape[1]}/{n} "
        f"successful pFBA solutions"
    )
    return flux_dist


# -- CI computation -------------------------------------------

def compute_ci(flux_dist):
    """Compute per-reaction CI from bootstrap solutions."""
    alpha    = 1 - CI_LEVEL
    mean_f   = flux_dist.mean(axis=1)
    sd_f     = flux_dist.std(axis=1)
    ci_lower = flux_dist.quantile(alpha / 2,     axis=1)
    ci_upper = flux_dist.quantile(1 - alpha / 2, axis=1)
    return pd.DataFrame({
        "mean_flux": mean_f,
        "sd_flux":   sd_f,
        "ci_lower":  ci_lower,
        "ci_upper":  ci_upper,
        "ci_width":  ci_upper - ci_lower,
        "cv":        sd_f / (mean_f.abs() + 1e-9),
    })


def add_original_comparison(ci_df, obs_mean,
                             base_model, gpr_rules):
    """
    Run pFBA with original observed mean expression
    and check what fraction falls within bootstrap CI.
    This is the direct statistical rebuttal to reviewer.
    """
    print("  Running pFBA with original mean expression...")
    obs_dict = obs_mean.to_dict()
    rxn_vals = compute_all_rxn_values(
        base_model, obs_dict, gpr_rules
    )
    model = base_model.copy()
    apply_bounds(model, rxn_vals)
    model.objective = get_biomass_reaction(model)
    orig_fluxes = run_pfba_safe(model, label="original")

    if orig_fluxes is None:
        print("  Original pFBA failed, skipping.")
        return ci_df

    common = ci_df.index.intersection(orig_fluxes.index)
    ci_df.loc[common, "original_flux"] = (
        orig_fluxes[common]
    )
    ci_df["original_within_ci"] = (
        (ci_df["original_flux"] >= ci_df["ci_lower"])
        & (ci_df["original_flux"] <= ci_df["ci_upper"])
    )
    return ci_df


def make_plot(ci_df, condition):
    """Plot CI width and CV distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{condition}: bootstrap flux uncertainty",
        fontsize=13,
    )
    axes[0].hist(
        ci_df["ci_width"].dropna(),
        bins=60, color="#7F77DD",
        edgecolor="white", linewidth=0.3,
    )
    axes[0].set_xlabel(
        f"{int(CI_LEVEL*100)}% CI width (flux units)"
    )
    axes[0].set_ylabel("Reactions")
    axes[0].set_title("CI width distribution")

    axes[1].hist(
        ci_df["cv"].clip(upper=5).dropna(),
        bins=60, color="#4CAF88",
        edgecolor="white", linewidth=0.3,
    )
    axes[1].set_xlabel("CV (sd / |mean flux|)")
    axes[1].set_ylabel("Reactions")
    axes[1].set_title("Flux uncertainty (CV)")

    if "original_within_ci" in ci_df.columns:
        n_tot = ci_df["original_within_ci"].notna().sum()
        n_in  = ci_df["original_within_ci"].sum()
        pct   = 100 * n_in / max(n_tot, 1)
        axes[1].text(
            0.97, 0.95,
            f"Original within CI:\n"
            f"{n_in}/{n_tot} ({pct:.1f}%)",
            transform=axes[1].transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="wheat", alpha=0.5,
            ),
        )

    plt.tight_layout()
    out = os.path.join(
        OUTPUT_DIR,
        f"{condition}_uncertainty_plot.pdf",
    )
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out}")


# -- Main -----------------------------------------------------

def main():
    print("=" * 60)
    print(
        "Step 01: Bootstrap expression uncertainty"
        " -> flux CI"
    )
    print(
        "GPR logic: AND=min, OR=sum, mixed=resolve"
        " parens first"
    )
    print("=" * 60)

    ci_frames   = {}
    report_rows = []

    for cond, gem_path in CONSENSUS_GEM_PATHS.items():
        print(f"\n-- Condition: {cond} --")

        boot_path = os.path.join(
            BOOT_DIR,
            f"{BOOT_PREFIX}{cond}_boot_means.csv"
        )
        obs_path = os.path.join(
            BOOT_DIR,
            f"{BOOT_PREFIX}{cond}_observed_mean.csv"
        )

        missing = [p for p in [gem_path, boot_path, obs_path]
                   if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f"  ERROR: missing -> {p}")
            continue

        print(f"  Loading GEM: {gem_path}")
        import logging
        logging.getLogger("cobra").setLevel(logging.ERROR)
        base_model = cobra.io.read_sbml_model(gem_path)
        print(
            f"  Reactions: {len(base_model.reactions)}"
        )

        # Detect and confirm biomass reaction
        try:
            biomass_rxn = get_biomass_reaction(base_model)
            print(
                f"  Biomass reaction: {biomass_rxn.id} "
                f"({biomass_rxn.name})"
            )
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        # Load GPR rules from external CSV
        print(f"  Loading GPR rules: {GPR_CSV_PATH}")
        gpr_rules = load_gpr_rules(GPR_CSV_PATH)

        boot_df  = pd.read_csv(
            boot_path, index_col="gene"
        )
        obs_mean = pd.read_csv(
            obs_path, index_col="gene"
        )["observed_mean"]

        # Check gene overlap (case-insensitive)
        gpr_genes = set(
            g.strip().upper()
            for rules in gpr_rules.values()
            for g in re.split(r"\band\b|\bor\b",
                              rules, flags=re.IGNORECASE)
            if g.strip() and not g.strip().isdigit()
        )
        expr_genes = set(g.upper() for g in boot_df.index)
        overlap    = gpr_genes.intersection(expr_genes)
        sample_expr = list(boot_df.index)[:5]
        sample_gpr  = [
            g for rules in gpr_rules.values()
            for g in re.split(r"\band\b|\bor\b", rules,
                              flags=re.IGNORECASE)
            if g.strip()
        ][:5]
        print(
            f"  GPR genes: {len(gpr_genes)} | "
            f"Expr genes: {len(expr_genes)} | "
            f"Overlap (case-insensitive): {len(overlap)}"
        )
        print(f"  Sample expr genes: {sample_expr}")
        print(f"  Sample GPR genes:  {sample_gpr}")
        if len(overlap) == 0:
            print(
                "  WARNING: Still no overlap. "
                "Proceeding -- unmatched genes get value 0."
            )

        flux_dist = run_bootstrap_fba(
            cond, base_model, boot_df,
            obs_mean, gpr_rules
        )

        dist_out = os.path.join(
            OUTPUT_DIR,
            f"{cond}_flux_distribution.csv",
        )
        flux_dist.to_csv(dist_out)
        print(f"  Saved distribution: {dist_out}")

        ci_df = compute_ci(flux_dist)
        ci_df = add_original_comparison(
            ci_df, obs_mean, base_model, gpr_rules
        )

        ci_out = os.path.join(
            OUTPUT_DIR, f"{cond}_flux_ci.csv"
        )
        ci_df.to_csv(ci_out)
        print(f"  Saved CI: {ci_out}")

        make_plot(ci_df, cond)
        ci_frames[cond] = ci_df["mean_flux"].rename(cond)

        row = {
            "condition":     cond,
            "n_reactions":   len(base_model.reactions),
            "n_boot_ok":     flux_dist.shape[1],
            "mean_ci_width": ci_df["ci_width"].mean(),
            "median_cv":     ci_df["cv"].median(),
        }
        if "original_within_ci" in ci_df.columns:
            n_tot = (
                ci_df["original_within_ci"].notna().sum()
            )
            n_in  = ci_df["original_within_ci"].sum()
            row["pct_original_within_ci"] = (
                100 * n_in / max(n_tot, 1)
            )
        report_rows.append(row)

    if ci_frames:
        wide  = pd.DataFrame(ci_frames)
        w_out = os.path.join(
            OUTPUT_DIR,
            "all_conditions_flux_ci_mean.csv",
        )
        wide.to_csv(w_out)
        print(f"\nWide CI mean matrix -> {w_out}")

    if report_rows:
        report  = pd.DataFrame(report_rows)
        rep_out = os.path.join(
            OUTPUT_DIR, "uncertainty_report.csv"
        )
        report.to_csv(rep_out, index=False)
        print(f"Uncertainty report  -> {rep_out}")
        print()
        print(report.to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 01 complete.")
    print(
        "Key output for reviewer: "
        "data/flux_uncertainty/uncertainty_report.csv"
    )
    print(
        "pct_original_within_ci = fraction of original "
        "GPR-scaled fluxes within bootstrap 95% CI"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()