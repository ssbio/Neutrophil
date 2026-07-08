"""
Step 0 - GEM XML -> pFBA + FVA flux tables
==========================================
Loads one XML GEM per condition, sets your chosen objective,
runs pFBA and FVA, and writes clean CSV files ready for
Module A (SHAP feature importance pipeline).

Outputs (in OUTPUT_DIR):
  {condition}_pFBA.csv          columns: reaction_id, flux
  {condition}_FVA.csv           columns: reaction_id, minimum,
                                  maximum, flux_range, flux_center
  {condition}_metadata.csv      model stats
  all_conditions_pFBA.csv       wide matrix: reactions x conditions
  all_conditions_FVA_center.csv wide matrix: FVA centers

Usage:
  python 00_gem_to_fluxes.py

"""

import os
import warnings
import pandas as pd
import cobra
from cobra.flux_analysis import (
    pfba,
    flux_variability_analysis,
)

warnings.filterwarnings("ignore")


# =============================================================
#  CONFIGURATION
# =============================================================

# Full paths to your 4 condition-specific XML GEMs
GEM_PATHS = {
    "healthy_ctrl": "../Normal.xml",
    "healthy_act":  "../PDAC.xml",
    "disease_ctrl": "../GCSF.xml",
    "disease_act":  "../HSCT.xml",
}

# -- Objective function ---------------------------------------
# Set OBJECTIVE to one of:
#
#   "default"
#       use whatever objective is already set in each XML
#
#   "reaction_id"
#       maximize a single reaction by its ID, e.g.:
#         OBJECTIVE = "MAR13082"
#         OBJECTIVE = "NADPH_tx"
#
#   dict
#       linear combination of reactions
#       positive coeff = maximize, negative = minimize, e.g.:
#         OBJECTIVE = {"NADPH_tx": 1.0, "ROS_production": -0.5}
#
#   "list_reactions"
#       print all reaction IDs in the first model and exit
#       so you can inspect what is available before choosing

OBJECTIVE = "default"

# -- FVA settings ---------------------------------------------
# Fraction of optimal objective that must be maintained.
# 1.0 = strict; 0.9 = allow 10% slack. Range: 0.9-1.0 typical.
FVA_FRACTION_OPTIMUM = 1.0

# Run FVA on ALL reactions, or only SLC transporters?
#   "all" -> full FVA (slower, complete)
#   "slc" -> only reactions whose ID contains SLC_SUBSTRING
FVA_SCOPE = "all"


# Number of parallel processes (set to SLURM cpus-per-task)
N_PROCESSES = 8

# Output directory
OUTPUT_DIR = "data/fluxes"

# =============================================================


os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Helpers --------------------------------------------------

def load_model(path):
    """Load SBML/XML GEM via COBRApy."""
    print(f"  Loading: {path}")
    model = cobra.io.read_sbml_model(path)
    print(
        f"  -> {len(model.reactions)} reactions, "
        f"{len(model.metabolites)} metabolites"
    )
    return model


def set_objective(model, objective):
    """
    Apply the user-defined objective to the model.
    Accepts: "default", a reaction ID string,
    or a dict of {rxn_id: coefficient}.
    """
    if objective == "default":
        print("  Objective: using model default")
        return

    if objective == "list_reactions":
        print("\n-- All reaction IDs in the model --")
        for rxn in model.reactions:
            print(f"  {rxn.id:40s}  {rxn.name}")
        raise SystemExit(
            "Listing done. "
            "Set OBJECTIVE to your chosen reaction ID and re-run."
        )

    if isinstance(objective, str):
        ids = [r.id for r in model.reactions]
        if objective not in ids:
            raise ValueError(
                f"Reaction '{objective}' not found in model.\n"
                "Set OBJECTIVE = 'list_reactions' to see all IDs."
            )
        model.objective = objective
        print(f"  Objective: maximize {objective}")

    elif isinstance(objective, dict):
        model.objective = {
            model.reactions.get_by_id(rxn_id): coeff
            for rxn_id, coeff in objective.items()
        }
        print(f"  Objective: linear combination {objective}")

    else:
        raise TypeError(
            "OBJECTIVE must be 'default', a reaction ID string, "
            f"or a dict. Got: {type(objective)}"
        )


def run_pfba(model):
    """Run pFBA and return a tidy DataFrame."""
    sol = pfba(model)
    if sol.status != "optimal":
        raise RuntimeError(
            f"pFBA did not reach optimal (status: {sol.status})"
        )

    df = sol.fluxes.reset_index()
    df.columns = ["reaction_id", "flux"]

    rxn_names = {r.id: r.name for r in model.reactions}
    rxn_subs  = {r.id: r.subsystem for r in model.reactions}

    df["reaction_name"] = df["reaction_id"].map(rxn_names)
    df["subsystem"]     = df["reaction_id"].map(rxn_subs)
    df = df[["reaction_id", "reaction_name", "subsystem", "flux"]]

    n_active = (df["flux"].abs() > 1e-9).sum()
    print(f"  pFBA: {n_active} active reactions (|flux| > 1e-9)")
    return df


def run_fva(model, scope, slc_substring, fraction_optimum, n_proc):
    """Run FVA and return a tidy DataFrame."""
    if scope == "slc":
        rxn_list = [
            r for r in model.reactions
            if slc_substring.lower() in r.id.lower()
        ]
        print(f"  FVA scope: SLC only -> {len(rxn_list)} reactions")
    else:
        rxn_list = model.reactions
        print(f"  FVA scope: all -> {len(rxn_list)} reactions")

    fva_raw = flux_variability_analysis(
        model,
        reaction_list=rxn_list,
        fraction_of_optimum=fraction_optimum,
        processes=n_proc,
    )

    df = fva_raw.reset_index()
    df.columns = ["reaction_id", "minimum", "maximum"]
    df["flux_range"]  = df["maximum"] - df["minimum"]
    df["flux_center"] = (df["maximum"] + df["minimum"]) / 2

    rxn_names = {r.id: r.name for r in model.reactions}
    rxn_subs  = {r.id: r.subsystem for r in model.reactions}

    df["reaction_name"] = df["reaction_id"].map(rxn_names)
    df["subsystem"]     = df["reaction_id"].map(rxn_subs)
    df = df[[
        "reaction_id", "reaction_name", "subsystem",
        "minimum", "maximum", "flux_range", "flux_center",
    ]]

    n_variable = (df["flux_range"] > 1e-9).sum()
    print(f"  FVA: {n_variable} variable reactions (range > 1e-9)")
    return df


def save_results(condition, pfba_df, fva_df, model):
    """Write per-condition CSVs."""
    pfba_path = os.path.join(OUTPUT_DIR, f"{condition}_pFBA.csv")
    fva_path  = os.path.join(OUTPUT_DIR, f"{condition}_FVA.csv")
    meta_path = os.path.join(OUTPUT_DIR, f"{condition}_metadata.csv")

    pfba_df.to_csv(pfba_path, index=False)
    fva_df.to_csv(fva_path,   index=False)

    meta = pd.DataFrame([{
        "condition":     condition,
        "n_reactions":   len(model.reactions),
        "n_metabolites": len(model.metabolites),
        "n_genes":       len(model.genes),
        "objective":     str(OBJECTIVE),
        "fva_fraction":  FVA_FRACTION_OPTIMUM,
        "fva_scope":     FVA_SCOPE,
    }])
    meta.to_csv(meta_path, index=False)

    print(f"  Saved: {pfba_path}")
    print(f"  Saved: {fva_path}")


def build_merged_matrices(conditions):
    """
    Build wide matrices across all conditions.
    One column per condition; missing reactions filled with 0.
    """
    pfba_frames = {}
    fva_frames  = {}

    for cond in conditions:
        p_path = os.path.join(OUTPUT_DIR, f"{cond}_pFBA.csv")
        f_path = os.path.join(OUTPUT_DIR, f"{cond}_FVA.csv")
        p = pd.read_csv(p_path, index_col="reaction_id")
        f = pd.read_csv(f_path, index_col="reaction_id")
        pfba_frames[cond] = p["flux"]
        fva_frames[cond]  = f["flux_center"]

    merged_pfba = pd.DataFrame(pfba_frames).fillna(0)
    merged_fva  = pd.DataFrame(fva_frames).fillna(0)

    out_pfba = os.path.join(OUTPUT_DIR, "all_conditions_pFBA.csv")
    out_fva  = os.path.join(
        OUTPUT_DIR, "all_conditions_FVA_center.csv"
    )

    merged_pfba.to_csv(out_pfba)
    merged_fva.to_csv(out_fva)

    print(f"\nMerged pFBA matrix:       {merged_pfba.shape} -> {out_pfba}")
    print(f"Merged FVA center matrix: {merged_fva.shape}  -> {out_fva}")


# -- Main -----------------------------------------------------

def main():
    print("=" * 60)
    print("Step 0: GEM XML -> pFBA + FVA flux tables")
    print("=" * 60)

    if OBJECTIVE == "list_reactions":
        first_path = list(GEM_PATHS.values())[0]
        model = load_model(first_path)
        set_objective(model, "list_reactions")

    successful = []

    for condition, gem_path in GEM_PATHS.items():
        print(f"\n-- Condition: {condition} --")

        if not os.path.exists(gem_path):
            print(f"  WARNING: file not found -> {gem_path}. Skipping.")
            continue

        try:
            model = load_model(gem_path)
            set_objective(model, OBJECTIVE)

            print("  Running pFBA...")
            pfba_df = run_pfba(model)

            print("  Running FVA...")
            fva_df = run_fva(
                model, FVA_SCOPE, SLC_SUBSTRING,
                FVA_FRACTION_OPTIMUM, N_PROCESSES,
            )

            save_results(condition, pfba_df, fva_df, model)
            successful.append(condition)

        except Exception as e:
            print(f"  ERROR in {condition}: {e}")
            import traceback
            traceback.print_exc()

    if len(successful) > 1:
        print("\n-- Building merged matrices --")
        build_merged_matrices(successful)

    print("\n" + "=" * 60)
    print(
        f"Step 0 complete. "
        f"{len(successful)}/{len(GEM_PATHS)} conditions processed."
    )
    print(f"Output folder: {OUTPUT_DIR}/")
    print("\nNext step:")
    print(
        "  Edit DATA_DIR in "
        "moduleA_shap/01_build_feature_matrix.py"
    )
    print("  then run: python moduleA_shap/01_build_feature_matrix.py")
    print("=" * 60)


if __name__ == "__main__":
    main()