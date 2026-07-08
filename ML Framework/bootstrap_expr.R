# =============================================================
# Step 00: Extract counts + bootstrap expression means
#          from Montaldo et al. Fig6 and Fig8 Seurat objects
# =============================================================
#
# This is the ONLY bootstrap script for this pipeline.
# It replaces the generic 01_bootstrap_expr.R which
# produced 0 cells because it looked for condition labels
# directly in the sample column (e.g. "Normal") instead
# of mapping sample IDs (e.g. HD_T2, HD_T4 -> Normal).
#
# Fig 6 — disease conditions (130,628 cells)
#   Conditions decoded from sample column:
#     Normal : HD_T2, HD_T4
#     HSCT   : EE_r14/16/18, LE_r14/16
#     GCSF   : PostG_d16/LDN_d11/T1/X6
#     PDAC   : PDAC_31/32/36/37/39
#
# Fig 8 — cord blood maturation axis (22,440 cells)
#   CB conditions via orig.ident:
#     CB_UT, CB_GCSF, CB_IFNB, CB_IFNG
#   Maturity via seurat_clusters -> paper annotation:
#     Scheme A binary:  immature (2,3) / mature (4,5,6)
#     Scheme B fine:    early (2,3) / mid (4,5) / late (6)
#
# Outputs -> data/bootstrap_expression/:
#   Fig6_{condition}_boot_means.csv
#   Fig6_{condition}_observed_mean.csv
#   Fig8_{cb_condition}_boot_means.csv
#   Fig8_{cb_condition}_observed_mean.csv
#   Fig8_binary_{group}_boot_means.csv
#   Fig8_binary_{group}_observed_mean.csv
#   Fig8_fine_{group}_boot_means.csv
#   Fig8_fine_{group}_observed_mean.csv
#   Fig8_{binary_group}_{cb_cond}_boot_means.csv
#   cell_counts_summary.csv
#
# Usage:
#   Rscript scripts/step00_extract/00_extract_bootstrap.R
# HPC:
#   sbatch slurm/00_extract_bootstrap.sh
# =============================================================

library(Seurat)
library(Matrix)
library(dplyr)
library(readr)
library(tibble)

# -- CONFIG ---------------------------------------------------

FIG6_PATH <- paste0(
  "..Fig6.rds"
)
FIG8_PATH <- paste0(
  "..Fig8.rds"
)

OUTPUT_DIR <- "data/bootstrap_expression"

# Assay and slot to pull expression from
# SCT = SCTransform normalized (recommended for GEM input)
# RNA = raw counts (use if SCT not available)
ASSAY <- "SCT"
SLOT  <- "data"

N_BOOT <- 100
SEED   <- 42

# -- Control which sections to run ---------------------------
# Set FALSE to skip a section (e.g. if Fig6 already done)
RUN_FIG6 <- TRUE
RUN_FIG8 <- TRUE

# -- Fig 6 condition definitions ------------------------------
# Maps condition label -> sample IDs in Seurat object
# This is why Step 00 finds cells but the old generic
# script found 0 — it searched for "Normal" directly
# in sample column instead of "HD_T2", "HD_T4"

FIG6_CONDITIONS <- list(
  Normal = c("HD_T2", "HD_T4"),
  HSCT   = c(
    "EE_r14", "EE_r16", "EE_r18",
    "LE_r14", "LE_r16"
  ),
  GCSF   = c(
    "PostG_d16", "PostG_LDN_d11",
    "PostG_T1",  "PostG_X6"
  ),
  PDAC   = c(
    "PDAC_31", "PDAC_32", "PDAC_36",
    "PDAC_37", "PDAC_39"
  )
)

# BM_Healthy saved as optional reference
# (excluded from main 4-condition analysis)
FIG6_BM_SAMPLES <- c(
  "BM_HD_d10", "BM_HD_d15",
  "BM_LE_r14", "BM_LE_r16",
  "BM_180_r14", "BM_180_r16"
)

# -- Fig 8 condition definitions ------------------------------
FIG8_CB_CONDITIONS <- c(
  "CB_UT", "CB_GCSF", "CB_IFNB", "CB_IFNG"
)

# seurat_clusters -> paper annotation mapping
FIG8_CLUSTER_MAP <- c(
  "0" = "cluster_2",
  "1" = "cluster_4",
  "2" = "cluster_5",
  "3" = "cluster_6",
  "4" = "cluster_3"
)

# Scheme A: binary (2 groups)
#   immature = paper clusters 2, 3
#   mature   = paper clusters 4, 5, 6
FIG8_MATURITY_BINARY <- list(
  immature = c("cluster_2", "cluster_3"),
  mature   = c("cluster_4", "cluster_5", "cluster_6")
)

# Scheme B: fine-grained (3 groups)
#   early = paper clusters 2, 3
#   mid   = paper clusters 4, 5
#   late  = paper cluster 6
FIG8_MATURITY_FINE <- list(
  early = c("cluster_2", "cluster_3"),
  mid   = c("cluster_4", "cluster_5"),
  late  = c("cluster_6")
)

# Optional: restrict to Human1 metabolic genes only
# Provide path to text file with one gene symbol per line
# Set NULL to use all genes
GENE_SUBSET_FILE <- NULL

# -------------------------------------------------------------

set.seed(SEED)
dir.create(
  OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE
)
count_summary <- list()


# -- Helper functions -----------------------------------------

bootstrap_means <- function(expr_mat, n_boot, label) {
  n_cells <- ncol(expr_mat)
  cat("  Cells:", n_cells, "\n")

  if (n_cells == 0) {
    stop(
      "0 cells found for '", label, "'. ",
      "Check sample IDs in condition definition."
    )
  }
  if (n_cells < 30) {
    warning(
      n_cells, " cells in '", label,
      "' - bootstrap CIs may be unreliable."
    )
  }

  obs_mean <- rowMeans(expr_mat)

  cat("  Bootstrapping", n_boot, "iterations...\n")
  boot_mat <- matrix(
    NA_real_,
    nrow     = nrow(expr_mat),
    ncol     = n_boot,
    dimnames = list(
      rownames(expr_mat),
      sprintf("boot_%03d", seq_len(n_boot))
    )
  )
  for (b in seq_len(n_boot)) {
    idx <- sample(
      seq_len(n_cells),
      size    = n_cells,
      replace = TRUE
    )
    boot_mat[, b] <- rowMeans(
      expr_mat[, idx, drop = FALSE]
    )
    if (b %% 25 == 0) {
      cat("    iteration", b, "/", n_boot, "\n")
    }
  }

  list(
    obs_mean = obs_mean,
    boot_mat = boot_mat,
    n_cells  = n_cells
  )
}


save_bootstrap <- function(results, prefix) {
  obs_df <- tibble(
    gene          = names(results$obs_mean),
    observed_mean = as.numeric(results$obs_mean)
  )
  write_csv(
    obs_df,
    file.path(
      OUTPUT_DIR,
      paste0(prefix, "_observed_mean.csv")
    )
  )

  boot_df <- as.data.frame(results$boot_mat) %>%
    rownames_to_column("gene")
  write_csv(
    boot_df,
    file.path(
      OUTPUT_DIR,
      paste0(prefix, "_boot_means.csv")
    )
  )
  cat("  Saved:", prefix, "\n")
}


get_expr_mat <- function(seu, assay, slot,
                         gene_subset = NULL) {
  mat <- GetAssayData(seu, assay = assay, slot = slot)
  if (!is.null(gene_subset)) {
    keep <- intersect(gene_subset, rownames(mat))
    mat  <- mat[keep, , drop = FALSE]
  }
  mat
}


add_to_summary <- function(prefix, dataset,
                           condition, results,
                           expr_mat, samples_str) {
  count_summary[[prefix]] <<- data.frame(
    dataset   = dataset,
    condition = condition,
    n_cells   = results$n_cells,
    n_genes   = nrow(expr_mat),
    samples   = samples_str
  )
}


# -- Load gene subset -----------------------------------------

gene_subset <- NULL
if (!is.null(GENE_SUBSET_FILE) &&
    file.exists(GENE_SUBSET_FILE)) {
  gene_subset <- readLines(GENE_SUBSET_FILE)
  cat(
    "Gene subset loaded:",
    length(gene_subset), "genes\n"
  )
}


# =============================================================
# Fig 6 — disease conditions
# =============================================================

if (RUN_FIG6) {
  cat("\n")
  cat(strrep("=", 50), "\n")
  cat("Processing Fig 6\n")
  cat(strrep("=", 50), "\n")

  cat("Loading:", FIG6_PATH, "\n")
  s6 <- readRDS(FIG6_PATH)
  cat(
    "Object:", nrow(s6), "genes x",
    ncol(s6), "cells\n"
  )

  # clustering_paper already has paper annotation numbers
  s6@meta.data$paper_cluster <- as.character(
    s6@meta.data$clustering_paper
  )

  expr6 <- get_expr_mat(s6, ASSAY, SLOT, gene_subset)
  cat("Expression matrix:", nrow(expr6), "genes\n")

  # Verify sample column matches expectations
  cat("\nSample IDs found in object:\n")
  print(sort(unique(s6@meta.data$sample)))

  for (cond_name in names(FIG6_CONDITIONS)) {
    cat("\n-- Fig6:", cond_name, "--\n")

    sample_ids <- FIG6_CONDITIONS[[cond_name]]
    cell_mask  <- s6@meta.data$sample %in% sample_ids
    cell_ids   <- colnames(s6)[cell_mask]

    cat(
      "  Looking for samples:",
      paste(sample_ids, collapse = ", "), "\n"
    )

    if (length(cell_ids) == 0) {
      warning(
        "No cells found for '", cond_name, "'. ",
        "Check sample IDs match object."
      )
      next
    }

    mat     <- expr6[, cell_ids, drop = FALSE]
    results <- bootstrap_means(mat, N_BOOT, cond_name)
    prefix  <- paste0("Fig6_", cond_name)
    save_bootstrap(results, prefix)
    add_to_summary(
      prefix, "Fig6", cond_name, results, mat,
      paste(sample_ids, collapse = ";")
    )
  }

  # BM Healthy as reference
  cat("\n-- Fig6: BM_Healthy (reference) --\n")
  bm_mask <- s6@meta.data$sample %in% FIG6_BM_SAMPLES
  bm_ids  <- colnames(s6)[bm_mask]
  if (length(bm_ids) > 0) {
    bm_mat  <- expr6[, bm_ids, drop = FALSE]
    results <- bootstrap_means(
      bm_mat, N_BOOT, "BM_Healthy"
    )
    save_bootstrap(results, "Fig6_BM_Healthy")
    add_to_summary(
      "Fig6_BM_Healthy", "Fig6", "BM_Healthy",
      results, bm_mat,
      paste(FIG6_BM_SAMPLES, collapse = ";")
    )
  }

  rm(s6, expr6); gc()
}


# =============================================================
# Fig 8 — cord blood maturation axis
# =============================================================

if (RUN_FIG8) {
  cat("\n")
  cat(strrep("=", 50), "\n")
  cat("Processing Fig 8\n")
  cat(strrep("=", 50), "\n")

  cat("Loading:", FIG8_PATH, "\n")
  s8 <- readRDS(FIG8_PATH)
  cat(
    "Object:", nrow(s8), "genes x",
    ncol(s8), "cells\n"
  )

  # Map seurat_clusters to paper annotations
  s8@meta.data$paper_cluster <- FIG8_CLUSTER_MAP[
    as.character(s8@meta.data$seurat_clusters)
  ]

  cat("\nPaper cluster distribution:\n")
  print(table(
    s8@meta.data$paper_cluster, useNA = "ifany"
  ))

  expr8 <- get_expr_mat(s8, ASSAY, SLOT, gene_subset)
  cat("Expression matrix:", nrow(expr8), "genes\n")

  # -- By CB condition ----------------------------------------
  for (cb_cond in FIG8_CB_CONDITIONS) {
    cat("\n-- Fig8:", cb_cond, "--\n")

    cell_mask <- s8@meta.data$orig.ident == cb_cond
    cell_ids  <- colnames(s8)[cell_mask]

    if (length(cell_ids) == 0) {
      warning("No cells for CB condition: ", cb_cond)
      next
    }

    mat     <- expr8[, cell_ids, drop = FALSE]
    results <- bootstrap_means(mat, N_BOOT, cb_cond)
    prefix  <- paste0("Fig8_", cb_cond)
    save_bootstrap(results, prefix)
    add_to_summary(
      prefix, "Fig8", cb_cond,
      results, mat, cb_cond
    )
  }

  # -- Scheme A: binary maturity groups ----------------------
  cat("\n-- Fig8 maturity: binary scheme --\n")
  for (grp in names(FIG8_MATURITY_BINARY)) {
    cat("\n  Group:", grp, "\n")

    clusters  <- FIG8_MATURITY_BINARY[[grp]]
    cell_mask <- s8@meta.data$paper_cluster %in% clusters
    cell_ids  <- colnames(s8)[cell_mask]

    if (length(cell_ids) == 0) {
      warning("No cells for binary group: ", grp)
      next
    }

    mat     <- expr8[, cell_ids, drop = FALSE]
    results <- bootstrap_means(mat, N_BOOT, grp)
    prefix  <- paste0("Fig8_binary_", grp)
    save_bootstrap(results, prefix)
    add_to_summary(
      prefix, "Fig8", paste0("binary_", grp),
      results, mat,
      paste(clusters, collapse = ";")
    )
  }

  # -- Scheme B: fine-grained maturity groups ----------------
  cat("\n-- Fig8 maturity: fine-grained scheme --\n")
  for (grp in names(FIG8_MATURITY_FINE)) {
    cat("\n  Group:", grp, "\n")

    clusters  <- FIG8_MATURITY_FINE[[grp]]
    cell_mask <- s8@meta.data$paper_cluster %in% clusters
    cell_ids  <- colnames(s8)[cell_mask]

    if (length(cell_ids) == 0) {
      warning("No cells for fine group: ", grp)
      next
    }

    mat     <- expr8[, cell_ids, drop = FALSE]
    results <- bootstrap_means(mat, N_BOOT, grp)
    prefix  <- paste0("Fig8_fine_", grp)
    save_bootstrap(results, prefix)
    add_to_summary(
      prefix, "Fig8", paste0("fine_", grp),
      results, mat,
      paste(clusters, collapse = ";")
    )
  }

  # -- Binary maturity x CB condition ------------------------
  cat("\n-- Fig8: binary maturity x CB condition --\n")
  for (grp in names(FIG8_MATURITY_BINARY)) {
    for (cb_cond in FIG8_CB_CONDITIONS) {
      clusters  <- FIG8_MATURITY_BINARY[[grp]]
      cell_mask <- (
        s8@meta.data$paper_cluster %in% clusters &
        s8@meta.data$orig.ident == cb_cond
      )
      cell_ids <- colnames(s8)[cell_mask]
      n        <- length(cell_ids)

      if (n < 50) {
        cat(
          "  Skipping", grp, "x", cb_cond,
          "- only", n, "cells\n"
        )
        next
      }

      cat(
        "  Processing:", grp, "x", cb_cond,
        "(", n, "cells)\n"
      )
      mat     <- expr8[, cell_ids, drop = FALSE]
      results <- bootstrap_means(
        mat, N_BOOT,
        label = paste(grp, cb_cond, sep = "_")
      )
      prefix <- paste0("Fig8_", grp, "_", cb_cond)
      save_bootstrap(results, prefix)
      add_to_summary(
        prefix, "Fig8",
        paste0(grp, "_", cb_cond),
        results, mat,
        paste(grp, cb_cond, sep = "_")
      )
    }
  }

  rm(s8, expr8); gc()
}


# =============================================================
# Save cell count summary
# =============================================================

summary_df <- bind_rows(count_summary)
write_csv(
  summary_df,
  file.path(OUTPUT_DIR, "cell_counts_summary.csv")
)

cat("\n")
cat(strrep("=", 50), "\n")
cat("Cell count summary:\n")
print(
  summary_df[, c("dataset", "condition", "n_cells")]
)
cat(strrep("=", 50), "\n")
cat("Step 00 complete.\n")
cat("Output folder:", OUTPUT_DIR, "\n")
cat("Files generated:\n")
cat(
  paste(
    list.files(OUTPUT_DIR, pattern = "\\.csv$"),
    collapse = "\n"
  ),
  "\n"
)
cat("Next: sbatch slurm/01_uncertainty_fba.sh\n")