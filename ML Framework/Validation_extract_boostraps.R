# =============================================================
# Step 00b: QC + cluster + identify neutrophils + bootstrap
#           for validation cohorts (GSE167363, GSE145926)
# =============================================================
#
# Unlike the primary Montaldo data (Step 00), these are RAW
# 10x Cell Ranger outputs (matrix.mtx + features.tsv +
# barcodes.tsv) with NO existing cell type annotation.
# This script must therefore do what Montaldo et al. already
# did for us in the primary dataset:
#   1. Load raw counts per sample
#   2. QC filter (min genes/cell, min cells/gene, mito %)
#   3. Normalize + cluster (PCA -> UMAP -> Leiden)
#   4. Identify neutrophil cluster(s) via marker genes
#   5. Subset to neutrophils only
#   6. Bootstrap expression means (same as Step 00)
#
# Neutrophil markers (standard, consistent with Montaldo
# et al. gating / annotation approach):
#   FCGR3B, CSF3R, S100A8, S100A9, MPO, ELANE,
#   CXCR2, FUT4, MNDA
#
# Outputs -> data/validation/{cohort}/bootstrap_expression/:
#   {cohort}_{condition}_boot_means.csv
#   {cohort}_{condition}_observed_mean.csv
#   {cohort}_neutrophil_qc_summary.csv
#   {cohort}_cluster_marker_scores.csv
#
# Usage:
#   Rscript scripts/step00_extract/00b_validation_extract_bootstrap.R \
#     --cohort GSE167363
#   Rscript scripts/step00_extract/00b_validation_extract_bootstrap.R \
#     --cohort GSE145926
# HPC:
#   sbatch slurm/00b_validation_extract.sh GSE167363
#   sbatch slurm/00b_validation_extract.sh GSE145926
# =============================================================

library(Seurat)
library(Matrix)
library(dplyr)
library(readr)
library(tibble)
library(stringr)

# -- CONFIG ---------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
cohort_arg <- args[which(args == "--cohort") + 1]
if (length(cohort_arg) == 0 || is.na(cohort_arg)) {
  stop("Usage: Rscript 00b_validation_extract_bootstrap.R --cohort <GSE_ID>")
}
COHORT <- cohort_arg
cat("Cohort:", COHORT, "\n")

RAW_DIR    <- file.path("data/validation", COHORT)
OUTPUT_DIR <- file.path(
  "data/validation", COHORT, "bootstrap_expression"
)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

N_BOOT <- 100
SEED   <- 42

# QC thresholds
MIN_GENES_PER_CELL <- 200
MAX_GENES_PER_CELL <- 6000
MIN_CELLS_PER_GENE <- 3
MAX_MITO_PCT       <- 10
MIN_UMI            <- 1000

# Clustering
N_PCS           <- 30
CLUSTER_RES     <- 0.5

# Neutrophil marker genes (standard panel, consistent with
# typical PBMC/BALF neutrophil gating used in Montaldo et al.
# and related literature)
NEUTROPHIL_MARKERS <- c(
  "FCGR3B", "CSF3R", "S100A8", "S100A9",
  "MPO", "ELANE", "CXCR2", "FUT4", "MNDA"
)

# Minimum mean marker score (z-scored) for a cluster to be
# called "neutrophil". Adjust if too strict/lenient based on
# QC summary output.
NEUTROPHIL_SCORE_THRESHOLD <- 0.5

# Condition mapping per cohort -- derived from GEO sample
# metadata (GSM titles). Maps directory name pattern -> label.
# GSE167363: sepsis (Qiu et al.) -- HC/NS/S labeled samples
# GSE145926: COVID-19 (Liao et al.) -- HC/M/S labeled samples
COHORT_CONDITION_MAP <- list(
  GSE167363 = list(
    Healthy  = c("HC1", "HC2"),
    Survivor = c("P50_T6","P50_T0", "NSES_T0", "NSES_T6"),
    Fatal    = c("S2_T0", "S2_T6", "P25_T6", "P25_T0", "S3_T0", "S3_T6")
  )
)
if (!COHORT %in% names(COHORT_CONDITION_MAP)) {
  stop(
    "Unknown cohort '", COHORT, "'. ",
    "Add its condition map to COHORT_CONDITION_MAP."
  )
}
CONDITION_MAP <- COHORT_CONDITION_MAP[[COHORT]]


# -- Helper: bootstrap (same logic as primary Step 00) --------

bootstrap_means <- function(expr_mat, n_boot, label) {
  n_cells <- ncol(expr_mat)
  cat("  Cells:", n_cells, "\n")

  if (n_cells == 0) {
    warning("0 cells for '", label, "' -- skipping.")
    return(NULL)
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
      seq_len(n_cells), size = n_cells, replace = TRUE
    )
    boot_mat[, b] <- rowMeans(expr_mat[, idx, drop = FALSE])
  }

  list(obs_mean = obs_mean, boot_mat = boot_mat,
       n_cells = n_cells)
}


save_bootstrap <- function(results, prefix) {
  obs_df <- tibble(
    gene          = names(results$obs_mean),
    observed_mean = as.numeric(results$obs_mean)
  )
  write_csv(
    obs_df,
    file.path(OUTPUT_DIR, paste0(prefix, "_observed_mean.csv"))
  )

  boot_df <- as.data.frame(results$boot_mat) %>%
    rownames_to_column("gene")
  write_csv(
    boot_df,
    file.path(OUTPUT_DIR, paste0(prefix, "_boot_means.csv"))
  )
  cat("  Saved:", prefix, "\n")
}


# =============================================================
# Step 1: Load all 10x samples for this cohort
# =============================================================

cat("\n", strrep("=", 50), "\n")
cat("Loading 10x samples for", COHORT, "\n")
cat(strrep("=", 50), "\n")

gsm_dirs <- list.dirs(RAW_DIR, recursive = FALSE)
gsm_dirs <- gsm_dirs[
  grepl("^GSM", basename(gsm_dirs))
]
cat("Found", length(gsm_dirs), "GSM sample folders\n")

seurat_list <- list()
for (d in gsm_dirs) {
  gsm_id <- basename(d)
  cat("\nLoading:", gsm_id, "from", d, "\n")

  # Find the matrix/features/barcodes files (handle naming
  # variants across GEO submissions -- files may be prefixed
  # with GSM ID and/or sample name, e.g.
  # GSM5511356_S3_T6_matrix.mtx, and may or may not be gzipped)
  mtx_file  <- list.files(
    d, pattern = "matrix\\.mtx(\\.gz)?$",
    full.names = TRUE, ignore.case = TRUE
  )
  feat_file <- list.files(
    d, pattern = "(features|genes)\\.tsv(\\.gz)?$",
    full.names = TRUE, ignore.case = TRUE
  )
  bc_file   <- list.files(
    d, pattern = "barcodes\\.tsv(\\.gz)?$",
    full.names = TRUE, ignore.case = TRUE
  )

  if (length(mtx_file) == 0 || length(feat_file) == 0 ||
      length(bc_file) == 0) {
    cat("  WARNING: missing 10x files in", d, "-- skipping\n")
    cat("  Folder contents:\n")
    all_files <- list.files(d, full.names = FALSE)
    if (length(all_files) == 0) {
      cat("    (empty folder)\n")
    } else {
      cat("   ", paste(all_files, collapse = "\n    "), "\n")
    }
    next
  }

  counts <- ReadMtx(
    mtx      = mtx_file[1],
    features = feat_file[1],
    cells    = bc_file[1]
  )

  # Extract sample title from the filename itself, since
  # the GSM folder name (e.g. GSM5511356) does not contain
  # the condition-identifying string -- that lives in the
  # filename instead (e.g. GSM5511356_S3_T6_matrix.mtx ->
  # sample title is "S3_T6")
  mtx_basename <- basename(mtx_file[1])
  sample_title <- sub(
    "^GSM[0-9]+_(.*)_matrix\\.mtx.*$", "\\1",
    mtx_basename, ignore.case = TRUE
  )
  if (sample_title == mtx_basename) {
    # Fallback: pattern didn't match, use folder name
    sample_title <- gsm_id
  }
  cat("  Sample title extracted:", sample_title, "\n")

  obj <- CreateSeuratObject(
    counts   = counts,
    project  = gsm_id,
    min.cells = 0, min.features = 0
  )
  obj$gsm_id <- gsm_id
  obj$sample_title <- sample_title
  obj$cohort <- COHORT

  seurat_list[[gsm_id]] <- obj
  cat("  Loaded:", ncol(obj), "cells x", nrow(obj), "genes\n")
}

if (length(seurat_list) == 0) {
  stop("No valid 10x samples found in ", RAW_DIR)
}

# Merge all samples into one object
cat("\nMerging", length(seurat_list), "samples...\n")
merged <- merge(
  seurat_list[[1]],
  y = seurat_list[-1],
  add.cell.ids = names(seurat_list)
)
cat(
  "Merged object:", ncol(merged), "cells x",
  nrow(merged), "genes\n"
)

cat("\nSample titles found (for condition matching):\n")
print(table(merged$sample_title))

rm(seurat_list); gc()

# CRITICAL: Seurat v5 keeps each merged sample as a SEPARATE
# layer (data.GSM1, data.GSM2, ...) instead of one combined
# matrix. Downstream functions like LayerData/GetAssayData
# will silently grab only the FIRST layer and ignore the
# rest, causing wrong results (e.g. neutrophil % computed
# from one sample only) or "subscript out of bounds" errors
# when indexing cells from other samples.
# JoinLayers() merges them back into a single "data" layer.
cat("\nJoining Seurat v5 layers across all samples...\n")
merged <- JoinLayers(merged)
cat("Layers after join:", paste(Layers(merged), collapse = ", "), "\n")


# =============================================================
# Step 2: QC filtering
# =============================================================

cat("\n", strrep("=", 50), "\n")
cat("QC filtering\n")
cat(strrep("=", 50), "\n")

merged[["percent.mt"]] <- PercentageFeatureSet(
  merged, pattern = "^MT-"
)

cat("Before QC:", ncol(merged), "cells\n")
merged <- subset(
  merged,
  subset = nFeature_RNA > MIN_GENES_PER_CELL &
           nFeature_RNA < MAX_GENES_PER_CELL &
           nCount_RNA > MIN_UMI &
           percent.mt < MAX_MITO_PCT
)
cat("After QC:", ncol(merged), "cells\n")

qc_summary <- data.frame(
  cohort         = COHORT,
  n_samples      = length(unique(merged$gsm_id)),
  n_cells_post_qc = ncol(merged),
  n_genes        = nrow(merged)
)


# =============================================================
# Step 3: Normalize + cluster
# =============================================================

cat("\n", strrep("=", 50), "\n")
cat("Normalization + clustering\n")
cat(strrep("=", 50), "\n")

merged <- NormalizeData(merged, verbose = FALSE)
merged <- FindVariableFeatures(
  merged, selection.method = "vst",
  nfeatures = 2000, verbose = FALSE
)
merged <- ScaleData(merged, verbose = FALSE)
merged <- RunPCA(
  merged, npcs = N_PCS, verbose = FALSE
)
merged <- FindNeighbors(
  merged, dims = 1:N_PCS, verbose = FALSE
)
merged <- FindClusters(
  merged, resolution = CLUSTER_RES, verbose = FALSE
)

cat("\nCluster sizes:\n")
print(table(merged$seurat_clusters))


# =============================================================
# Step 4: Identify neutrophil cluster(s) via marker genes
# =============================================================

cat("\n", strrep("=", 50), "\n")
cat("Neutrophil identification\n")
cat(strrep("=", 50), "\n")

available_markers <- intersect(
  NEUTROPHIL_MARKERS, rownames(merged)
)
cat(
  "Neutrophil markers found in data:",
  paste(available_markers, collapse = ", "), "\n"
)
if (length(available_markers) == 0) {
  stop(
    "None of the neutrophil marker genes found in this ",
    "dataset. Check gene symbol format (e.g. case)."
  )
}

merged <- AddModuleScore(
  merged,
  features = list(available_markers),
  name     = "neutrophil_score"
)

# Mean neutrophil score per cluster
cluster_scores <- merged@meta.data %>%
  group_by(seurat_clusters) %>%
  summarise(
    mean_neutrophil_score = mean(neutrophil_score1),
    n_cells = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_neutrophil_score))

cat("\nCluster neutrophil scores:\n")
print(cluster_scores)

write_csv(
  cluster_scores,
  file.path(
    OUTPUT_DIR,
    paste0(COHORT, "_cluster_marker_scores.csv")
  )
)

neutrophil_clusters <- cluster_scores$seurat_clusters[
  cluster_scores$mean_neutrophil_score >
    NEUTROPHIL_SCORE_THRESHOLD
]
cat(
  "\nClusters called as neutrophils:",
  paste(neutrophil_clusters, collapse = ", "), "\n"
)

if (length(neutrophil_clusters) == 0) {
  warning(
    "No clusters passed the neutrophil score threshold ",
    "(", NEUTROPHIL_SCORE_THRESHOLD, "). Consider lowering ",
    "NEUTROPHIL_SCORE_THRESHOLD and re-running, or inspect ",
    "cluster_marker_scores.csv manually."
  )
}

merged$is_neutrophil <- merged$seurat_clusters %in%
  neutrophil_clusters

n_neutrophils <- sum(merged$is_neutrophil)
cat(
  "\nTotal neutrophils identified:", n_neutrophils,
  "/", ncol(merged), "cells",
  sprintf("(%.1f%%)", 100 * n_neutrophils / ncol(merged)),
  "\n"
)

qc_summary$n_neutrophils <- n_neutrophils
qc_summary$pct_neutrophils <- 100 * n_neutrophils / ncol(merged)
write_csv(
  qc_summary,
  file.path(
    OUTPUT_DIR, paste0(COHORT, "_neutrophil_qc_summary.csv")
  )
)


# =============================================================
# Step 5: Subset to neutrophils, assign conditions, bootstrap
# =============================================================

cat("\n", strrep("=", 50), "\n")
cat("Bootstrap expression per condition (neutrophils only)\n")
cat(strrep("=", 50), "\n")

neutro <- subset(merged, subset = is_neutrophil == TRUE)
rm(merged); gc()

# Defensive: subset() can sometimes re-fragment layers in
# Seurat v5. Re-join to guarantee a single unified matrix
# before extraction.
if (length(Layers(neutro, search = "data")) > 1) {
  cat("  Re-joining layers after subset...\n")
  neutro <- JoinLayers(neutro)
}

expr_mat <- LayerData(object=neutro, assay = "RNA", layer = "data")
cat("Neutrophil expression matrix:", nrow(expr_mat), "genes x",
    ncol(expr_mat), "cells\n")

set.seed(SEED)
count_summary <- list()

for (cond_name in names(CONDITION_MAP)) {
  cat("\n--", COHORT, ":", cond_name, "--\n")

  gsm_patterns <- CONDITION_MAP[[cond_name]]
  # Match by sample title substring (case-insensitive,
  # partial) -- NOT gsm_id, since the GSM accession itself
  # does not encode condition/timepoint information
  cell_mask <- rep(FALSE, ncol(neutro))
  for (pat in gsm_patterns) {
    cell_mask <- cell_mask | grepl(
      pat, neutro$sample_title, ignore.case = TRUE
    )
  }
  cell_ids <- colnames(neutro)[cell_mask]

  cat(
    "  Looking for sample title patterns:",
    paste(gsm_patterns, collapse = ", "), "\n"
  )

  if (length(cell_ids) == 0) {
    warning(
      "No neutrophils found for condition '", cond_name,
      "'. Check GSM title matching against ",
      "COHORT_CONDITION_MAP."
    )
    next
  }

  mat     <- expr_mat[, cell_ids, drop = FALSE]
  results <- bootstrap_means(mat, N_BOOT, cond_name)
  if (is.null(results)) next

  prefix <- paste0(COHORT, "_", cond_name)
  save_bootstrap(results, prefix)

  count_summary[[prefix]] <- data.frame(
    cohort    = COHORT,
    condition = cond_name,
    n_cells   = results$n_cells,
    n_genes   = nrow(mat)
  )
}

summary_df <- bind_rows(count_summary)
write_csv(
  summary_df,
  file.path(
    OUTPUT_DIR, paste0(COHORT, "_cell_counts_summary.csv")
  )
)

cat("\n", strrep("=", 50), "\n")
cat("Cell count summary:\n")
print(summary_df)
cat(strrep("=", 50), "\n")
cat("Step 00b complete for", COHORT, "\n")
cat("Output folder:", OUTPUT_DIR, "\n")
cat(
  "Next: run Step 01 (bootstrap FBA) pointing BOOT_DIR to\n  ",
  OUTPUT_DIR, "\n"
)