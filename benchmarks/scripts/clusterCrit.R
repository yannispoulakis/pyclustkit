# ---------- Writable per-user library ----------
user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  if (.Platform$OS.type == "windows") {
    ver <- paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep=".")
    user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", ver)
  } else {
    user_lib <- file.path("~", "R", paste0("library-", R.version$major, ".", strsplit(R.version$minor, "\\.")[[1]][1]))
  }
}
user_lib <- normalizePath(path.expand(user_lib), mustWork = FALSE)
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing '", pkg, "' into: ", user_lib)
    ok <- FALSE
    try({ install.packages(pkg, lib = user_lib, type = "binary", quiet = TRUE); ok <- TRUE }, silent = TRUE)
    if (!ok) install.packages(pkg, lib = user_lib)  # source as fallback
  }
}

install_if_missing("jsonlite")
install_if_missing("clusterCrit")
library(jsonlite)
library(clusterCrit)


# ---------- Paths ----------
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir  <- dirname(script_path)
project_root <- normalizePath(file.path(script_dir, "..", ".."))
data_path    <- file.path(project_root, "benchmarks","data")
out_dir      <- file.path(project_root,"benchmarks", "results", "vsClusterCrit")
out_file     <- file.path(out_dir, "clustercrit_results.json")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- helpers ----
extract_meta <- function(path) {
  # expects filenames like "1000_10_something.csv" -> nsamples="1000", nfeatures="10"
  base_noext <- tools::file_path_sans_ext(basename(path))
  parts <- strsplit(base_noext, "_", fixed = TRUE)[[1]]
  if (length(parts) < 2) {
    stop(sprintf("Filename must start with 'nsamples_nfeatures_...': %s", path))
  }
  nsamples  <- parts[1]
  nfeatures <- parts[2]
  list(nsamples = nsamples, nfeatures = nfeatures)
}

read_dataset <- function(csv_path) {
  df <- utils::read.csv(csv_path, check.names = FALSE)
  if (!"target" %in% names(df)) {
    stop(sprintf("File '%s' must contain a 'target' column for labels.", csv_path))
  }
  y <- as.integer(factor(df[["target"]]))           # ensure labels 1..K
  X <- df[, setdiff(names(df), "target"), drop = FALSE]
  
  # Ensure all features are numeric
  non_num <- !vapply(X, is.numeric, logical(1))
  if (any(non_num)) {
    bad <- paste(names(X)[non_num], collapse = ", ")
    stop(sprintf("Non-numeric feature columns in '%s': %s", csv_path, bad))
  }
  list(X = as.matrix(X), y = y)
}

# ---- main ----
if (!dir.exists(data_path)) stop(sprintf("Data path not found: %s", data_path))

subdirs <- list.dirs(data_path, full.names = FALSE, recursive = FALSE)
results <- list()

for (folder in subdirs) {
  folder_path <- file.path(data_path, folder)
  csvs <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(csvs) == 0) next
  if (is.null(results[[folder]])) results[[folder]] <- list()

  for (csv_path in csvs) {
    meta <- extract_meta(csv_path)
    ns <- meta$nsamples
    nf <- meta$nfeatures
    
    if (is.null(results[[folder]][[ns]])) results[[folder]][[ns]] <- list()

    ds <- read_dataset(csv_path)

    start_time <- Sys.time()
    intCriteria(ds$X, ds$y, getCriteriaNames(TRUE))
    end_time <- Sys.time()
    time_taken_sec <- as.numeric(difftime(end_time, start_time, units = "secs"))
    print(ns)
    print(time_taken_sec)
    results[[folder]][[ns]][[nf]] <- time_taken_sec

    
  }
}

# ---- write JSON ----
write_json(results, out_file, pretty = TRUE, auto_unbox = TRUE)
message("Saved JSON: ", out_file)
