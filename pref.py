# ─────────────────────────────────────────────────────────────────────────────
# pref.py  –  preference / hyperparameter grid for the ablation study
# ─────────────────────────────────────────────────────────────────────────────

# Fixed environment shared across all runs
START      = [2, 2]
FINISH     = [13, 13]
BOUNDS     = [0, 25]       # same as original `range`
OBSTACLES  = [
    (12, 7, 1), (4, 6, 2), (8, 8, 1), (10, 11, 2),
    (6, 13, 1.5), (9, 4, 1.5), (3, 11, 1), (13, 4, 1),
]

# How many independent trials per configuration (for statistical stability)
TRIALS_PER_CONFIG = 100

# Random seed base (trial k uses seed BASE_SEED + k)
BASE_SEED = 42

# Output paths
CSV_OUTPUT  = "ablation_results.csv"
FIG_DIR     = "figures"   # directory where per-config plots are saved

# ─────────────────────────────────────────────────────────────────────────────
# Ablation grid: list of (epoch, expand_dis) pairs to sweep
# ─────────────────────────────────────────────────────────────────────────────

EPOCH_VALUES       = [100, 250, 500, 1000]
EXPAND_DIS_VALUES  = [1, 2, 3, 5]

# Build full Cartesian product automatically
CONFIGS = [
    {"epoch": e, "expand_dis": d}
    for e in EPOCH_VALUES
    for d in EXPAND_DIS_VALUES
]
