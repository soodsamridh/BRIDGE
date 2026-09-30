import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Architecture ──────────────────────────────────────────
GRU_HIDDEN = 32
MLP_WIDTH  = 32
GAMMA_RHO  = 3.0

# ── Loss weights (scalar problems) ───────────────────────
LAMBDA_DATA   = 1.0
LAMBDA_CONS   = 1.0
LAMBDA_ANCH   = 5.0
LAMBDA_FD     = 3.0
LAMBDA_POS    = 2.0

# ── Training (scalar problems) ────────────────────────────
EPOCHS          = 1000
LR              = 2e-3
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
EPOCHS_PREINIT  = 300
LR_PREINIT      = 3e-3
TAU_CONS        = 0.05
B_PAIRS         = 64
TBPTT_WINDOW    = 15
N_BINS          = 30
MIN_BIN_SAMPLES = 5
QUALITY_THRESH  = 0.25

PROBLEMS = {
    # ── Scalar benchmarks ──────────────────────────────────
    "fisher": {
        "name":          "Fisher-KPP",
        "eps_diff":      1.0,
        "domain":        (0.0, 1.0),
        "T":             0.3,
        "Nx":            128,
        "Nt":            60,
        "dt":            0.005,
        "bc_type":       "dirichlet",
        "Lambda":        2.0,
        "s_res":         1.0,
        "u_min":         0.0,
        "u_max":         1.0,
        "mesh_type":     "shishkin",
        "beta":          2.0,
        "grad_clip":     1.0,
        "positivity":    True,
        "fd_filter":     "slow",
        "fd_pct":        50,
        "ics":           ["original", "shifted_05", "shifted_08", "shifted_095"],
        "n_rollout_ics": 4,
        "anchors":       [(0.0, 0.0), (1.0, 0.0)],
        "tbptt":         15,
    },
    "allen_cahn": {
        "name":          "Allen-Cahn",
        "eps_diff":      1e-4,
        "domain":        (-1.0, 1.0),
        "T":             1.0,
        "Nx":            128,
        "Nt":            1000,
        "dt":            0.001,
        "bc_type":       "periodic",
        "Lambda":        3.0,
        "s_res":         1.5,
        "u_min":         -1.0,
        "u_max":          1.0,
        "mesh_type":     "uniform",
        "beta":          2.0,
        "grad_clip":     1.0,
        "positivity":    False,
        "fd_filter":     "lap",
        "fd_pct":        60,
        "ics":           ["x2cosx", "tanh0", "tanh_neg05"],
        "n_rollout_ics": 1,
        "anchors":       [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        "tbptt":         50,
    },
    "fhn": {
        "name":          "FitzHugh-Nagumo",
        "eps_diff":      0.05,
        "a_fhn":         0.25,
        "domain":        (0.0, 1.0),
        "T":             0.2,
        "Nx":            128,
        "Nt":            40,
        "dt":            0.005,
        "bc_type":       "dirichlet",
        "Lambda":        20.0,
        "s_res":         10.0,
        "u_min":         0.0,
        "u_max":         1.0,
        "mesh_type":     "shishkin",
        "beta":          2.0,
        "grad_clip":     0.3,
        "positivity":    False,
        "fd_filter":     "lap",
        "fd_pct":        60,
        "ics":           ["front03", "front06"],
        "n_rollout_ics": 2,
        "lambda_fd":     5.0,
        "anchors":       [(0.0, 0.0), (0.25, 0.0), (1.0, 0.0)],
        "tbptt":         15,
    }
}