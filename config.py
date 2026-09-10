"""
config.py  —  ε-US-GRU v5 — all problem configurations.

Problems:
  fisher, allen_cahn, fhn, bistable  — scalar singularly perturbed RDEs
  fhn_bvp                            — Bonhoeffer-Van der Pol semi-known system
"""

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
    },
    "bistable": {
        "name":          "Bistable (Ginzburg-Landau)",
        "eps_diff":      0.01,
        "domain":        (0.0, 1.0),
        "T":             0.3,
        "Nx":            128,
        "Nt":            60,
        "dt":            0.005,
        "bc_type":       "dirichlet",
        "Lambda":        1.0,
        "s_res":         0.5,
        "u_min":         -1.0,
        "u_max":          1.0,
        "mesh_type":     "shishkin",
        "beta":          2.0,
        "grad_clip":     1.0,
        "positivity":    False,
        "fd_filter":     "lap",
        "fd_pct":        60,
        "ics":           ["tanh03", "tanh04", "tanh05", "tanh06"],
        "n_rollout_ics": 4,
        "anchors":       [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        "tbptt":         15,
    },

    # ── FHN Semi-known system (Keener-Sneyd form) ────────
    #
    #  ∂u/∂t = ε·u_xx + R_u(u,v)         ← IDENTIFY R_u(u,v) = u(1-u)(u-a) - v
    #  ∂v/∂t = δ(u − v)                   ← KNOWN ODE, solved explicitly
    #
    #  ε=0.01 (genuine sharp layers → Shishkin mesh),  a=0.10,  δ=0.5
    #  Domain [0,1],  T=0.5,  Neumann BCs
    #  ICs: u and v chosen INDEPENDENTLY for diverse (u,v) state-space coverage
    #  max|R_u_true| = 1.0 over [0,1]²
    #
    #  Citations:
    #    Keener J, Sneyd J (2009) Mathematical Physiology I. Springer. Ch.5.2
    #    FitzHugh R (1961) Biophysical Journal 1(6):445-466
    "fhn_semi": {
        "name":           "FHN Semi-known (Keener-Sneyd)",
        "eps_diff":       0.01,    # activator diffusion — sharp layers
        "a_fhn":          0.10,    # excitation threshold
        "delta_v":        0.50,    # recovery coupling strength
        "domain":         (0.0, 1.0),
        "T":              0.5,
        "Nx":             128,
        "Nt":             100,
        "dt":             0.005,
        "bc_type":        "neumann",
        "Lambda":         1.5,     # max|R_u| ≈ 1.0 → Lambda=1.5 safe
        "u_min":          0.0,
        "u_max":          1.0,
        "v_min":          0.0,
        "v_max":          1.0,
        "mesh_type":      "shishkin",
        "beta":           2.0,     # Shishkin transition parameter
        "grad_clip":      0.5,
        "positivity":     False,
        "fd_filter":      "lap",
        "fd_pct":         50,
        "ics":            ["front_v0", "step_v05", "bump_v02", "front2_v08"],
        "n_rollout_ics":  4,
        # Equilibrium: (u*,v*) = (0,0) only in [0,1]² (R_u(0,0)=0)
        "anchors_uv":     [(0.0, 0.0)],
        "tbptt":          20,
        "n_bins_2d":      12,
        # Loss ramps — all start small, anch and data dominate early
        "ramp": {
            "data":  {"start": 0.50,  "end": 1.0,  "over": 100},
            "fd":    {"start": 1.0,   "end": 3.0,  "over": 200},
            "cons":  {"start": 0.01,  "end": 1.0,  "over": 300},
            "anch":  {"start": 0.50,  "end": 5.0,  "over": 300},
        },
        "lr":             1e-3,    # standard LR — ε=0.01 is well-conditioned
        "lr_peak":        2e-3,
        "lr_warmup":      50,
        "epochs":         500,
        "preinit_epochs": 400,
        "gru_detach_epochs": 0,   # 0 = no detach phase needed (ε=0.01 stable)
    },
}