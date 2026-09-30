import sys, os, time, json, copy, argparse
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from numerics import build_shishkin_mesh, build_compact_laplacian

SEED      = 42
EPOCHS    = 2000
OUT_DIR   = "eps_study_systems"
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Runtime switches, set once in main().
USE_CONS      = True       # --no-cons disables
LEGACY_RHO    = False      # --legacy-rho
LAMBDA_MODE   = "auto"     # --lambda-mode {auto,config}
LAMBDA_SCALE  = 1.0        # --lambda-scale, multiplies the auto bound
LAMBDA_SAFETY = 1.25       # headroom over the observed FD residual
LAMBDA_MIN_PER_BIN = 8     # samples a bin needs before it can set the scale


# ======================================================================
#  EPS VALUES AND STANDARD VALUE PER SYSTEM
# ======================================================================

EPS_VALUES = {
    "fhn_partial":   [0.5, 0.1, 0.05, 0.01, 0.005],
    "predator_prey": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
}
STANDARD_EPS = {
    "fhn_partial":   0.05,
    "predator_prey": 1e-3,
}


# ======================================================================
#  BASE SYSTEM CONFIGS
# ======================================================================

BASE_CFGS = {

    "fhn_partial": {
        "name": "FitzHugh-Nagumo Partial (Neuroscience)",
        "eps_diff": 0.05, "delta_v": 0.1, "beta_v": 1.0,
        "gamma_v": 0.5, "a_fhn": 0.25,
        # Lambda here is only the fallback for --lambda-mode config; in
        # auto mode it is replaced per eps by the FD-derived bound.
        "Lambda": 2.0, "u_min": 0.0, "u_max": 1.0,
        "v_min": 0.0, "v_max": 0.6,
        # Nx = 128, matching Table 1 and comparison_systems. The earlier
        # value of 52 in this file was inconsistent with both.
        "domain": (0.0, 1.0), "T": 0.3, "Nx": 128, "Nt": 60, "dt": 0.005,
        "bc_type": "dirichlet", "beta_mesh": 2.0,
        "tbptt": 1, "n_bins_2d": 12,
        "anchors_uv": [(0.0, 0.0), (0.25, 0.0), (1.0, 0.0)],
        "ics": [("front", 0.3, 0.0), ("front", 0.6, 0.0),
                ("front", 0.3, 0.3), ("step", 0.5, 0.5)],
        "input_dim": 5,
    },

    "predator_prey": {
        "name": "Predator-Prey Holling Type II (Ecology)",
        "alpha_pp": 1.0, "beta_pp": 0.1, "gamma_pp": 0.5, "delta_pp": 0.25,
        "D1": 1e-3, "D2": 1e-2, "eps_diff": 1e-3, "Lambda": 0.35,
        "u_min": 0.0, "u_max": 0.5, "v_min": 0.0, "v_max": 0.4,
        "u_star": 0.1, "v_star": 0.18,
        "domain": (0.0, 1.0), "T": 2.0, "Nx": 128, "Nt": 200, "dt": 0.01,
        "bc_type": "neumann", "beta_mesh": 2.0,
        "tbptt": 1, "n_bins_2d": 10,
        "anchors_uv": [(0.0, 0.1), (0.1, 0.18), (0.0, 0.0)],
        "ics": [(0.30, 0.12, 1, 1), (0.25, 0.10, 1, 1), (0.20, 0.08, 2, 2),
                (0.28, 0.11, 1, 2), (0.35, 0.14, 1, 1), (0.22, 0.09, 2, 1),
                (0.18, 0.07, 1, 2), (0.32, 0.13, 2, 2)],
        "input_dim": 5,
    },
}


def build_eps_cfg(system, eps, base, nx_override=None):
    cfg = copy.deepcopy(base)
    cfg["eps_diff"] = float(eps)
    if system == "predator_prey":
        cfg["D1"] = float(eps)          # eps = D1 for PP
    if nx_override is not None:
        cfg["Nx"] = int(nx_override)
    return cfg


# ======================================================================
#  TRUE REACTIONS (evaluation only)
# ======================================================================

def get_R_true(system, u, v, cfg):
    if system == "fhn_partial":
        return (1.0 / cfg["eps_diff"]) * u * (u - cfg["a_fhn"]) * (1.0 - u) - v
    if system == "predator_prey":
        alp = cfg["alpha_pp"]; bet = cfg["beta_pp"]
        return np.clip(u, 0, None) * (1 - np.clip(u, 0, None)) - \
            alp * np.clip(u, 0, None) * np.clip(v, 0, None) / \
            (bet + np.clip(u, 0.001, None))
    raise ValueError(f"Unknown system: {system}")


# ======================================================================
#  MESH, IMEX, AND A FACTORISED SOLVER
# ======================================================================

def build_laplacian(cfg):
    xl, xr = cfg["domain"]
    x, _ = build_shishkin_mesh(cfg["Nx"], xl, xr,
                               cfg["eps_diff"], cfg["beta_mesh"])
    return x, build_compact_laplacian(x, periodic=False)


def make_imex(D, L_np, cfg, neumann=True):
    dt = cfg["dt"]; N = cfg["Nx"]
    A = np.eye(N + 1) - (dt / 2) * D * L_np
    if neumann:
        A[0, :] = 0; A[0, 0] = 1; A[0, 1] = -1
        A[-1, :] = 0; A[-1, -1] = 1; A[-1, -2] = -1
    else:
        A[0, :] = 0; A[0, 0] = 1
        A[-1, :] = 0; A[-1, -1] = 1
    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv, torch.tensor(A, dtype=torch.float32)


def make_solver(A_t):
    try:
        LU, piv = torch.linalg.lu_factor(A_t)
        return ("lu", LU, piv)
    except Exception as e:                      # very old torch
        print(f"    (lu_factor unavailable: {e}; falling back to solve)")
        return ("dense", A_t, None)


def imex_solve(solver, rhs):
    kind, a, b = solver
    if kind == "lu":
        return torch.linalg.lu_solve(a, b, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(a, rhs.unsqueeze(-1)).squeeze(-1)


# ======================================================================
#  INITIAL CONDITIONS AND REFERENCE TRAJECTORIES
# ======================================================================

def make_ics(system, x, cfg):
    if system == "fhn_partial":
        eps = cfg["eps_diff"]; ics = []
        for (kind, loc, v0) in cfg["ics"]:
            u0 = (0.5 * (1 + np.tanh((x - loc) / np.sqrt(eps)))
                  if kind == "front" else np.where(x < loc, 0.9, 0.05))
            ics.append((u0.copy(), v0 * np.ones_like(x)))
        return ics
    if system == "predator_prey":
        us = cfg["u_star"]; vs = cfg["v_star"]; ics = []
        for (au, av, mu, mv) in cfg["ics"]:
            u0 = np.clip(us + au * np.cos(mu * np.pi * x),
                         cfg["u_min"] + 0.001, cfg["u_max"])
            v0 = np.clip(vs + av * np.sin(mv * np.pi * x),
                         cfg["v_min"] + 0.001, cfg["v_max"])
            ics.append((u0.copy(), v0.copy()))
        return ics


def _lus(lu, piv, rhs, lo, hi):
    return np.clip(scipy.linalg.lu_solve((lu, piv), rhs), lo, hi)


def generate_reference(system, x, cfg, L_np):
    dt = cfg["dt"]; Nt = cfg["Nt"]
    t_arr = np.linspace(0, cfg["T"], Nt + 1); refs = []
    if system == "fhn_partial":
        eps = cfg["eps_diff"]; dv = cfg["delta_v"]
        bv = cfg["beta_v"]; gv = cfg["gamma_v"]
        _, lu_u, pu, _ = make_imex(eps, L_np, cfg, neumann=False)
        _, lu_v, pv, _ = make_imex(dv, L_np, cfg, neumann=False)
        for u0, v0 in make_ics(system, x, cfg):
            ur = np.zeros((Nt + 1, len(x))); vr = np.zeros((Nt + 1, len(x)))
            ur[0] = u0; vr[0] = v0
            for n in range(Nt):
                un = ur[n]; vn = vr[n]
                Ru = (1 / eps) * un * (un - cfg["a_fhn"]) * (1 - un) - vn
                rhs_u = un + (dt / 2) * eps * (L_np @ un) + dt * Ru
                rhs_u[0] = 0.0; rhs_u[-1] = 1.0
                ur[n + 1] = _lus(lu_u, pu, rhs_u, -0.05, 1.05)
                rhs_v = vn + (dt / 2) * dv * (L_np @ vn) + dt * (bv * un - gv * vn)
                rhs_v[0] = 0.0; rhs_v[-1] = 0.0
                vr[n + 1] = _lus(lu_v, pv, rhs_v, -0.05, 1.05)
            refs.append((ur, vr))
    elif system == "predator_prey":
        D1 = cfg["D1"]; D2 = cfg["D2"]
        alp = cfg["alpha_pp"]; bet = cfg["beta_pp"]
        gam = cfg["gamma_pp"]; dlt = cfg["delta_pp"]
        _, lu_u, pu, _ = make_imex(D1, L_np, cfg, neumann=True)
        _, lu_v, pv, _ = make_imex(D2, L_np, cfg, neumann=True)
        for u0, v0 in make_ics(system, x, cfg):
            ur = np.zeros((Nt + 1, len(x))); vr = np.zeros((Nt + 1, len(x)))
            ur[0] = np.clip(u0, 0.001, None); vr[0] = np.clip(v0, 0.001, None)
            for n in range(Nt):
                un = np.clip(ur[n], 0.001, None)
                vn = np.clip(vr[n], 0.001, None)
                Rprey = alp * un * vn / (bet + un)
                Rfull = un * (1 - un) - Rprey
                rhs_u = un + (dt / 2) * D1 * (L_np @ un) + dt * Rfull
                rhs_u[0] = 0; rhs_u[-1] = 0
                ur[n + 1] = _lus(lu_u, pu, rhs_u, -0.01, 0.65)
                rhs_v = vn + (dt / 2) * D2 * (L_np @ vn) + \
                    dt * (gam * Rprey - dlt * vn)
                rhs_v[0] = 0; rhs_v[-1] = 0
                vr[n + 1] = _lus(lu_v, pv, rhs_v, -0.01, 0.65)
            refs.append((ur, vr))
    return refs, t_arr


# ======================================================================
#  FD EXTRACTION AND THE DATA-DERIVED OUTPUT BOUND
# ======================================================================

def _fd_residual(ur, L_np, D, dt, n):
    return (ur[n + 1] - ur[n - 1]) / (2 * dt) - D * (L_np @ ur[n])


def estimate_lambda(refs, L_np, cfg, system, safety=LAMBDA_SAFETY):
    dt = cfg["dt"]
    D = cfg["eps_diff"] if system == "fhn_partial" else cfg["D1"]
    nb = cfg["n_bins_2d"]
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    x_mesh, _ = build_laplacian(cfg)
    u, v, R = fd_samples(refs, L_np, x_mesh, cfg, system)
    if len(u) == 0:
        return float(safety * 1.0)

    ui = np.clip(np.digitize(u, ue) - 1, 0, nb - 1)
    vi = np.clip(np.digitize(v, ve) - 1, 0, nb - 1)
    peaks = []
    for i in range(nb):
        for j in range(nb):
            m = (ui == i) & (vi == j)
            if m.sum() >= LAMBDA_MIN_PER_BIN:
                peaks.append(abs(float(np.median(R[m]))))
    if not peaks:                      # degenerate: nothing well sampled
        return float(safety * np.median(np.abs(R)) * 10.0)
    return float(safety * max(peaks))


def lambda_coverage(system, cfg, Lambda, ng=60):
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg)
    Rt = get_R_true(system, UU.ravel(), VV.ravel(), cfg)
    best = np.clip(Rt, -Lambda, Lambda)
    floor = float(np.sqrt(np.mean((best - Rt) ** 2)) /
                  (np.sqrt(np.mean(Rt ** 2)) + 1e-8))
    return float(np.abs(Rt).max()), floor

FD_TOL_TIME  = 0.10   # --fd-tol-time
FD_TOL_SPACE = 0.25   # --fd-tol-space
FD_SKIP      = 3      # interior nodes dropped at each end, see below


def fd_accept_mask(ur, x, L_np, D, dt, n,
                   tol_t=None, tol_x=None, skip=None):
    tol_t = FD_TOL_TIME if tol_t is None else tol_t
    tol_x = FD_TOL_SPACE if tol_x is None else tol_x
    skip = FD_SKIP if skip is None else skip

    R1 = (ur[n + 1] - ur[n - 1]) / (2.0 * dt) - D * (L_np @ ur[n])
    R2 = (ur[n + 2] - ur[n - 2]) / (4.0 * dt) - D * (L_np @ ur[n])
    s = float(np.median(np.abs(R1[2:-2]))) + 1e-12
    m = np.abs(R2 - R1) <= tol_t * (np.abs(R1) + 0.01 * s)

    u = ur[n]
    lap_f = L_np @ u
    lap_c = np.zeros_like(u)
    lap_c[2:-2] = (u[:-4] - 2 * u[2:-2] + u[4:]) / \
                  (((x[4:] - x[:-4]) / 2.0) ** 2)
    sc = float(np.median(np.abs(lap_f[2:-2]))) + 1e-12
    m = m & (np.abs(lap_c - lap_f) <= tol_x * (np.abs(lap_f) + 0.05 * sc))

    m = m.copy()
    if skip > 0:
        m[:skip] = False
        m[-skip:] = False
    return m, R1


def fd_samples(refs, L_np, x, cfg, system, **kw):
    dt = cfg["dt"]
    D = cfg["eps_diff"] if system == "fhn_partial" else cfg["D1"]
    U, V, R = [], [], []
    for ur, vr in refs:
        Nt = ur.shape[0] - 1
        for n in range(2, Nt - 1):
            m, R1 = fd_accept_mask(ur, x, L_np, D, dt, n, **kw)
            if m.sum():
                U.append(ur[n][m]); V.append(vr[n][m]); R.append(R1[m])
    if not U:
        z = np.zeros(0)
        return z, z, z
    return np.concatenate(U), np.concatenate(V), np.concatenate(R)


def visited_region(refs, cfg, nb=None):
    nb = nb or (cfg["n_bins_2d"] * 5)
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    uc = 0.5 * (ue[:-1] + ue[1:]); vc = 0.5 * (ve[:-1] + ve[1:])
    U = np.concatenate([ur[1:, 1:-1].ravel() for ur, vr in refs])
    V = np.concatenate([vr[1:, 1:-1].ravel() for ur, vr in refs])
    ui = np.clip(np.digitize(U, ue) - 1, 0, nb - 1)
    vi = np.clip(np.digitize(V, ve) - 1, 0, nb - 1)
    key = np.unique(ui.astype(np.int64) * nb + vi.astype(np.int64))
    return (uc[key // nb].astype(np.float32),
            vc[key % nb].astype(np.float32))


def extract_fd(refs, L_np, cfg, system, x=None):
    if x is None:
        x, _ = build_laplacian(cfg)
    return fd_samples(refs, L_np, x, cfg, system)


def aggregate_fd(u_fd, v_fd, R_fd, cfg, min_u_cols=4, verbose=True):
    nb = cfg["n_bins_2d"]
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    uc = 0.5 * (ue[:-1] + ue[1:]); vc = 0.5 * (ve[:-1] + ve[1:])
    ui = np.clip(np.digitize(u_fd, ue) - 1, 0, nb - 1)
    vi = np.clip(np.digitize(v_fd, ve) - 1, 0, nb - 1)
    u_t, v_t, R_t = [], [], []
    for i in range(nb):
        for j in range(nb):
            m = (ui == i) & (vi == j)
            if m.sum() >= 4:
                u_t.append(uc[i]); v_t.append(vc[j])
                R_t.append(np.median(R_fd[m]))
    u_t = np.array(u_t, dtype=np.float32)
    v_t = np.array(v_t, dtype=np.float32)
    R_t = np.array(R_t, dtype=np.float32)
    n_cols = len(np.unique(u_t)) if len(u_t) else 0
    if verbose:
        print(f"    FD bins: {len(u_t)} of {nb*nb}   distinct u columns: "
              f"{n_cols} of {nb}   max|R_target| = "
              f"{(np.abs(R_t).max() if len(R_t) else 0.0):.4g}")
        if n_cols < min_u_cols:
            print(f"    *** WARNING: the FD targets span only {n_cols} "
                  f"distinct u values. The supervision cannot constrain "
                  f"how R varies with u, the identified law will collapse "
                  f"toward a constant, and eps_R will approach 1.0. This "
                  f"row says nothing about the method. ***")
    return u_t, v_t, R_t


# ======================================================================
#  MODEL
# ======================================================================

class SystemModel(nn.Module):

    def __init__(self, hidden=32, mlp_w=64, Lambda=2.0, input_dim=5):
        super().__init__()
        self.hidden = hidden; self.mlp_w = mlp_w; self.Lambda = Lambda
        self.gru_out = nn.GRUCell(input_dim, hidden)
        self.gru_lay = nn.GRUCell(input_dim, hidden)
        self.gate = nn.Linear(hidden, mlp_w)
        self.mlp1 = nn.Linear(2, mlp_w)
        self.mlp2 = nn.Linear(mlp_w, mlp_w)
        self.mlp3 = nn.Linear(mlp_w, 1)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.mlp1.weight, std=0.10)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)

    def init_hidden(self, Nx):
        h = torch.zeros(Nx, self.hidden)
        return h, h.clone()

    def _decode(self, u, v, gamma):
        inp = torch.stack([u, v], dim=-1)
        h1 = F.silu(self.mlp1(inp)); h2 = F.silu(h1 + gamma)
        h3 = F.silu(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    def forward(self, u, v, z_out, z_lay, rho, H_out, H_lay):
        rho_e = rho.unsqueeze(-1)
        Hoc = self.gru_out(torch.clamp(z_out, -10, 10), H_out)
        Hlc = self.gru_lay(torch.clamp(z_lay, -10, 10), H_lay)
        Hon = (1 - rho_e) * Hoc + rho_e * H_out
        Hln = rho_e * Hlc + (1 - rho_e) * H_lay
        Hb = (1 - rho_e) * Hon + rho_e * Hln
        return self._decode(u, v, self.gate(Hb)), Hon, Hln

    def react_grad(self, u, v):
        gamma = torch.zeros(u.shape[0], self.mlp_w)
        return self._decode(u, v, gamma)

    @torch.no_grad()
    def react_nograd(self, u, v):
        return self.react_grad(u, v)


# ======================================================================
#  2-D DISTILLATION
# ======================================================================

class ReactionMLP2D(nn.Module):

    def __init__(self, hidden=64, Lambda=1.0):
        super().__init__()
        self.Lambda = Lambda
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u, v):
        uv = torch.stack([u, v], dim=-1)
        return self.Lambda * torch.tanh(self.net(uv)).squeeze(-1)


def distil_2d(model, system, cfg, Lambda, ng=60, epochs=2000,
              hidden=64, visited=None):
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg)
    uf = UU.ravel().astype(np.float32); vf = VV.ravel().astype(np.float32)

    u_t = torch.tensor(uf); v_t = torch.tensor(vf)
    with torch.no_grad():
        R_target = model.react_nograd(u_t, v_t)

    anchors = cfg.get("anchors_uv", [])
    if anchors:
        au = torch.tensor([a[0] for a in anchors], dtype=torch.float32)
        av = torch.tensor([a[1] for a in anchors], dtype=torch.float32)
        ar = torch.zeros(len(anchors), dtype=torch.float32)
        u_t = torch.cat([u_t, au]); v_t = torch.cat([v_t, av])
        R_target = torch.cat([R_target, ar])
        w = torch.cat([torch.ones(ng * ng), 20.0 * torch.ones(len(anchors))])
    else:
        w = torch.ones(ng * ng)
    w = w / w.mean()

    mlp = ReactionMLP2D(hidden=hidden, Lambda=Lambda)
    opt = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                 eta_min=1e-6)

    best = float("inf"); best_state = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = (w * (mlp(u_t, v_t) - R_target).pow(2)).mean()
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    if best_state:
        mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        R_dist = mlp(torch.tensor(uf), torch.tensor(vf)).numpy()
    R_true = get_R_true(system, uf.astype(np.float64),
                        vf.astype(np.float64), cfg)
    l2 = float(np.sqrt(np.mean((R_dist - R_true) ** 2)) /
               (np.sqrt(np.mean(R_true ** 2)) + 1e-8))
    l2_vis = float("nan")
    if visited is not None and len(visited[0]):
        uv, vv = visited
        with torch.no_grad():
            Rv = mlp(torch.tensor(uv), torch.tensor(vv)).numpy()
        Rtv = get_R_true(system, uv.astype(np.float64),
                         vv.astype(np.float64), cfg)
        l2_vis = float(np.sqrt(np.mean((Rv - Rtv) ** 2)) /
                       (np.sqrt(np.mean(Rtv ** 2)) + 1e-8))

    print(f"    distillation: fit loss={best:.3e}   "
          f"l2_distil(box)={l2:.4f}   l2_distil(visited)={l2_vis:.4f}")
    return mlp, UU, VV, R_dist.reshape(ng, ng), l2, l2_vis


@torch.no_grad()
def conditioning_ratio(model, system, refs, cfg, x_t, phi_t, solver, L_t,
                       max_ics=2):
    num = 0.0; den = 0.0
    for ur_np, vr_np in refs[:max_ics]:
        ur_t = torch.tensor(ur_np, dtype=torch.float32)
        vr_t = torch.tensor(vr_np, dtype=torch.float32)
        H_out, H_lay = model.init_hidden(x_t.shape[0])
        u_n = ur_t[0]; u_prev = ur_t[0]
        for step in range(cfg["Nt"]):
            v_ref_n = vr_t[step]
            u_next, R_ctx, H_out, H_lay = imex_step(
                system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, solver, L_t, cfg)
            R_id = model.react_grad(u_n, v_ref_n)
            num += float((R_ctx - R_id).pow(2).sum())
            den += float(R_id.pow(2).sum())
            u_prev = u_n; u_n = u_next
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-12))


# ======================================================================
#  HELPERS
# ======================================================================

def regime_indicator(dhu, eps):
    l = torch.log(torch.abs(dhu) + 1e-12)
    if LEGACY_RHO or l.shape[0] < 5:
        mu = torch.median(l)
        return torch.sigmoid(3.0 * (l - mu) /
                             (float(np.log(1 / (eps + 1e-12))) + 1e-8))
    mu = torch.median(l[1:-1])
    denom = max(float(np.log(1 / (eps + 1e-12))), 1.0)
    rho = torch.sigmoid(3.0 * (l - mu) / denom).clone()
    rho[0] = rho[1]; rho[-1] = rho[-2]
    return rho


def layer_envelope(x_t, xl, xr, eps):
    xr_t = torch.as_tensor(xr, dtype=torch.float32)
    return torch.exp(-torch.minimum(x_t - xl, xr_t - x_t)
                     / (eps ** 0.5 + 1e-12))


def _neu(r):
    r = r.clone(); r[0] = 0.0; r[-1] = 0.0; return r


def _dir(r, gL, gR):
    r = r.clone(); r[0] = gL; r[-1] = gR; return r


TAU = 0.05


def l_data_fn(u_next, u_ref, su):
    return ((u_next - u_ref) / (su + 1e-8)).pow(2).mean()


def l_cons_fn(ua, va, Ra, B=48):
    
    idx = torch.argsort(ua); us = ua[idx]; vs = va[idx]; Rs = Ra[idx]
    du = torch.sqrt((us[1:] - us[:-1]).pow(2)
                    + (vs[1:] - vs[:-1]).pow(2) + 1e-12)
    mask = du < TAU
    if mask.sum() < 2:
        return torch.zeros((), device=ua.device)
    valid = torch.where(mask)[0]
    if len(valid) > B:
        valid = valid[torch.randperm(len(valid))[:B]]
    w = (1 - du[valid] / TAU).clamp(0, 1).pow(2)
    return (w * (Rs[valid] - Rs[valid + 1]).pow(2)).mean()


def l_anch_fn(model, cfg):
    pts = cfg["anchors_uv"]
    pu = torch.tensor([p[0] for p in pts], dtype=torch.float32)
    pv = torch.tensor([p[1] for p in pts], dtype=torch.float32)
    return model.react_grad(pu, pv).pow(2).mean()


def l_fd_fn(model, u_t, v_t, R_t, Lambda):
    return (model.react_grad(u_t, v_t) - R_t).pow(2).mean() \
        / (Lambda ** 2 + 1e-8)


# ======================================================================
#  IMEX STEP
# ======================================================================

def imex_step(system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
              x_t, phi_t, solver, L_t, cfg):
    eps = cfg["eps_diff"]; dt = cfg["dt"]
    with torch.no_grad():
        dhu = L_t @ u_n
    rho = regime_indicator(dhu, eps)
    dt_u = (u_n.detach() - u_prev.detach()) / (dt + 1e-12)
    z_out = torch.stack([u_n, v_ref_n, dhu, dt_u, x_t], dim=-1)
    z_lay = torch.stack([u_n, v_ref_n, eps * dhu, dt_u, phi_t], dim=-1)
    R, Hon, Hln = model(u_n, v_ref_n, z_out, z_lay, rho, H_out, H_lay)
    if system == "fhn_partial":
        rhs = _dir(u_n + (dt / 2) * eps * (L_t @ u_n) + dt * R, 0.0, 1.0)
    else:
        rhs = _neu(u_n + (dt / 2) * cfg["D1"] * (L_t @ u_n) + dt * R)
    u_next = imex_solve(solver, rhs)
    return (torch.clamp(u_next, cfg["u_min"] - 0.05, cfg["u_max"] + 0.05),
            R, Hon, Hln)


# ======================================================================
#  EVALUATION
# ======================================================================

def evaluate(model, system, cfg, u_fd=None, v_fd=None,
             visited=None):
    ng = 60
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg); uf = UU.ravel(); vf = VV.ravel()
    Rp = model.react_nograd(torch.tensor(uf, dtype=torch.float32),
                            torch.tensor(vf, dtype=torch.float32)).numpy()
    Rt = get_R_true(system, uf, vf, cfg)
    l2_full = float(np.sqrt(np.mean((Rp - Rt) ** 2)) /
                    (np.sqrt(np.mean(Rt ** 2)) + 1e-8))
    l2_in = l2_full
    if u_fd is not None and len(u_fd) > 0:
        Rp2 = model.react_nograd(torch.tensor(u_fd.astype(np.float32)),
                                 torch.tensor(v_fd.astype(np.float32))).numpy()
        Rt2 = get_R_true(system, u_fd, v_fd, cfg)
        l2_in = float(np.sqrt(np.mean((Rp2 - Rt2) ** 2)) /
                      (np.sqrt(np.mean(Rt2 ** 2)) + 1e-8))
    l2_vis = l2_full
    if visited is not None and len(visited[0]):
        uv, vv = visited
        Rpv = model.react_nograd(torch.tensor(uv),
                                 torch.tensor(vv)).numpy()
        Rtv = get_R_true(system, uv.astype(np.float64),
                         vv.astype(np.float64), cfg)
        l2_vis = float(np.sqrt(np.mean((Rpv - Rtv) ** 2)) /
                       (np.sqrt(np.mean(Rtv ** 2)) + 1e-8))
    return (l2_full, l2_in, UU, VV, Rp.reshape(ng, ng), Rt.reshape(ng, ng),
            l2_vis)


# ======================================================================
#  TRAINING
# ======================================================================

def train_one(system, model, refs, cfg, x_t, phi_t, solver, L_t,
              u_fd_t, v_fd_t, R_fd_t, epochs):
    su = max(float(np.mean([np.abs(r[0]).mean() for r in refs])), 0.01)
    Lambda = cfg["Lambda"]
    Nt = cfg["Nt"]; n_refs = len(refs)

    for n, p in model.named_parameters():
        if "gru" in n or "gate" in n:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if params:
        pi_opt = optim.Adam(params, lr=1e-3)
        for _ in range(1000):
            pi_opt.zero_grad()
            loss = (model.react_grad(u_fd_t, v_fd_t) - R_fd_t) \
                .pow(2).mean() / (Lambda ** 2 + 1e-8)
            loss.backward(); pi_opt.step()
    for p in model.parameters():
        p.requires_grad_(True)

    ref_t = [(torch.tensor(ur, dtype=torch.float32),
              torch.tensor(vr, dtype=torch.float32)) for ur, vr in refs]

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=500, eta_min=1e-5)
    best_l = float("inf"); best_state = None; nan_c = 0; history = []

    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        f = min((epoch - 1) / (epochs - 1 + 1e-8), 1.0)
        lam_d = 0.8 + 0.2 * f
        lam_f = 3.0 + 2.0 * f
        lam_c = 0.01 + 0.99 * f
        lam_a = 0.001 + 7.999 * f
        ep = ep_d = ep_f = ep_c = ep_a = 0.0

        Lf = l_fd_fn(model, u_fd_t, v_fd_t, R_fd_t, Lambda)
        La = l_anch_fn(model, cfg)
        aux = lam_f * Lf + lam_a * La
        aux.backward()
        ep += float(aux.detach())
        ep_f += float(Lf.detach()); ep_a += float(La.detach())

        for ur_t, vr_t in ref_t:
            H_out, H_lay = model.init_hidden(x_t.shape[0])
            u_n = ur_t[0]; u_prev = ur_t[0]
            for step in range(Nt):
                H_out = H_out.detach(); H_lay = H_lay.detach()
                u_n = u_n.detach(); u_prev = u_prev.detach()
                v_ref_n = vr_t[step].detach()
                u_next, R, H_out, H_lay = imex_step(
                    system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
                    x_t, phi_t, solver, L_t, cfg)
                Ld = l_data_fn(u_next, ur_t[step + 1], su)
                if USE_CONS:
                    Lc = l_cons_fn(u_n.detach(), v_ref_n, R)
                else:
                    Lc = torch.zeros(())
                Lw = lam_d * Ld + lam_c * Lc
                (Lw / Nt / n_refs).backward()
                ep += float(Lw.detach()) / Nt / n_refs
                ep_d += float(Ld.detach()) / Nt / n_refs
                ep_c += float(Lc) / Nt / n_refs
                u_prev = u_n; u_n = u_next

        if not np.isfinite(ep):
            nan_c += 1
            if best_state:
                model.load_state_dict(best_state)
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            if nan_c >= 5:
                break
            continue
        nan_c = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step(); sched.step()
        history.append({"epoch": epoch, "total": ep, "data": ep_d,
                        "fd": ep_f, "cons": ep_c, "anch": ep_a})
        unw = ep_d + ep_f
        if unw < best_l:
            best_l = unw
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return history


# ======================================================================
#  RUN ONE (system, eps) PAIR
# ======================================================================

def run_one(system, eps, cfg, epochs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    x, L_np = build_laplacian(cfg)
    x_t = torch.tensor(x, dtype=torch.float32)
    L_t = torch.tensor(L_np, dtype=torch.float32)
    xl, xr = cfg["domain"]
    phi_t = layer_envelope(x_t, xl, xr, cfg["eps_diff"])
    D_u = cfg["eps_diff"] if system == "fhn_partial" else cfg["D1"]
    _, _, _, A_u_t = make_imex(D_u, L_np, cfg,
                               neumann=(cfg["bc_type"] == "neumann"))
    solver = make_solver(A_u_t)

    refs, t_arr = generate_reference(system, x, cfg, L_np)

    # ---- output bound -------------------------------------------------
    lam_cfg = float(cfg["Lambda"])
    lam_eff = (LAMBDA_SCALE * estimate_lambda(refs, L_np, cfg, system)
               if LAMBDA_MODE == "auto" else lam_cfg)
    cfg["Lambda"] = lam_eff
    r_max, floor = lambda_coverage(system, cfg, lam_eff)
    ok = lam_eff >= r_max
    extra = f", config was {lam_cfg:.3f}" if LAMBDA_MODE == "auto" else ""
    status = ("coverage OK" if ok else
              f"*** INSUFFICIENT: no model bounded by Lambda can score "
              f"below {floor:.4f} ***")
    print(f"    Lambda={lam_eff:.3f} ({LAMBDA_MODE}{extra})   "
          f"max|R_true|={r_max:.3f}   {status}")

    u_fd, v_fd, R_fd = extract_fd(refs, L_np, cfg, system, x=x)
    u_t, v_t, R_t = aggregate_fd(u_fd, v_fd, R_fd, cfg)
    n_bins = len(u_t)
    u_fd_t = torch.tensor(u_t, dtype=torch.float32)
    v_fd_t = torch.tensor(v_t, dtype=torch.float32)
    R_fd_t = torch.tensor(R_t, dtype=torch.float32)

    model = SystemModel(hidden=32, mlp_w=64, Lambda=lam_eff,
                        input_dim=cfg["input_dim"])
    t0 = time.time()
    history = train_one(system, model, refs, cfg, x_t, phi_t, solver, L_t,
                        u_fd_t, v_fd_t, R_fd_t, epochs)
    t_train = time.time() - t0

    vis = visited_region(refs, cfg)
    l2_full, l2_in, UU, VV, Rp, Rt, l2_vis = evaluate(
        model, system, cfg, u_fd, v_fd, visited=vis)
    print(f"    eps_R(R^id): box={l2_full:.4f}   visited={l2_vis:.4f}   "
          f"({len(vis[0])} occupied state bins)")
    cond = conditioning_ratio(model, system, refs, cfg, x_t, phi_t,
                              solver, L_t)
    print(f"    conditioning: ||R^ctx - R^id|| / ||R^id|| = {cond:.4f}")

    _, _, _, R_dist, l2_distil, l2_distil_vis = distil_2d(
        model, system, cfg, lam_eff, visited=vis)

    ckpt = os.path.join(MODEL_DIR, f"{system}_eps{eps:.0e}_nx{cfg['Nx']}.pt")
    torch.save({"state_dict": model.state_dict(),
                "cfg": {k: v for k, v in cfg.items()
                        if not isinstance(v, (list, tuple))},
                "eps": float(eps), "system": system,
                "Lambda": lam_eff, "use_cons": USE_CONS}, ckpt)

    return {
        "eps": float(eps),
        "l2_full": float(l2_full),
        "l2_in": float(l2_in),
        "l2_vis": float(l2_vis),
        "l2_distil": float(l2_distil),
        "l2_distil_vis": float(l2_distil_vis),
        "n_u_cols": int(len(np.unique(u_t))) if len(u_t) else 0,
        "cond_ratio": float(cond),
        "Lambda": float(lam_eff),
        "lambda_needed": float(r_max),
        "lambda_ok": bool(ok),
        "bound_floor": float(floor),
        "n_bins": int(n_bins),
        "Nx": int(cfg["Nx"]),
        "use_cons": bool(USE_CONS),
        "lambda_mode": LAMBDA_MODE,
        "train_s": float(t_train),
        "UU": UU, "VV": VV, "Rp": Rp, "Rt": Rt, "Rd": R_dist,
        "history": history,
        "ckpt": ckpt,
    }


# ======================================================================
#  PLOTS
# ======================================================================

def plot_distilled(results, system, cfg_name, out_dir):
    n = len(results); fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    std_eps = STANDARD_EPS[system]
    for col, r in enumerate(results):
        eps = r["eps"]; UU = r["UU"]; VV = r["VV"]
        Rd = r["Rd"]; Rt = r["Rt"]
        mark = " *" if abs(eps - std_eps) / std_eps < 0.01 else ""
        vm = float(np.abs(Rt).max())
        im = axes[0, col].contourf(UU, VV, Rd, levels=30, cmap="RdBu_r",
                                   vmin=-vm, vmax=vm)
        plt.colorbar(im, ax=axes[0, col])
        axes[0, col].set_title(f"Distilled R(u,v)  eps={eps:.0e}{mark}\n"
                               f"L2={r['l2_distil']:.4f}  |  "
                               f"ctx/id={r['cond_ratio']:.3f}", fontsize=8)
        axes[0, col].set_xlabel("u"); axes[0, col].set_ylabel("v")
        im2 = axes[1, col].contourf(UU, VV, np.abs(Rd - Rt), levels=30,
                                    cmap="YlOrRd")
        plt.colorbar(im2, ax=axes[1, col])
        axes[1, col].set_title(f"|Distilled - True|  eps={eps:.0e}{mark}",
                               fontsize=8)
        axes[1, col].set_xlabel("u"); axes[1, col].set_ylabel("v")
    fig.suptitle(f"{cfg_name}\nDistilled state-only reaction surface "
                 f"$R^{{id}}_\\theta(u,v)$ across eps",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, f"eps_distilled_{system}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ======================================================================
#  RUN ONE SYSTEM
# ======================================================================

def run_system(system, epochs, nx_override):
    cfg_base = BASE_CFGS[system]; cfg_name = cfg_base["name"]
    eps_list = EPS_VALUES[system]
    print(f"\n{'='*65}")
    print(f"  System: {cfg_name}")
    print(f"  eps values: {[f'{e:.0e}' for e in eps_list]}")
    print(f"  Epochs: {epochs} per eps value")
    print(f"  Nx: {nx_override if nx_override else cfg_base['Nx']}   "
          f"L_cons: {'ON' if USE_CONS else 'OFF'}   "
          f"Lambda: {LAMBDA_MODE}   "
          f"rho: {'legacy' if LEGACY_RHO else 'boundary-corrected'}")
    if system == "fhn_partial":
        print("  NOTE: for FHN, eps appears in the reaction as 1/eps, so this")
        print("        sweep varies the layer AND the reaction amplitude. For")
        print("        Predator-Prey, eps = D1 is diffusion only and R_true is")
        print("        eps-independent. The two columns answer different")
        print("        questions and the manuscript should say so.")
    print(f"{'='*65}")

    results = []; histories = []
    for eps in eps_list:
        cfg = build_eps_cfg(system, eps, cfg_base, nx_override)
        print(f"\n  eps={eps:.1e}  (standard={STANDARD_EPS[system]:.1e})",
              flush=True)
        try:
            r = run_one(system, eps, cfg, epochs)
            print(f"    L2(full)={r['l2_full']:.4f}  L2(in)={r['l2_in']:.4f}  "
                  f"L2(distil)={r['l2_distil']:.4f}  bins={r['n_bins']}  "
                  f"{r['train_s']:.0f}s")
            results.append(r); histories.append(r["history"])
        except Exception:
            import traceback
            traceback.print_exc()
            results.append({"eps": float(eps), "l2_full": float("nan"),
                            "l2_in": float("nan"), "l2_distil": float("nan"),
                            "cond_ratio": float("nan"),
                            "Lambda": float("nan"),
                            "lambda_needed": float("nan"), "lambda_ok": True,
                            "bound_floor": 0.0, "n_bins": 0, "Nx": cfg["Nx"],
                            "use_cons": USE_CONS, "lambda_mode": LAMBDA_MODE,
                            "train_s": 0,
                            "UU": None, "VV": None, "Rp": None, "Rt": None,
                            "Rd": None, "history": [], "ckpt": None})
            histories.append([])

    valid = [r for r in results
             if not np.isnan(r["l2_full"]) and r["UU"] is not None]
    if len(valid) >= 2:
        plot_distilled(valid, system, cfg_name, OUT_DIR)

    save = [{k: v for k, v in r.items()
             if k not in ("UU", "VV", "Rp", "Rt", "Rd", "history")}
            for r in results]
    with open(os.path.join(OUT_DIR, f"eps_summary_{system}.json"), "w") as f:
        json.dump(save, f, indent=2)
    return results


# ======================================================================
#  MAIN
# ======================================================================

def main():
    global USE_CONS, LEGACY_RHO, LAMBDA_MODE, LAMBDA_SCALE
    parser = argparse.ArgumentParser(
        description="BRIDGE: epsilon sensitivity study for coupled systems")
    parser.add_argument("--system", default="all",
                        choices=["fhn_partial", "predator_prey", "all"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--nx", type=int, default=None,
                        help="override Nx (both systems now default to 128)")
    parser.add_argument("--no-cons", action="store_true",
                        help="disable L_cons, reproducing the earlier runs "
                             "in which it was defined but never applied")
    parser.add_argument("--lambda-mode", default="auto",
                        choices=["auto", "config"],
                        help="auto: set the decoder bound per eps from the "
                             "FD residual scale (default). config: use the "
                             "fixed value, which saturates for FHN at "
                             "eps <= 1e-2")
    parser.add_argument("--lambda-scale", type=float, default=1.0,
                        help="multiply the auto bound, if the coverage line "
                             "reports it is short (default 1.0)")
    parser.add_argument("--legacy-rho", action="store_true",
                        help="restore the previous regime indicator")
    args = parser.parse_args()

    USE_CONS    = not args.no_cons
    LEGACY_RHO  = bool(args.legacy_rho)
    LAMBDA_MODE  = args.lambda_mode
    LAMBDA_SCALE = float(args.lambda_scale)

    systems = (["fhn_partial", "predator_prey"] if args.system == "all"
               else [args.system])

    print("\nBRIDGE epsilon sensitivity study (coupled systems)")
    print(f"Systems: {systems}")
    print(f"Epochs per run: {args.epochs}")
    print(f"L_cons: {'ON' if USE_CONS else 'OFF'}   Lambda: {LAMBDA_MODE}")
    if not USE_CONS:
        print("NOTE: L_cons is OFF. The headline results train with it, so "
              "this sweep is not comparable to them.")
    if LAMBDA_MODE == "config":
        print("NOTE: fixed Lambda. For FHN at eps <= 1e-2 the reaction "
              "exceeds the decoder bound and the resulting rows measure "
              "that bound, not the method.")
    for s in systems:
        cfg = BASE_CFGS[s]; eps_list = EPS_VALUES[s]
        t_est = args.epochs * cfg["Nt"] * len(cfg["ics"]) * 12.5 / 1000 / 3600
        print(f"  {cfg['name']}: {len(eps_list)} eps values x ~{t_est:.1f}h "
              f"= ~{len(eps_list)*t_est:.1f}h  (before the solver and "
              f"loss-hoisting speedups)")

    all_results = {}
    for s in systems:
        all_results[s] = run_system(s, args.epochs, args.nx)

    print(f"\n{'='*65}\n  FINAL SUMMARY\n{'='*65}")
    for s in systems:
        print(f"\n  {BASE_CFGS[s]['name']}:")
        print(f"  {'eps':>10}  {'Lambda':>8}  {'L2(full)':>10}  "
              f"{'L2(in)':>10}  {'L2(distil)':>11}  {'ctx/id':>8}  "
              f"{'bins':>6}")
        print("  " + "-" * 78)
        for r in all_results[s]:
            mk = ("  <- std"
                  if abs(r["eps"] - STANDARD_EPS[s]) / STANDARD_EPS[s] < 0.01
                  else "")
            if not r.get("lambda_ok", True):
                mk += "  [bound too small]"

            def _f(key, width=10):
                v = r.get(key, float("nan"))
                return (f"{v:.4f}".rjust(width) if not np.isnan(v)
                        else "FAILED".rjust(width))

            lam = r.get("Lambda", float("nan"))
            lam_s = f"{lam:.3f}".rjust(8) if not np.isnan(lam) else " " * 8
            print(f"  {r['eps']:>10.1e}  {lam_s}  {_f('l2_full')}  "
                  f"{_f('l2_in')}  {_f('l2_distil', 11)}  "
                  f"{_f('cond_ratio', 8)}  {r['n_bins']:>6}{mk}")


if __name__ == "__main__":
    main()
