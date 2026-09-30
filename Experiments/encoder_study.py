import sys, os, time, json, copy, argparse, math
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.linalg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from numerics import (build_shishkin_mesh, build_uniform_mesh,
                              shishkin_sigma, build_compact_laplacian,
                              build_imex_matrix, layer_mask_boundary,
                              layer_mask_curvature, ac_nx_for_resolution,
                              interface_width, measured_layer_width,
                              layer_on_refined_region, nx_for_width,
                              fhn_front_width,
                              boundary_skip, layer_budget_report)
from model import (EncoderReactionNet, build_pair, count_parameters,
                           STREAMS)

OUT_DIR = "encoder_study"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")           # set in main()

TAU         = 0.05     # consistency-loss state window
METRIC_SKIP = boundary_skip()          

LAMBDA_SAFETY      = 1.25   # headroom over the observed FD residual
LAMBDA_MIN_PER_BIN = 8     
COND_FLOOR_FRAC    = 0.05  


# ======================================================================
#  ENCODER GEOMETRY
# ======================================================================


GEOM = {
    False: dict(mlp_w=32, input_dim=4, state_dim=1, gate="film",
                act="relu", Lambda=1.0),      # scalar
    True:  dict(mlp_w=64, input_dim=5, state_dim=2, gate="additive",
                act="silu", Lambda=1.0),      # coupled
}


# ======================================================================
#  SCALAR REACTION LAWS
# ======================================================================

REACTIONS = {
    "paper": {
        "fisher":     dict(amp=1.0, Lambda=0.5, max_R=0.25),
        "allen_cahn": dict(amp=1.0, Lambda=0.8, max_R=2.0 / (3 * np.sqrt(3))),
    },
    "code": {
        "fisher":     dict(amp=6.0, Lambda=2.0, max_R=6.0 * 0.25),
        "allen_cahn": dict(amp=5.0, Lambda=3.0,
                           max_R=5.0 * 2.0 / (3 * np.sqrt(3))),
    },
}


def R_true_np(cfg, u, v=None):
    """R_true as a numpy field. Evaluation only -- never seen in training."""
    p = cfg["problem"]; amp = cfg.get("amp", 1.0)
    if p == "fisher":
        return amp * u * (1.0 - u)
    if p == "allen_cahn":
        return amp * (u - u ** 3)
    if p == "fhn":
        return ((1.0 / cfg["eps"]) * u * (u - cfg["a_fhn"]) * (1.0 - u) - v)
    if p == "predator_prey":
        uu = np.clip(u, 0.0, None); vv = np.clip(v, 0.0, None)
        return uu * (1 - uu) - cfg["alpha"] * uu * vv / \
            (cfg["beta_pp"] + np.clip(uu, 0.001, None))
    raise ValueError(p)


BASE = {
    "fisher": {
        "name": "Fisher-KPP", "coupled": False,
        "domain": (0.0, 1.0), "T": 5.0, "Nx": 128, "Nt": 1000, "dt": 0.005,
        "bc_type": "dirichlet", "bc_values": (0.0, 0.0),
        "mesh_type": "shishkin", "mesh_scale": "sqrt", "beta": 2.0,
        "u_min": 0.0, "u_max": 1.0, "clip_pad": 0.0,
        "positivity": True, "fd_pct": 50, "n_bins": 24,
        "mask": "boundary", "layer_pct": 90.0,
        "ic_amps": [1.0, 0.70, 0.45, 0.25],
        "n_rollout_ics": 3,
        "anchors": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        "tbptt": 15, "nwin": 3, "lambda_fd": 3.0, "epochs": 1200,
        "eps_list": [1e-2, 1e-3, 1e-4, 1e-5],
    },

    "allen_cahn": {
        "name": "Allen-Cahn", "coupled": False,
        "domain": (-1.0, 1.0), "T": 1.0, "Nx": 128, "Nt": 400, "dt": 0.0025,
        "bc_type": "periodic", "bc_values": (0.0, 0.0),
        "mesh_type": "uniform", "mesh_scale": "sqrt", "beta": 2.0,
        "u_min": -1.0, "u_max": 1.0, "clip_pad": 0.0,
        "positivity": False, "fd_pct": 60, "n_bins": 24,
        "mask": "curvature", "layer_pct": 90.0,
        "n_rollout_ics": 3,
        "eval_ic": 1,
        "anchors": [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        "tbptt": 40, "nwin": 2, "lambda_fd": 3.0, "epochs": 900,
        "eps_list": [1e-2, 3e-3, 1e-3, 3e-4],
    },

    "fhn": {
        "name": "FitzHugh-Nagumo", "coupled": True,
        "domain": (0.0, 1.0), "T": 0.3, "Nx": 128, "Nt": 60, "dt": 0.005,
        "bc_type": "dirichlet", "bc_values": (0.0, 1.0),
        "bc_values_v": (0.0, 0.0),
        "mesh_type": "uniform", "mesh_scale": "lin", "beta": 2.0,
        "u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 0.6,
        "clip_pad": 0.05,
        "a_fhn": 0.25, "delta_v": 0.1, "beta_v": 1.0, "gamma_v": 0.5,
        "positivity": False, "fd_pct": 50, "n_bins_2d": 10,
        "mask": "curvature", "layer_pct": 90.0,
        "ics": [("front", 0.3, 0.0), ("front", 0.6, 0.0),
                ("front", 0.3, 0.3), ("step", 0.5, 0.5)],
        "n_rollout_ics": 3,
        "anchors": [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (1.0, 0.0, 0.0)],
        "tbptt": 12, "nwin": 3, "lambda_fd": 3.0, "epochs": 700,
        "eps_list": [5e-2, 2e-2, 1e-2, 5e-3],
    },

    "predator_prey": {
        "name": "Predator-Prey", "coupled": True,
        "domain": (0.0, 1.0), "T": 2.0, "Nx": 128, "Nt": 200, "dt": 0.01,
        "bc_type": "neumann", "bc_values": (0.0, 0.0),
        "bc_values_v": (0.0, 0.0),
        "mesh_type": "shishkin", "mesh_scale": "sqrt", "beta": 2.0,
        "u_min": 0.0, "u_max": 0.5, "v_min": 0.0, "v_max": 0.4,
        "clip_pad": 0.05,
        "alpha": 1.0, "beta_pp": 0.1, "gamma_pp": 0.5, "delta_pp": 0.25,
        "D2": 1e-2, "u_star": 0.1, "v_star": 0.18,
        "positivity": False, "fd_pct": 50, "n_bins_2d": 10,
        "mask": "curvature", "layer_pct": 90.0,
        "ics": [(0.30, 0.12, 1, 1), (0.25, 0.10, 1, 1), (0.20, 0.08, 2, 2),
                (0.28, 0.11, 1, 2), (0.35, 0.14, 1, 1), (0.22, 0.09, 2, 1),
                (0.18, 0.07, 1, 2), (0.32, 0.13, 2, 2)],
        "n_rollout_ics": 3,
        "anchors": [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.1, 0.18, 0.0)],
        "tbptt": 10, "nwin": 3, "lambda_fd": 3.0, "epochs": 500,
        "eps_list": [1e-2, 5e-3, 1e-3, 1e-4],
    },
}

ALL_PROBLEMS = ["fisher", "allen_cahn", "fhn", "predator_prey"]

PROFILES = {              
    "fast":     (0.5, 3),
    "standard": (1.0, 5),
    "full":     (1.0, 5),
}


def _round4(nx, args):
    nx = int(min(max(int(nx), args.min_nx), args.max_nx))
    return int(nx + (-nx % 4))


NODE_TARGET = {"allen_cahn": "ac_nodes", "fhn": "fhn_nodes"}


def choose_nx(problem, eps, args):

    if args.nx or problem not in NODE_TARGET:
        return None
    target = float(getattr(args, NODE_TARGET[problem]))
    nx = None
    hist = []
    for _ in range(4):
        cfg = build_config(problem, eps, args, nx_override=nx)
        x, sigma, L_np, A_np, lu, piv = build_operators(cfg)
        refs, _, _ = generate_reference(cfg, x, L_np, lu, piv)
        w, nodes, _, _ = measured_layer_width(refs[cfg["eval_ic"]][0], x)
        hist.append((cfg["Nx"], float(w), float(nodes)))
        if not np.isfinite(w) or w <= 0:
            break
        want = _round4(nx_for_width(w, target, cfg["domain"]), args)
        if want <= cfg["Nx"] * 1.08:
            nx = max(cfg["Nx"], want)
            break
        nx = want
        if nx >= args.max_nx:
            break
    return {"Nx": _round4(nx or cfg["Nx"], args), "history": hist,
            "target": target}


def _kv_map(items):
    """['fhn=2000', 'allen_cahn=1500'] -> {'fhn': 2000, ...}"""
    out = {}
    for it in (items or []):
        k, _, v = it.replace(":", "=").partition("=")
        if not v:
            raise ValueError(f"expected problem=value, got {it!r}")
        out[k.strip()] = int(v)
    return out


def per_problem(problem, mapping, default):
    return _kv_map(mapping).get(problem, default)


def build_config(problem, eps, args, nx_override=None):
    cfg = copy.deepcopy(BASE[problem])
    cfg.update(eps=float(eps), problem=problem, ic_scale=args.ic_scale)
    if problem == "fisher" and args.fisher_T:
        cfg["T"] = float(args.fisher_T)
        cfg["Nt"] = int(round(cfg["T"] / cfg["dt"]))
    if not cfg["coupled"]:
        rx = REACTIONS[args.reaction][problem]
        cfg.update(amp=rx["amp"], Lambda=rx["Lambda"], max_R=rx["max_R"],
                   reaction=args.reaction)
    else:
        cfg.update(amp=1.0, reaction="paper")      # coupled laws agree
        cfg["Lambda"] = None                       # set from data
        cfg["max_R"] = None
    if args.mesh_scale != "auto":
        cfg["mesh_scale"] = args.mesh_scale

    if problem == "fhn":
        cfg["mesh_type"] = args.fhn_mesh
    if nx_override:
        cfg["Nx"] = _round4(nx_override, args)
    elif args.nx:
        cfg["Nx"] = int(args.nx)
    elif problem == "fhn":
        cfg["Nx"] = _round4(nx_for_width(fhn_front_width(eps),
                                         args.fhn_nodes, cfg["domain"]), args)
    elif problem == "allen_cahn":
        cfg["Nx"] = _round4(ac_nx_for_resolution(eps, nodes=args.ac_nodes,
                                                 domain=cfg["domain"],
                                                 scale=cfg["amp"]), args)

    if args.tbptt:
        cfg["tbptt"] = int(args.tbptt)
    if args.nwin:
        cfg["nwin"] = int(args.nwin)
    if args.n_ics:
        cfg["n_rollout_ics"] = int(args.n_ics)
    scale = PROFILES[args.profile][0]
    ep_override = per_problem(problem, getattr(args, "epochs_map", None),
                              args.epochs)
    cfg["epochs"] = (int(ep_override) if ep_override
                     else max(50, int(round(cfg["epochs"] * scale))))
    cfg["preinit"] = per_problem(problem, getattr(args, "preinit_map", None),
                                 args.preinit)
    ei = cfg.get("eval_ic", 0) if args.eval_ic is None else int(args.eval_ic)
    cfg["eval_ic"] = int(min(max(ei, 0), cfg["n_rollout_ics"] - 1))
    return cfg


# ======================================================================
#  MESH AND OPERATORS
# ======================================================================

def build_operators(cfg):
    N = cfg["Nx"]; xl, xr = cfg["domain"]; eps = cfg["eps"]
    if cfg["mesh_type"] == "shishkin":
        x, sigma = build_shishkin_mesh(N, xl, xr, eps, cfg["beta"],
                                       cfg["mesh_scale"])
    else:
        x, sigma = build_uniform_mesh(N, xl, xr)
    periodic = cfg["bc_type"] == "periodic"
    L_np = build_compact_laplacian(x, periodic=periodic)
    A_np, lu, piv = build_imex_matrix_bc(L_np, cfg, cfg["dt"], eps)
    return x, sigma, L_np, A_np, lu, piv


def build_imex_matrix_bc(L_np, cfg, dt, diff):
    Nx1 = L_np.shape[0]
    if cfg["bc_type"] == "neumann":
        A = np.eye(Nx1) - (dt / 2) * diff * L_np
        A[0, :] = 0; A[0, 0] = 1; A[0, 1] = -1
        A[-1, :] = 0; A[-1, -1] = 1; A[-1, -2] = -1
        lu, piv = scipy.linalg.lu_factor(A)
        return A, lu, piv
    return build_imex_matrix(L_np, dt, diff,
                             periodic=(cfg["bc_type"] == "periodic"))


def bc_apply_np(cfg, rhs, gL, gR):
    t = cfg["bc_type"]
    if t == "periodic":
        return rhs
    r = rhs.copy()
    if t == "neumann":
        r[0] = 0.0; r[-1] = 0.0
    else:
        r[0] = gL; r[-1] = gR
    return r


def bc_apply_t(cfg, rhs, gL, gR):
    t = cfg["bc_type"]
    if t == "periodic":
        return rhs
    r = rhs.clone()
    dev = rhs.device
    if t == "neumann":
        z = torch.zeros((), dtype=rhs.dtype, device=dev)
        r[..., 0] = z; r[..., -1] = z
    else:
        r[..., 0] = torch.as_tensor(gL, dtype=rhs.dtype, device=dev)
        r[..., -1] = torch.as_tensor(gR, dtype=rhs.dtype, device=dev)
    return r


# ======================================================================
#  INITIAL CONDITIONS
# ======================================================================

def make_ics(cfg, x):
    p = cfg["problem"]; eps = cfg["eps"]

    if p == "fisher":
        return [(a * np.sin(np.pi * x), None) for a in cfg["ic_amps"]]

    if p == "allen_cahn":
        w = (np.sqrt(2.0 * eps / cfg["amp"])
             if cfg["ic_scale"] == "eps" else 0.1)
        w = max(w, 1e-6)

        def double_front(a, b):
            return np.tanh((x - a) / w) - np.tanh((x - b) / w) - 1.0

        return [(x ** 2 * np.cos(np.pi * x), None),
                (double_front(-0.5, 0.5), None),
                (double_front(-0.3, 0.2), None)]

    if p == "fhn":
        w = eps if cfg["mesh_scale"] == "lin" else np.sqrt(eps)
        w = max(w, 1e-6)
        out = []
        for (kind, loc, v0) in cfg["ics"]:
            u0 = (0.5 * (1 + np.tanh((x - loc) / w)) if kind == "front"
                  else np.where(x < loc, 0.9, 0.05).astype(float))
            out.append((u0.copy(), v0 * np.ones_like(x)))
        return out

    if p == "predator_prey":
        us = cfg["u_star"]; vs = cfg["v_star"]; out = []
        for (au, av, mu, mv) in cfg["ics"]:
            u0 = np.clip(us + au * np.cos(mu * np.pi * x),
                         cfg["u_min"] + 0.001, cfg["u_max"])
            v0 = np.clip(vs + av * np.sin(mv * np.pi * x),
                         cfg["v_min"] + 0.001, cfg["v_max"])
            out.append((u0.copy(), v0.copy()))
        return out

    raise ValueError(p)


# ======================================================================
#  REFERENCE TRAJECTORIES
# ======================================================================

def _lus(lu, piv, rhs, lo, hi):
    return np.clip(scipy.linalg.lu_solve((lu, piv), rhs), lo, hi)


def generate_reference(cfg, x, L_np, lu, piv):
    p = cfg["problem"]; eps = cfg["eps"]; dt = cfg["dt"]; Nt = cfg["Nt"]
    pad = 0.05
    lo, hi = cfg["u_min"] - pad, cfg["u_max"] + pad
    gL, gR = cfg["bc_values"]
    t_arr = np.linspace(0, cfg["T"], Nt + 1)
    refs = []

    if not cfg["coupled"]:
        for u0, _ in make_ics(cfg, x):
            ur = np.zeros((Nt + 1, len(x))); ur[0] = u0
            if cfg["bc_type"] == "dirichlet":
                ur[0, 0] = gL; ur[0, -1] = gR
            for n in range(Nt):
                un = ur[n]
                rhs = un + (dt / 2) * eps * (L_np @ un) + dt * R_true_np(cfg, un)
                ur[n + 1] = _lus(lu, piv, bc_apply_np(cfg, rhs, gL, gR), lo, hi)
            refs.append((ur, None))
        return refs, t_arr, (gL, gR)

    # ---- coupled: the v equation is KNOWN and integrated alongside ----
    vlo, vhi = cfg["v_min"] - pad, cfg["v_max"] + pad
    gLv, gRv = cfg["bc_values_v"]
    Dv = cfg["delta_v"] if p == "fhn" else cfg["D2"]
    _, lu_v, piv_v = build_imex_matrix_bc(L_np, cfg, dt, Dv)

    for u0, v0 in make_ics(cfg, x):
        ur = np.zeros((Nt + 1, len(x))); vr = np.zeros((Nt + 1, len(x)))
        ur[0] = np.clip(u0, lo, hi); vr[0] = np.clip(v0, vlo, vhi)
        if cfg["bc_type"] == "dirichlet":
            ur[0, 0] = gL; ur[0, -1] = gR
            vr[0, 0] = gLv; vr[0, -1] = gRv
        for n in range(Nt):
            un = ur[n]; vn = vr[n]
            Ru = R_true_np(cfg, un, vn)
            rhs_u = un + (dt / 2) * eps * (L_np @ un) + dt * Ru
            ur[n + 1] = _lus(lu, piv, bc_apply_np(cfg, rhs_u, gL, gR), lo, hi)
            if p == "fhn":
                Rv = cfg["beta_v"] * un - cfg["gamma_v"] * vn
            else:
                uu = np.clip(un, 0.0, None); vv = np.clip(vn, 0.0, None)
                Rprey = cfg["alpha"] * uu * vv / \
                    (cfg["beta_pp"] + np.clip(uu, 0.001, None))
                Rv = cfg["gamma_pp"] * Rprey - cfg["delta_pp"] * vn
            rhs_v = vn + (dt / 2) * Dv * (L_np @ vn) + dt * Rv
            vr[n + 1] = _lus(lu_v, piv_v,
                             bc_apply_np(cfg, rhs_v, gLv, gRv), vlo, vhi)
        refs.append((ur, vr))
    return refs, t_arr, (gL, gR)


# ======================================================================
#  FD SUPERVISION TARGETS                                       
# ======================================================================

def _fd_residual(ur, L_np, D, dt, n, k=1):
    """Central time difference at half-width k, minus the diffusion term.

    k = 1 is the (n-1, n+1) stencil the method uses. k = 2 is the same
    estimate with twice the spacing and therefore four times the
    truncation error; comparing the two tests whether the estimate has
    converged at this node, which is what the curvature filter was
    standing in for.
    """
    return (ur[n + k] - ur[n - k]) / (2 * k * dt) - D * (L_np @ ur[n])


def _accept_mask(ur, L_np, D, dt, n, cfg, kind, tol):
    """Which nodes give a trustworthy FD reaction estimate at level n."""
    R1 = _fd_residual(ur, L_np, D, dt, n, 1)
    if kind == "pct":
        lap = np.abs(L_np @ ur[n])
        m = lap < np.percentile(lap[1:-1], cfg["fd_pct"])
    else:
        R2 = _fd_residual(ur, L_np, D, dt, n, 2)
        scale = float(np.median(np.abs(R1[1:-1]))) + 1e-12
        m = np.abs(R2 - R1) <= tol * (np.abs(R1) + 0.01 * scale)
    m = m.copy()
    m[0] = False; m[-1] = False
    return m, R1


def _collect(cfg, refs, L_np, kind, tol):
    dt = cfg["dt"]; D = cfg["eps"]
    lo = 1 if kind == "pct" else 2
    us, vs, Rs = [], [], []
    for ur, vr in refs:
        Nt = ur.shape[0] - 1
        hi = Nt if kind == "pct" else Nt - 1
        for n in range(lo, max(lo, hi)):
            m, R1 = _accept_mask(ur, L_np, D, dt, n, cfg, kind, tol)
            if m.sum():
                us.append(ur[n][m]); Rs.append(R1[m])
                if vr is not None:
                    vs.append(vr[n][m])
    if not us:
        return None
    return (np.concatenate(us),
            np.concatenate(vs) if vs else None,
            np.concatenate(Rs))


def _bin_1d(cfg, u, R, nb, min_count):
    edges = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(u, edges) - 1, 0, nb - 1)
    ut, Rt, wt = [], [], []
    for i in range(nb):
        m = idx == i
        if m.sum() >= min_count:
            ut.append(ctr[i]); Rt.append(np.median(R[m])); wt.append(m.sum())
    return ut, None, Rt, wt


def _bin_2d(cfg, u, v, R, nb, min_count):
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    uc = 0.5 * (ue[:-1] + ue[1:]); vc = 0.5 * (ve[:-1] + ve[1:])
    ui = np.clip(np.digitize(u, ue) - 1, 0, nb - 1)
    vi = np.clip(np.digitize(v, ve) - 1, 0, nb - 1)
    ut, vt, Rt, wt = [], [], [], []
    for i in range(nb):
        for j in range(nb):
            m = (ui == i) & (vi == j)
            if m.sum() >= min_count:
                ut.append(uc[i]); vt.append(vc[j])
                Rt.append(np.median(R[m])); wt.append(m.sum())
    return ut, vt, Rt, wt


def extract_fd_targets(cfg, refs, L_np, kind="reliability", tol=0.10,
                       min_bins=12):
    coupled = cfg["coupled"]
    nb = cfg["n_bins_2d"] if coupled else cfg["n_bins"]
    rep = {"filter": kind, "tol": tol, "escalations": [],
           "n_raw": 0, "n_bins_total": nb ** 2 if coupled else nb}

    plan = [(kind, 4), (kind, 2), ("none", 2)]
    best = None
    for (k, mc) in plan:
        raw = _collect(cfg, refs, L_np, k, 1e9 if k == "none" else tol)
        if raw is None:
            continue
        u, v, R = raw
        rep["n_raw"] = int(u.size)
        ut, vt, Rt, wt = (_bin_2d(cfg, u, v, R, nb, mc) if coupled
                          else _bin_1d(cfg, u, R, nb, mc))
        best = (ut, vt, Rt, wt, k, mc)
        if len(ut) >= min_bins:
            break
        rep["escalations"].append(
            f"{k}/min{mc} gave {len(ut)} bins (< {min_bins})")

    if best is None or not best[0]:
        z = np.zeros(0, np.float32)
        rep["n_targets"] = 0
        return z, (z if coupled else None), z, z, rep

    ut, vt, Rt, wt, k_used, mc_used = best
    rep.update(filter_used=k_used, min_count_used=mc_used,
               n_targets=len(ut))
    w = np.array(wt, dtype=np.float64)
    return (np.array(ut, np.float32),
            (np.array(vt, np.float32) if coupled else None),
            np.array(Rt, np.float32),
            (w / w.mean()).astype(np.float32),
            rep)


def estimate_lambda(cfg, refs, L_np, safety=LAMBDA_SAFETY):
    dt = cfg["dt"]; D = cfg["eps"]; nb = cfg["n_bins_2d"]
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    us, vs, Rs = [], [], []
    for ur, vr in refs:
        Nt = ur.shape[0] - 1
        for n in range(2, Nt - 1):
            us.append(ur[n][1:-1]); vs.append(vr[n][1:-1])
            Rs.append(_fd_residual(ur, L_np, D, dt, n)[1:-1])
    u = np.concatenate(us); v = np.concatenate(vs); R = np.concatenate(Rs)
    ui = np.clip(np.digitize(u, ue) - 1, 0, nb - 1)
    vi = np.clip(np.digitize(v, ve) - 1, 0, nb - 1)
    peaks = []
    for i in range(nb):
        for j in range(nb):
            m = (ui == i) & (vi == j)
            if m.sum() >= LAMBDA_MIN_PER_BIN:
                peaks.append(abs(float(np.median(R[m]))))
    if not peaks:
        return float(safety * np.median(np.abs(R)) * 10.0)
    return float(safety * max(peaks))


def lambda_coverage(cfg, Lambda, visited):
    uv, vv = visited
    Rt = (R_true_np(cfg, uv.astype(np.float64), vv.astype(np.float64))
          if vv is not None else R_true_np(cfg, uv.astype(np.float64)))
    best = np.clip(Rt, -Lambda, Lambda)
    floor = float(np.sqrt(np.mean((best - Rt) ** 2)) /
                  (np.sqrt(np.mean(Rt ** 2)) + 1e-12))
    return float(np.abs(Rt).max()), floor


def visited_states(cfg, refs, nb=64):
    us, vs = [], []
    for ur, vr in refs:
        us.append(ur[1:, 1:-1].ravel())
        if vr is not None:
            vs.append(vr[1:, 1:-1].ravel())
    u = np.concatenate(us)
    ue = np.linspace(cfg["u_min"], cfg["u_max"], nb + 1)
    uc = 0.5 * (ue[:-1] + ue[1:])
    ui = np.clip(np.digitize(u, ue) - 1, 0, nb - 1)

    if not vs:
        occ = np.unique(ui)
        return uc[occ].astype(np.float32), None

    v = np.concatenate(vs)
    ve = np.linspace(cfg["v_min"], cfg["v_max"], nb + 1)
    vc = 0.5 * (ve[:-1] + ve[1:])
    vi = np.clip(np.digitize(v, ve) - 1, 0, nb - 1)
    key = np.unique(ui.astype(np.int64) * nb + vi.astype(np.int64))
    return (uc[key // nb].astype(np.float32),
            vc[key % nb].astype(np.float32))


# ======================================================================
#  REGIME GATE, IMEX STEP, ROLLOUTS
# ======================================================================

def regime_gate(dh, eps):
    l = torch.log(torch.abs(dh) + 1e-12)
    denom = max(float(np.log(1.0 / (eps + 1e-12))), 1.0)
    if l.shape[-1] < 5:
        mu = torch.median(l, dim=-1, keepdim=True).values
        return torch.sigmoid(3.0 * (l - mu) / denom)
    mu = torch.median(l[..., 1:-1], dim=-1, keepdim=True).values
    rho = torch.sigmoid(3.0 * (l - mu) / denom).clone()
    rho[..., 0] = rho[..., 1]
    rho[..., -1] = rho[..., -2]
    return rho

GATES = ("true", "scram", "const")

def apply_gate(rho, sh):
    """Intervene on the regime gate. Identity for the unmodified study."""
    g = sh.get("gate", "true")
    if g == "true":
        return rho
    if g == "const":
        return torch.full_like(rho, float(sh.get("gate_const", 0.5)))
    perm = sh.get("gate_perm")
    if perm is None:
        return rho
    r = rho[..., perm]
    if r.shape[-1] >= 5:
        # Keep the endpoint closure identical to the unpermuted gate, so
        # the arms differ in alignment and in nothing else.
        r = r.clone()
        r[..., 0] = r[..., 1]
        r[..., -1] = r[..., -2]
    return r


def _solve(sh, rhs):
    """u = A^{-1} rhs, batched over the leading axis."""
    if sh["Ainv_T"] is not None:
        return rhs @ sh["Ainv_T"]
    LU, piv = sh["solver"]
    flat = rhs.reshape(-1, rhs.shape[-1]).T          # (Nx1, B)
    sol = torch.linalg.lu_solve(LU, piv, flat)
    return sol.T.reshape(rhs.shape)


def imex_step(u_n, u_prev, v_n, model, H_out, H_lay, sh, cfg,
              stream="both", want_rho=False):
    eps = cfg["eps"]; dt = cfg["dt"]
    L_t = sh["L_t"]; gL, gR = sh["bc"]
    with torch.no_grad():
        dh = u_n @ L_t.T
    rho = apply_gate(regime_gate(dh, eps), sh)
    dt_u = (u_n.detach() - u_prev.detach()) / (dt + 1e-12)
    x_t = sh["x_t"].expand_as(u_n)
    phi_t = sh["phi_t"].expand_as(u_n)

    if cfg["coupled"]:
        z_out = torch.stack([u_n, v_n, dh, dt_u, x_t], -1)
        z_lay = torch.stack([u_n, v_n, eps * dh, dt_u, phi_t], -1)
        z_out = torch.clamp(z_out, -10, 10)
        z_lay = torch.clamp(z_lay, -10, 10)
    else:
        se = sh["layer_scale"]
        z_out = torch.stack([u_n, dh, dt_u, x_t], -1)
        z_lay = torch.stack([u_n, eps * dh, phi_t, se * torch.abs(dh)], -1)

    R, Ho, Hl = model(u_n, z_out, z_lay, rho, H_out, H_lay, v=v_n,
                      stream=stream)
    rhs = u_n + (dt / 2) * eps * (u_n @ L_t.T) + dt * R
    rhs = bc_apply_t(cfg, rhs, gL, gR)
    u_next = _solve(sh, rhs)
    pad = cfg["clip_pad"]
    u_next = torch.clamp(u_next, cfg["u_min"] - pad, cfg["u_max"] + pad)
    return (u_next, R, Ho, Hl, rho) if want_rho else (u_next, R, Ho, Hl)


def sample_windows(cfg, args, rng):
    Nt = cfg["Nt"]; w = cfg["tbptt"]
    if args.full_rollout:
        return 0, int(np.ceil(Nt / w))
    nw = min(cfg["nwin"], int(np.ceil(Nt / w)))
    hi = max(1, Nt - nw * w)
    return int(rng.randint(hi)), nw


def train_windows(model, U_ref, V_ref, sh, cfg, start, nwin):
    Nt = cfg["Nt"]; w = cfg["tbptt"]
    B, _, Nx1 = U_ref.shape
    H_out, H_lay = model.init_hidden(B * Nx1, U_ref.device)
    u_n = U_ref[:, start]; u_prev = U_ref[:, start]
    n = start
    for _ in range(nwin):
        e = min(n + w, Nt)
        if e <= n:
            break
        u_n = u_n.detach(); u_prev = u_prev.detach()
        H_out = H_out.detach(); H_lay = H_lay.detach()
        ut = [u_n]; vt = []; Rt = []
        for s in range(n, e):
            v_n = None if V_ref is None else V_ref[:, s].detach()
            u_next, R, H_out, H_lay = imex_step(
                u_n, u_prev, v_n, model, H_out, H_lay, sh, cfg)
            ut.append(u_next); Rt.append(R)
            vt.append(v_n if v_n is not None else u_n.detach())
            u_prev = u_n; u_n = u_next
        yield ut, vt, Rt, n
        n = e


@torch.no_grad()
def rollout_full(model, u_ref_t, v_ref_t, sh, cfg, stream="both",
                 want_rho=False):
    Nt = cfg["Nt"]
    Nx1 = u_ref_t.shape[-1]
    H_out, H_lay = model.init_hidden(Nx1, u_ref_t.device)
    u_n = u_ref_t[0:1]; u_prev = u_ref_t[0:1]
    U = [u_n[0].cpu().numpy()]; RC = []; RI = []; V = []; RHO = []
    for s in range(Nt):
        v_n = None if v_ref_t is None else v_ref_t[s:s + 1]
        RI.append(model.react_id_nograd(u_n, v_n)[0].cpu().numpy())
        V.append((v_n if v_n is not None else u_n)[0].cpu().numpy())
        u_next, R, H_out, H_lay, rho = imex_step(
            u_n, u_prev, v_n, model, H_out, H_lay, sh, cfg,
            stream=stream, want_rho=True)
        RC.append(R[0].cpu().numpy()); U.append(u_next[0].cpu().numpy())
        if want_rho:
            RHO.append(rho[0].cpu().numpy())
        u_prev = u_n; u_n = u_next
    out = (np.stack(U), np.stack(V), np.stack(RC), np.stack(RI))
    return out + (np.stack(RHO),) if want_rho else out


# ======================================================================
#  LOSSES
# ======================================================================

def l_data(ut, U_ref, first, su):
    return torch.stack([((ut[i] - U_ref[:, first + i]) / (su + 1e-8))
                        .pow(2).mean() for i in range(len(ut))]).mean()


def l_cons(ut, vt, Rt, coupled, tau=TAU, B=64):
    Nt = len(Rt)
    if Nt == 0:
        return torch.zeros((), device=ut[0].device)
    step = max(1, Nt // 10)
    idxs = list(range(0, Nt, step))
    ua = torch.cat([ut[n].reshape(-1) for n in idxs])
    Ra = torch.cat([Rt[n].reshape(-1) for n in idxs])
    order = torch.argsort(ua)
    us = ua[order]; Rs = Ra[order]
    if coupled:
        va = torch.cat([vt[n].reshape(-1) for n in idxs])
        vs = va[order]
        du = torch.sqrt((us[1:] - us[:-1]).pow(2)
                        + (vs[1:] - vs[:-1]).pow(2) + 1e-12)
    else:
        du = torch.abs(us[1:] - us[:-1])
    m = du < tau
    if int(m.sum()) < 2:
        return torch.zeros((), device=ua.device)
    v_idx = torch.where(m)[0]
    if len(v_idx) > B:
        v_idx = v_idx[torch.randperm(len(v_idx), device=ua.device)[:B]]
    w = (1 - du[v_idx] / tau).clamp(0, 1).pow(2)
    return (w * (Rs[v_idx] - Rs[v_idx + 1]).pow(2)).mean()


def l_anch(model, cfg, device):
    a = cfg["anchors"]
    if not a:
        return torch.zeros((), device=device)
    ua = torch.tensor([p[0] for p in a], dtype=torch.float32, device=device)
    Ra = torch.tensor([p[2] for p in a], dtype=torch.float32, device=device)
    va = (torch.tensor([p[1] for p in a], dtype=torch.float32, device=device)
          if cfg["coupled"] else None)
    return (model.react_id(ua, va) - Ra).pow(2).mean()


def l_fd(model, u_t, v_t, R_t, w_t, Lambda):
    if u_t.numel() == 0:
        return next(model.parameters()).new_zeros(())
    Rp = model.react_id(u_t, v_t)
    return (w_t * (Rp - R_t).pow(2) / (Lambda ** 2 + 1e-12)).mean()


def l_pos(Rt):
    return torch.stack([torch.relu(-R).pow(2).mean() for R in Rt]).mean()


# ======================================================================
#  EXPERIMENT GRADIENT IMBALANCE
# ======================================================================

def gamma_diagnostic(model, u_ref_t, v_ref_t, sh, cfg, mask_fn, valid_t,
                     su, steps=None):
    steps = steps or min(max(cfg["tbptt"], 8), 20, cfg["Nt"])
    params = [p for p in model.parameters() if p.requires_grad]
    Nx1 = u_ref_t.shape[-1]
    H_out, H_lay = model.init_hidden(Nx1, u_ref_t.device)
    u_n = u_ref_t[0:1]; u_prev = u_ref_t[0:1]
    lay, out = [], []
    for s in range(steps):
        v_n = None if v_ref_t is None else v_ref_t[s:s + 1].detach()
        u_next, _, H_out, H_lay = imex_step(
            u_n, u_prev, v_n, model, H_out, H_lay, sh, cfg)
        ml = mask_fn(s + 1) & valid_t
        mo = (~mask_fn(s + 1)) & valid_t
        d2 = ((u_next[0] - u_ref_t[s + 1]) / (su + 1e-8)).pow(2)
        if bool(ml.any()):
            lay.append(d2[ml].mean())
        if bool(mo.any()):
            out.append(d2[mo].mean())
        u_prev = u_n; u_n = u_next
    if not lay or not out:
        return float("nan")

    def gnorm(gs):
        return float(torch.sqrt(sum((g.detach() ** 2).sum()
                                    for g in gs if g is not None)))

    g_l = torch.autograd.grad(torch.stack(lay).mean(), params,
                              retain_graph=True, allow_unused=True)
    g_o = torch.autograd.grad(torch.stack(out).mean(), params,
                              retain_graph=False, allow_unused=True)
    return gnorm(g_l) / (gnorm(g_o) + 1e-30)


# ======================================================================
#  EXPERIMENT REGIONAL METRICS
# ======================================================================

def valid_mask(cfg, Nx1, skip):
    v = np.ones(Nx1, dtype=bool)
    if cfg["bc_type"] != "periodic" and skip > 0:
        v[:skip] = False; v[-skip:] = False
    return v


def region_masks(cfg, shape, mask_np, skip):
    M = (np.broadcast_to(mask_np, shape).copy() if mask_np.ndim == 1
         else mask_np[:shape[0]].copy())
    Vm = valid_mask(cfg, shape[1], skip)
    M &= Vm
    return M, (~M) & Vm


def regional_metrics(cfg, U, V, RC, RI, u_ref, mask_np, skip, Lambda):
    Nt = min(len(U), len(u_ref)) - 1
    U = U[:Nt + 1]; ur = u_ref[:Nt + 1]
    M, OUT = region_masks(cfg, U.shape, mask_np, skip)
    ADMIT = M | OUT

    def rms(field, m):
        return (float(np.sqrt(np.mean(field[m] ** 2))) if m.sum()
                else float("nan"))

    def pair(num, den, m_lay, m_out, m_all):
        g = rms(den, m_all)
        if not np.isfinite(g) or g <= 0:
            n = float("nan"); return n, n, n, n, True
        dl, do = rms(den, m_lay), rms(den, m_out)
        ok_l = np.isfinite(dl) and dl > 1e-6 * g
        ok_o = np.isfinite(do) and do > 1e-6 * g
        return (rms(num, m_lay) / g, rms(num, m_out) / g,
                rms(num, m_lay) / dl if ok_l else float("nan"),
                rms(num, m_out) / do if ok_o else float("nan"),
                not (ok_l and ok_o))

    du = U - ur
    Ms = M[:Nt]; Os = OUT[:Nt]; As = ADMIT[:Nt]
    Us = U[:Nt]; Vs = V[:Nt]
    Rt_true = (R_true_np(cfg, Us, Vs) if cfg["coupled"]
               else R_true_np(cfg, Us))
    dR = RC[:Nt] - Rt_true
    dcond = RC[:Nt] - RI[:Nt]

    ul, uo, ulr, uor, thin_u = pair(du, ur, M, OUT, ADMIT)
    rl, ro, rlr, ror, thin_r = pair(dR, Rt_true, Ms, Os, As)

    cond_abs = float(np.sqrt(np.mean(dcond ** 2)))
    id_abs = float(np.sqrt(np.mean(RI[:Nt] ** 2)))
    floor = COND_FLOOR_FRAC * float(Lambda)
    return dict(eps_u_layer=ul, eps_u_outer=uo,
                eps_u_layer_rn=ulr, eps_u_outer_rn=uor,
                Rctx_layer=rl, Rctx_outer=ro,
                Rctx_layer_rn=rlr, Rctx_outer_rn=ror,
                degenerate=bool(thin_u or thin_r),
                cond_abs=cond_abs, id_abs=id_abs,
                cond_ratio=cond_abs / max(id_abs, floor),
                cond_degenerate=bool(id_abs < floor),
                layer_frac=float(M.mean()))


# ======================================================================
#  EXPERIMENT STREAM ABLATION INSIDE THE TRAINED DUAL MODEL
# ======================================================================

def stream_ablation(model, cfg, sh, refs_t, refs, ei, mask_np, skip, Lambda):
    if model.mode != "dual":
        return {}
    out = {}
    for s in STREAMS:
        U, V, RC, RI = rollout_full(model, refs_t[ei][0], refs_t[ei][1],
                                    sh, cfg, stream=s)
        m = regional_metrics(cfg, U, V, RC, RI, refs[ei][0], mask_np,
                             skip, Lambda)
        out[s] = {k: m[k] for k in ("eps_u_layer", "eps_u_outer",
                                    "Rctx_layer", "Rctx_outer")}
    b, o = out["both"], out["outer"]

    def rel(k):
        d = b[k]
        return float((o[k] - d) / d) if np.isfinite(d) and d > 0 else float("nan")

    out["cost_of_dropping_layer_stream"] = {
        "eps_u_layer": rel("eps_u_layer"),
        "eps_u_outer": rel("eps_u_outer"),
        "Rctx_layer": rel("Rctx_layer"),
        "Rctx_outer": rel("Rctx_outer"),
    }
    return out


def stream_divergence(model, cfg, sh, refs_t, ei):
    
    if model.mode != "dual":
        return {}
    u_ref_t, v_ref_t = refs_t[ei]
    Nx1 = u_ref_t.shape[-1]
    H_out, H_lay = model.init_hidden(Nx1, u_ref_t.device)
    u_n = u_ref_t[0:1]; u_prev = u_ref_t[0:1]
    num, den = [], []
    for st in range(cfg["Nt"]):
        v_n = None if v_ref_t is None else v_ref_t[st:st + 1]
        u_next, _, H_out, H_lay = imex_step(
            u_n, u_prev, v_n, model, H_out, H_lay, sh, cfg)
        d = (H_out - H_lay).detach()
        num.append(float(torch.linalg.vector_norm(d)))
        den.append(float(torch.linalg.vector_norm(H_out.detach())))
        u_prev = u_n; u_n = u_next
    n = float(np.mean(num)); d = float(np.mean(den))
    return {"stream_gap": n / (d + 1e-12), "stream_gap_abs": n}


# ======================================================================
#  EXPERIMENT 
# ======================================================================

def gate_diagnostic(model, cfg, sh, refs_t, refs, ei, mask_np, skip):
    if model.mode != "dual":
        return {}
    U, V, RC, RI, RHO = rollout_full(model, refs_t[ei][0], refs_t[ei][1],
                                     sh, cfg, want_rho=True)
    Ntr = RHO.shape[0]
    M, OUT = region_masks(cfg, (Ntr, RHO.shape[1]), mask_np, skip)
    if not M.any() or not OUT.any():
        return {}
    r_in = float(RHO[M].mean()); r_out = float(RHO[OUT].mean())
    adm = M | OUT
    r = RHO[adm].astype(np.float64); lab = M[adm].astype(np.float64)
    sr = r.std(); sl = lab.std()
    corr = (float(((r - r.mean()) * (lab - lab.mean())).mean() / (sr * sl))
            if sr > 1e-12 and sl > 1e-12 else float("nan"))
    return {"rho_layer": r_in, "rho_outer": r_out,
            "rho_separation": r_in - r_out, "rho_mask_corr": corr}


# ======================================================================
#  IDENTIFIED-LAW ERROR
# ======================================================================

def _rel_l2(Rp, Rt):
    return float(np.sqrt(np.mean((Rp - Rt) ** 2)) /
                 (np.sqrt(np.mean(Rt ** 2)) + 1e-12))


def eval_eps_R(model, cfg, dev, sh, ng=60):
    if cfg["coupled"]:
        ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
        vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
        UU, VV = np.meshgrid(ug, vg)
        uf = UU.ravel().astype(np.float32); vf = VV.ravel().astype(np.float32)
        Rp = model.react_id_nograd(torch.tensor(uf, device=dev),
                                   torch.tensor(vf, device=dev)).cpu().numpy()
        Rt = R_true_np(cfg, uf.astype(np.float64), vf.astype(np.float64))
        grid = (uf, vf)
    else:
        ug = np.linspace(cfg["u_min"], cfg["u_max"], 400)
        Rp = model.react_id_nograd(
            torch.tensor(ug, dtype=torch.float32, device=dev)).cpu().numpy()
        Rt = R_true_np(cfg, ug)
        grid = (ug, None)

    uv, vv = sh["visited"]
    tu = torch.tensor(uv, device=dev)
    tv = torch.tensor(vv, device=dev) if vv is not None else None
    Rpv = model.react_id_nograd(tu, tv).cpu().numpy()
    Rtv = (R_true_np(cfg, uv.astype(np.float64), vv.astype(np.float64))
           if vv is not None else R_true_np(cfg, uv.astype(np.float64)))
    return _rel_l2(Rp, Rt), _rel_l2(Rpv, Rtv), grid, Rp, Rt


# ======================================================================
#  ONE RUN
# ======================================================================

def split_arm(arm):
    for g in GATES:
        if g != "true" and arm.endswith("_" + g):
            return arm[: -(len(g) + 1)], g
    return arm, "true"


def run_one(arm, seed, args, sh, info):
    mode, gate = split_arm(arm)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed + 9173)
    cfg = sh["cfg"]; dev = DEVICE
    refs = sh["refs"]; su = sh["su"]; mask_np = sh["mask"]
    Lambda = cfg["Lambda"]; coupled = cfg["coupled"]

    # Per-run shallow copy: the gate is a property of the RUN, and sh is
    # shared across every run at this eps. Copying the dict rebinds the
    # gate keys without touching the tensors or the shared object.
    sh = dict(sh)
    sh["gate"] = gate
    if gate == "scram":
        # Drawn from this run's seed, so the permutation is reproducible
        # and differs between seeds -- five seeds give five misalignments
        # rather than five repeats of one.
        nx1 = sh["L_t"].shape[0]
        perm = np.random.RandomState(seed + 4441).permutation(nx1)
        sh["gate_perm"] = torch.tensor(perm, dtype=torch.long, device=dev)

    hidden = info["dual_hidden"] if mode == "dual" else info["single_hidden"]
    geom = dict(GEOM[coupled]); geom["Lambda"] = Lambda
    model = EncoderReactionNet(mode=mode, hidden=hidden, **geom).to(dev)

    # The matched budget is only protective if the model that actually
    # trains carries it. Assert rather than trust.
    n_params = count_parameters(model)
    expected = info["dual_params"] if mode == "dual" else info["single_params"]
    assert n_params == expected, (
        f"{mode} encoder built with {n_params} parameters, matching check "
        f"expected {expected}. The architectures have drifted apart.")

    n_roll = min(cfg["n_rollout_ics"], len(refs))
    ei = min(cfg["eval_ic"], n_roll - 1)
    U_ref = torch.tensor(np.stack([refs[i][0] for i in range(n_roll)]),
                         dtype=torch.float32, device=dev)
    V_ref = (torch.tensor(np.stack([refs[i][1] for i in range(n_roll)]),
                          dtype=torch.float32, device=dev)
             if refs[0][1] is not None else None)
    refs_t = [(U_ref[i], None if V_ref is None else V_ref[i])
              for i in range(n_roll)]

    u_fd_t = torch.tensor(sh["u_fd"], dtype=torch.float32, device=dev)
    v_fd_t = (torch.tensor(sh["v_fd"], dtype=torch.float32, device=dev)
              if sh["v_fd"] is not None else None)
    R_fd_t = torch.tensor(sh["R_fd"], dtype=torch.float32, device=dev)
    w_fd_t = torch.tensor(sh["w_fd"], dtype=torch.float32, device=dev)

    for n, p in model.named_parameters():
        if n.startswith("gru") or n.startswith("proj"):
            p.requires_grad_(False)
    pre = [p for p in model.parameters() if p.requires_grad]
    if pre and u_fd_t.numel():
        opt0 = optim.Adam(pre, lr=3e-3)
        for _ in range(cfg.get("preinit", args.preinit)):
            opt0.zero_grad()
            l_fd(model, u_fd_t, v_fd_t, R_fd_t, w_fd_t, Lambda).backward()
            opt0.step()
    for p in model.parameters():
        p.requires_grad_(True)

    if mask_np.ndim == 1:
        m_static = torch.tensor(mask_np, dtype=torch.bool, device=dev)

        def mask_fn(n):
            return m_static
    else:
        m_all = torch.tensor(mask_np, dtype=torch.bool, device=dev)

        def mask_fn(n):
            return m_all[min(n, m_all.shape[0] - 1)]

    valid_t = torch.tensor(valid_mask(cfg, len(sh["x"]), args.metric_skip),
                           dtype=torch.bool, device=dev)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(100, cfg["epochs"] // 8), eta_min=1e-5)
    epochs = cfg["epochs"]
    hist = []; gammas = []; nan_c = 0; n_skip = 0
    best = float("inf"); best_state = None
    t0 = time.time(); n_steps = 0

    print(f"    [{mode:<6} seed {seed}] hidden={hidden:<3} params={n_params}"
          f"  {epochs} ep  win={cfg['tbptt']}x{cfg['nwin']}  "
          f"Lambda={Lambda:.4g}  {dev}", flush=True)

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); tot = 0.0

        aux = (args.lambda_anch * l_anch(model, cfg, dev)
               + cfg["lambda_fd"] * l_fd(model, u_fd_t, v_fd_t, R_fd_t,
                                         w_fd_t, Lambda))
        aux.backward()
        tot += float(aux.detach())

        start, nwin = sample_windows(cfg, args, rng)
        nw_seen = 0
        for ut, vt, Rt, first in train_windows(model, U_ref, V_ref, sh, cfg,
                                               start, nwin):
            Lw = l_data(ut, U_ref, first, su)
            if args.cons:
                Lw = Lw + l_cons(ut[:-1], vt, Rt, coupled)
            if cfg["positivity"]:
                Lw = Lw + 2.0 * l_pos(Rt)
            lw = float(Lw.detach())
            n_steps += len(Rt)
            if not np.isfinite(lw):
                n_skip += 1
                continue
            (Lw / max(nwin, 1)).backward()
            tot += lw / max(nwin, 1)
            nw_seen += 1

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if nw_seen == 0 or not np.isfinite(tot) or not torch.isfinite(gn):
            nan_c += 1
            opt.zero_grad(set_to_none=True)
            if best_state is not None:
                model.load_state_dict(best_state)
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            sched.step()
            if nan_c >= 8:
                print(f"      ! non-finite loss/grad 8x at epoch {ep}; "
                      f"stopping")
                break
            continue
        nan_c = 0

        opt.step(); sched.step()
        hist.append(tot)
        if tot < best:
            best = tot
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        if ep % args.gamma_every == 0 or ep == 1:
            g = gamma_diagnostic(model, refs_t[ei][0], refs_t[ei][1], sh, cfg,
                                 mask_fn, valid_t, su)
            gammas.append({"epoch": ep, "gamma": g})
            model.zero_grad(set_to_none=True)
        if ep % max(50, epochs // 8) == 0:
            gl = gammas[-1]["gamma"] if gammas else float("nan")
            print(f"      ep {ep}/{epochs}  loss={tot:.5f}  Gamma={gl:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_s = time.time() - t0

    eps_R, eps_R_vis, grid, R_pred, R_ref = eval_eps_R(model, cfg, dev, sh)
    U, V, RC, RI = rollout_full(model, refs_t[ei][0], refs_t[ei][1], sh, cfg)
    reg = regional_metrics(cfg, U, V, RC, RI, refs[ei][0], mask_np,
                           args.metric_skip, Lambda)
    reg0 = regional_metrics(cfg, U, V, RC, RI, refs[ei][0], mask_np, 0,
                            Lambda)
    dstream = stream_ablation(model, cfg, sh, refs_t, refs, ei, mask_np,
                              args.metric_skip, Lambda)
    gate_diag = gate_diagnostic(model, cfg, sh, refs_t, refs, ei, mask_np,
                                args.metric_skip)
    sgap = stream_divergence(model, cfg, sh, refs_t, ei)

    out = dict(mode=arm, arch=mode, gate=gate,
               seed=int(seed), eps=float(cfg["eps"]),
               n_params=n_params, hidden=hidden,
               eps_R=eps_R, eps_R_vis=eps_R_vis,
               Lambda=float(Lambda), loss_history=hist, gammas=gammas,
               Nx=int(cfg["Nx"]), metric_skip=int(args.metric_skip),
               n_fd=int(len(sh["u_fd"])), train_seconds=train_s,
               n_steps=n_steps, n_window_skips=n_skip,
               nodes_across=float(sh["nodes_across"]),
               layer_width=float(sh["layer_width"]),
               stream=dstream, gate_diag=gate_diag, sgap=sgap,
               noskip={k: v for k, v in reg0.items()})
    out.update(reg)
    if not cfg["coupled"]:
        out["u_grid"] = grid[0].tolist()
        out["R_pred"] = np.asarray(R_pred).tolist()
        out["R_true"] = np.asarray(R_ref).tolist()

    gv = [d["gamma"] for d in gammas if np.isfinite(d["gamma"])]
    tail = gv[-8:] if len(gv) >= 8 else gv
    out["gamma"] = float(np.median(tail)) if tail else float("nan")
    out["gamma_lo"] = float(np.percentile(tail, 25)) if tail else float("nan")
    out["gamma_hi"] = float(np.percentile(tail, 75)) if tail else float("nan")

    msg = (f"      -> eps_R={eps_R:.4f} (vis {eps_R_vis:.4f})  "
           f"eps_u(out/lay)={reg['eps_u_outer']:.4f}/{reg['eps_u_layer']:.4f}"
           f"  Rctx(out/lay)={reg['Rctx_outer']:.4f}/{reg['Rctx_layer']:.4f}"
           f"  ctx/id={reg['cond_ratio']:.3f}"
           f"  Gamma={out['gamma']:.3f}"
           f"  [{train_s/60:.1f} min]")
    if reg["cond_degenerate"]:
        msg += "  *** ||R^id|| BELOW FLOOR: ctx/id MEANINGLESS ***"
    if reg["degenerate"]:
        msg += "  *** DEGENERATE REGION ***"
    if dstream:
        c = dstream["cost_of_dropping_layer_stream"]
        msg += (f"\n         drop layer stream -> eps_u_layer "
                f"{c['eps_u_layer']:+.1%}, eps_u_outer "
                f"{c['eps_u_outer']:+.1%}")
    if gate_diag:
        msg += (f";  rho layer/outer = {gate_diag['rho_layer']:.3f}/"
                f"{gate_diag['rho_outer']:.3f}  "
                f"corr={gate_diag['rho_mask_corr']:.3f}")
    if sgap:
        msg += f";  stream gap = {sgap['stream_gap']:.4f}"
    print(msg, flush=True)
    return out


# ======================================================================
#  PREPARE ONE (problem, eps)
# ======================================================================

def prepare(problem, eps, args, quiet=False):
    sizing = choose_nx(problem, eps, args)
    cfg = build_config(problem, eps, args,
                       nx_override=(sizing["Nx"] if sizing else None))
    x, sigma, L_np, A_np, lu, piv = build_operators(cfg)
    refs, t_arr, bc = generate_reference(cfg, x, L_np, lu, piv)

    n_roll0 = min(cfg["n_rollout_ics"], len(refs))
    vis = visited_states(cfg, refs[:n_roll0])

    if cfg["coupled"]:
        if args.lambda_mode == "auto":
            cfg["Lambda"] = estimate_lambda(cfg, refs, L_np)
        else:
            cfg["Lambda"] = float(args.lambda_fixed or 2.0)
    cfg["max_R"], cfg["repr_floor"] = lambda_coverage(cfg, cfg["Lambda"], vis)

    u_fd, v_fd, R_fd, w_fd, fd_rep = extract_fd_targets(
        cfg, refs, L_np, kind=args.fd_filter, tol=args.fd_tol,
        min_bins=args.fd_min_bins)

    ei = cfg["eval_ic"]
    u_eval = refs[ei][0]
    if cfg["mask"] == "boundary":
        mask = layer_mask_boundary(x, sigma, cfg["domain"])
    else:
        mask = layer_mask_curvature(u_eval, L_np, cfg["layer_pct"])

    dev = DEVICE
    L_t = torch.tensor(L_np, dtype=torch.float32, device=dev)
    A_t = torch.tensor(A_np, dtype=torch.float32, device=dev)
    solver = torch.linalg.lu_factor(A_t)
    Ainv_T = None
    if args.solver == "inv":
        Ainv = torch.linalg.inv(A_t)
        res = float(torch.linalg.matrix_norm(Ainv @ A_t
                                             - torch.eye(A_t.shape[0],
                                                         device=dev)))
        if res > 1e-3:
            print(f"    ! inverse residual {res:.2e} is large; "
                  f"falling back to the LU solve")
        else:
            Ainv_T = Ainv.T.contiguous()

    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    xl, xr = cfg["domain"]
    layer_scale = (np.sqrt(cfg["eps"]) if cfg["mesh_scale"] == "sqrt"
                   else cfg["eps"])
    phi_t = torch.exp(-torch.minimum(
        x_t - xl, torch.as_tensor(xr, dtype=torch.float32, device=dev) - x_t)
        / (layer_scale + 1e-12))

    n_roll = min(cfg["n_rollout_ics"], len(refs))
    su = max(float(np.mean(np.abs(np.concatenate(
        [r[0].flatten() for r in refs[:n_roll]])))), 1e-3)

    w_meas, nodes_meas, x_peak, h_local = measured_layer_width(u_eval, x)
    w_anal = (interface_width(problem, cfg["eps"], cfg["amp"])
              if problem in ("fisher", "allen_cahn") else float("nan"))
    h_fine = float(np.min(np.diff(x)))
    on_fine = layer_on_refined_region(x_peak, sigma, cfg["domain"])
    budget = layer_budget_report(u_eval, L_np, mask,
                                 0 if cfg["bc_type"] == "periodic"
                                 else args.metric_skip)

    sh = dict(cfg=cfg, x=x, sigma=sigma, L_np=L_np, refs=refs, bc=bc,
              t_arr=t_arr, u_fd=u_fd, v_fd=v_fd, R_fd=R_fd, w_fd=w_fd,
              fd_report=fd_rep, mask=mask, L_t=L_t, solver=solver,
              Ainv_T=Ainv_T, x_t=x_t, phi_t=phi_t,
              su=su, layer_scale=layer_scale, budget=budget,
              visited=vis, sizing=sizing,
              layer_width=float(w_meas), nodes_across=float(nodes_meas),
              layer_width_analytic=float(w_anal), h_fine=h_fine,
              x_peak=float(x_peak), h_local=float(h_local),
              layer_on_fine=bool(on_fine))

    if not quiet:
        report_prepare(sh, args)
    return sh


def report_prepare(sh, args):
    cfg = sh["cfg"]; x = sh["x"]; mask = sh["mask"]; refs = sh["refs"]
    sigma = sh["sigma"]; fr = sh["fd_report"]
    print(f"\n  eps={cfg['eps']:.0e}  Nx={cfg['Nx']}  "
          f"Lambda={cfg['Lambda']:.4g}  mesh={cfg['mesh_type']}"
          f"({cfg['mesh_scale']})  "
          f"sigma={'n/a' if sigma is None else f'{sigma:.4f}'}  "
          f"eval IC #{cfg['eval_ic']} of "
          f"{min(cfg['n_rollout_ics'], len(refs))}")
    print(f"    measured layer width={sh['layer_width']:.5f} at "
          f"x={sh['x_peak']:.4f}   local h={sh['h_local']:.6f}  "
          f"(fine h={sh['h_fine']:.6f})   "
          f"nodes across={sh['nodes_across']:.1f}"
          + ("" if not np.isfinite(sh["layer_width_analytic"]) else
             f"   [analytic width {sh['layer_width_analytic']:.5f}]"))
    if sh.get("sizing"):
        h = sh["sizing"]["history"]
        if len(h) > 1:
            print("      mesh resized from the measured width: "
                  + " -> ".join(f"Nx={a} ({c:.1f} nodes)" for a, b, c in h))
    if sigma is not None and not sh["layer_on_fine"]:
        print("    *** WARNING: the steepest part of the solution sits "
              "OUTSIDE the Shishkin refined region. Nodes-across above "
              "already uses the LOCAL spacing, so it is the number that "
              "counts. ***")
    # THE LAYER MAY SIMPLY NOT HAVE FORMED. If the measured width is far
    # wider than the asymptotic one, the reference is still governed by
    # the initial transient and is eps-INDEPENDENT -- no encoder can be
    # boundary-layer-aware about a layer that is not there, and the
    # eps sweep is measuring the reaction clock rather than eps.
    wa = sh["layer_width_analytic"]
    if np.isfinite(wa) and wa > 0 and sh["layer_width"] > 3.0 * wa:
        print(f"    *** WARNING: measured width {sh['layer_width']:.4f} is "
              f"{sh['layer_width']/wa:.0f}x the asymptotic "
              f"{wa:.4f}. THE LAYER HAS NOT FORMED WITHIN T={cfg['T']:g}. "
              f"The reference is set by the initial transient, not by "
              f"eps, and this row cannot demonstrate anything about a "
              f"regime-aware encoder. Increase T. ***")
    print(f"    FD targets={fr.get('n_targets', 0)} of "
          f"{fr['n_bins_total']} bins from {fr['n_raw']} accepted nodes "
          f"[filter={fr.get('filter_used', 'n/a')}, "
          f"min/bin={fr.get('min_count_used', '-')}]  "
          f"sigma_u={sh['su']:.4f}  max|R_true|={cfg['max_R']:.4g}  "
          f"representation floor={cfg['repr_floor']:.2e}")
    for e in fr["escalations"]:
        print(f"      coverage guard: {e}")
    if fr.get("n_targets", 0) < args.fd_min_bins:
        print("    *** WARNING: the FD supervision is starved even after "
              "the coverage guard. The reaction cannot be identified "
              "from this few targets and eps_R will reflect that, not "
              "the encoder. ***")

    V0 = valid_mask(cfg, len(x), args.metric_skip)
    u0 = refs[cfg["eval_ic"]][0]
    M0 = (np.broadcast_to(mask, u0.shape) if mask.ndim == 1
          else mask[:u0.shape[0]])
    M0 = M0 & V0; O0 = (~M0) & V0
    g0 = float(np.sqrt(np.mean(u0[M0 | O0] ** 2)))
    dl0 = float(np.sqrt(np.mean(u0[M0] ** 2))) if M0.sum() else np.nan
    do0 = float(np.sqrt(np.mean(u0[O0] ** 2))) if O0.sum() else np.nan
    print(f"    reference magnitude: global={g0:.3e}  layer={dl0:.3e}  "
          f"outer={do0:.3e}")
    if np.nanmin([dl0, do0]) < 1e-3 * g0:
        print("    *** WARNING: one region carries almost no reference "
              "signal. A region-normalised error there is meaningless. ***")
    bud = sh["budget"]
    print(f"    layer nodes={100*mask.mean():.1f}%   metric skip="
          f"{0 if cfg['bc_type']=='periodic' else args.metric_skip}/end  "
          f"-> drops {100*bud['nodes']:.1f}% of layer nodes, "
          f"{100*bud['curvature']:.1f}% of its curvature")
    if sh["nodes_across"] < 2.0:
        print("    *** WARNING: fewer than 2 nodes across the measured "
              "layer. This eps is not resolved; the encoder cannot act "
              "on it. ***")
    if cfg["repr_floor"] > 0.05:
        print(f"    *** WARNING: Lambda={cfg['Lambda']:.4g} clips R_true; "
              f"no model can score below {cfg['repr_floor']:.1%}. ***")


# ======================================================================
#  PREFLIGHT  (no training)
# ======================================================================

def verify_benchmark(problem, args):
    """Inspect the benchmark before spending compute on it."""
    cfg0 = BASE[problem]; pname = cfg0["name"]
    eps_list = args.eps if args.eps else cfg0["eps_list"]
    print(f"\n{'='*74}\n  PREFLIGHT: {pname}  "
          f"(reaction = {args.reaction}, no training)\n{'='*74}")

    n = len(eps_list)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8), squeeze=False)
    rows = []
    for j, eps in enumerate(eps_list):
        sh = prepare(problem, eps, args)
        cfg = sh["cfg"]; x = sh["x"]; ei = cfg["eval_ic"]
        ur = sh["refs"][ei][0]
        Nt = cfg["Nt"]
        snaps = [0, Nt // 4, Nt // 2, Nt]
        for s in snaps:
            axes[0, j].plot(x, ur[s], lw=1.6,
                            label=f"t={sh['t_arr'][s]:.3g}")
        if cfg["coupled"]:
            axes[0, j].plot(x, sh["refs"][ei][1][Nt], "k--", lw=1.0,
                            label="v at T")
        axes[0, j].set_title(f"eps={eps:.0e}   "
                             f"{sh['nodes_across']:.1f} nodes/layer",
                             fontsize=10)
        axes[0, j].set_xlabel("x"); axes[0, j].grid(True, alpha=0.3)
        if j == 0:
            axes[0, j].set_ylabel("u(x,t)"); axes[0, j].legend(fontsize=7)

        if cfg["coupled"]:
            axes[1, j].scatter(sh["visited"][0][::7], sh["visited"][1][::7],
                               s=2, c="0.8", label="visited")
            if len(sh["u_fd"]):
                axes[1, j].scatter(sh["u_fd"], sh["v_fd"], s=26, c="#C0392B",
                                   marker="s", label="FD targets")
            axes[1, j].set_xlabel("u"); axes[1, j].set_ylabel("v")
            axes[1, j].legend(fontsize=7)
        else:
            ug = np.linspace(cfg["u_min"], cfg["u_max"], 300)
            axes[1, j].plot(ug, R_true_np(cfg, ug), "k-", lw=1.4,
                            label="R_true")
            if len(sh["u_fd"]):
                axes[1, j].plot(sh["u_fd"], sh["R_fd"], "s", ms=5,
                                color="#C0392B", label="FD targets")
            axes[1, j].set_xlabel("u"); axes[1, j].legend(fontsize=7)
        axes[1, j].set_title(f"{len(sh['u_fd'])} FD targets", fontsize=9)
        axes[1, j].grid(True, alpha=0.3)

        rows.append((eps, cfg["Nx"], sh["layer_width"], sh["nodes_across"],
                     len(sh["u_fd"]), sh["fd_report"]["n_bins_total"],
                     cfg["Lambda"], cfg["max_R"], cfg["repr_floor"],
                     sh["budget"], sh["layer_on_fine"]))

    fig.suptitle(f"{pname} preflight - references and FD supervision "
                 f"({args.reaction} reaction law)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"preflight_{problem}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Saved: {p}")

    print(f"\n  {'eps':>9}{'Nx':>7}{'layer w':>10}{'nodes':>8}"
          f"{'FD/bins':>11}{'Lambda':>10}{'max|R|':>10}{'floor':>9}"
          f"{'skip drops':>12}")
    print("  " + "-" * 88)
    for (eps, nx, w, nd, nb, nbt, lam, mr, fl, bud, onf) in rows:
        flag = "" if nd >= 2 else "  <-- UNRESOLVED"
        if not onf:
            flag += "  <-- LAYER OFF THE REFINED REGION"
        if nb < args.fd_min_bins:
            flag += "  <-- FD SUPERVISION STARVED"
        print(f"  {eps:>9.0e}{nx:>7}{w:>10.5f}{nd:>8.1f}"
              f"{f'{nb}/{nbt}':>11}{lam:>10.4g}{mr:>10.4g}{fl:>8.1%}"
              f"{100*bud['curvature']:>11.1f}%{flag}")
    print("\n  'layer w' is measured from the reference as "
          "(max u - min u)/max|du/dx|.")
    print("  'nodes' uses the LOCAL mesh spacing at the steepest point.")
    print("  'FD/bins' is populated bins over total bins. A starved row "
          "cannot identify")
    print("  the reaction at all, and its eps_R says nothing about the "
          "encoder -- this")
    print("  is the check that the previous revision did not have and "
          "that FitzHugh-")
    print("  Nagumo failed silently.")
    print("  The lower row of panels shows WHERE in state space the "
          "targets sit.")


# ======================================================================
#  AGGREGATION, PAIRED STATISTICS                            
# ======================================================================

SCALARS = ["eps_R", "eps_R_vis", "eps_u_outer", "eps_u_layer", "Rctx_outer",
           "Rctx_layer", "cond_ratio", "gamma", "layer_frac"]

# Metrics on which "dual better" means "smaller".
PAIRED_KEYS = ["eps_R_vis", "eps_u_outer", "eps_u_layer", "Rctx_outer",
               "Rctx_layer", "gamma"]


def _sign_test_p(k, n):
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def arms_present(runs):
    have = {r["mode"] for r in runs}
    order = ["single", "dual"] + [f"dual_{g}" for g in GATES if g != "true"]
    return [a for a in order if a in have] + sorted(have - set(order))


def aggregate(runs):
    out = []
    eps_vals = sorted({r["eps"] for r in runs}, reverse=True)
    for e in eps_vals:
        for mode in arms_present(runs):
            rs = [r for r in runs if r["eps"] == e and r["mode"] == mode]
            if not rs:
                continue
            rec = {"eps": e, "mode": mode, "n_seeds": len(rs),
                   "n_params": rs[0]["n_params"], "hidden": rs[0]["hidden"],
                   "Lambda": rs[0]["Lambda"], "Nx": rs[0]["Nx"],
                   "n_fd": rs[0].get("n_fd", 0),
                   "nodes_across": rs[0].get("nodes_across", float("nan")),
                   "layer_width": rs[0].get("layer_width", float("nan")),
                   "degenerate": any(r.get("degenerate") for r in rs),
                   "cond_degenerate": any(r.get("cond_degenerate")
                                          for r in rs)}
            for k in SCALARS:
                v = np.array([r[k] for r in rs], dtype=float)
                v = v[np.isfinite(v)]
                rec[k] = float(np.median(v)) if v.size else float("nan")
                rec[k + "_lo"] = (float(np.percentile(v, 25)) if v.size
                                  else float("nan"))
                rec[k + "_hi"] = (float(np.percentile(v, 75)) if v.size
                                  else float("nan"))
            for k in ("eps_u_outer", "eps_u_layer", "Rctx_outer",
                      "Rctx_layer"):
                z = np.array([r["noskip"][k] for r in rs], dtype=float)
                z = z[np.isfinite(z)]
                rec["ns_" + k] = float(np.median(z)) if z.size else float("nan")
            out.append(rec)

    for e in eps_vals:
        sd = {r["seed"]: r for r in runs if r["eps"] == e
              and r["mode"] == "single"}
        dd = {r["seed"]: r for r in runs if r["eps"] == e
              and r["mode"] == "dual"}
        common = sorted(set(sd) & set(dd))
        wins = {}
        for k in PAIRED_KEYS:
            w = sum(1 for s in common
                    if np.isfinite(sd[s][k]) and np.isfinite(dd[s][k])
                    and dd[s][k] < sd[s][k])
            wins[k] = (w, len(common))
        for rec in out:
            if rec["eps"] == e:
                rec["wins"] = wins
    return out


def pooled_paired(runs, key, a="single", b="dual"):
    pairs = []
    for e in sorted({r["eps"] for r in runs}):
        sd = {r["seed"]: r for r in runs if r["eps"] == e
              and r["mode"] == a}
        dd = {r["seed"]: r for r in runs if r["eps"] == e
              and r["mode"] == b}
        for s in sorted(set(sd) & set(dd)):
            a, b = sd[s][key], dd[s][key]
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                pairs.append((a, b))
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    w = sum(1 for a, b in pairs if b < a)
    lr = [math.log10(a / b) for a, b in pairs]
    return {"n": n, "wins": w, "p": _sign_test_p(w, n),
            "median_log10_ratio": float(np.median(lr)),
            "median_ratio": float(10 ** np.median(lr))}


def eps_slopes(runs, key):
    out = {}
    for mode in ("single", "dual"):
        per_seed = {}
        for r in runs:
            if r["mode"] != mode:
                continue
            if not np.isfinite(r[key]) or r[key] <= 0:
                continue
            per_seed.setdefault(r["seed"], []).append(
                (math.log10(r["eps"]), math.log10(r[key])))
        sl = []
        for s, pts in per_seed.items():
            if len(pts) >= 3:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                sl.append(float(np.polyfit(xs, ys, 1)[0]))
        out[mode] = sl
    a, b = out.get("single", []), out.get("dual", [])
    res = {"single": a, "dual": b,
           "single_med": float(np.median(a)) if a else float("nan"),
           "dual_med": float(np.median(b)) if b else float("nan")}
    common = min(len(a), len(b))
    if common:
        w = sum(1 for i in range(common) if abs(b[i]) < abs(a[i]))
        res.update(n=common, wins=w, p=_sign_test_p(w, common))
    return res


def stream_summary(runs):
    rows = []
    for r in runs:
        if r["mode"] != "dual" or not r.get("stream"):
            continue
        c = r["stream"]["cost_of_dropping_layer_stream"]
        rows.append((r["eps"], r["seed"], c["eps_u_layer"], c["eps_u_outer"],
                     c["Rctx_layer"], c["Rctx_outer"]))
    if not rows:
        return None
    arr = np.array([[x[2], x[3], x[4], x[5]] for x in rows], dtype=float)
    out = {"n": len(rows), "rows": rows}
    for i, k in enumerate(("eps_u_layer", "eps_u_outer",
                           "Rctx_layer", "Rctx_outer")):
        v = arr[:, i][np.isfinite(arr[:, i])]
        out[k] = float(np.median(v)) if v.size else float("nan")
        out[k + "_pos"] = int((v > 0).sum())
        out[k + "_n"] = int(v.size)
    return out


def gate_summary(runs):
    vals = [r["gate_diag"] for r in runs
            if r["mode"] == "dual" and r.get("gate_diag")]
    if not vals:
        return None
    out = {"n": len(vals)}
    for k in ("rho_layer", "rho_outer", "rho_separation", "rho_mask_corr"):
        v = np.array([g[k] for g in vals], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = float(np.median(v)) if v.size else float("nan")
    return out


def get(agg, eps, mode, key, default=float("nan")):
    for r in agg:
        if r["eps"] == eps and r["mode"] == mode:
            return r.get(key, default)
    return default


def gap(single, dual, floor):
    if not (np.isfinite(single) and np.isfinite(dual)) or single <= 0:
        return float("nan"), False
    if max(abs(single), abs(dual)) < floor:
        return (single - dual) / single, False
    return (single - dual) / single, True


def stream_gap_summary(runs, arm="dual"):
    v = [r["sgap"]["stream_gap"] for r in runs
         if r["mode"] == arm and r.get("sgap")]
    return float(np.median(v)) if v else float("nan")


MECH_KEYS = ("eps_u_layer", "eps_u_outer", "Rctx_layer", "eps_R_vis")


def mechanism_test(runs, keys=MECH_KEYS):
    have = set(r["mode"] for r in runs)
    out = {}
    for a, b, tag in (("single", "dual", "capacity"),
                      ("dual_scram", "dual", "alignment"),
                      ("dual_const", "dual", "gating")):
        if a not in have or b not in have:
            continue
        out[tag] = {"a": a, "b": b,
                    **{k: pooled_paired(runs, k, a=a, b=b) for k in keys}}
    return out


def report_mechanism(problem, runs):
    mt = mechanism_test(runs)
    if not mt:
        return
    print(f"\n  {'-'*70}")
    print("  GATE-INTERVENTION CONTROLS ")
    print(f"  {'-'*70}")
    print(f"  {'comparison':<34}{'metric':>14}{'wins':>10}{'p':>8}"
          f"{'ratio':>10}")
    for tag, d in mt.items():
        lbl = "{} -> {}  ({})".format(d["a"], d["b"], tag)
        first = True
        for k in MECH_KEYS:
            st = d[k]
            if not st.get("n"):
                continue
            print(f"  {(lbl if first else ''):<34}{k:>14}"
                  f"{'{}/{}'.format(st['wins'], st['n']):>10}"
                  f"{st['p']:>8.3f}{st['median_ratio']:>10.3f}")
            first = False
    g_true = stream_gap_summary(runs, "dual")
    g_scr = stream_gap_summary(runs, "dual_scram")
    print(f"\n  stream gap ||H_out - H_lay|| / ||H_out||:  "
          f"true rho {g_true:.4f}", end="")
    if np.isfinite(g_scr):
        print(f",  scrambled rho {g_scr:.4f}")
    else:
        print()
    if np.isfinite(g_true) and g_true < 1e-3:
        print("  *** THE TWO STREAMS COLLAPSED. The blend is a no-op, so "
              "rho cannot")
        print("      matter and this control CANNOT SPEAK. Report the "
              "collapse, not")
        print("      a null result about alignment.")
    else:
        print("  The streams carry different states, so the blend is "
              "live and the")
        print("  alignment test is meaningful.")
    print("\n  Read only the alignment row for the title's claim. The "
          "capacity row")
    print("  says the dual encoder helps; it does not say WHY, and a ")
    print("  answer it with 'two streams are more expressive'. The "
          "alignment row")
    print("  holds architecture and parameter count fixed and varies "
          "only whether")
    print("  rho points at the layer.")


# ======================================================================
#  PLOTS
# ======================================================================

C_DUAL, C_SINGLE = "#C0392B", "#2980B9"


def _band(ax, eps, agg, mode, key, color, marker, log=True):
    y = np.array([get(agg, e, mode, key) for e in eps], float)
    lo = np.array([get(agg, e, mode, key + "_lo") for e in eps], float)
    hi = np.array([get(agg, e, mode, key + "_hi") for e in eps], float)
    (ax.semilogx if log else ax.plot)(eps, y, marker, color=color, lw=2,
                                      ms=7, label=mode)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(np.array(eps)[ok], lo[ok], hi[ok],
                        color=color, alpha=0.18, lw=0)


def make_plots(problem, agg, runs, args):
    cfgn = BASE[problem]["name"]
    eps = sorted({r["eps"] for r in agg}, reverse=True)

    # Experiment A + B
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("eps_R_vis", "identified law  eps_R (visited states)", True),
              ("eps_u_outer", "trajectory error, OUTER region", True),
              ("eps_u_layer", "trajectory error, LAYER region", True),
              ("Rctx_outer", "|R^ctx - R_true|, OUTER", True),
              ("Rctx_layer", "|R^ctx - R_true|, LAYER", True),
              ("gamma", "Gamma = |grad|_layer / |grad|_outer", True)]
    for a, (k, title, lg) in zip(ax.ravel(), panels):
        _band(a, eps, agg, "single", k, C_SINGLE, "o-")
        _band(a, eps, agg, "dual", k, C_DUAL, "s-")
        a.set_xscale("log")
        if lg:
            a.set_yscale("log")
        a.set_xlabel("eps"); a.set_title(title, fontsize=10)
        a.grid(True, which="both", alpha=0.3); a.legend(fontsize=8)
        a.invert_xaxis()
    fig.suptitle(f"{cfgn}: matched single vs dual encoder "
                 f"(median over {agg[0]['n_seeds']} seeds, IQR band)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"expAB_{problem}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")

    # Experiment D
    ss = stream_summary(runs)
    if ss:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        rows = ss["rows"]
        ex = [r[0] for r in rows]
        for a, (i, ttl) in zip(ax, [(2, "eps_u LAYER"), (3, "eps_u OUTER")]):
            a.semilogx(ex, [100 * r[i] for r in rows], "o", color=C_DUAL,
                       ms=6, alpha=0.8)
            a.axhline(0, color="k", lw=1)
            a.set_xlabel("eps"); a.set_ylabel("% error increase")
            a.set_title(f"cost of dropping the layer stream: {ttl}",
                        fontsize=10)
            a.grid(True, alpha=0.3); a.invert_xaxis()
        fig.suptitle(f"{cfgn}: Experiment D -- same weights, decoder routed "
                     f"from one stream", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"expD_{problem}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {p}")


# ======================================================================
#  TABLES AND READING
# ======================================================================

def emit_tables(problem, agg, runs, args):
    name = BASE[problem]["name"]
    eps = sorted({r["eps"] for r in agg}, reverse=True)
    L = []
    W = ("+" + "+".join(["-" * 9, "-" * 8, "-" * 8, "-" * 9, "-" * 9,
                         "-" * 9, "-" * 9, "-" * 9, "-" * 8, "-" * 8]) + "+")
    hdr = ("|{:^9}|{:^8}|{:^8}|{:^9}|{:^9}|{:^9}|{:^9}|{:^9}|{:^8}|{:^8}|"
           .format("eps", "encoder", "FDbins", "eps_R", "eR_vis", "eu_out",
                   "eu_lay", "Rc_lay", "ctx/id", "Gamma"))
    L.append(f"\n  {name} - encoder study")
    L.append(f"  reaction law: {args.reaction}   metric skip: "
             f"{args.metric_skip} nodes/end   seeds: {agg[0]['n_seeds']} "
             f"(median shown)")
    L.append(W); L.append(hdr); L.append(W)
    for e in eps:
        for mode in ("single", "dual"):
            r = [z for z in agg if z["eps"] == e and z["mode"] == mode]
            if not r:
                continue
            r = r[0]
            cd = "  n/a" if r["cond_degenerate"] else f"{r['cond_ratio']:.3f}"
            L.append("|{:^9}|{:^8}|{:^8}|{:^9.4f}|{:^9.4f}|{:^9.4f}"
                     "|{:^9.4f}|{:^9.4f}|{:^8}|{:^8.3f}|".format(
                         f"{e:.0e}", mode, r["n_fd"], r["eps_R"],
                         r["eps_R_vis"], r["eps_u_outer"], r["eps_u_layer"],
                         r["Rctx_layer"], cd, r["gamma"]))
        L.append(W)

    L.append("\n  Discretisation actually used at each eps:")
    L.append(f"      {'eps':>9}{'Nx':>7}{'layer width':>14}"
             f"{'nodes/layer':>13}{'Lambda':>10}{'FD bins':>9}")
    for e in eps:
        L.append(f"      {e:>9.0e}{int(get(agg,e,'dual','Nx')):>7}"
                 f"{get(agg,e,'dual','layer_width'):>14.5f}"
                 f"{get(agg,e,'dual','nodes_across'):>13.1f}"
                 f"{get(agg,e,'dual','Lambda'):>10.4g}"
                 f"{int(get(agg,e,'dual','n_fd')):>9}")
    L.append("  A trend in the errors is a trend in eps only if "
             "nodes/layer is flat.")
    if any(get(agg, e, "dual", "cond_degenerate") for e in eps):
        L.append("\n  ctx/id is printed as n/a wherever ||R^id|| falls "
                 "below 5% of Lambda.")
        L.append("  There the ratio is 0/0 and says nothing about the "
                 "encoder; the row's")
        L.append("  eps_R will show that the reaction was not identified "
                 "at all.")

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT_DIR, f"encoder_table_{problem}.txt"), "w") as f:
        f.write(txt + "\n")
    return txt


def read_result(problem, agg, runs, args):
    name = BASE[problem]["name"]
    eps = sorted({r["eps"] for r in agg}, reverse=True)
    print(f"\n{'='*74}\n  READING THE RESULT - {name}\n{'='*74}")

    print(f"\n  Per-eps gaps (positive = dual better). A gap is printed "
          f"only when the")
    print(f"  larger of the two errors exceeds the floor "
          f"{args.gap_floor:.0e}; below that both")
    print(f"  encoders reproduce the trajectory and the ratio is noise.")
    print(f"\n  {'eps':>9}{'eR_vis':>10}{'outer':>10}{'layer':>10}"
          f"{'Rc_lay':>10}{'Gamma s->d':>16}{'dual wins (layer)':>20}")
    print("  " + "-" * 86)
    for e in eps:
        cells = []
        for k, fl in (("eps_R_vis", 1e-3), ("eps_u_outer", args.gap_floor),
                      ("eps_u_layer", args.gap_floor),
                      ("Rctx_layer", 1e-3)):
            g, ok = gap(get(agg, e, "single", k), get(agg, e, "dual", k), fl)
            cells.append(f"{g:>9.1%}" if ok else f"{'tie':>9}")
        gs = get(agg, e, "single", "gamma"); gd = get(agg, e, "dual", "gamma")
        w = get(agg, e, "dual", "wins", {}).get("eps_u_layer", (0, 0))
        print(f"  {e:>9.0e}" + "".join(cells) +
              f"{f'{gs:.2f} -> {gd:.2f}':>16}{f'{w[0]}/{w[1]}':>20}")

    print(f"\n  POOLED PAIRED TEST over every (eps, seed) pair. This is "
          f"the statistic;")
    print(f"  a win count at one eps is not. p is a two-sided exact "
          f"sign test.")
    print(f"\n  {'metric':>14}{'dual wins':>12}{'p':>9}"
          f"{'median single/dual':>22}")
    print("  " + "-" * 58)
    for k in PAIRED_KEYS:
        s = pooled_paired(runs, k)
        if not s.get("n"):
            continue
        wn = "{}/{}".format(s["wins"], s["n"])
        print(f"  {k:>14}{wn:>12}{s['p']:>9.3f}{s['median_ratio']:>22.3f}")

    print(f"\n  eps-UNIFORMITY as a slope, d log(error) / d log(eps), "
          f"fitted per seed.")
    print(f"  A slope nearer zero is more eps-uniform. This pools the "
          f"sweep into one")
    print(f"  number per seed instead of four noisy per-eps gaps.")
    print(f"\n  {'metric':>14}{'single':>10}{'dual':>10}"
          f"{'dual flatter':>15}{'p':>8}")
    print("  " + "-" * 58)
    for k in ("eps_u_layer", "eps_u_outer", "Rctx_layer"):
        s = eps_slopes(runs, k)
        if not np.isfinite(s["single_med"]):
            continue
        wn = "{}/{}".format(s.get("wins", "-"), s.get("n", "-")) \
            if "n" in s else "-"
        pv = "{:.3f}".format(s["p"]) if "p" in s else "  -"
        print(f"  {k:>14}{s['single_med']:>10.2f}{s['dual_med']:>10.2f}"
              f"{wn:>15}{pv:>8}")

    ss = stream_summary(runs)
    if ss:
        print(f"\n  EXPERIMENT D -- the decoder routed from one stream, on "
              f"the SAME trained")
        print(f"  dual weights. 'outer only' is the information path the "
              f"single encoder")
        print(f"  has. Positive = the error grows when the layer stream is "
              f"removed, i.e.")
        print(f"  the second stream was contributing. {ss['n']} models "
              f"(eps x seed).")
        print(f"\n  {'region':>14}{'median change':>16}{'models worse':>15}")
        print("  " + "-" * 47)
        for k, lab in (("eps_u_layer", "trajectory LAYER"),
                       ("eps_u_outer", "trajectory OUTER"),
                       ("Rctx_layer", "reaction LAYER"),
                       ("Rctx_outer", "reaction OUTER")):
            frac = "{}/{}".format(ss[k + "_pos"], ss[k + "_n"])
            print(f"  {lab:>14}{ss[k]:>15.1%}{frac:>15}")
        print("\n  If the LAYER rows are clearly positive and the OUTER "
              "rows are near zero,")
        print("  the two streams are specialised as Section 2.4 describes. "
              "If every row is")
        print("  near zero the second stream is inert and the paper must "
              "say so.")

    gsum = gate_summary(runs)
    if gsum:
        print(f"\n  EXPERIMENT E -- the regime gate against the "
              f"model-independent mask.")
        print(f"    mean rho inside the layer : {gsum['rho_layer']:.3f}")
        print(f"    mean rho outside          : {gsum['rho_outer']:.3f}")
        print(f"    separation                : {gsum['rho_separation']:+.3f}")
        print(f"    point-biserial corr       : {gsum['rho_mask_corr']:+.3f}")
        print("  rho is never used to define a region. A separation near "
              "zero would mean")
        print("  the gate cannot tell the regimes apart, and no routing "
              "built on it can.")

    print("\n  Pre-committed reading:")
    print("   - dual flatter slope in eps_u_layer   -> eps-uniformity "
          "holds")
    print("   - Experiment D positive on LAYER only -> the streams are "
          "specialised")
    print("   - pooled sign test p < 0.05           -> the effect is not "
          "seed noise")
    print("   - same gaps on Predator-Prey          -> not about singular "
          "structure")
    print("   - Gamma_dual < Gamma_single           -> the motivation in "
          "2.4 is validated")
    print("   - eps_R ~ 1.0 anywhere                -> that benchmark "
          "identified nothing;")
    print("                                            read no encoder "
          "conclusion from it")


# ======================================================================
#  COST ESTIMATE
# ======================================================================

def estimate_cost(problems, args):
    seeds = args.seeds or PROFILES[args.profile][1]
    total = 0; rows = []
    for p in problems:
        c0 = BASE[p]
        eps_list = args.eps if args.eps else c0["eps_list"]
        cfg = build_config(p, eps_list[0], args)
        w = cfg["tbptt"]
        nw = (int(np.ceil(cfg["Nt"] / w)) if args.full_rollout
              else min(cfg["nwin"], int(np.ceil(cfg["Nt"] / w))))
        per_run = cfg["epochs"] * nw * w
        n_runs = len(eps_list) * len(args.arms) * seeds
        sub = per_run * n_runs
        total += sub
        rows.append((c0["name"], cfg["epochs"], w, nw, per_run, n_runs, sub))
    print(f"\n{'='*74}\n  COST ESTIMATE  (profile={args.profile}, "
          f"seeds={seeds}, arms={'+'.join(args.arms)})\n{'='*74}")
    print(f"  {'problem':<18}{'epochs':>8}{'win':>6}{'nwin':>6}"
          f"{'steps/run':>12}{'runs':>7}{'steps':>14}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r[0]:<18}{r[1]:>8}{r[2]:>6}{r[3]:>6}{r[4]:>12,}"
              f"{r[5]:>7}{r[6]:>14,}")
    print("  " + "-" * 72)
    print(f"  {'TOTAL':<18}{'':>8}{'':>6}{'':>6}{'':>12}{'':>7}"
          f"{total:>14,}")
    print("\n  Steps are BATCHED over the rollout trajectories, so one "
          "step advances")
    print("  all of them. Wall time is steps x (forward + backward) and "
          "depends on the")
    print("  device; measure it with a short run:")
    print("    python encoder_experiment.py --problem fisher --epochs 40 "
          "--seeds 1")
    print("  then scale. On a current GPU expect roughly 1.5-4 ms per "
          "batched step,")
    print(f"  i.e. about {total*1.5e-3/3600:.1f}-{total*4e-3/3600:.1f} "
          f"hours for the whole sweep.")
    print("  Use --profile fast to halve it, or --seeds 3.")
    return total


# ======================================================================
#  MAIN
# ======================================================================

def run_problem(problem, args):
    cfg0 = BASE[problem]
    eps_list = args.eps if args.eps else cfg0["eps_list"]
    seeds = list(range(args.seed0, args.seed0 + (
        args.seeds or PROFILES[args.profile][1])))
    arms = list(args.arms)
    print(f"\n{'#'*74}\n#  {cfg0['name']}  --  encoder study, "
          f"{len(eps_list)} eps x {len(arms)} arms x {len(seeds)} seeds"
          f"\n#  arms: {', '.join(arms)}"
          f"\n{'#'*74}")

    geom = dict(GEOM[cfg0["coupled"]])
    _, _, info = build_pair(geom, dual_hidden=args.hidden, device=DEVICE)

    runs = []
    for eps in eps_list:
        sh = prepare(problem, eps, args)
        for seed in seeds:
            for arm in arms:
                runs.append(run_one(arm, seed, args, sh, info))

    agg = aggregate(runs)
    with open(os.path.join(OUT_DIR, f"encoder_runs_{problem}.json"), "w") as f:
        json.dump({"problem": problem, "agg": agg,
                   "runs": [{k: v for k, v in r.items()
                             if k not in ("loss_history",)} for r in runs]},
                  f, indent=1, default=float)
    emit_tables(problem, agg, runs, args)
    make_plots(problem, agg, runs, args)
    read_result(problem, agg, runs, args)
    report_mechanism(problem, runs)
    return runs, agg


def cross_summary(all_runs, args):
    print(f"\n{'='*74}\n  CROSS-PROBLEM SUMMARY\n{'='*74}")
    print(f"\n  {'problem':<18}{'metric':>14}{'dual wins':>12}{'p':>8}"
          f"{'single/dual':>14}")
    print("  " + "-" * 68)
    for p, runs in all_runs.items():
        for k in ("eps_u_layer", "Rctx_layer", "gamma"):
            s = pooled_paired(runs, k)
            if not s.get("n"):
                continue
            wn = "{}/{}".format(s["wins"], s["n"])
            nm = BASE[p]["name"]
            print(f"  {nm:<18}{k:>14}{wn:>12}{s['p']:>8.3f}"
                  f"{s['median_ratio']:>14.3f}")
        print()

    align = {p: mechanism_test(r).get("alignment")
             for p, r in all_runs.items()}
    if any(align.values()):
        print(f"  MECHANISM: scrambled rho -> true rho, same architecture "
              f"and parameters")
        print(f"  {'problem':<18}{'metric':>14}{'true wins':>12}{'p':>8}"
              f"{'scram/true':>14}")
        print("  " + "-" * 68)
        for p, d in align.items():
            if not d:
                continue
            for k in ("eps_u_layer", "eps_u_outer"):
                st = d[k]
                if not st.get("n"):
                    continue
                print(f"  {BASE[p]['name']:<18}{k:>14}"
                      f"{'{}/{}'.format(st['wins'], st['n']):>12}"
                      f"{st['p']:>8.3f}{st['median_ratio']:>14.3f}")
        print("\n  The claim in the title is supported if true rho beats "
              "scrambled rho")
        print("  in the LAYER on the singularly perturbed benchmarks and "
              "not on")
        print("  Predator-Prey, whose layer is not sharp. A gain that is "
              "the same")
        print("  everywhere is expressiveness, not boundary-layer "
              "awareness.\n")

    print(f"  {'problem':<18}{'D: layer cost':>16}{'D: outer cost':>16}"
          f"{'rho separation':>17}")
    print("  " + "-" * 68)
    for p, runs in all_runs.items():
        ss = stream_summary(runs); g = gate_summary(runs)
        if not ss:
            continue
        print(f"  {BASE[p]['name']:<18}{ss['eps_u_layer']:>15.1%}"
              f"{ss['eps_u_outer']:>16.1%}"
              f"{(g['rho_separation'] if g else float('nan')):>17.3f}")

    print("\n  Read this against the preflight, not on its own. A row can "
          "only speak")
    print("  about the encoder if its benchmark (i) resolved the layer -- "
          "nodes-across")
    print("  flat across the sweep -- and (ii) identified the reaction at "
          "all -- eps_R")
    print("  well below 1. A benchmark that failed either test is "
          "reporting the")
    print("  discretisation or the supervision, not the architecture.")
    print("\n  Predator-Prey is the negative control: eps is the prey "
          "diffusivity and the")
    print("  reaction carries no 1/eps, so nothing sharpens as eps falls. "
          "If the dual")
    print("  encoder shows the same gain there as on Allen-Cahn and "
          "FitzHugh-Nagumo,")
    print("  the gain is not about singular structure and Section 2.4's "
          "mechanism claim")
    print("  does not survive. Its rho separation should be near zero "
          "while the others")
    print("  are not.")
    p = os.path.join(OUT_DIR, "encoder_summary.json")
    with open(p, "w") as f:
        json.dump({k: {"pooled": {m: pooled_paired(v, m)
                                  for m in PAIRED_KEYS},
                       "mechanism": mechanism_test(v),
                       "stream_gap": {a: stream_gap_summary(v, a)
                                      for a in arms_present(v)},
                       "slopes": {m: eps_slopes(v, m)
                                  for m in ("eps_u_layer", "eps_u_outer")},
                       "stream": stream_summary(v),
                       "gate": gate_summary(v)}
                   for k, v in all_runs.items()}, f, indent=1, default=float)
    print(f"\n  Saved: {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Encoder study")
    ap.add_argument("--problem", default="fisher",
                    choices=ALL_PROBLEMS + ["all"])
    ap.add_argument("--profile", default="standard",
                    choices=list(PROFILES))
    ap.add_argument("--eps", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs-map", nargs="*", default=[], metavar="P=N",
                    help="per-problem epoch override, e.g. "
                         "--epochs-map fhn=2000. Overrides --epochs for "
                         "those problems only, so --problem all can run "
                         "in one process")
    ap.add_argument("--preinit-map", nargs="*", default=[], metavar="P=N",
                    help="per-problem Stage-0 override, e.g. "
                         "--preinit-map fhn=1500")
    ap.add_argument("--arms", nargs="+", default=["single", "dual"],
                    choices=["single", "dual", "dual_scram", "dual_const"],
                    help="encoder arms to run. single+dual (default) is "
                         "the capacity control and reproduces the "
                         "submitted study exactly; add dual_scram for the "
                         "gate-alignment mechanism test, dual_const for "
                         "no gating at all")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--preinit", type=int, default=400)
    ap.add_argument("--lambda-anch", type=float, default=1.0)
    ap.add_argument("--cons", type=int, default=1)
    ap.add_argument("--tbptt", type=int, default=None)
    ap.add_argument("--nwin", type=int, default=None,
                    help="TBPTT windows sampled per epoch")
    ap.add_argument("--full-rollout", action="store_true",
                    help="integrate every window every epoch (slow)")
    ap.add_argument("--n-ics", type=int, default=None)
    ap.add_argument("--eval-ic", type=int, default=None)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--min-nx", type=int, default=64)
    ap.add_argument("--max-nx", type=int, default=512)
    ap.add_argument("--ac-nodes", type=float, default=9.0)
    ap.add_argument("--fhn-nodes", type=float, default=5.0)
    ap.add_argument("--fisher-T", type=float, default=None,
                    help="final time for Fisher-KPP; the manuscript's 0.3 "
                         "develops no eps-dependent layer (see BASE)")
    ap.add_argument("--fhn-mesh", default="uniform",
                    choices=["uniform", "shishkin"])
    ap.add_argument("--mesh-scale", default="auto",
                    choices=["auto", "sqrt", "lin"])
    ap.add_argument("--reaction", default="paper", choices=["paper", "code"])
    ap.add_argument("--ic-scale", default="eps", choices=["fixed", "eps"])
    ap.add_argument("--lambda-mode", default="auto", choices=["auto", "fixed"])
    ap.add_argument("--lambda-fixed", type=float, default=None)
    ap.add_argument("--fd-filter", default="reliability",
                    choices=["reliability", "pct"],
                    help="how unreliable FD estimates are rejected")
    ap.add_argument("--fd-tol", type=float, default=0.10)
    ap.add_argument("--fd-min-bins", type=int, default=12)
    ap.add_argument("--metric-skip", type=int, default=METRIC_SKIP)
    ap.add_argument("--gap-floor", type=float, default=5e-3,
                    help="below this error both encoders are at the floor")
    ap.add_argument("--gamma-every", type=int, default=50)
    ap.add_argument("--solver", default="inv", choices=["inv", "lu"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--verify-benchmark", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    if args.profile == "full":
        args.full_rollout = True

    global DEVICE
    DEVICE = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True

    problems = ALL_PROBLEMS if args.problem == "all" else [args.problem]

    if args.estimate:
        estimate_cost(problems, args)
        return
    if args.verify_benchmark:
        for p in problems:
            verify_benchmark(p, args)
        return

    estimate_cost(problems, args)
    t0 = time.time()
    all_runs = {}
    for p in problems:
        runs, agg = run_problem(p, args)
        all_runs[p] = runs
    if len(all_runs) > 1:
        cross_summary(all_runs, args)
    print(f"\n  Total wall time: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
