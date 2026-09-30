import sys, os, time, json, argparse
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from numerics import build_shishkin_mesh, build_compact_laplacian

SEED    = 42
EPOCHS  = 2000
OUT_DIR = "ablation_results_systems"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  

LEGACY_RHO = False
ANCHOR_BAND = 0.12

TAU = 0.05


# ======================================================================
#  RUN DEFINITIONS
# ======================================================================

FULL_FLAGS = dict(dual_gru=True, silu=True, additive_gate=True,
                  l_fd=True, l_anch=True, l_cons=True, preinit=True)

STEPS = [
    ("Step 1\nData Loss",
     dict(dual_gru=False, silu=False, additive_gate=False,
          l_fd=False, l_anch=False, l_cons=False, preinit=False),
     "Data Loss"),
    ("Step 2\n+ $L_{fd}$\n(FD Guidance)",
     dict(dual_gru=False, silu=False, additive_gate=False,
          l_fd=True, l_anch=False, l_cons=False, preinit=True),
     "+ L_fd  (FD Guidance)"),
    ("Step 3\n+ Dual GRU",
     dict(dual_gru=True, silu=True, additive_gate=True,
          l_fd=True, l_anch=False, l_cons=False, preinit=True),
     "+ Dual GRU"),
    ("Step 4\n+ $L_{anch}$\n(Anchoring)",
     dict(dual_gru=True, silu=True, additive_gate=True,
          l_fd=True, l_anch=True, l_cons=False, preinit=True),
     "+ L_anch  (Anchoring)"),
    ("Step 5\nFull Model\n(+ $L_{cons}$)",
     dict(FULL_FLAGS),
     "Full Model (Ours)"),
]
N_STEPS = len(STEPS)
FULL_MODEL_IDX = N_STEPS - 1

# "Full - L_cons" 
LOO_STEPS = [
    ("Full $-$ $L_{anch}$", dict(FULL_FLAGS, l_anch=False), "Full - L_anch"),
]

STEP_COLORS = ["#AED6F1", "#5DADE2", "#2471A3", "#1A5276", "#C0392B"]
LOO_COLOR   = "#E67E22"


def _check_run_design():
    keys = sorted(FULL_FLAGS)
    trans = []
    for i in range(1, N_STEPS):
        a, b = STEPS[i - 1][1], STEPS[i][1]
        trans.append([k for k in keys if a.get(k) != b.get(k)])
    assert ["l_anch"] in trans, "no transition isolates L_anch"
    assert ["l_cons"] in trans, "no transition isolates L_cons"
    for _, fl, short in LOO_STEPS:
        diff = [k for k in keys if FULL_FLAGS[k] != fl[k]]
        assert len(diff) == 1, f"{short} differs from Full in {diff}"


_check_run_design()


# ======================================================================
#  SYSTEM CONFIGS
# ======================================================================

SYSTEM_CFGS = {
    "fhn_partial": {
        "name": "FitzHugh-Nagumo Partial (Neuroscience)",
        "eps_diff": 0.05, "delta_v": 0.1, "beta_v": 1.0,
        "gamma_v": 0.5, "a_fhn": 0.25,
        "Lambda": 2.0, "u_min": 0.0, "u_max": 1.0,
        "v_min": 0.0, "v_max": 0.6,
        "domain": (0.0, 1.0), "T": 0.3, "Nx": 52, "Nt": 60, "dt": 0.005,
        "bc_type": "dirichlet", "beta_mesh": 2.0, "n_bins_2d": 12,
        "anchors_uv": [(0.0, 0.0), (0.25, 0.0), (1.0, 0.0)],
        "ics": [("front", 0.2, 0.0), ("front", 0.3, 0.0),
                ("front", 0.5, 0.0), ("front", 0.6, 0.0),
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
        "bc_type": "neumann", "beta_mesh": 2.0, "n_bins_2d": 10,
        "anchors_uv": [(0.0, 0.1), (0.1, 0.18), (0.0, 0.0)],
        "ics": [(0.30, 0.12, 1, 1), (0.25, 0.10, 1, 1), (0.20, 0.08, 2, 2),
                (0.28, 0.11, 1, 2), (0.35, 0.14, 1, 1), (0.22, 0.09, 2, 1)],
        "input_dim": 5,
    },
}


# ======================================================================
#  TRUE REACTIONS
# ======================================================================

def get_R_true(system, u, v, cfg):
    if system == "fhn_partial":
        return (1.0 / cfg["eps_diff"]) * u * (u - cfg["a_fhn"]) * (1.0 - u) - v
    if system == "predator_prey":
        a = cfg["alpha_pp"]; b = cfg["beta_pp"]
        return np.clip(u, 0, None) * (1 - np.clip(u, 0, None)) - \
            a * np.clip(u, 0, None) * np.clip(v, 0, None) / \
            (b + np.clip(u, 0.001, None))


# ======================================================================
#  MESH AND IMEX
# ======================================================================

def build_laplacian(cfg):
    xl, xr = cfg["domain"]
    x, _ = build_shishkin_mesh(cfg["Nx"], xl, xr,
                               cfg["eps_diff"], cfg["beta_mesh"])
    return x, build_compact_laplacian(x, periodic=False)


def make_imex(D, L_np, cfg, neumann=True, device=None):
    dt = cfg["dt"]; N = cfg["Nx"]
    A = np.eye(N + 1) - (dt / 2) * D * L_np
    if neumann:
        A[0, :] = 0; A[0, 0] = 1; A[0, 1] = -1
        A[-1, :] = 0; A[-1, -1] = 1; A[-1, -2] = -1
    else:
        A[0, :] = 0; A[0, 0] = 1; A[-1, :] = 0; A[-1, -1] = 1
    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv, torch.tensor(A, dtype=torch.float32,
                                    device=device or DEVICE)


def layer_envelope(x_t, xl, xr, eps):
    xr_t = torch.as_tensor(xr, dtype=torch.float32, device=x_t.device)
    return torch.exp(-torch.minimum(x_t - xl, xr_t - x_t)
                     / (eps ** 0.5 + 1e-12))


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
        D1 = cfg["D1"]; D2 = cfg["D2"]; alp = cfg["alpha_pp"]
        bet = cfg["beta_pp"]; gam = cfg["gamma_pp"]; dlt = cfg["delta_pp"]
        _, lu_u, pu, _ = make_imex(D1, L_np, cfg, neumann=True)
        _, lu_v, pv, _ = make_imex(D2, L_np, cfg, neumann=True)
        for u0, v0 in make_ics(system, x, cfg):
            ur = np.zeros((Nt + 1, len(x))); vr = np.zeros((Nt + 1, len(x)))
            ur[0] = np.clip(u0, 0.001, None); vr[0] = np.clip(v0, 0.001, None)
            for n in range(Nt):
                un = np.clip(ur[n], 0.001, None)
                vn = np.clip(vr[n], 0.001, None)
                Rp = alp * un * vn / (bet + un); Rf = un * (1 - un) - Rp
                rhs_u = un + (dt / 2) * D1 * (L_np @ un) + dt * Rf
                rhs_u[0] = 0; rhs_u[-1] = 0
                ur[n + 1] = _lus(lu_u, pu, rhs_u, -0.01, 0.65)
                rhs_v = vn + (dt / 2) * D2 * (L_np @ vn) + dt * (gam * Rp - dlt * vn)
                rhs_v[0] = 0; rhs_v[-1] = 0
                vr[n + 1] = _lus(lu_v, pv, rhs_v, -0.01, 0.65)
            refs.append((ur, vr))
    return refs, t_arr


# ======================================================================
#  FD EXTRACTION
# ======================================================================

def extract_fd(refs, L_np, cfg, system):
    dt = cfg["dt"]; ua, va, Ra = [], [], []
    for ur, vr in refs:
        Nt = ur.shape[0] - 1
        for n in range(1, Nt - 1):
            un = ur[n]; vn = vr[n]; dtu = (ur[n + 1] - ur[n - 1]) / (2 * dt)
            Rfd = (dtu - cfg["eps_diff"] * (L_np @ un) if system == "fhn_partial"
                   else dtu - cfg["D1"] * (L_np @ un))
            lap = np.abs(L_np @ un)
            mask = lap < np.percentile(lap, 50)
            mask[0] = False; mask[-1] = False
            if mask.sum() > 0:
                ua.append(un[mask]); va.append(vn[mask]); Ra.append(Rfd[mask])
    return np.concatenate(ua), np.concatenate(va), np.concatenate(Ra)


def aggregate_fd(u_fd, v_fd, R_fd, cfg):
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
    print(f"    FD bins: {len(u_t)}")
    return (np.array(u_t, dtype=np.float32), np.array(v_t, dtype=np.float32),
            np.array(R_t, dtype=np.float32))


# ======================================================================
#  ABLATION MODEL
# ======================================================================

class AblModel(nn.Module):
    """Configurable model.

    dual_gru:       True -> dual GRU regime encoder, False -> single GRU
    silu:           True -> SiLU activation, False -> ReLU
    additive_gate:  True -> additive gate, False -> multiplicative FiLM
    """

    def __init__(self, Lambda, flags, hidden=32, mlp_w=64, input_dim=5):
        super().__init__()
        self.Lambda = Lambda; self.flags = flags
        self.hidden = hidden; self.mlp_w = mlp_w
        self._act = F.silu if flags["silu"] else F.relu

        if flags["dual_gru"]:
            self.gru_out = nn.GRUCell(input_dim, hidden)
            self.gru_lay = nn.GRUCell(input_dim, hidden)
        else:
            self.gru = nn.GRUCell(input_dim, hidden)

        if flags["additive_gate"]:
            self.gate = nn.Linear(hidden, mlp_w)
        else:
            self.film = nn.Linear(hidden, 2 * mlp_w)

        self.mlp1 = nn.Linear(2, mlp_w)
        self.mlp2 = nn.Linear(mlp_w, mlp_w)
        self.mlp3 = nn.Linear(mlp_w, 1)
        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)
        nn.init.normal_(self.mlp1.weight, std=0.10)
        nn.init.zeros_(self.mlp1.bias)
        if flags["additive_gate"]:
            nn.init.normal_(self.gate.weight, std=0.01)
            nn.init.zeros_(self.gate.bias)
        else:
            nn.init.normal_(self.film.weight, std=0.01)
            nn.init.zeros_(self.film.bias)

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden, device=device)
        return h, h.clone()

    def _decode(self, u, v, context):
        inp = torch.stack([u, v], dim=-1); act = self._act
        if self.flags["additive_gate"]:
            gamma = self.gate(context)
            h1 = act(self.mlp1(inp)); h2 = act(h1 + gamma)
        else:
            f = self.film(context)
            a = f[:, :self.mlp_w]; b = f[:, self.mlp_w:]
            h1 = act(self.mlp1(inp)); h2 = act((1 + a) * h1 + b)
        h3 = act(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    def forward(self, u, v, z_out, z_lay, rho, H_out, H_lay):
        rho_e = rho.unsqueeze(-1)
        if self.flags["dual_gru"]:
            Hoc = self.gru_out(torch.clamp(z_out, -10, 10), H_out)
            Hlc = self.gru_lay(torch.clamp(z_lay, -10, 10), H_lay)
            Hon = (1 - rho_e) * Hoc + rho_e * H_out
            Hln = rho_e * Hlc + (1 - rho_e) * H_lay
            Hb = (1 - rho_e) * Hon + rho_e * Hln
            H_out_n = Hon; H_lay_n = Hln
        else:
            Hb = self.gru(torch.clamp(z_out, -10, 10), H_out)
            H_out_n = Hb; H_lay_n = Hb
        return self._decode(u, v, Hb), H_out_n, H_lay_n

    def react_grad(self, u, v):
        """Identified law: the decoder evaluated at neutral context."""
        dev = u.device
        context = torch.zeros(u.shape[0], self.hidden, device=dev)
        if self.flags["additive_gate"]:
            gamma = torch.zeros(u.shape[0], self.mlp_w, device=dev)
        else:
            f = self.film(context)
            a = f[:, :self.mlp_w]; b = f[:, self.mlp_w:]
        act = self._act
        inp = torch.stack([u, v], dim=-1)
        h1 = act(self.mlp1(inp))
        h2 = act(h1 + gamma) if self.flags["additive_gate"] \
            else act((1 + a) * h1 + b)
        h3 = act(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    @torch.no_grad()
    def react_nograd(self, u, v):
        return self.react_grad(u, v)


# ======================================================================
#  REGIME GATE AND IMEX STEP
# ======================================================================

def regime_indicator(dhu, eps):
    """Continuous regime indicator. See LEGACY_RHO in the docstring."""
    l = torch.log(torch.abs(dhu) + 1e-12)
    if LEGACY_RHO:
        mu = torch.median(l)
        denom = float(np.log(1 / (eps + 1e-12))) + 1e-8
        return torch.sigmoid(3.0 * (l - mu) / denom)
    mu = torch.median(l[1:-1])
    denom = max(float(np.log(1 / (eps + 1e-12))), 1.0)
    rho = torch.sigmoid(3.0 * (l - mu) / denom).clone()
    rho[0] = rho[1]; rho[-1] = rho[-2]
    return rho


def _neu(r):
    r = r.clone(); r[0] = 0.0; r[-1] = 0.0; return r


def _dir(r, gL, gR):
    r = r.clone(); r[0] = gL; r[-1] = gR; return r


def imex_step_abl(system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
                  x_t, phi_t, A_u_t, L_t, cfg):
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
    u_next = torch.linalg.solve(A_u_t, rhs.unsqueeze(-1)).squeeze(-1)
    return (torch.clamp(u_next, cfg["u_min"] - 0.05, cfg["u_max"] + 0.05),
            R, Hon, Hln)


# ======================================================================
#  EVALUATION, INCLUDING THE ANCHOR SPLIT
# ======================================================================

def anchor_split_error(Rp_flat, Rt_flat, u_flat, v_flat, cfg,
                       band=ANCHOR_BAND):
    """Split the reaction error by distance to the equilibrium anchors.

    Distance is measured in coordinates normalised by the (u,v) box, so
    the two variables contribute comparably regardless of their ranges.
    BOTH pieces are normalised by the SAME global ||R_true||: R_true
    vanishes at the anchors by construction, so a locally normalised
    error there would divide by almost nothing and mean nothing.
    """
    anchors = cfg.get("anchors_uv") or []
    den = float(np.sqrt(np.mean(Rt_flat ** 2))) + 1e-8
    if not anchors:
        return float("nan"), float("nan"), 0.0
    du_s = cfg["u_max"] - cfg["u_min"]; dv_s = cfg["v_max"] - cfg["v_min"]
    near = np.zeros_like(u_flat, dtype=bool)
    for (ua, va) in anchors:
        d = np.sqrt(((u_flat - ua) / du_s) ** 2 + ((v_flat - va) / dv_s) ** 2)
        near |= d <= band
    far = ~near

    def rms(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((Rp_flat[mask] - Rt_flat[mask]) ** 2))
                     / den)

    return rms(near), rms(far), float(near.mean())


def evaluate(model, system, cfg, u_fd=None, v_fd=None):
    ng = 60
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg); uf = UU.ravel(); vf = VV.ravel()
    Rp = model.react_nograd(
        torch.tensor(uf, dtype=torch.float32, device=DEVICE),
        torch.tensor(vf, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    Rt = get_R_true(system, uf, vf, cfg)
    l2 = float(np.sqrt(np.mean((Rp - Rt) ** 2)) /
               (np.sqrt(np.mean(Rt ** 2)) + 1e-8))
    e_near, e_far, frac = anchor_split_error(Rp, Rt, uf, vf, cfg)

    l2_in = l2
    if u_fd is not None and len(u_fd) > 0:
        Rp2 = model.react_nograd(
            torch.tensor(u_fd.astype(np.float32), device=DEVICE),
            torch.tensor(v_fd.astype(np.float32), device=DEVICE)).cpu().numpy()
        Rt2 = get_R_true(system, u_fd, v_fd, cfg)
        l2_in = float(np.sqrt(np.mean((Rp2 - Rt2) ** 2)) /
                      (np.sqrt(np.mean(Rt2 ** 2)) + 1e-8))
    return dict(l2=l2, l2_in=l2_in, eps_near=e_near, eps_far=e_far,
                band_frac=frac, UU=UU, VV=VV,
                Rp=Rp.reshape(ng, ng), Rt=Rt.reshape(ng, ng))


# ======================================================================
#  TRAINING
# ======================================================================

def train_step(system, model, flags, refs, cfg, x_t, phi_t, A_u_t, L_t,
               u_fd_t, v_fd_t, R_fd_t, u_fd, v_fd, epochs):
    su = max(float(np.mean([np.abs(r[0]).mean() for r in refs])), 0.01)
    Lambda = cfg["Lambda"]
    dev = DEVICE

    # Stage 0 -- pre-init the decoder on the FD bins.
    if flags["preinit"] and flags["l_fd"] and u_fd_t is not None:
        for n, p in model.named_parameters():
            freeze = ("gru" in n) \
                or ("gate" in n and flags["additive_gate"]) \
                or ("film" in n and not flags["additive_gate"])
            if freeze:
                p.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            pi_opt = optim.Adam(params, lr=1e-3); best_s0 = float("inf")
            for _ in range(1000):
                pi_opt.zero_grad()
                loss = (model.react_grad(u_fd_t, v_fd_t) - R_fd_t) \
                    .pow(2).mean() / (Lambda ** 2 + 1e-8)
                loss.backward(); pi_opt.step()
                best_s0 = min(best_s0, loss.item())
            print(f"    Stage 0 done. FD loss={best_s0:.5f}")
        for p in model.parameters():
            p.requires_grad_(True)

    anchors = cfg.get("anchors_uv") or []
    if flags["l_anch"] and anchors:
        pu = torch.tensor([p[0] for p in anchors],
                          dtype=torch.float32, device=dev)
        pv = torch.tensor([p[1] for p in anchors],
                          dtype=torch.float32, device=dev)
    else:
        pu = pv = None

    ref_t = [(torch.tensor(ur, dtype=torch.float32, device=dev),
              torch.tensor(vr, dtype=torch.float32, device=dev))
             for ur, vr in refs]

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=400, eta_min=1e-6)
    history = []; best_l = float("inf"); best_state = None; nan_c = 0
    n_refs = len(ref_t); Nt = cfg["Nt"]

    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        f = min((epoch - 1) / (epochs - 1 + 1e-8), 1.0)
        lam_d = 1.0
        lam_f = (3.0 + 2.0 * f) if flags["l_fd"] else 0.0
        lam_a = (2.0 + 3.0 * f) if flags["l_anch"] else 0.0
        lam_c = (0.01 + 0.49 * f) if flags["l_cons"] else 0.0
        ep = 0.0

        if lam_f > 0 and u_fd_t is not None:
            Lf = (model.react_grad(u_fd_t, v_fd_t) - R_fd_t) \
                .pow(2).mean() / (Lambda ** 2 + 1e-8)
        else:
            Lf = None
        La = model.react_grad(pu, pv).pow(2).mean() \
            if (lam_a > 0 and pu is not None) else None
        aux = 0.0
        if Lf is not None:
            aux = aux + lam_f * Lf
        if La is not None:
            aux = aux + lam_a * La
        if torch.is_tensor(aux):
            aux.backward()
            ep += float(aux.detach())

        for ur_t, vr_t in ref_t:
            H_out, H_lay = model.init_hidden(x_t.shape[0], dev)
            u_n = ur_t[0]; u_prev = ur_t[0]
            ua_buf = []; va_buf = []
            for step in range(Nt):
                H_out = H_out.detach(); H_lay = H_lay.detach()
                u_n = u_n.detach(); u_prev = u_prev.detach()
                v_ref_n = vr_t[step].detach()
                u_next, R, H_out, H_lay = imex_step_abl(
                    system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
                    x_t, phi_t, A_u_t, L_t, cfg)
                Ld = ((u_next - ur_t[step + 1]) / (su + 1e-8)).pow(2).mean()
                ua_buf.append(u_n.detach()); va_buf.append(v_ref_n)
                Lw = lam_d * Ld
                (Lw / Nt / n_refs).backward()
                ep += float(Lw.detach()) / Nt / n_refs
                u_prev = u_n; u_n = u_next

            if flags["l_cons"] and len(ua_buf) > 2:
                ua_c = torch.cat(ua_buf); va_c = torch.cat(va_buf)
                if len(ua_c) > 512:
                    perm = torch.randperm(len(ua_c), device=dev)[:512]
                    ua_c = ua_c[perm]; va_c = va_c[perm]
                Rs_fresh = model.react_grad(ua_c, va_c)
                sort_idx = torch.argsort(ua_c)
                us2 = ua_c[sort_idx]; Rs2 = Rs_fresh[sort_idx]
                du2 = torch.abs(us2[1:] - us2[:-1]); mask2 = du2 < TAU
                if mask2.sum() >= 2:
                    valid2 = torch.where(mask2)[0]
                    if len(valid2) > 64:
                        valid2 = valid2[
                            torch.randperm(len(valid2), device=dev)[:64]]
                    w2 = (1 - du2[valid2] / TAU).clamp(0, 1).pow(2)
                    Lc = (w2 * (Rs2[valid2] - Rs2[valid2 + 1]).pow(2)).mean()
                    (lam_c * Lc / n_refs).backward()
                    ep += lam_c * float(Lc.detach()) / n_refs

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
        history.append(ep)
        if ep < best_l:
            best_l = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0 or epoch <= 3 or epoch == epochs:
            m = evaluate(model, system, cfg, u_fd, v_fd)
            print(f"    ep {epoch:>5}  loss={ep:.5f}  "
                  f"L2={m['l2']:.4f}  L2_in={m['l2_in']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return history


# ======================================================================
#  ROLLOUT FOR PLOTS
# ======================================================================

def compute_v_pred_simple(system, u_pred, vr_ref, cfg, L_np):
    dt = cfg["dt"]; Nt = u_pred.shape[0] - 1
    v_pred = np.zeros_like(u_pred); v_pred[0] = vr_ref[0]
    import scipy.linalg as sla
    if not np.isfinite(u_pred).all():
        return vr_ref.copy()
    if system == "fhn_partial":
        dv = cfg["delta_v"]; bv = cfg["beta_v"]; gv = cfg["gamma_v"]
        A = np.eye(len(vr_ref[0])) - (dt / 2) * dv * L_np
        A[0, :] = 0; A[0, 0] = 1; A[-1, :] = 0; A[-1, -1] = 1
        lu, piv = sla.lu_factor(A)
        for n in range(Nt):
            vn = v_pred[n]; un = u_pred[n + 1]
            rhs = vn + (dt / 2) * dv * (L_np @ vn) + dt * (bv * un - gv * vn)
            rhs[0] = 0; rhs[-1] = 0
            v_pred[n + 1] = np.clip(sla.lu_solve((lu, piv), rhs), -0.05, 1.05)
    elif system == "predator_prey":
        D2 = cfg["D2"]; alp = cfg["alpha_pp"]; bet = cfg["beta_pp"]
        gam = cfg["gamma_pp"]; dlt = cfg["delta_pp"]
        A = np.eye(len(vr_ref[0])) - (dt / 2) * D2 * L_np
        A[0, :] = 0; A[0, 0] = 1; A[0, 1] = -1
        A[-1, :] = 0; A[-1, -1] = 1; A[-1, -2] = -1
        lu, piv = sla.lu_factor(A)
        for n in range(Nt):
            vn = v_pred[n]; un = u_pred[n + 1]
            R_est = alp * np.clip(un, 0.001, None) * vn / \
                (bet + np.clip(un, 0.001, None))
            rhs = vn + (dt / 2) * D2 * (L_np @ vn) + dt * (gam * R_est - dlt * vn)
            rhs[0] = 0; rhs[-1] = 0
            v_pred[n + 1] = np.clip(sla.lu_solve((lu, piv), rhs), -0.01, 0.65)
    return v_pred


def rollout_for_plot(system, model, refs, cfg, x_t, phi_t, A_u_t, L_t, L_np):
    ur_ref, vr_ref = refs[0]
    ur_t = torch.tensor(ur_ref, dtype=torch.float32, device=DEVICE)
    vr_t = torch.tensor(vr_ref, dtype=torch.float32, device=DEVICE)
    H_out, H_lay = model.init_hidden(x_t.shape[0], DEVICE)
    u_n = ur_t[0]; u_prev = ur_t[0]
    u_traj = [u_n.detach().cpu().numpy()]
    for step in range(cfg["Nt"]):
        v_ref_n = vr_t[step].detach()
        with torch.no_grad():
            u_next, _, H_out, H_lay = imex_step_abl(
                system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, A_u_t, L_t, cfg)
        if not torch.isfinite(u_next).all():
            return ur_ref, vr_ref, ur_ref.copy(), vr_ref.copy()
        u_traj.append(u_next.detach().cpu().numpy())
        u_prev = u_n; u_n = u_next
    u_pred = np.array(u_traj)
    return ur_ref, vr_ref, u_pred, \
        compute_v_pred_simple(system, u_pred, vr_ref, cfg, L_np)


# ======================================================================
#  RUN ONE CONFIGURATION
# ======================================================================

def run_one(system, label, flags, cfg, refs, L_np, x_t, phi_t, A_u_t, L_t,
            u_fd_t, v_fd_t, R_fd_t, u_fd, v_fd, epochs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = AblModel(cfg["Lambda"], flags, hidden=32, mlp_w=64,
                     input_dim=cfg["input_dim"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}   ({DEVICE})")
    t0 = time.time()
    history = train_step(system, model, flags, refs, cfg, x_t, phi_t,
                         A_u_t, L_t, u_fd_t, v_fd_t, R_fd_t,
                         u_fd, v_fd, epochs)
    t_train = time.time() - t0
    m = evaluate(model, system, cfg, u_fd, v_fd)
    print(f"  -> L2={m['l2']:.4f}  L2_in={m['l2_in']:.4f}  "
          f"near={m['eps_near']:.4f}  far={m['eps_far']:.4f}  "
          f"({t_train:.0f}s)")
    ur, vr, up, vp = rollout_for_plot(system, model, refs, cfg,
                                      x_t, phi_t, A_u_t, L_t, L_np)
    out = dict(m)
    out.update(history=history, short=label, train_s=t_train,
               n_params=n_params, u_pred=up, v_pred=vp,
               u_ref=ur, v_ref=vr)
    return out


# ======================================================================
#  PLOTS
# ======================================================================

def plot_reaction(steps, system, cfg, out_dir):
    fig, axes = plt.subplots(2, N_STEPS, figsize=(4.5 * N_STEPS, 9))
    fig.suptitle(f"{cfg['name']} - ablation: "
                 f"R_theta(u,v) and |error| at each step",
                 fontsize=12, fontweight="bold")
    vmax = max(np.abs(r["Rt"]).max() for r in steps)
    anchors = cfg.get("anchors_uv") or []

    for col, r in enumerate(steps):
        is_full = (col == FULL_MODEL_IDX)
        title = STEPS[col][0].replace("$L_{fd}$", "L_fd") \
            .replace("$L_{anch}$", "L_anch").replace("$L_{cons}$", "L_cons")
        im = axes[0, col].contourf(r["UU"], r["VV"], r["Rp"], levels=30,
                                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=axes[0, col])
        axes[0, col].set_title(
            f"{title}\nL2={r['l2']:.4f} | L2_in={r['l2_in']:.4f}",
            fontsize=8, fontweight="bold" if is_full else "normal")
        err = np.abs(r["Rp"] - r["Rt"])
        im2 = axes[1, col].contourf(r["UU"], r["VV"], err, levels=30,
                                    cmap="YlOrRd")
        plt.colorbar(im2, ax=axes[1, col])
        axes[1, col].set_title(f"|error|  max={err.max():.3f}", fontsize=8)
        for row in (0, 1):
            for (ua, va) in anchors:
                axes[row, col].plot([ua], [va], "o", ms=7, mfc="none",
                                    mec="#117A65", mew=1.8)
            axes[row, col].set_xlabel("u")
            axes[row, col].set_ylabel("v" if col == 0 else "")
            if is_full:
                for sp in axes[row, col].spines.values():
                    sp.set_edgecolor("#C0392B"); sp.set_linewidth(2.5)

    fig.text(0.5, -0.02,
             "Adding one component at a time  ->  Full Model"
             "     (circles mark the equilibrium anchors)",
             ha="center", fontsize=11, style="italic", color="#444")
    plt.tight_layout()
    p = os.path.join(out_dir, f"ablation_reaction_{system}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_full_model_errors(steps, system, cfg, t_arr, x_np, out_dir):
    from scipy.stats import gaussian_kde
    r = steps[FULL_MODEL_IDX]
    u_pred = r["u_pred"]; u_ref = r["u_ref"]
    v_pred = r["v_pred"]; v_ref = r["v_ref"]
    Nt = cfg["Nt"]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)
    fig.suptitle(f"{cfg['name']} - Full Model (BRIDGE): error analysis",
                 fontsize=13, fontweight="bold")

    ax_v = fig.add_subplot(gs[0, 0])
    quart = [range(0, Nt // 4), range(Nt // 4, Nt // 2),
             range(Nt // 2, 3 * Nt // 4), range(3 * Nt // 4, Nt + 1)]
    q_labels = ["t in [0,T/4]", "t in [T/4,T/2]",
                "t in [T/2,3T/4]", "t in [3T/4,T]"]
    vdata = [np.abs(u_pred[list(q)] - u_ref[list(q)]).ravel() for q in quart]
    vdata = [d[d > 1e-14] if (d > 1e-14).any() else np.array([1e-14])
             for d in vdata]
    vp = ax_v.violinplot(vdata, positions=range(1, 5), showmedians=True)
    for pc in vp["bodies"]:
        pc.set_alpha(0.7); pc.set_facecolor("#3498DB")
    vp["cmedians"].set_color("red"); vp["cmedians"].set_lw(2)
    ax_v.set_yscale("log"); ax_v.set_xticks(range(1, 5))
    ax_v.set_xticklabels(q_labels, fontsize=7)
    ax_v.set_ylabel("Absolute error", fontsize=9)
    ax_v.set_title("Error distribution by time quartile", fontsize=9)
    ax_v.grid(True, alpha=0.3, axis="y")

    ax_t = fig.add_subplot(gs[0, 1])
    err_u = np.abs(u_pred - u_ref)
    mean_err = err_u.mean(axis=1); std_err = err_u.std(axis=1)
    ax_t.plot(t_arr, mean_err, "b-", lw=2, label="Mean |error|")
    ax_t.fill_between(t_arr, np.maximum(mean_err - 1.96 * std_err, 0),
                      mean_err + 1.96 * std_err, alpha=0.25, color="C0",
                      label="95% CI")
    ax_t.set_xlabel("t", fontsize=9)
    ax_t.set_ylabel("Mean absolute error", fontsize=9)
    ax_t.set_title("Temporal error evolution", fontsize=9)
    ax_t.legend(fontsize=8); ax_t.grid(True, alpha=0.3)

    ax_d = fig.add_subplot(gs[0, 2])
    l2_t = np.sqrt(np.mean((u_pred - u_ref) ** 2, axis=1)) / \
        (np.sqrt(np.mean(u_ref ** 2, axis=1)) + 1e-8)
    pos = l2_t[l2_t > 1e-8]
    if len(pos) > 5:
        k = gaussian_kde(np.log10(pos + 1e-10))
        xs = np.linspace(np.log10(pos.min()) - 0.5,
                         np.log10(pos.max()) + 0.5, 200)
        ax_d.fill_between(10 ** xs, k(xs), alpha=0.6, color="#3498DB",
                          label="Rel. L2")
    ax_d.axvline(l2_t.mean(), color="navy", lw=2, ls="--",
                 label=f"Mean={l2_t.mean():.4f}")
    ax_d.set_xscale("log"); ax_d.set_xlabel("Relative L2 error", fontsize=9)
    ax_d.set_ylabel("Density", fontsize=9)
    ax_d.set_title(f"L2 error distribution\nMean L2={l2_t.mean():.4f}",
                   fontsize=9)
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)

    ax_vd = fig.add_subplot(gs[0, 3])
    l2v = np.sqrt(np.mean((v_pred - v_ref) ** 2, axis=1)) / \
        (np.sqrt(np.mean(v_ref ** 2, axis=1)) + 1e-8)
    vpos = l2v[l2v > 1e-8]
    if len(vpos) > 5:
        kv = gaussian_kde(np.log10(vpos + 1e-10))
        xs2 = np.linspace(np.log10(vpos.min()) - 0.5,
                          np.log10(vpos.max()) + 0.5, 200)
        ax_vd.fill_between(10 ** xs2, kv(xs2), alpha=0.6, color="#E07B54",
                           label="Rel. L2 (v)")
    ax_vd.axvline(l2v.mean(), color="darkred", lw=2, ls="--",
                  label=f"Mean={l2v.mean():.4f}")
    ax_vd.set_xscale("log")
    ax_vd.set_xlabel("Relative L2 error (v)", fontsize=9)
    ax_vd.set_ylabel("Density", fontsize=9)
    ax_vd.set_title(f"v error distribution\nMean L2={l2v.mean():.4f}",
                    fontsize=9)
    ax_vd.legend(fontsize=8); ax_vd.grid(True, alpha=0.3)

    ax_traj = fig.add_subplot(gs[1, :2])
    for n, clr in ((0, "C0"), (Nt // 2, "C1"), (Nt, "C2")):
        lbl = f"t={t_arr[n]:.2f}"
        ax_traj.plot(x_np, u_ref[n], color=clr, lw=2.0, ls="-",
                     label=f"u_ref {lbl}")
        ax_traj.plot(x_np, u_pred[n], color=clr, lw=1.5, ls="--",
                     label=f"u_pred {lbl}")
    ax_traj.set_xlabel("x", fontsize=10); ax_traj.set_ylabel("u", fontsize=10)
    ax_traj.set_title("u_ref (solid) vs u_pred (dashed)", fontsize=10)
    ax_traj.legend(fontsize=8, ncol=2, loc="best")
    ax_traj.grid(True, alpha=0.3)

    ax_vt = fig.add_subplot(gs[1, 2:])
    for n, clr in ((0, "C0"), (Nt // 2, "C1"), (Nt, "C2")):
        ax_vt.plot(x_np, v_ref[n], color=clr, lw=2.0, ls="-")
        ax_vt.plot(x_np, v_pred[n], color=clr, lw=1.5, ls="--")
    ax_vt.set_xlabel("x", fontsize=10); ax_vt.set_ylabel("v", fontsize=10)
    ax_vt.set_title("v_ref (solid) vs v_pred (dashed)", fontsize=10)
    ax_vt.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax_vt.legend(handles=[Line2D([0], [0], color="gray", lw=2, label="ref"),
                          Line2D([0], [0], color="gray", lw=1.5, ls="--",
                                 label="pred")], fontsize=9)

    p = os.path.join(out_dir, f"full_model_error_analysis_{system}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ======================================================================
#  RUN ONE SYSTEM
# ======================================================================

def run_system(system, epochs, run_loo=True):
    cfg = SYSTEM_CFGS[system]
    print(f"\n{'='*65}\n  System: {cfg['name']}\n  Epochs: {epochs}\n{'='*65}")
    x, L_np = build_laplacian(cfg)
    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=DEVICE)
    xl, xr = cfg["domain"]
    phi_t = layer_envelope(x_t, xl, xr, cfg["eps_diff"])
    D_u = cfg["eps_diff"] if system == "fhn_partial" else cfg["D1"]
    _, _, _, A_u_t = make_imex(D_u, L_np, cfg,
                               neumann=(cfg["bc_type"] == "neumann"),
                               device=DEVICE)

    print("Generating references...")
    refs, t_arr = generate_reference(system, x, cfg, L_np)
    print("Extracting FD pairs...")
    u_fd, v_fd, R_fd = extract_fd(refs, L_np, cfg, system)
    u_tt, v_tt, R_tt = aggregate_fd(u_fd, v_fd, R_fd, cfg)
    u_fd_t = torch.tensor(u_tt, dtype=torch.float32, device=DEVICE)
    v_fd_t = torch.tensor(v_tt, dtype=torch.float32, device=DEVICE)
    R_fd_t = torch.tensor(R_tt, dtype=torch.float32, device=DEVICE)

    common = dict(cfg=cfg, refs=refs, L_np=L_np, x_t=x_t, phi_t=phi_t,
                  A_u_t=A_u_t, L_t=L_t, u_fd_t=u_fd_t, v_fd_t=v_fd_t,
                  R_fd_t=R_fd_t, u_fd=u_fd, v_fd=v_fd, epochs=epochs)

    steps = []
    for i, (_, flags, short) in enumerate(STEPS):
        print(f"\n  [Step {i+1}/{N_STEPS}] {short}")
        steps.append(run_one(system, short, flags, **common))

    loos = []
    if run_loo:
        for k, (_, flags, short) in enumerate(LOO_STEPS):
            print(f"\n  [Leave-one-out {k+1}/{len(LOO_STEPS)}] {short}")
            r = run_one(system, short, flags, **common)
            fl = steps[FULL_MODEL_IDX]["l2"]
            print(f"     {(r['l2']-fl)/(fl+1e-12)*100:+.1f}% vs full model")
            loos.append(r)

    print("\n  Monotone check across the additive sequence:")
    for i in range(1, N_STEPS):
        prev = steps[i - 1]["l2"]; curr = steps[i]["l2"]
        ok = "ok" if curr <= prev * 1.05 else "DEGRADED"
        print(f"    Step {i}->{i+1}: {prev:.4f} -> {curr:.4f}  "
              f"({(prev-curr)/prev*100:+.1f}%)  {ok}")

    t_arr_np = np.linspace(0, cfg["T"], cfg["Nt"] + 1)
    plot_reaction(steps, system, cfg, OUT_DIR)
    plot_full_model_errors(steps, system, cfg, t_arr_np, x, OUT_DIR)

    def _pack(r):
        return {k: (float(v) if isinstance(v, (int, float, np.floating))
                    else v)
                for k, v in r.items()
                if k in ("l2", "l2_in", "eps_near", "eps_far", "band_frac",
                         "short", "train_s", "n_params")}

    with open(os.path.join(OUT_DIR, f"ablation_{system}.json"), "w") as f:
        json.dump({"legacy_rho": LEGACY_RHO, "anchor_band": ANCHOR_BAND,
                   "steps": [_pack(r) for r in steps],
                   "loos": [_pack(r) for r in loos]}, f, indent=2)
    return {"steps": steps, "loos": loos}


# ======================================================================
#  MAIN
# ======================================================================

def main():
    global DEVICE
    ap = argparse.ArgumentParser(
        description="Component ablation for BRIDGE (coupled systems)")
    ap.add_argument("--system", default="all",
                    choices=["fhn_partial", "predator_prey", "all"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-loo", action="store_true",
                    help="skip the leave-one-out control (not recommended: "
                         "it is what referee 3 asked for)")
    args = ap.parse_args()

    DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"\nablation_systems.py  |  epochs={args.epochs}  |  {DEVICE}")
    print(f"Regime gate: {'LEGACY' if LEGACY_RHO else 'corrected'} "
          "(interior median, ln(1/eps) floored at 1)")

    systems = (["fhn_partial", "predator_prey"] if args.system == "all"
               else [args.system])
    n_runs = N_STEPS + (0 if args.no_loo else len(LOO_STEPS))
    for s in systems:
        cfg = SYSTEM_CFGS[s]
        t_est = args.epochs * cfg["Nt"] * len(cfg["ics"]) * 12.5 / 1000 / 3600
        print(f"  {cfg['name']}: ~{t_est:.1f}h per run x {n_runs} runs "
              f"= ~{t_est*n_runs:.1f}h  (CPU estimate)")

    all_results = {}
    for s in systems:
        all_results[s] = run_system(s, args.epochs, run_loo=not args.no_loo)

    print(f"\n{'='*65}\n  ANCHOR CONTRIBUTION\n{'='*65}")
    for s in systems:
        steps = all_results[s]["steps"]; loos = all_results[s]["loos"]
        full = steps[FULL_MODEL_IDX]
        print(f"\n  {SYSTEM_CFGS[s]['name']}")
        print(f"    additive  (Step 2 -> Step 3, L_anch added): "
              f"{steps[1]['l2']:.4f} -> {steps[2]['l2']:.4f}  "
              f"({(steps[2]['l2']-steps[1]['l2'])/(steps[1]['l2']+1e-12)*100:+.1f}%)")
        for k, d in enumerate(loos):
            print(f"    leave-one-out ({LOO_STEPS[k][2]}): "
                  f"{full['l2']:.4f} -> {d['l2']:.4f}  "
                  f"({(d['l2']-full['l2'])/(full['l2']+1e-12)*100:+.1f}%)")
            print(f"      near anchors: {full['eps_near']:.4f} -> "
                  f"{d['eps_near']:.4f}")
            print(f"      away:         {full['eps_far']:.4f} -> "
                  f"{d['eps_far']:.4f}")


if __name__ == "__main__":
    main()
