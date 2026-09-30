import sys, os, time, json, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from numerics  import build_all_operators
from data      import generate_reference, extract_fd_pairs, aggregate_bins
from equations import get_R_true, get_bc_values

DEVICE  = torch.device("cpu")       
SEED    = 42
EPOCHS  = 2000
LR      = 2e-3
OUT_DIR = "ablation_results"
os.makedirs(OUT_DIR, exist_ok=True)


LEGACY_RHO = False
ANCHOR_BAND = 0.10


# ══════════════════════════════════════════════════════════
#  Problem configs
# ══════════════════════════════════════════════════════════

PROBLEMS = {
    "fisher": {
        "name": "Fisher-KPP", "eps_diff": 1.0,
        "domain": (0.0, 1.0), "T": 0.3, "Nx": 128,
        "Nt": 60, "dt": 0.005, "bc_type": "dirichlet",
        "Lambda": 2.0, "u_min": 0.0, "u_max": 1.0,
        "mesh_type": "shishkin", "beta": 2.0,
        "positivity": True, "fd_filter": "slow", "fd_pct": 50,
        "ics": ["original", "shifted_05", "shifted_08", "shifted_095"],
        "n_rollout_ics": 3, "anchors": [(0.0, 0.0), (1.0, 0.0)],
        "tbptt": 15, "lambda_fd": 3.0,
    },
    "allen_cahn": {
        "name": "Allen-Cahn", "eps_diff": 1e-4,
        "domain": (-1.0, 1.0), "T": 1.0, "Nx": 128,
        "Nt": 1000, "dt": 0.001, "bc_type": "periodic",
        "Lambda": 3.0, "u_min": -1.0, "u_max": 1.0,
        "mesh_type": "uniform", "beta": 2.0,
        "positivity": False, "fd_filter": "lap", "fd_pct": 60,
        "ics": ["x2cosx", "tanh0", "tanh_neg05"],
        "n_rollout_ics": 1, "anchors": [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        "tbptt": 50, "lambda_fd": 3.0,
    },
}


# ══════════════════════════════════════════════════════════
#  Run definitions
# ══════════════════════════════════════════════════════════

FULL_FLAGS = dict(dual_gru=True, film=True,
                  l_cons=True, l_anch=True, l_fd=True, preinit=True)

STEPS = [
    ("Step 1\nTrajectory Only",
     dict(dual_gru=False, film=False,
          l_cons=False, l_anch=False, l_fd=False, preinit=False),
     "Trajectory Only"),
    ("Step 2\n+ $\\mathcal{L}_{\\mathrm{fd}}$\n(FD Guidance)",
     dict(dual_gru=False, film=False,
          l_cons=False, l_anch=False, l_fd=True, preinit=True),
     "+ L_fd  (FD Guidance)"),
    ("Step 3\n+ Dual GRU\n(Regime Encoder)",
     dict(dual_gru=True, film=True,
          l_cons=False, l_anch=False, l_fd=True, preinit=True),
     "+ Dual GRU  (Regime Encoder)"),
    ("Step 4\n+ $\\mathcal{L}_{\\mathrm{anch}}$\n(Equilibrium Anchors)",
     dict(dual_gru=True, film=True,
          l_cons=False, l_anch=True, l_fd=True, preinit=True),
     "+ L_anch  (Anchors)"),
    ("Step 5\nFull Model (Ours)\n+ $\\mathcal{L}_{\\mathrm{cons}}$",
     dict(FULL_FLAGS),
     "Full Model (Ours)"),
]
N_STEPS = len(STEPS)
FULL_MODEL_IDX = N_STEPS - 1

# Leave-one-out controls. "Full - L_cons" is deliberately absent: it is
# already Step 4 of the additive sequence, so running it again would
# only burn compute.
LOO_STEPS = [
    ("Full $-$ $\\mathcal{L}_{\\mathrm{anch}}$",
     dict(FULL_FLAGS, l_anch=False),
     "Full - L_anch"),
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


# ══════════════════════════════════════════════════════════
#  Model  
# ══════════════════════════════════════════════════════════

class AblModel(nn.Module):
    def __init__(self, Lambda, hidden=32, mlp_w=32,
                 dual_gru=True, film=True, input_dim=4):
        super().__init__()
        self.Lambda = Lambda; self.dual_gru = dual_gru
        self.use_film = film; self.hidden = hidden; self.mlp_w = mlp_w

        if dual_gru:
            self.gru_out = nn.GRUCell(input_dim, hidden)
            self.gru_lay = nn.GRUCell(input_dim, hidden)
        else:
            self.gru_single = nn.GRUCell(input_dim, hidden)

        if film:
            self.film_proj = nn.Linear(hidden, 2 * mlp_w)
            self.mlp1 = nn.Linear(1, mlp_w)
            self.mlp2 = nn.Linear(mlp_w, mlp_w)
            self.mlp3 = nn.Linear(mlp_w, 1)
            nn.init.normal_(self.mlp3.weight, std=0.01)
            nn.init.zeros_(self.mlp3.bias)
        else:
            self.direct = nn.Sequential(
                nn.Linear(1 + hidden, mlp_w), nn.ReLU(),
                nn.Linear(mlp_w, mlp_w),      nn.ReLU(),
                nn.Linear(mlp_w, 1))
            nn.init.normal_(self.direct[-1].weight, std=0.01)
            nn.init.zeros_(self.direct[-1].bias)

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden, device=device)
        return h, h.clone()

    def forward(self, u, z_out, z_lay, rho, H_out, H_lay):
        rho_e = rho.unsqueeze(-1)
        if self.dual_gru:
            Ho = self.gru_out(z_out, H_out)
            Hl = self.gru_lay(z_lay, H_lay)
            H_out_n = (1 - rho_e) * Ho + rho_e * H_out
            H_lay_n =      rho_e  * Hl + (1 - rho_e) * H_lay
            H_blend = (1 - rho_e) * H_out_n + rho_e * H_lay_n
        else:
            H_out_n = self.gru_single(z_out, H_out)
            H_lay_n = H_out_n; H_blend = H_out_n

        if self.use_film:
            f = self.film_proj(H_blend)
            a = f[:, :self.mlp_w]; b = f[:, self.mlp_w:]
            h1 = F.relu(self.mlp1(u.unsqueeze(-1)))
            h2 = F.relu((1 + a) * h1 + b)
            h3 = F.relu(self.mlp2(h2))
            R  = self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)
        else:
            inp = torch.cat([u.unsqueeze(-1), H_blend], dim=-1)
            R = self.Lambda * torch.tanh(self.direct(inp)).squeeze(-1)

        return R, H_out_n, H_lay_n

    def reaction_from_u_only(self, u_vals):
        Nx = u_vals.shape[0]; dev = u_vals.device
        if self.use_film:
            a = torch.zeros(Nx, self.mlp_w, device=dev)
            b = torch.zeros(Nx, self.mlp_w, device=dev)
            h1 = F.relu(self.mlp1(u_vals.unsqueeze(-1)))
            h2 = F.relu((1 + a) * h1 + b)
            h3 = F.relu(self.mlp2(h2))
            return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)
        H0  = torch.zeros(Nx, self.hidden, device=dev)
        inp = torch.cat([u_vals.unsqueeze(-1), H0], dim=-1)
        return self.Lambda * torch.tanh(self.direct(inp)).squeeze(-1)

    @torch.no_grad()
    def reaction_nograd(self, u_vals):
        return self.reaction_from_u_only(u_vals)


# ══════════════════════════════════════════════════════════
#  Regime gate, IMEX step, rollouts
# ══════════════════════════════════════════════════════════

def regime_gate(dh, eps):
    """Continuous regime indicator rho in (0,1). See REGIME GATE above."""
    l_i = torch.log(torch.abs(dh) + 1e-8)
    if LEGACY_RHO:
        mu = torch.median(l_i)
        denom = float(np.log(1.0 / (eps + 1e-12))) + 1e-8
        return torch.sigmoid(3.0 * (l_i - mu) / denom)
    mu = torch.median(l_i[1:-1])
    denom = max(float(np.log(1.0 / (eps + 1e-12))), 1.0)
    rho = torch.sigmoid(3.0 * (l_i - mu) / denom).clone()
    rho[0] = rho[1]; rho[-1] = rho[-2]
    return rho


def imex_step(u_n, u_prev, model, H_out, H_lay,
              x_t, phi_t, A_t, L_t, cfg, problem, t_next):
    eps = cfg["eps_diff"]; dt = cfg["dt"]
    periodic = cfg["bc_type"] == "periodic"
    dev = u_n.device
    with torch.no_grad():
        dh = L_t @ u_n
    rho   = regime_gate(dh, eps)
    dt_u  = (u_n.detach() - u_prev.detach()) / (dt + 1e-12)
    se    = eps ** 0.5
    z_out = torch.stack([u_n, dh, dt_u, x_t], -1)
    z_lay = torch.stack([u_n, eps * dh, phi_t, se * torch.abs(dh)], -1)
    R_n, Ho, Hl = model(u_n, z_out, z_lay, rho, H_out, H_lay)
    rhs = u_n + (dt / 2) * eps * (L_t @ u_n) + dt * R_n
    if not periodic:
        gL, gR = get_bc_values(problem, t_next, cfg)
        rhs = rhs.clone()
        rhs[0]  = torch.as_tensor(gL, dtype=torch.float32, device=dev)
        rhs[-1] = torch.as_tensor(gR, dtype=torch.float32, device=dev)
    u_next = torch.linalg.solve(A_t, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(u_next, cfg["u_min"], cfg["u_max"]), R_n, Ho, Hl


def rollout_tbptt(model, u0_t, u_ref_t, x_t, phi_t, A_t, L_t,
                  cfg, problem, t_arr, window):
    Nt = cfg["Nt"]
    H_out, H_lay = model.init_hidden(len(x_t), x_t.device)
    u_n = u0_t; u_prev = u0_t; n = 0
    while n < Nt:
        w = min(n + window, Nt)
        u_n = u_n.detach(); u_prev = u_prev.detach()
        H_out = H_out.detach(); H_lay = H_lay.detach()
        utw = [u_n]; Rtw = []
        for s in range(n, w):
            u_next, R_n, H_out, H_lay = imex_step(
                u_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, A_t, L_t, cfg, problem, float(t_arr[s + 1]))
            utw.append(u_next); Rtw.append(R_n)
            u_prev = u_n; u_n = u_next
        yield utw, Rtw, u_ref_t[n:w + 1]
        n = w


def rollout_full_nograd(model, u0_t, x_t, phi_t, A_t, L_t,
                        cfg, problem, t_arr):
    """Complete autonomous rollout, used by the error-analysis panel."""
    Nt = cfg["Nt"]
    H_out, H_lay = model.init_hidden(len(x_t), x_t.device)
    u_n = u0_t; u_prev = u0_t
    u_traj = [u_n.detach().clone()]
    with torch.no_grad():
        for s in range(Nt):
            u_next, _, H_out, H_lay = imex_step(
                u_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, A_t, L_t, cfg, problem, float(t_arr[s + 1]))
            u_traj.append(u_next.detach().clone())
            u_prev = u_n; u_n = u_next
    return torch.stack(u_traj).cpu().numpy()


# ══════════════════════════════════════════════════════════
#  Losses  
# ══════════════════════════════════════════════════════════

TAU = 0.05


def l_data(ut, ur, su):
    return torch.stack([((ut[i] - ur[i]) / (su + 1e-8)).pow(2).mean()
                        for i in range(len(ut))]).mean()


def l_cons(ut, Rt, tau=TAU, B=64):
    Nt = len(Rt); step = max(1, Nt // 10)
    ua = torch.cat([ut[n] for n in range(0, Nt, step)])
    Ra = torch.cat([Rt[n] for n in range(0, Nt, step)])
    idx = torch.argsort(ua); us = ua[idx]; Rs = Ra[idx]
    du = torch.abs(us[1:] - us[:-1]); mask = du < tau
    if mask.sum() < 2:
        return ua.new_zeros(())
    valid = torch.where(mask)[0]
    if len(valid) > B:
        valid = valid[torch.randperm(len(valid), device=ua.device)[:B]]
    w = (1 - du[valid] / tau).clamp(0, 1).pow(2)
    return (w * (Rs[valid] - Rs[valid + 1]).pow(2)).mean()


def l_anch(model, anchors, device):
    if not anchors:
        return torch.zeros((), device=device)
    ua = torch.tensor([a[0] for a in anchors],
                      dtype=torch.float32, device=device)
    Ra = torch.tensor([a[1] for a in anchors],
                      dtype=torch.float32, device=device)
    return (model.reaction_from_u_only(ua) - Ra).pow(2).mean()


def l_fd(model, u_t, R_t, w_t, Lambda):
    if u_t is None:
        # Unreachable via compute_loss, which guards on u_fd_t, but a
        # bare CPU scalar would break the CUDA path if it ever were.
        return next(model.parameters()).new_zeros(())
    Rp = model.reaction_from_u_only(u_t)
    return (w_t * (Rp - R_t).pow(2) / (Lambda ** 2 + 1e-8)).mean()


def l_pos(Rt):
    return torch.stack([torch.relu(-Rt[n]).pow(2).mean()
                        for n in range(len(Rt))]).mean()


def compute_loss(ut, Rt, ur, model, cfg, su, anchors,
                 u_fd_t, R_fd_t, w_fd_t, flags, device):
    L = cfg["Lambda"]
    tot = l_data(ut, ur, su)
    if flags["l_cons"]:
        tot = tot + l_cons(ut, Rt)
    if flags["l_anch"]:
        tot = tot + 5.0 * l_anch(model, anchors, device)
    if flags["l_fd"] and u_fd_t is not None:
        tot = tot + cfg["lambda_fd"] * l_fd(model, u_fd_t, R_fd_t, w_fd_t, L)
    if cfg.get("positivity"):
        tot = tot + 2.0 * l_pos(Rt)
    return tot


def preinit(model, u_t, R_t, w_t, Lambda, epochs=300):
    for n, p in model.named_parameters():
        if any(g in n for g in ["gru_out", "gru_lay", "gru_single",
                                "film_proj"]):
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        for p in model.parameters():
            p.requires_grad_(True)
        params = list(model.parameters())
    opt = optim.Adam(params, lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (w_t * (model.reaction_from_u_only(u_t) - R_t).pow(2)
                / (Lambda ** 2 + 1e-8)).mean()
        loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


# ══════════════════════════════════════════════════════════
#  Anchor-neighbourhood error split 
# ══════════════════════════════════════════════════════════

def anchor_split_error(R_pred, R_true, u_grid, cfg, band=ANCHOR_BAND):
    anchors = cfg.get("anchors") or []
    den = float(np.sqrt(np.mean(R_true ** 2))) + 1e-8
    if not anchors:
        return float("nan"), float("nan"), 0.0
    half = band * (cfg["u_max"] - cfg["u_min"])
    near = np.zeros_like(u_grid, dtype=bool)
    for (ua, _) in anchors:
        near |= np.abs(u_grid - ua) <= half
    far = ~near

    def rms(mask):
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((R_pred[mask] - R_true[mask]) ** 2))
                     / den)

    return rms(near), rms(far), float(near.mean())


# ══════════════════════════════════════════════════════════
#  Run one configuration
# ══════════════════════════════════════════════════════════

def run_config(problem, cfg, u_refs_np, t_arr_np, L_np, A_t, L_t,
               u_tgt, R_tgt, weights, flags, su, x_np,
               epochs, need_rollout=False):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = DEVICE

    x_t   = torch.tensor(x_np, dtype=torch.float32, device=dev)
    xl, xr = cfg["domain"]; eps = cfg["eps_diff"]
    phi_t = torch.exp(-torch.minimum(
        x_t - xl,
        torch.as_tensor(xr, dtype=torch.float32, device=dev) - x_t)
        / (eps ** 0.5 + 1e-12))
    t_arr = torch.tensor(t_arr_np, dtype=torch.float32, device=dev)

    use_fd = flags["l_fd"] and len(u_tgt) > 0
    u_fd_t = torch.tensor(u_tgt, dtype=torch.float32, device=dev) if use_fd else None
    R_fd_t = torch.tensor(R_tgt, dtype=torch.float32, device=dev) if use_fd else None
    w_fd_t = torch.tensor(weights, dtype=torch.float32, device=dev) if use_fd else None

    model = AblModel(cfg["Lambda"],
                     dual_gru=flags["dual_gru"],
                     film=flags["film"],
                     input_dim=4).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    if flags["preinit"] and u_fd_t is not None:
        preinit(model, u_fd_t, R_fd_t, w_fd_t, cfg["Lambda"])

    anchors  = cfg["anchors"] if flags["l_anch"] else []
    n_roll   = min(cfg["n_rollout_ics"], len(u_refs_np))
    u_refs_t = [torch.tensor(u, dtype=torch.float32, device=dev)
                for u in u_refs_np[:n_roll]]
    window   = cfg["tbptt"]; n_ics = len(u_refs_t)

    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=100, eta_min=1e-5)
    lhist = []

    print(f"      ({n_params} params, {epochs} ep, {dev})", flush=True)
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(); ep_loss = 0.0
        for u_ref_t in u_refs_t:
            n_wins = max(1, cfg["Nt"] // window)
            ic = 0.0
            for uw, Rw, urw in rollout_tbptt(
                    model, u_ref_t[0], u_ref_t, x_t, phi_t, A_t, L_t,
                    cfg, problem, t_arr, window):
                Lw = compute_loss(uw, Rw, urw, model, cfg, su, anchors,
                                  u_fd_t, R_fd_t, w_fd_t, flags, dev)
                (Lw / n_wins).backward()
                ic += float(Lw.detach()) / n_wins
            ep_loss += ic / n_ics
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); lhist.append(ep_loss)
        if epoch % 100 == 0:
            print(f"      ep {epoch}/{epochs}  loss={ep_loss:.5f}")

    model.eval()
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    R_pred = model.reaction_nograd(
        torch.tensor(u_grid, dtype=torch.float32, device=dev)).cpu().numpy()
    R_true = get_R_true(problem, u_grid, cfg)
    l2 = float(np.sqrt(np.mean((R_pred - R_true) ** 2)) /
               (np.sqrt(np.mean(R_true ** 2)) + 1e-8))
    e_near, e_far, frac = anchor_split_error(R_pred, R_true, u_grid, cfg)

    extra = {}
    if need_rollout:
        all_u_pred, all_u_ref = [], []
        for u_ref_np in u_refs_np:
            u_ref_t_full = torch.tensor(u_ref_np, dtype=torch.float32,
                                        device=dev)
            all_u_pred.append(rollout_full_nograd(
                model, u_ref_t_full[0], x_t, phi_t, A_t, L_t,
                cfg, problem, t_arr))
            all_u_ref.append(u_ref_np)
        extra = dict(u_pred_trajs=all_u_pred, u_ref_trajs=all_u_ref,
                     t_arr=t_arr_np, x=x_np)

    return dict(l2=l2, eps_near=e_near, eps_far=e_far, band_frac=frac,
                R_pred=R_pred, u_grid=u_grid, R_true=R_true,
                loss_history=lhist, n_params=n_params, extra=extra)


# ══════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════

def plot_reaction(steps, problem, pname, cfg):
    fig, axes = plt.subplots(1, N_STEPS, figsize=(3.5 * N_STEPS, 5.0),
                             sharey=True)
    for col, (title, _, _) in enumerate(STEPS):
        ax = axes[col]; d = steps[col]
        full = (col == FULL_MODEL_IDX)
        ax.plot(d["u_grid"], d["R_true"], "k-", lw=2.4,
                label="$R_{\\mathrm{true}}$", zorder=5)
        ax.plot(d["u_grid"], d["R_pred"], "--", color=STEP_COLORS[col],
                lw=2.0, label="$R_\\theta$", zorder=4)
        for (ua, Ra) in (cfg.get("anchors") or []):
            ax.plot([ua], [Ra], "o", ms=6, mfc="none", mec="#117A65",
                    mew=1.8, zorder=6)
        ax.set_title(f"{title}\n$\\varepsilon_R = {d['l2']:.3f}$",
                     fontsize=9.5, fontweight="bold" if full else "normal")
        ax.set_xlabel("$u$", fontsize=11)
        if col == 0:
            ax.set_ylabel("$R(u)$", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)
        if full:
            for sp in ax.spines.values():
                sp.set_edgecolor("#C0392B"); sp.set_linewidth(2.5)
    fig.text(0.5, -0.04,
             "Adding one component at a time  ->  Full Model"
             "     (circles mark the equilibrium anchors)",
             ha="center", fontsize=10.5, style="italic", color="#444444")
    fig.suptitle(f"{pname} - Ablation: identified $R_\\theta(u)$ vs "
                 f"$R_{{\\mathrm{{true}}}}(u)$",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig



def plot_full_model_errors(full_result, extra, problem, pname, cfg):
    """6-panel error analysis for the full model only."""
    import scipy.stats as st

    u_pred_trajs = extra["u_pred_trajs"]; u_ref_trajs = extra["u_ref_trajs"]
    t_arr = extra["t_arr"]; x = extra["x"]; Nt1 = len(t_arr)

    abs_err = np.stack([np.abs(p - r)
                        for p, r in zip(u_pred_trajs, u_ref_trajs)])
    rel_l2_t = np.stack([
        np.sqrt(np.mean((p - r) ** 2, axis=1)) /
        (np.sqrt(np.mean(r ** 2, axis=1)) + 1e-8)
        for p, r in zip(u_pred_trajs, u_ref_trajs)])
    rel_l2_full = np.array([
        np.sqrt(np.mean((p - r) ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-8)
        for p, r in zip(u_pred_trajs, u_ref_trajs)])

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05],
                          hspace=0.32, wspace=0.30)
    fig.suptitle(f"{pname} - Full Model (BRIDGE): error analysis",
                 fontsize=17, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    quarts = [(0, Nt1 // 4), (Nt1 // 4, Nt1 // 2),
              (Nt1 // 2, 3 * Nt1 // 4), (3 * Nt1 // 4, Nt1)]
    labels_q = ["t in [0,T/4]", "t in [T/4,T/2]",
                "t in [T/2,3T/4]", "t in [3T/4,T]"]
    data_q = []
    for (a, b) in quarts:
        vals = abs_err[:, a:b, :].ravel()
        vals = vals[vals > 1e-12]
        data_q.append(vals if len(vals) else np.array([1e-12]))
    parts = ax1.violinplot(data_q, showmedians=False, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#5DADE2"); pc.set_alpha(0.75)
    for i, vals in enumerate(data_q):
        ax1.hlines(np.median(vals), i + 0.75, i + 1.25,
                   color="red", lw=2.2, zorder=5)
    ax1.set_yscale("log"); ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(labels_q, fontsize=8.5)
    ax1.set_ylabel("Absolute error", fontsize=11)
    ax1.set_title("Error distribution by time quartile", fontsize=12)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    mean_err_t = abs_err.mean(axis=(0, 2))
    ci_lo = np.percentile(abs_err, 2.5, axis=(0, 2))
    ci_hi = np.percentile(abs_err, 97.5, axis=(0, 2))
    ax2.plot(t_arr, mean_err_t, color="#2563C9", lw=2.2, label="Mean |error|")
    ax2.fill_between(t_arr, ci_lo, ci_hi, color="#2563C9", alpha=0.18,
                     label="95% CI")
    ax2.set_xlabel("t", fontsize=11)
    ax2.set_ylabel("Mean absolute error", fontsize=11)
    ax2.set_title("Temporal error evolution", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left"); ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[0, 2])
    rl2 = rel_l2_t.ravel()
    rl2 = rl2[(rl2 > 0) & np.isfinite(rl2)]
    if len(rl2) > 3:
        lv = np.log10(rl2 + 1e-12)
        kde = st.gaussian_kde(lv)
        xs = np.linspace(lv.min(), lv.max(), 400)
        ax3.fill_between(10 ** xs, kde(xs), color="#5DADE2", alpha=0.55,
                         label="Rel. $L^2$")
        ax3.axvline(float(np.mean(rl2)), color="navy", lw=2, ls="--",
                    label=f"Mean={np.mean(rl2):.4f}")
    ax3.set_xscale("log")
    ax3.set_xlabel("Relative $L^2$ error", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title(f"$L^2$ error distribution\n"
                  f"Mean $L^2$={np.mean(rel_l2_full):.4f}", fontsize=12)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(full_result["u_grid"], full_result["R_true"], "k-", lw=2.4,
             label="$R_{\\mathrm{true}}$")
    ax4.plot(full_result["u_grid"], full_result["R_pred"], "--",
             color="#C0392B", lw=2.2, label="$R_\\theta$ (full model)")
    ax4.set_xlabel("$u$", fontsize=11); ax4.set_ylabel("$R(u)$", fontsize=11)
    ax4.set_title(f"Identified reaction law\n"
                  f"$\\varepsilon_R$={full_result['l2']:.4f}", fontsize=12)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)
    for sp in ax4.spines.values():
        sp.set_edgecolor("#C0392B"); sp.set_linewidth(1.8)

    ax5 = fig.add_subplot(gs[1, 0:2])
    snap_idx = [0, Nt1 // 2, Nt1 - 1]
    snap_colors = ["#2563C9", "#E07B30", "#1E8449"]
    u_ref0, u_pred0 = u_ref_trajs[0], u_pred_trajs[0]
    for k, si in enumerate(snap_idx):
        ax5.plot(x, u_ref0[si], "-", color=snap_colors[k], lw=2.0,
                 label=f"u_ref t={t_arr[si]:.2f}")
        ax5.plot(x, u_pred0[si], "--", color=snap_colors[k], lw=2.0,
                 label=f"u_pred t={t_arr[si]:.2f}")
    ax5.set_xlabel("x", fontsize=11); ax5.set_ylabel("u", fontsize=11)
    ax5.set_title("u_ref (solid) vs u_pred (dashed)", fontsize=12)
    ax5.legend(fontsize=8, ncol=2); ax5.grid(True, alpha=0.25)

    ax6 = fig.add_subplot(gs[1, 2:4])
    im = ax6.imshow(abs_err[0], aspect="auto", origin="lower", cmap="YlOrRd",
                    extent=[x.min(), x.max(), t_arr.min(), t_arr.max()])
    ax6.set_xlabel("x", fontsize=11); ax6.set_ylabel("t", fontsize=11)
    ax6.set_title("|u_pred - u_ref| over (x, t)", fontsize=12)
    fig.colorbar(im, ax=ax6, label="Absolute error")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    global DEVICE
    ap = argparse.ArgumentParser(
        description="Component ablation for BRIDGE (scalar problems)")
    ap.add_argument("--problem", default="all",
                    choices=["fisher", "allen_cahn", "all"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-loo", action="store_true",
                    help="skip the leave-one-out controls (not "
                         "recommended)")
    args = ap.parse_args()

    DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})"
             if DEVICE.type == "cuda" else ""))
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Regime gate: {'LEGACY' if LEGACY_RHO else 'corrected'}"
          " (interior median, ln(1/eps) floored at 1)")

    problems = (["fisher", "allen_cahn"] if args.problem == "all"
                else [args.problem])
    loo_list = [] if args.no_loo else LOO_STEPS
    all_results = {}

    for problem in problems:
        cfg = copy.deepcopy(PROBLEMS[problem]); pname = cfg["name"]
        print(f"\n{'='*64}\n  Problem: {pname}\n{'='*64}")

        x, L_np, _, lu, piv, L_t, A_t = build_all_operators(
            problem, cfg, DEVICE)
        print("  Generating trajectories...")
        u_refs, t_arr = generate_reference(problem, x, cfg, L_np, lu, piv)
        print("  Extracting FD pairs...")
        u_fd_raw, R_fd_raw = extract_fd_pairs(u_refs, L_np, cfg)
        u_tgt, R_tgt, weights = aggregate_bins(u_fd_raw, R_fd_raw, cfg,
                                               problem)
        print(f"  FD targets: {len(u_tgt)}")

        n_roll = min(cfg["n_rollout_ics"], len(u_refs))
        su = float(np.mean(np.abs(
            np.concatenate([r.flatten() for r in u_refs[:n_roll]]))))

        common = dict(problem=problem, cfg=cfg, u_refs_np=u_refs,
                      t_arr_np=t_arr, L_np=L_np, A_t=A_t, L_t=L_t,
                      u_tgt=u_tgt, R_tgt=R_tgt, weights=weights,
                      su=su, x_np=x, epochs=args.epochs)

        steps = []
        for i, (_, flags, short) in enumerate(STEPS):
            print(f"\n  [Step {i+1}/{N_STEPS}] {short}")
            t0 = time.time()
            r = run_config(flags=flags,
                           need_rollout=(i == FULL_MODEL_IDX), **common)
            print(f"  -> eps_R = {r['l2']:.4f}   "
                  f"near={r['eps_near']:.4f}  far={r['eps_far']:.4f}   "
                  f"({time.time()-t0:.0f}s)")
            steps.append(r)

        loos = []
        for k, (_, flags, short) in enumerate(loo_list):
            print(f"\n  [Leave-one-out {k+1}/{len(loo_list)}] {short}")
            t0 = time.time()
            r = run_config(flags=flags, need_rollout=False, **common)
            fl = steps[FULL_MODEL_IDX]["l2"]
            print(f"  -> eps_R = {r['l2']:.4f}   "
                  f"near={r['eps_near']:.4f}  far={r['eps_far']:.4f}   "
                  f"({(r['l2']-fl)/(fl+1e-12)*100:+.1f}% vs full)   "
                  f"({time.time()-t0:.0f}s)")
            loos.append(r)

        all_results[problem] = {"steps": steps, "loos": loos}

        figs = [(plot_reaction(steps, problem, pname, cfg),
                 f"ablation_reaction_{problem}.png")]
        extra = steps[FULL_MODEL_IDX]["extra"]
        if extra:
            figs.append((plot_full_model_errors(
                steps[FULL_MODEL_IDX], extra, problem, pname, cfg),
                f"full_model_error_analysis_{problem}.png"))
        for fig, fname in figs:
            p = os.path.join(OUT_DIR, fname)
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  Saved: {p}")

    def _pack(d, label):
        return {"label": label, "l2": d["l2"],
                "eps_near": d["eps_near"], "eps_far": d["eps_far"],
                "band_frac": d["band_frac"], "n_params": d["n_params"],
                "R_pred": np.asarray(d["R_pred"]).tolist(),
                "u_grid": np.asarray(d["u_grid"]).tolist(),
                "R_true": np.asarray(d["R_true"]).tolist()}

    dump = {p: {"legacy_rho": LEGACY_RHO,
                "anchor_band": ANCHOR_BAND,
                "steps": [_pack(all_results[p]["steps"][i], STEPS[i][2])
                          for i in range(N_STEPS)],
                "loos": [_pack(all_results[p]["loos"][k], loo_list[k][2])
                         for k in range(len(all_results[p]["loos"]))]}
            for p in all_results}
    with open(os.path.join(OUT_DIR, "ablation_results.json"), "w") as f:
        json.dump(dump, f, indent=2)

    print(f"\n{'='*64}\n  ANCHOR CONTRIBUTION \n{'='*64}")
    for p in problems:
        full = all_results[p]["steps"][FULL_MODEL_IDX]
        add = all_results[p]["steps"][3]["l2"] - all_results[p]["steps"][2]["l2"]
        print(f"\n  {PROBLEMS[p]['name']}")
        print(f"    additive  (Step 3 -> Step 4, L_anch added): "
              f"{all_results[p]['steps'][2]['l2']:.4f} -> "
              f"{all_results[p]['steps'][3]['l2']:.4f}  ({add:+.4f})")
        for k, d in enumerate(all_results[p]["loos"]):
            print(f"    leave-one-out ({loo_list[k][2]}): "
                  f"{full['l2']:.4f} -> {d['l2']:.4f}  "
                  f"({(d['l2']-full['l2'])/(full['l2']+1e-12)*100:+.1f}%)")
            print(f"      near anchors: {full['eps_near']:.4f} -> "
                  f"{d['eps_near']:.4f}")
            print(f"      away:         {full['eps_far']:.4f} -> "
                  f"{d['eps_far']:.4f}")


if __name__ == "__main__":
    main()
