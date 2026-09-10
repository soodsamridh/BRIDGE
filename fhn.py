"""
fhn_bvp_runner.py  —  Semi-known FHN system (Keener-Sneyd form).

EQUATIONS:
  ∂u/∂t = ε·∂²u/∂x²  +  R_u(u, v)     ← IDENTIFY
  ∂v/∂t = δ(u − v)                       ← KNOWN explicit ODE

  R_u_true(u,v) = u(1−u)(u−a) − v       [FHN activator kinetics]
  ε = 0.01,  a = 0.10,  δ = 0.50
  Domain: x∈[0,1],  T=0.5
  BC: Neumann (no-flux) at x=0,1
  ICs: u₀ and v₀ chosen INDEPENDENTLY for diverse (u,v) coverage

WHY THIS SYSTEM IS SOLVABLE:
  1. ε=0.01 → genuine sharp boundary layers → Shishkin mesh fully engaged
  2. Domain [0,1] is small → dense 2D FD bin coverage
  3. v-equation has NO diffusion → simple explicit torch update
  4. ICs are NOT on the nullcline → diverse (u,v) state-space coverage
  5. R_u depends primarily on u (cubic) with linear v correction
     → FD estimates are clean and informative

CITATIONS:
  [1] Keener J, Sneyd J (2009) Mathematical Physiology I. Springer. Ch.5.2
  [2] FitzHugh R (1961) Biophysical Journal 1(6):445-466

Run:
  python fhn_bvp_runner.py
  python fhn_bvp_runner.py --epochs 3000
"""

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

import config as C
from config    import PROBLEMS
from numerics  import build_all_operators
from equations import get_ics, R_u_fhn_semi, R_v_fhn_semi
from model     import EpsUSGRU_SemiKnown
from regime    import regime_indicator, layer_envelope

DEVICE  = torch.device("cpu")
SEED    = 42
OUT_DIR = "results_v5/fhn_semi"


# ══════════════════════════════════════════════════════════
#  Reference trajectory
# ══════════════════════════════════════════════════════════

def generate_reference(x, cfg, L_np, lu, piv):
    """Generate reference (u, v) trajectories using true reactions."""
    Nt = cfg["Nt"]; dt = cfg["dt"]; eps = cfg["eps_diff"]; D = cfg["eps_diff"]
    t_arr = np.linspace(0, cfg["T"], Nt+1)
    ics   = get_ics("fhn_semi", x, cfg)   # list of (u0, v0) tuples
    refs  = []

    for k, (u0, v0) in enumerate(ics):
        ur = np.zeros((Nt+1, len(x))); vr = np.zeros((Nt+1, len(x)))
        ur[0] = u0; vr[0] = v0
        for n in range(Nt):
            un = ur[n]; vn = vr[n]
            Ru = R_u_fhn_semi(un, vn, cfg["a_fhn"])
            Rv = R_v_fhn_semi(un, vn, cfg)
            # IMEX for u
            rhs = un + (dt/2)*eps*(L_np@un) + dt*Ru
            rhs[0] = 0; rhs[-1] = 0                  # Neumann
            ur[n+1] = np.clip(
                scipy.linalg.lu_solve((lu, piv), rhs),
                cfg["u_min"]-0.05, cfg["u_max"]+0.05)
            # Explicit for v (no diffusion — pure ODE)
            vr[n+1] = np.clip(
                vn + dt*Rv,
                cfg["v_min"]-0.05, cfg["v_max"]+0.05)
        refs.append((ur, vr))
        print(f"    IC {k+1}: u∈[{ur.min():.3f},{ur.max():.3f}]"
              f"  v∈[{vr.min():.3f},{vr.max():.3f}]")
    return refs, t_arr


# ══════════════════════════════════════════════════════════
#  FD extraction and 2D binning
# ══════════════════════════════════════════════════════════

def extract_fd_pairs(refs, L_np, cfg):
    """R̂_u = ∂_t u - ε·L_h u at outer nodes, tagged with (u,v)."""
    dt = cfg["dt"]; eps = cfg["eps_diff"]
    ua, va, Ru_a = [], [], []
    for ur, vr in refs:
        Nt = ur.shape[0]-1
        for n in range(1, Nt-1):
            un = ur[n]; vn = vr[n]
            dtu  = (ur[n+1]-ur[n-1])/(2*dt)
            Ru_fd = dtu - eps*(L_np@un)
            lap   = np.abs(L_np@un)
            mask  = lap < np.percentile(lap, cfg["fd_pct"])
            mask[0] = False; mask[-1] = False
            if mask.sum() > 0:
                ua.append(un[mask]); va.append(vn[mask])
                Ru_a.append(Ru_fd[mask])
    return (np.concatenate(ua), np.concatenate(va), np.concatenate(Ru_a))


def aggregate_bins_2d(u_fd, v_fd, Ru_fd, cfg):
    """Bin-medians over 2D (u,v) grid."""
    nb  = cfg["n_bins_2d"]
    ue  = np.linspace(cfg["u_min"], cfg["u_max"], nb+1)
    ve  = np.linspace(cfg["v_min"], cfg["v_max"], nb+1)
    uc  = 0.5*(ue[:-1]+ue[1:]); vc = 0.5*(ve[:-1]+ve[1:])
    ui  = np.clip(np.digitize(u_fd, ue)-1, 0, nb-1)
    vi  = np.clip(np.digitize(v_fd, ve)-1, 0, nb-1)
    u_t=[]; v_t=[]; Ru_t=[]
    for i in range(nb):
        for j in range(nb):
            m = (ui==i)&(vi==j)
            if m.sum() >= 4:
                u_t.append(uc[i]); v_t.append(vc[j])
                Ru_t.append(np.median(Ru_fd[m]))
    if not u_t:
        # 1D fallback
        ue1 = np.linspace(cfg["u_min"], cfg["u_max"], 21)
        uc1 = 0.5*(ue1[:-1]+ue1[1:])
        ui1 = np.clip(np.digitize(u_fd, ue1)-1, 0, 19)
        for i in range(20):
            m = ui1==i
            if m.sum() >= 4:
                u_t.append(uc1[i]); v_t.append(0.5)
                Ru_t.append(np.median(Ru_fd[m]))
        print("  WARNING: 1D fallback used")
    print(f"  2D FD bins: {len(u_t)}")
    return (np.array(u_t), np.array(v_t),
            np.array(Ru_t), np.ones(len(u_t)))


# ══════════════════════════════════════════════════════════
#  IMEX step — u: IMEX (differentiable), v: explicit (detached)
# ══════════════════════════════════════════════════════════

def _neumann_rhs(rhs):
    rhs = rhs.clone(); rhs[0] = 0.0; rhs[-1] = 0.0; return rhs


def imex_step_semi(u_n, v_n, u_prev, model, H_out, H_lay,
                   x_t, phi_t, A_t, L_t, cfg, device):
    eps = cfg["eps_diff"]; dt = cfg["dt"]

    with torch.no_grad():
        dhu = L_t @ u_n
    rho  = regime_indicator(dhu, eps)
    dt_u = (u_n.detach()-u_prev.detach())/(dt+1e-12)

    # 5-feature GRU input: [u, v, Δ_hu, ∂_tu, x]
    z_out = torch.stack([u_n, v_n, dhu, dt_u, x_t], dim=-1)
    z_lay = torch.stack([u_n, v_n, eps*dhu, dt_u, phi_t], dim=-1)

    R_u, H_out_n, H_lay_n = model(u_n, v_n, z_out, z_lay, rho, H_out, H_lay)

    # IMEX solve for u — differentiable
    rhs_u  = _neumann_rhs(u_n + (dt/2)*eps*(L_t@u_n) + dt*R_u)
    u_next = torch.linalg.solve(A_t, rhs_u.unsqueeze(-1)).squeeze(-1)
    u_next = torch.clamp(u_next, cfg["u_min"]-0.05, cfg["u_max"]+0.05)

    # Explicit ODE for v — detached, no gradient
    with torch.no_grad():
        delta_v = cfg["delta_v"]
        v_next  = v_n.detach() + dt*delta_v*(u_n.detach()-v_n.detach())
        v_next  = torch.clamp(v_next, cfg["v_min"]-0.05, cfg["v_max"]+0.05)

    return u_next, v_next, R_u, H_out_n, H_lay_n


# ══════════════════════════════════════════════════════════
#  TBPTT rollout
# ══════════════════════════════════════════════════════════

def rollout_tbptt(model, u0, v0, ur_t, vr_t, x_t, phi_t,
                  A_t, L_t, cfg, device, window):
    Nt = cfg["Nt"]
    H_out, H_lay = model.init_hidden(len(x_t), device)
    u_n = u0; v_n = v0; u_prev = u0; n = 0
    while n < Nt:
        ww = min(n+window, Nt)
        u_n=u_n.detach(); v_n=v_n.detach()
        u_prev=u_prev.detach()
        H_out=H_out.detach(); H_lay=H_lay.detach()
        utw=[u_n]; vtw=[v_n]; Rtuw=[]
        for s in range(n, ww):
            u_next,v_next,R_u,H_out,H_lay = imex_step_semi(
                u_n,v_n,u_prev,model,H_out,H_lay,
                x_t,phi_t,A_t,L_t,cfg,device)
            utw.append(u_next); vtw.append(v_next); Rtuw.append(R_u)
            u_prev=u_n; u_n=u_next; v_n=v_next
        yield utw, vtw, Rtuw, ur_t[n:ww+1], vr_t[n:ww+1]
        n = ww


# ══════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════

TAU = 0.05

def l_data(utw, vtw, urw, vrw, su, sv):
    Lu = torch.stack([((utw[i]-urw[i])/(su+1e-8)).pow(2).mean()
                      for i in range(len(utw))]).mean()
    Lv = torch.stack([((vtw[i]-vrw[i])/(sv+1e-8)).pow(2).mean()
                      for i in range(len(vtw))]).mean()
    return Lu + Lv

def l_fd(model, u_t, v_t, Ru_t, w_t, Lambda):
    Ru_p = model.reaction_grad(u_t, v_t)
    return (w_t*(Ru_p-Ru_t).pow(2)/(Lambda**2+1e-8)).mean()

def l_cons(utw, vtw, Rtuw, B=48):
    step = max(1, len(Rtuw)//8)
    ua = torch.cat([utw[n] for n in range(0,len(utw)-1,step)])
    va = torch.cat([vtw[n] for n in range(0,len(vtw)-1,step)])
    Ra = torch.cat([Rtuw[n] for n in range(0,len(Rtuw),step)])
    idx = torch.argsort(ua)
    us=ua[idx]; vs=va[idx]; Rs=Ra[idx]
    du = torch.sqrt((us[1:]-us[:-1]).pow(2)+(vs[1:]-vs[:-1]).pow(2))
    mask = du < TAU
    if mask.sum() < 2: return torch.tensor(0.0)
    valid = torch.where(mask)[0]
    if len(valid)>B: valid=valid[torch.randperm(len(valid))[:B]]
    w = (1-du[valid]/TAU).clamp(0,1).pow(2)
    return (w*(Rs[valid]-Rs[valid+1]).pow(2)).mean()

def l_anch(model, cfg):
    pts = cfg["anchors_uv"]
    pu  = torch.tensor([p[0] for p in pts], dtype=torch.float32)
    pv  = torch.tensor([p[1] for p in pts], dtype=torch.float32)
    return model.reaction_grad(pu, pv).pow(2).mean()

def get_lams(epoch, cfg):
    lams = {}
    for key, r in cfg["ramp"].items():
        frac = min((epoch-1)/max(r["over"]-1,1), 1.0)
        lams[key] = r["start"] + (r["end"]-r["start"])*frac
    return lams


# ══════════════════════════════════════════════════════════
#  Stage 0: pre-init MLP from FD estimates
# ══════════════════════════════════════════════════════════

def preinit(model, u_t, v_t, Ru_t, w_t, cfg, epochs=400):
    for name, p in model.named_parameters():
        if "gru" in name or "film" in name:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        for p in model.parameters(): p.requires_grad_(True); return
    opt = optim.Adam(params, lr=1e-3)
    Lambda = cfg["Lambda"]; best = float("inf")
    print(f"  [Stage 0] {len(u_t)} 2D FD targets  ({epochs} ep)")
    for ep in range(1, epochs+1):
        opt.zero_grad()
        Ru_p = model.reaction_grad(u_t, v_t)
        loss = (w_t*(Ru_p-Ru_t).pow(2)/(Lambda**2+1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if loss.item()<best: best=loss.item()
        if ep%100==0 or ep==1:
            print(f"    ep {ep:>4}  loss={loss.item():.5f}")
    for p in model.parameters(): p.requires_grad_(True)
    print(f"  [Stage 0] Done. Best={best:.5f}")


# ══════════════════════════════════════════════════════════
#  Stage 1: training
# ══════════════════════════════════════════════════════════

def train(model, refs, t_arr, x_t, phi_t, A_t, L_t, cfg,
          u_fd_t, v_fd_t, Ru_fd_t, w_fd_t, epochs=2000):
    window  = cfg["tbptt"]
    su = max(float(np.mean([np.abs(r[0]).mean() for r in refs])), 0.01)
    sv = max(float(np.mean([np.abs(r[1]).mean() for r in refs])), 0.01)
    LR   = cfg.get("lr", 1e-3)
    CLIP = cfg.get("grad_clip", 0.5)

    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=500, eta_min=1e-5)

    best_loss=float("inf"); best_state=None; history=[]
    nan_count=0; MAX_NAN=8

    print(f"\n  [Stage 1] {epochs} ep  TBPTT={window}  LR={LR}  clip={CLIP}")
    print(f"  su={su:.4f}  sv={sv:.4f}")
    print(f"\n  {'Ep':>5}  {'Total':>8}  {'Data':>8}  "
          f"{'FD':>8}  {'Cons':>8}  {'Anch':>8}")
    print("  "+"-"*55)

    for epoch in range(1, epochs+1):
        lams = get_lams(epoch, cfg)
        model.train(); opt.zero_grad()
        ep=ep_d=ep_f=ep_c=ep_a=0.0

        for ur_np, vr_np in refs:
            ur_t = torch.tensor(ur_np, dtype=torch.float32)
            vr_t = torch.tensor(vr_np, dtype=torch.float32)
            n_wins = max(1, cfg["Nt"]//window)

            for utw,vtw,Rtuw,urw,vrw in rollout_tbptt(
                    model, ur_t[0], vr_t[0], ur_t, vr_t,
                    x_t, phi_t, A_t, L_t, cfg, DEVICE, window):

                Ld = l_data(utw,vtw,urw,vrw,su,sv)
                Lf = l_fd(model,u_fd_t,v_fd_t,Ru_fd_t,w_fd_t,cfg["Lambda"])
                Lc = l_cons(utw,vtw,Rtuw)
                La = l_anch(model,cfg)
                Lw = (lams["data"]*Ld + lams["fd"]*Lf +
                      lams["cons"]*Lc + lams["anch"]*La)
                (Lw/n_wins).backward()
                ep+=Lw.item()/n_wins/len(refs)
                ep_d+=Ld.item()/n_wins/len(refs)
                ep_f+=Lf.item()/n_wins/len(refs)
                ep_c+=Lc.item()/n_wins/len(refs)
                ep_a+=La.item()/n_wins/len(refs)

        if not np.isfinite(ep):
            nan_count+=1
            print(f"  NaN ep {epoch} ({nan_count}/{MAX_NAN}) — restore+halve LR")
            if best_state: model.load_state_dict(best_state)
            for pg in opt.param_groups: pg["lr"]*=0.5
            if nan_count>=MAX_NAN: print("  Stopping."); break
            continue

        nan_count=0
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step(); sched.step()

        history.append({"epoch":epoch,"total":ep,"data":ep_d,
                         "fd":ep_f,"cons":ep_c,"anch":ep_a})
        if ep<best_loss:
            best_loss=ep
            best_state={k:v.clone() for k,v in model.state_dict().items()}

        if epoch%100==0 or epoch<=3:
            print(f"  {epoch:>5}  {ep:>8.5f}  {ep_d:>8.5f}  "
                  f"{ep_f:>8.5f}  {ep_c:>8.5f}  {ep_a:>8.5f}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Best restored (loss={best_loss:.5f})")
    return history


# ══════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════

def evaluate(model, cfg):
    ug = np.linspace(cfg["u_min"], cfg["u_max"], 40)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], 40)
    UU,VV = np.meshgrid(ug, vg)
    uf=UU.ravel(); vf=VV.ravel()
    u_t = torch.tensor(uf, dtype=torch.float32)
    v_t = torch.tensor(vf, dtype=torch.float32)
    Ru_p    = model.reaction_nograd(u_t, v_t).numpy()
    Ru_true = R_u_fhn_semi(uf, vf, cfg["a_fhn"])
    l2 = float(np.sqrt(np.mean((Ru_p-Ru_true)**2)) /
               (np.sqrt(np.mean(Ru_true**2))+1e-8))
    return l2, UU, VV, Ru_p.reshape(40,40), Ru_true.reshape(40,40)


# ══════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════

def make_plots(model, refs, t_arr, x_np, cfg, history, l2):
    os.makedirs(OUT_DIR, exist_ok=True)
    l2_v, UU, VV, Ru_p, Ru_true = evaluate(model, cfg)

    fig, axes = plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f"FHN Semi-known — Reaction Identification  L²={l2:.4f}",
                 fontsize=13, fontweight="bold")
    for ax,Z,title,cmap in [
        (axes[0],Ru_true,"$R_u^\\mathrm{true}=u(1-u)(u-a)-v$","RdBu_r"),
        (axes[1],Ru_p,   "$R_u^\\theta$ (model)","RdBu_r"),
        (axes[2],np.abs(Ru_p-Ru_true),"|Error|","YlOrRd")]:
        im=ax.contourf(UU,VV,Z,levels=20,cmap=cmap)
        plt.colorbar(im,ax=ax)
        ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_title(title,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"reaction_fhn_semi.png"),dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved: reaction_fhn_semi.png")

    # Trajectory
    ur0,vr0=refs[0]; Nt=cfg["Nt"]
    snaps=[0,Nt//4,Nt//2,3*Nt//4,Nt]
    fig,axes=plt.subplots(2,5,figsize=(22,7),sharey="row")
    fig.suptitle("FHN Semi-known — Trajectory (IC 1)",fontsize=12,fontweight="bold")
    for col,n in enumerate(snaps):
        axes[0,col].plot(x_np,ur0[n],"k-",lw=2)
        axes[0,col].set_title(f"u, t={t_arr[n]:.3f}",fontsize=9)
        axes[1,col].plot(x_np,vr0[n],"C1-",lw=2)
        axes[1,col].set_title(f"v, t={t_arr[n]:.3f}",fontsize=9)
    axes[0,0].set_ylabel("u(x,t)"); axes[1,0].set_ylabel("v(x,t)")
    for ax in axes.ravel(): ax.set_xlabel("x",fontsize=8); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"trajectory_fhn_semi.png"),dpi=150,
                bbox_inches="tight"); plt.close()
    print("  Saved: trajectory_fhn_semi.png")

    # Loss
    if not history: return
    fig,ax=plt.subplots(figsize=(10,5))
    ep=[h["epoch"] for h in history]
    for k,c,ls in [("total","k","-"),("data","C0","--"),
                   ("fd","C1","-."),("cons","C2",":"),("anch","C3","--")]:
        ax.semilogy(ep,[max(h[k],1e-10) for h in history],
                    color=c,linestyle=ls,lw=1.5,label=k)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("FHN Semi-known — Training Loss",fontsize=12,fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"loss_fhn_semi.png"),dpi=150); plt.close()
    print("  Saved: loss_fhn_semi.png")


# ══════════════════════════════════════════════════════════
#  Main entry point (called from main.py or standalone)
# ══════════════════════════════════════════════════════════

def run_fhn_bvp(epochs=None, preinit_epochs=None, return_metrics=False):
    """Entry point — name kept as run_fhn_bvp for registry compatibility."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = PROBLEMS["fhn_semi"]
    ep  = epochs        or cfg.get("epochs", 2000)
    pi  = preinit_epochs or cfg.get("preinit_epochs", 400)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*65)
    print("  FHN Semi-known System (Keener-Sneyd) — ε-US-GRU")
    print("="*65)
    print(f"  IDENTIFY:  ∂u/∂t = {cfg['eps_diff']}·u_xx + R_u(u,v)")
    print(f"             R_u_true = u(1-u)(u-a) - v   [a={cfg['a_fhn']}]")
    print(f"  KNOWN:     ∂v/∂t = {cfg['delta_v']}·(u - v)")
    print(f"  Domain [0,1],  T={cfg['T']},  Nx={cfg['Nx']},  Nt={cfg['Nt']}")
    print(f"  Lambda={cfg['Lambda']},  ε={cfg['eps_diff']},  δ={cfg['delta_v']}")
    print()

    print("Building operators...")
    x, L_np, A_np, lu, piv, L_t, A_t = build_all_operators(
        "fhn_semi", cfg, DEVICE)
    x_t   = torch.tensor(x, dtype=torch.float32)
    xl,xr = cfg["domain"]; eps = cfg["eps_diff"]
    phi_t = layer_envelope(x_t, xl, xr, eps)

    print("Generating reference trajectories...")
    refs, t_arr = generate_reference(x, cfg, L_np, lu, piv)

    print("Extracting 2D FD pairs...")
    u_fd, v_fd, Ru_fd = extract_fd_pairs(refs, L_np, cfg)
    u_t, v_t, Ru_t, w_t = aggregate_bins_2d(u_fd, v_fd, Ru_fd, cfg)

    u_fd_t  = torch.tensor(u_t,  dtype=torch.float32)
    v_fd_t  = torch.tensor(v_t,  dtype=torch.float32)
    Ru_fd_t = torch.tensor(Ru_t, dtype=torch.float32)
    w_fd_t  = torch.tensor(w_t,  dtype=torch.float32)

    model = EpsUSGRU_SemiKnown(
        hidden_dim=32, mlp_width=32,
        Lambda=cfg["Lambda"], input_dim=5)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_p:,}")

    preinit(model, u_fd_t, v_fd_t, Ru_fd_t, w_fd_t, cfg, epochs=pi)

    t0 = time.time()
    history = train(model, refs, t_arr, x_t, phi_t, A_t, L_t, cfg,
                    u_fd_t, v_fd_t, Ru_fd_t, w_fd_t, epochs=ep)
    t_train = time.time()-t0

    l2, *_ = evaluate(model, cfg)
    print(f"\n  Results:")
    print(f"    Relative L² (R_u):  {l2:.4f}")
    print(f"    Training time:      {t_train:.0f}s ({t_train/60:.1f} min)")

    make_plots(model, refs, t_arr, x, cfg, history, l2)

    results = {"problem":"fhn_semi","l2_Ru":float(l2),
                "params":n_p,"train_s":float(t_train),"epochs":ep}
    with open(os.path.join(OUT_DIR,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2)
    print(f"  Saved: {OUT_DIR}/metrics.json")

    if return_metrics: return results
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FHN semi-known system — eps-US-GRU")
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--preinit_epochs", type=int, default=None)
    args = parser.parse_args()
    run_fhn_bvp(args.epochs, args.preinit_epochs)