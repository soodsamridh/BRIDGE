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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from numerics import build_shishkin_mesh, build_compact_laplacian

SEED      = 42
HIDDEN    = 32
MLP_W     = 64
LR        = 1e-3
WD        = 1e-5
GRAD_CLIP = 0.5
EPOCHS    = 2000
OUT_DIR   = "comparison_results_systems"
os.makedirs(OUT_DIR, exist_ok=True)

FDPINN_MLP_W = 64
FDPINN_DEPTH = 2

METHODS = ["PINN", "FDPINN", "PI-RNN", "Proposed"]
STYLES  = {
    "PINN":     {"color":"#E07B54", "ls":"--", "lw":1.8},
    "FDPINN":   {"color":"#8E44AD", "ls":":",  "lw":2.0},
    "PI-RNN":   {"color":"#27AE60", "ls":"-.", "lw":2.0},
    "Proposed": {"color":"#C0392B", "ls":"-",  "lw":2.5},
}
REF_STYLE = {"color":"black", "ls":"-", "lw":2.5}

# DATA-MISFIT FIGURE STYLE.  The misfit figure in the submitted
# manuscript draws every method as a dashed curve and separates them by
# colour, so it is kept that way here rather than inheriting the scheme
# used by the reaction and trajectory figures.  Proposed keeps a heavier
# stroke so it is identifiable in greyscale.
MISFIT_STYLES = {
    "PINN":     {"color":"#E07B54", "ls":"--", "lw":1.8},
    "FDPINN":   {"color":"#8E44AD", "ls":"--", "lw":2.0},
    "PI-RNN":   {"color":"#27AE60", "ls":"--", "lw":2.0},
    "Proposed": {"color":"#C0392B", "ls":"--", "lw":2.8},
}

SYSTEM_CFGS = {
    "fhn_partial": {
        "name":    "FitzHugh-Nagumo Partial (Neuroscience)",
        "citation":"FitzHugh (1961) Biophys J 1(6):445",
        "eps_diff":0.05, "delta_v":0.1, "beta_v":1.0, "gamma_v":0.5, "a_fhn":0.25,
        "Lambda":2.0, "u_min":0.0, "u_max":1.0, "v_min":0.0, "v_max":0.6,
        "domain":(0.0,1.0), "T":0.3, "Nx":128, "Nt":60, "dt":0.005,
        "bc_type":"dirichlet", "beta_mesh":2.0,
        "tbptt":1, "n_bins_2d":12,
        "anchors_uv":[(0.0,0.0),(0.25,0.0),(1.0,0.0)],
        "ics":[("front",0.2,0.0),("front",0.3,0.0),("front",0.4,0.0),
               ("front",0.5,0.0),("front",0.6,0.0),("front",0.7,0.0),
               ("front",0.3,0.3),("step",0.5,0.5)],
        "input_dim":5,
    },
    "predator_prey": {
        "name":    "Predator-Prey Holling Type II (Ecology)",
        "citation":"Murray (2003) Math Bio II | Holling (1959) Can Entomol",
        "alpha_pp":1.0, "beta_pp":0.1, "gamma_pp":0.5, "delta_pp":0.25,
        "D1":1e-3, "D2":1e-2, "eps_diff":1e-3, "Lambda":0.35,
        "u_min":0.0, "u_max":0.5, "v_min":0.0, "v_max":0.4,
        "u_star":0.1, "v_star":0.18,
        "domain":(0.0,1.0), "T":2.0, "Nx":128, "Nt":200, "dt":0.01,
        "bc_type":"neumann", "beta_mesh":2.0,
        "tbptt":1, "n_bins_2d":10,
        "anchors_uv":[(0.0,0.1),(0.1,0.18),(0.0,0.0)],
        "ics":[(0.30,0.12,1,1),(0.25,0.10,1,1),(0.20,0.08,2,2),(0.28,0.11,1,2),
               (0.35,0.14,1,1),(0.22,0.09,2,1),(0.18,0.07,1,2),(0.32,0.13,2,2)],
        "input_dim":5,
    },
}

def get_R_true(system, u, v, cfg):
    if system == "fhn_partial":
        return (1.0/cfg["eps_diff"])*u*(u-cfg["a_fhn"])*(1.0-u) - v
    if system == "predator_prey":
        alp=cfg["alpha_pp"]; bet=cfg["beta_pp"]
        return np.clip(u,0,None)*(1-np.clip(u,0,None)) - \
               alp*np.clip(u,0,None)*np.clip(v,0,None)/(bet+np.clip(u,0.001,None))
    raise ValueError(f"Unknown system: {system}")

def build_laplacian(cfg):
    xl,xr=cfg["domain"]
    x,_=build_shishkin_mesh(cfg["Nx"],xl,xr,cfg["eps_diff"],cfg["beta_mesh"])
    return x, build_compact_laplacian(x,periodic=False)

def make_imex(D,L_np,cfg,neumann=True,device=None):
    dt=cfg["dt"]; N=cfg["Nx"]
    A=np.eye(N+1)-(dt/2)*D*L_np
    if neumann:
        A[0,:]=0; A[0,0]=1; A[0,1]=-1
        A[-1,:]=0; A[-1,-1]=1; A[-1,-2]=-1
    else:
        A[0,:]=0; A[0,0]=1; A[-1,:]=0; A[-1,-1]=1
    lu,piv=scipy.linalg.lu_factor(A)
    return torch.tensor(A,dtype=torch.float32,device=device),lu,piv

def make_ics(system, x, cfg):
    if system == "fhn_partial":
        eps=cfg["eps_diff"]; ics=[]
        for (kind,loc,v0) in cfg["ics"]:
            u0=0.5*(1+np.tanh((x-loc)/np.sqrt(eps))) if kind=="front" else np.where(x<loc,0.9,0.05)
            ics.append((u0.copy(), v0*np.ones_like(x)))
        return ics
    if system == "predator_prey":
        us=cfg["u_star"]; vs=cfg["v_star"]; ics=[]
        for (au,av,mu,mv) in cfg["ics"]:
            u0=np.clip(us+au*np.cos(mu*np.pi*x),cfg["u_min"]+0.001,cfg["u_max"])
            v0=np.clip(vs+av*np.sin(mv*np.pi*x),cfg["v_min"]+0.001,cfg["v_max"])
            ics.append((u0.copy(),v0.copy()))
        return ics

def _lus(lu,piv,rhs,lo,hi):
    return np.clip(scipy.linalg.lu_solve((lu,piv),rhs),lo,hi)

def generate_reference(system,x,cfg,L_np):
    dt=cfg["dt"]; Nt=cfg["Nt"]
    t_arr=np.linspace(0,cfg["T"],Nt+1); refs=[]
    if system=="fhn_partial":
        eps=cfg["eps_diff"]; dv=cfg["delta_v"]; bv=cfg["beta_v"]; gv=cfg["gamma_v"]
        _,lu_u,pu=make_imex(eps,L_np,cfg,neumann=False)
        _,lu_v,pv=make_imex(dv, L_np,cfg,neumann=False)
        for u0,v0 in make_ics(system,x,cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x)))
            ur[0]=u0; vr[0]=v0
            for n in range(Nt):
                un=ur[n]; vn=vr[n]
                Ru=(1/eps)*un*(un-cfg["a_fhn"])*(1-un)-vn
                rhs_u=un+(dt/2)*eps*(L_np@un)+dt*Ru; rhs_u[0]=0.0; rhs_u[-1]=1.0
                ur[n+1]=_lus(lu_u,pu,rhs_u,-0.05,1.05)
                rhs_v=vn+(dt/2)*dv*(L_np@vn)+dt*(bv*un-gv*vn); rhs_v[0]=0.0; rhs_v[-1]=0.0
                vr[n+1]=_lus(lu_v,pv,rhs_v,-0.05,1.05)
            refs.append((ur,vr))
    elif system=="predator_prey":
        D1=cfg["D1"]; D2=cfg["D2"]
        alp=cfg["alpha_pp"]; bet=cfg["beta_pp"]; gam=cfg["gamma_pp"]; dlt=cfg["delta_pp"]
        _,lu_u,pu=make_imex(D1,L_np,cfg,neumann=True)
        _,lu_v,pv=make_imex(D2,L_np,cfg,neumann=True)
        for u0,v0 in make_ics(system,x,cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x)))
            ur[0]=np.clip(u0,0.001,None); vr[0]=np.clip(v0,0.001,None)
            for n in range(Nt):
                un=np.clip(ur[n],0.001,None); vn=np.clip(vr[n],0.001,None)
                Rprey=alp*un*vn/(bet+un); Rfull=un*(1-un)-Rprey
                rhs_u=un+(dt/2)*D1*(L_np@un)+dt*Rfull; rhs_u[0]=0; rhs_u[-1]=0
                ur[n+1]=_lus(lu_u,pu,rhs_u,-0.01,0.65)
                rhs_v=vn+(dt/2)*D2*(L_np@vn)+dt*(gam*Rprey-dlt*vn); rhs_v[0]=0; rhs_v[-1]=0
                vr[n+1]=_lus(lu_v,pv,rhs_v,-0.01,0.65)
            refs.append((ur,vr))
    print(f"    {len(refs)} reference trajectories generated")
    return refs, t_arr

def extract_fd(refs,L_np,cfg,system):
    dt=cfg["dt"]; ua,va,Ra=[],[],[]
    for ur,vr in refs:
        Nt=ur.shape[0]-1
        for n in range(1,Nt-1):
            un=ur[n]; vn=vr[n]; dtu=(ur[n+1]-ur[n-1])/(2*dt)
            if system=="fhn_partial": Rfd=dtu-cfg["eps_diff"]*(L_np@un)
            elif system=="predator_prey": Rfd=dtu-cfg["D1"]*(L_np@un)
            lap=np.abs(L_np@un); mask=lap<np.percentile(lap,50)
            mask[0]=False; mask[-1]=False
            if mask.sum()>0: ua.append(un[mask]); va.append(vn[mask]); Ra.append(Rfd[mask])
    return np.concatenate(ua),np.concatenate(va),np.concatenate(Ra)

def aggregate_fd(u_fd,v_fd,R_fd,cfg):
    nb=cfg["n_bins_2d"]
    ue=np.linspace(cfg["u_min"],cfg["u_max"],nb+1)
    ve=np.linspace(cfg["v_min"],cfg["v_max"],nb+1)
    uc=0.5*(ue[:-1]+ue[1:]); vc=0.5*(ve[:-1]+ve[1:])
    ui=np.clip(np.digitize(u_fd,ue)-1,0,nb-1)
    vi=np.clip(np.digitize(v_fd,ve)-1,0,nb-1)
    u_t,v_t,R_t=[],[],[]
    for i in range(nb):
        for j in range(nb):
            m=(ui==i)&(vi==j)
            if m.sum()>=4: u_t.append(uc[i]); v_t.append(vc[j]); R_t.append(np.median(R_fd[m]))
    print(f"    FD bins: {len(u_t)}")
    return (np.array(u_t,dtype=np.float32), np.array(v_t,dtype=np.float32),
            np.array(R_t,dtype=np.float32))

def evaluate_l2(model_fn, system, cfg, u_fd=None, v_fd=None):
    ng=80
    ug=np.linspace(cfg["u_min"],cfg["u_max"],ng)
    vg=np.linspace(cfg["v_min"],cfg["v_max"],ng)
    UU,VV=np.meshgrid(ug,vg); uf=UU.ravel(); vf=VV.ravel()
    Rp=model_fn(uf,vf); Rt=get_R_true(system,uf,vf,cfg)
    l2=float(np.sqrt(np.mean((Rp-Rt)**2))/(np.sqrt(np.mean(Rt**2))+1e-8))
    l2_in=l2
    if u_fd is not None and len(u_fd)>0:
        Rp2=model_fn(u_fd,v_fd); Rt2=get_R_true(system,u_fd,v_fd,cfg)
        l2_in=float(np.sqrt(np.mean((Rp2-Rt2)**2))/(np.sqrt(np.mean(Rt2**2))+1e-8))
    return l2,l2_in,UU,VV,Rp.reshape(ng,ng),Rt.reshape(ng,ng)

def regime_indicator(dhu,eps):
    l=torch.log(torch.abs(dhu)+1e-12); mu=torch.median(l)
    return torch.sigmoid(3.0*(l-mu)/(float(np.log(1/(eps+1e-12)))+1e-8))

def layer_envelope(x_t,xl,xr,eps):
    # Keep all tensors on the same device as the spatial grid (CPU/GPU).
    xr_t=torch.as_tensor(xr,dtype=x_t.dtype,device=x_t.device)
    xl_t=torch.as_tensor(xl,dtype=x_t.dtype,device=x_t.device)
    return torch.exp(-torch.minimum(x_t-xl_t,xr_t-x_t)/(eps**0.5+1e-12))

def _neu(r): r=r.clone(); r[0]=0.0; r[-1]=0.0; return r
def _dir(r,gL,gR): r=r.clone(); r[0]=gL; r[-1]=gR; return r

TAU=0.05

def l_anch_fn(model,cfg):
    pts=cfg["anchors_uv"]
    dev=next(model.parameters()).device
    pu=torch.tensor([p[0] for p in pts],dtype=torch.float32,device=dev)
    pv=torch.tensor([p[1] for p in pts],dtype=torch.float32,device=dev)
    return model.react_grad(pu,pv).pow(2).mean()

def l_fd_fn(model,u_t,v_t,R_t,Lambda):
    return (model.react_grad(u_t,v_t)-R_t).pow(2).mean()/(Lambda**2+1e-8)

def l_cons_pair_fn(u0,v0,R0,u1,v1,R1):
    du=torch.sqrt((u1-u0).pow(2)+(v1-v0).pow(2)+1e-12)
    w=(1.0-du/TAU).clamp(0.0,1.0).pow(2)
    return (w*(R1-R0).pow(2)).mean()

def imex_step_shared(system,u_n,v_ref_n,u_prev,react_fn,x_t,phi_t,A_u_t,L_t,cfg):
    eps=cfg["eps_diff"]; dt=cfg["dt"]
    with torch.no_grad(): dhu=L_t@u_n
    rho=regime_indicator(dhu,eps)
    dt_u=(u_n.detach()-u_prev.detach())/(dt+1e-12)
    z_out=torch.stack([u_n,v_ref_n,dhu,dt_u,x_t],dim=-1)
    z_lay=torch.stack([u_n,v_ref_n,eps*dhu,dt_u,phi_t],dim=-1)
    R=react_fn(u_n,v_ref_n,z_out,z_lay,rho)
    if system=="fhn_partial": rhs=_dir(u_n+(dt/2)*eps*(L_t@u_n)+dt*R,0.0,1.0)
    elif system=="predator_prey": rhs=_neu(u_n+(dt/2)*cfg["D1"]*(L_t@u_n)+dt*R)
    u_next=torch.linalg.solve(A_u_t,rhs.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(u_next,cfg["u_min"]-0.05,cfg["u_max"]+0.05),R

def rollout_nograd_shared(system,model_forward,u0,ur_t,vr_t,x_t,phi_t,A_u_t,L_t,cfg):
    Nt=cfg["Nt"]; u_n=u0.detach(); u_prev=u0.detach(); u_traj=[u_n]
    for step in range(Nt):
        v_ref_n=vr_t[step].detach()
        with torch.no_grad():
            u_next,_=imex_step_shared(system,u_n,v_ref_n,u_prev,model_forward,x_t,phi_t,A_u_t,L_t,cfg)
        if not torch.isfinite(u_next).all():
            return ur_t.detach().cpu().numpy()
        u_traj.append(u_next.detach()); u_prev=u_n; u_n=u_next
    return torch.stack(u_traj).detach().cpu().numpy()


def count_parameters(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


class SolutionNet(nn.Module):
    def __init__(self, width=56, depth=5, domain=None, T=None):
        super().__init__()
        self.xl, self.xr = (domain if domain is not None else (0.0, 1.0))
        self.T = float(T) if T is not None else 1.0
        self.normalise = domain is not None and T is not None
        layers = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, v, traj_id=None):
        if traj_id is None:
            traj_id = torch.zeros_like(x)
        if self.normalise:
            x = 2.0 * (x - self.xl) / (self.xr - self.xl + 1e-12) - 1.0
            t = 2.0 * t / (self.T + 1e-12) - 1.0
        return self.net(torch.stack([x, t, v, traj_id], -1)).squeeze(-1)


class ReactionNet(nn.Module):
    """Shared coupled reaction law R_theta(u,v), matching scalar baseline."""
    def __init__(self, width=64, depth=3, Lambda=2.0):
        super().__init__()
        self.Lambda = float(Lambda)
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u, v):
        return self.Lambda * torch.tanh(
            self.net(torch.stack([u, v], -1))
        ).squeeze(-1)

    def reaction_nograd(self, u, v):
        with torch.no_grad():
            return self.forward(u, v)


def _assemble_system(refs, t_arr, x, cfg, device, max_snaps=60,
                      state_scale=None, traj_indices=None, n_refs_total=None):
    Nt = cfg["Nt"]
    snap = np.linspace(0, Nt, min(Nt + 1, max_snaps), dtype=int)
    xs, ts, vs, us, cs = [], [], [], [], []
    n_refs = len(refs)
    if traj_indices is None:
        traj_indices = list(range(n_refs))
    if len(traj_indices) != n_refs:
        raise ValueError("traj_indices must have the same length as refs")
    if n_refs_total is None:
        n_refs_total = (max(traj_indices) + 1) if traj_indices else 1
    n_refs_total = max(1, int(n_refs_total))

    for j, (ur, vr) in enumerate(refs):
        global_j = int(traj_indices[j])
        cid = 0.0 if n_refs_total == 1 else global_j / (n_refs_total - 1)
        for n in snap:
            xs.append(x)
            ts.append(np.full_like(x, t_arr[n]))
            vs.append(vr[n])
            us.append(ur[n])
            cs.append(np.full_like(x, cid, dtype=float))

    def cat(a):
        return torch.tensor(np.concatenate(a), dtype=torch.float32, device=device)

    if state_scale is None:
        state_scale = float(
            np.mean(np.abs(np.concatenate([r[0].flatten() for r in refs])))
            + 1e-8
        )
    return dict(
        xd=cat(xs), td=cat(ts), vd=cat(vs), ud=cat(us), cid=cat(cs),
        state_scale=float(state_scale), n_refs_total=n_refs_total
    )


def residual_autodiff_system(u_net, R_net, x, t, v, traj_id, D):
    """Coupled PINN residual for the observed-u equation."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_net(x, t, v, traj_id)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True
    )[0]
    return u_t - D * u_xx - R_net(u, v)


def residual_fd_system(u_net, R_net, x_t, time_idx, t_arr, L_t, dt, D,
                       refs, traj_ids):
    res = []
    for j, cid_value in enumerate(traj_ids):
        c = torch.full_like(x_t, float(cid_value))
        vr = refs[j][1]
        for k in range(1, len(time_idx) - 1):
            im = int(time_idx[k - 1])
            ic = int(time_idx[k])
            ip = int(time_idx[k + 1])
            tp = torch.full_like(x_t, float(t_arr[im]))
            tc = torch.full_like(x_t, float(t_arr[ic]))
            tn = torch.full_like(x_t, float(t_arr[ip]))
            vp = torch.tensor(vr[im], dtype=torch.float32, device=x_t.device)
            vc = torch.tensor(vr[ic], dtype=torch.float32, device=x_t.device)
            vn = torch.tensor(vr[ip], dtype=torch.float32, device=x_t.device)

            up = u_net(x_t, tp, vp, c)
            uc = u_net(x_t, tc, vc, c)
            un = u_net(x_t, tn, vn, c)
            R = R_net(uc, vc)
            # t[ic]-t[im] = (ic-im)*dt; central difference denominator
            # is 2*(t[ic]-t[im]).
            g = (un - up) / (2.0 * dt * (ic - im)) \
                - D * (L_t @ uc) - R
            # Interior spatial points only.
            res.append(g[1:-1])
    return torch.cat(res)


def train_pinn_family_system(method, cfg, refs, t_arr, x, L_t, epochs,
                             device=None, w_f=1.0, lr=1e-3, wd=0.0,
                             seed=42, state_scale=None,
                             traj_indices=None, n_refs_total=None,
                             verbose=True):
    device = device or torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    D = cfg["D1"] if "D1" in cfg else cfg["eps_diff"]
    res_scale = float(cfg["Lambda"])
    S = _assemble_system(refs, t_arr, x, cfg, device,
                          state_scale=state_scale,
                          traj_indices=traj_indices,
                          n_refs_total=n_refs_total)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    u_net = SolutionNet(domain=cfg["domain"], T=cfg["T"]).to(device)
    R_net = ReactionNet(Lambda=cfg["Lambda"]).to(device)
    params = list(u_net.parameters()) + list(R_net.parameters())
    n_params = count_parameters(u_net, R_net)

    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5
    )

    stride = max(1, cfg["Nt"] // 40)
    time_idx = np.arange(0, cfg["Nt"] + 1, stride, dtype=int)
    if time_idx[-1] != cfg["Nt"]:
        time_idx = np.append(time_idx, cfg["Nt"])
    dt_fd = float(t_arr[1] - t_arr[0])

    history = []
    best = float("inf")
    best_state = None

    if verbose:
        print(f"    {method.upper()} ({n_params} params, {epochs} ep, "
              f"w_f={w_f:g})")

    if traj_indices is None:
        traj_indices = list(range(len(refs)))
    total_refs = (n_refs_total if n_refs_total is not None
                  else (max(traj_indices) + 1 if traj_indices else 1))
    traj_ids = [0.0 if total_refs == 1 else int(j) / (total_refs - 1)
                for j in traj_indices]

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        up = u_net(S["xd"], S["td"], S["vd"], S["cid"])
        mse_u = ((up - S["ud"]) / S["state_scale"]).pow(2).mean()

        if method == "pinn":
            f = residual_autodiff_system(
                u_net, R_net, S["xd"], S["td"], S["vd"], S["cid"], D
            )
        else:
            f = residual_fd_system(
                u_net, R_net, x_t, time_idx, t_arr, L_t, dt_fd, D,
                refs, traj_ids
            )

        mse_f = (f / res_scale).pow(2).mean()
        loss = mse_u + w_f * mse_f
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        dm = float(mse_u.item())
        history.append({
            "epoch": ep,
            "total": float(loss.item()),
            "data_misfit": dm,
            "residual": float(mse_f.item())
        })

        if dm < best:
            best = dm
            best_state = (
                {k: v.detach().clone() for k, v in u_net.state_dict().items()},
                {k: v.detach().clone() for k, v in R_net.state_dict().items()}
            )

        if verbose and (ep % 250 == 0 or ep == 1):
            print(f"      ep {ep}/{epochs}  MSE_u={dm:.5f}  "
                  f"MSE_f={mse_f.item():.5f}")

    if best_state is not None:
        u_net.load_state_dict(best_state[0])
        R_net.load_state_dict(best_state[1])

    return u_net, R_net, history, n_params


def run_pinn_system(cfg, refs, t_arr, x, L_t, epochs, **kw):
    return train_pinn_family_system(
        "pinn", cfg, refs, t_arr, x, L_t, epochs, **kw
    )


def run_fdpinn_system(cfg, refs, t_arr, x, L_t, epochs, **kw):
    return train_pinn_family_system(
        "fdpinn", cfg, refs, t_arr, x, L_t, epochs, **kw
    )


# ── PI-RNN: corrected scalar-baseline analogue ────────────────────────

class ZhengSystemRNN(nn.Module):
    def __init__(self, n_state, hidden=32):
        super().__init__()
        self.n_state = n_state
        self.hidden = hidden
        self.cell = nn.RNNCell(2 * n_state, hidden, nonlinearity="tanh")
        self.dec = nn.Linear(hidden, n_state)
        # Unit-scale/default initialization is used instead of the tiny
        # std=0.01 initialization because the present problems use 60-200
        # time steps and the incremental update must be trainable.
        nn.init.zeros_(self.dec.bias)

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden, device=device)

    def step(self, u_n, v_n, h, dt):
        inp = torch.cat([u_n, v_n]).unsqueeze(0)
        h = self.cell(inp, h)
        return u_n + dt * self.dec(h).squeeze(0), h


def train_pirnn_system(cfg, refs, t_arr, x, L_t, epochs, L_np_res=None,
                       seed=42, device=None, w_G=1e-3, lr=1e-2,
                       wd=0.0, state_scale=None, window=None, verbose=True):
    device = device or torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    D = cfg["D1"] if "D1" in cfg else cfg["eps_diff"]
    dt = float(cfg["dt"]); Nt = cfg["Nt"]
    n_state = len(x)
    if L_np_res is None:
        L_np_res = L_t.detach().cpu().numpy()
    _rs = []
    for _ur, _vr in refs:
        for _n in range(1, cfg["Nt"]):
            _rs.append(((_ur[_n + 1] - _ur[_n - 1]) / (2.0 * float(cfg["dt"]))
                        - D * (L_np_res @ _ur[_n]))[1:-1])
    res_scale = (float(np.sqrt((np.concatenate(_rs) ** 2).mean())) + 1e-12
                 if _rs else float(cfg["Lambda"]))
    window = max(3, int(window or cfg.get("tbptt", 15)))
    warm = int(0.25 * epochs)
    ramp_len = max(1, int(0.15 * epochs))

    u_refs = [
        (torch.tensor(ur, dtype=torch.float32, device=device),
         torch.tensor(vr, dtype=torch.float32, device=device))
        for ur, vr in refs
    ]
    if state_scale is None:
        state_scale = float(np.mean(
            np.abs(np.concatenate([ur.flatten() for ur, _ in refs]))
        ) + 1e-8)

    rnn = ZhengSystemRNN(n_state, hidden=32).to(device)
    R_net = ReactionNet(Lambda=cfg["Lambda"]).to(device)
    params = list(rnn.parameters()) + list(R_net.parameters())
    n_params = count_parameters(rnn, R_net)

    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5
    )

    history = []
    best = float("inf")
    best_state = None

    if verbose:
        print(f"    PI-RNN ({n_params} params, {epochs} ep, "
              f"w_G={w_G:g}, window={window})")

    for ep in range(1, epochs + 1):
        w_G_ep = w_G * (0.0 if ep <= warm
                        else min(1.0, (ep - warm) / ramp_len))
        opt.zero_grad()
        total_se = 0.0
        total_count = 0
        total_loss = 0.0
        n_windows = 0

        for ur, vr in u_refs:
            h = rnn.init_hidden(device)
            u_n = ur[0]
            n = 0

            while n < Nt:
                w_end = min(n + window, Nt)
                u_n = u_n.detach()
                h = h.detach()

                seq = [u_n]
                for s in range(n, w_end):
                    u_n, h = rnn.step(u_n, vr[s], h, dt)
                    seq.append(u_n)
                U = torch.stack(seq)

                diff = U - ur[n:w_end + 1]
                se = (diff / (state_scale + 1e-8)).pow(2).sum()
                total_se += float(se.detach())
                total_count += U.numel()
                mse_X = se / U.numel()

                if U.shape[0] >= 3:
                    Uc = U[1:-1]
                    Vc = vr[n + 1:w_end]
                    dU = (U[2:] - U[:-2]) / (2.0 * dt)
                    lap = Uc @ L_t.T
                    R = R_net(Uc.reshape(-1), Vc.reshape(-1)).reshape(Uc.shape)
                    g = dU - D * lap - R
                    mse_G = (g[:, 1:-1] / res_scale).pow(2).mean()
                else:
                    mse_G = torch.zeros((), device=device)

                Lw = mse_X + w_G_ep * mse_G
                Lw.backward()
                total_loss += float(Lw.detach())
                n_windows += 1
                n = w_end

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        data_misfit = total_se / max(total_count, 1)
        avg_loss = total_loss / max(n_windows, 1)
        history.append({
            "epoch": ep, "total": avg_loss, "data_misfit": data_misfit
        })

        if data_misfit < best:
            best = data_misfit
            best_state = (
                {k: v.detach().clone() for k, v in rnn.state_dict().items()},
                {k: v.detach().clone() for k, v in R_net.state_dict().items()}
            )

        if verbose and (ep % 250 == 0 or ep == 1):
            print(f"      ep {ep}/{epochs}  MSE_X={data_misfit:.5f} "
                  f"total={avg_loss:.5f}")

    if best_state is not None:
        rnn.load_state_dict(best_state[0])
        R_net.load_state_dict(best_state[1])

    return rnn, R_net, history, n_params


@torch.no_grad()
def pirnn_system_rollout(rnn, u0_np, v_ref_np, cfg, device):
    """Roll out PI-RNN using the observed v trajectory."""
    u_n = torch.tensor(u0_np, dtype=torch.float32, device=device)
    v_ref = torch.tensor(v_ref_np, dtype=torch.float32, device=device)
    h = rnn.init_hidden(device)
    traj = [u_n.cpu().numpy()]
    for n in range(cfg["Nt"]):
        u_n, h = rnn.step(u_n, v_ref[n], h, cfg["dt"])
        traj.append(u_n.cpu().numpy())
    return np.stack(traj)


@torch.no_grad()
def reaction_grid(model, cfg, device, ng=80):
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg)
    uf = torch.tensor(UU.ravel(), dtype=torch.float32, device=device)
    vf = torch.tensor(VV.ravel(), dtype=torch.float32, device=device)
    Rp = model.reaction_nograd(uf, vf).cpu().numpy().reshape(ng, ng)
    Rt = get_R_true(cfg["name_key"], UU.ravel(), VV.ravel(), cfg).reshape(ng, ng)
    return UU, VV, Rp, Rt


def _react_nograd(model, u, v):
    """Evaluate the identified state-only law on whichever API the model has.

    PRE-EXISTING BUG.  reaction_errors() called model.reaction_nograd(),
    which the baseline ReactionNet defines but the proposed model does
    not -- it exposes react_nograd().  Every coupled run therefore
    crashed with AttributeError before reporting a single number for the
    proposed method.  Resolved here rather than renaming either class.
    """
    fn = getattr(model, "reaction_nograd", None) or getattr(model, "react_nograd")
    with torch.no_grad():
        return fn(u, v)


def reaction_errors(model, system, cfg, u_fd, v_fd, device):
    ng = 80
    ug = np.linspace(cfg["u_min"], cfg["u_max"], ng)
    vg = np.linspace(cfg["v_min"], cfg["v_max"], ng)
    UU, VV = np.meshgrid(ug, vg)
    uf = UU.ravel(); vf = VV.ravel()
    Rp = _react_nograd(
        model,
        torch.tensor(uf, dtype=torch.float32, device=device),
        torch.tensor(vf, dtype=torch.float32, device=device)).cpu().numpy()
    Rt = get_R_true(system, uf, vf, cfg)
    l2 = float(np.sqrt(np.mean((Rp-Rt)**2)) /
               (np.sqrt(np.mean(Rt**2)) + 1e-8))
    if len(u_fd):
        with torch.no_grad():
            Ri = _react_nograd(
                model,
                torch.tensor(u_fd, dtype=torch.float32, device=device),
                torch.tensor(v_fd, dtype=torch.float32, device=device)
            ).cpu().numpy()
        Rti = get_R_true(system, u_fd, v_fd, cfg)
        l2_in = float(np.sqrt(np.mean((Ri-Rti)**2)) /
                      (np.sqrt(np.mean(Rti**2)) + 1e-8))
    else:
        l2_in = l2
    return l2, l2_in, UU, VV, Rp.reshape(ng, ng), Rt.reshape(ng, ng)


def reaction_from_pinn(u_net, R_net, refs, t_arr, x, cfg, device):
    """A-posteriori PINN reaction law is simply R_net(u,v)."""
    return R_net


def rollout_reaction_model(system, model, u0_np, vr_ref, x_t, L_t, A_u_t, cfg):
    """Common evaluation rollout for PINN/FDPINN reaction laws only."""
    u_n = torch.tensor(u0_np, dtype=torch.float32, device=x_t.device)
    v_ref = torch.tensor(vr_ref, dtype=torch.float32, device=x_t.device)
    traj = [u_n.cpu().numpy()]
    with torch.no_grad():
        for n in range(cfg["Nt"]):
            R = model.reaction_nograd(u_n, v_ref[n])
            rhs = u_n + (cfg["dt"]/2) * (
                (cfg["D1"] if system == "predator_prey" else cfg["eps_diff"])
                * (L_t @ u_n)
            ) + cfg["dt"] * R
            if cfg["bc_type"] == "neumann":
                rhs = rhs.clone()
                rhs[0] = 0.0; rhs[-1] = 0.0
            else:
                rhs = rhs.clone()
                rhs[0] = 0.0; rhs[-1] = 1.0
            u_n = torch.linalg.solve(A_u_t, rhs.unsqueeze(-1)).squeeze(-1)
            u_n = torch.clamp(u_n, cfg["u_min"], cfg["u_max"])
            traj.append(u_n.cpu().numpy())
    return np.stack(traj)


# ── METHOD 4: PROPOSED (BRIDGE) ──────────────────────────────────────

class ProposedModel(nn.Module):
    def __init__(self,hidden=HIDDEN,mlp_w=MLP_W,Lambda=2.0,input_dim=5):
        super().__init__()
        self.hidden=hidden; self.mlp_w=mlp_w; self.Lambda=Lambda
        self.gru_out=nn.GRUCell(input_dim,hidden); self.gru_lay=nn.GRUCell(input_dim,hidden)
        self.gate=nn.Linear(hidden,mlp_w)
        self.mlp1=nn.Linear(2,mlp_w); self.mlp2=nn.Linear(mlp_w,mlp_w); self.mlp3=nn.Linear(mlp_w,1)
        nn.init.normal_(self.gate.weight,std=0.01); nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.mlp1.weight,std=0.10); nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.mlp3.weight,std=0.01); nn.init.zeros_(self.mlp3.bias)
    def init_hidden(self,Nx,device=None):
        device=device or next(self.parameters()).device
        h=torch.zeros(Nx,self.hidden,device=device)
        return h,h.clone()
    def _decode(self,u,v,gamma):
        inp=torch.stack([u,v],dim=-1); h1=F.silu(self.mlp1(inp)); h2=F.silu(h1+gamma); h3=F.silu(self.mlp2(h2))
        return self.Lambda*torch.tanh(self.mlp3(h3)).squeeze(-1)
    def forward_dual(self,u,v,z_out,z_lay,rho,H_out,H_lay):
        rho_e=rho.unsqueeze(-1)
        Hoc=self.gru_out(torch.clamp(z_out,-10,10),H_out); Hlc=self.gru_lay(torch.clamp(z_lay,-10,10),H_lay)
        Hon=(1-rho_e)*Hoc+rho_e*H_out; Hln=rho_e*Hlc+(1-rho_e)*H_lay
        Hb=(1-rho_e)*Hon+rho_e*Hln; gamma=self.gate(Hb)
        return self._decode(u,v,gamma),Hon,Hln
    def react_grad(self,u,v):
        gamma=torch.zeros(u.shape[0],self.mlp_w,
                          dtype=u.dtype,device=u.device)
        return self._decode(u,v,gamma)
    def react_nograd(self,u,v):
        with torch.no_grad(): return self.react_grad(u,v)

def train_proposed(system,model,cfg,refs,A_u_t,L_t,x_t,phi_t,
                   u_fd_t,v_fd_t,R_fd_t,epochs=EPOCHS,LR=LR,WD=WD,SEED=SEED,
                   state_scale=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"  [Proposed] {epochs} epochs...")
    dev=x_t.device
    su=(float(state_scale) if state_scale is not None
        else max(float(np.mean([np.abs(r[0]).mean() for r in refs])),0.01))
    su=max(su,0.01); Lambda=cfg["Lambda"]
    for n,p in model.named_parameters():
        if "gru" in n or "gate" in n: p.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    if params:
        pi_opt=optim.Adam(params,lr=1e-3); best_s0=float("inf"); best_s0_state=None
        for step in range(2000):
            pi_opt.zero_grad()
            loss=(model.react_grad(u_fd_t,v_fd_t)-R_fd_t).pow(2).mean()/(Lambda**2+1e-8)
            loss.backward(); pi_opt.step()
            if loss.item()<best_s0:
                best_s0=loss.item(); best_s0_state={k:v.clone() for k,v in model.state_dict().items()}
        if best_s0_state: model.load_state_dict(best_s0_state)
        print(f"    Stage 0 done. Best FD loss={best_s0:.5f}")
    for p in model.parameters(): p.requires_grad_(True)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=400,eta_min=1e-6)
    history=[]; best_l=float("inf"); best_state=None; nan_c=0; n_gskip=0
    def react_fn(u,v,z_out,z_lay,rho):
        R,Ho,Hl=model.forward_dual(u,v,z_out,z_lay,rho,model._Ho,model._Hl)
        model._Ho=Ho; model._Hl=Hl; return R
    print(f"  {'Ep':>5}  {'Total':>8}  {'Data':>8}  {'FD':>8}  {'Cons':>8}  {'Anch':>8}")
    print("  "+"-"*52)
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad()
        ep=ep_d=ep_f=ep_c=ep_a=0.0; f=min((epoch-1)/(epochs-1+1e-8),1.0)
        lam_d=0.5+0.5*f; lam_f=4.0+1.0*f; lam_c=0.01+0.99*f; lam_a=1.0+2.0*f
        for ur_np,vr_np in refs:
            ur_t=torch.tensor(ur_np,dtype=torch.float32,device=dev); vr_t=torch.tensor(vr_np,dtype=torch.float32,device=dev)
            H_out,H_lay=model.init_hidden(x_t.shape[0],device=dev); u_n=ur_t[0]; u_prev=ur_t[0]
            R_prev=None
            for step in range(cfg["Nt"]):
                H_out=H_out.detach(); H_lay=H_lay.detach(); u_n=u_n.detach(); u_prev=u_prev.detach()
                if R_prev is not None: R_prev=R_prev.detach()
                model._Ho=H_out; model._Hl=H_lay; v_ref_n=vr_t[step].detach()
                u_next,R=imex_step_shared(system,u_n,v_ref_n,u_prev,react_fn,x_t,phi_t,A_u_t,L_t,cfg)
                H_out=model._Ho; H_lay=model._Hl
                Ld=((u_next-ur_t[step+1])/(su+1e-8)).pow(2).mean()
                if R_prev is None:
                    Lc=torch.zeros((),device=dev)
                else:
                    Lc=l_cons_pair_fn(u_n,vr_t[step],R_prev,
                                      u_next,vr_t[step+1],R)
                La=l_anch_fn(model,cfg); Lf=l_fd_fn(model,u_fd_t,v_fd_t,R_fd_t,Lambda)
                Lw=lam_d*Ld+lam_f*Lf+lam_c*Lc+lam_a*La
                if not torch.isfinite(Lw):
                    R_prev=R.detach() if R is not None else None
                    u_prev=u_n; u_n=u_next.detach()
                    continue
                (Lw/cfg["Nt"]/len(refs)).backward()
                ep+=Lw.item()/cfg["Nt"]/len(refs); ep_d+=Ld.item()/cfg["Nt"]/len(refs)
                ep_f+=Lf.item()/cfg["Nt"]/len(refs); ep_c+=Lc.item()/cfg["Nt"]/len(refs)
                ep_a+=La.item()/cfg["Nt"]/len(refs); R_prev=R; u_prev=u_n; u_n=u_next
        if not np.isfinite(ep):
            nan_c+=1; print(f"  NaN ({nan_c}/5)")
            if best_state: model.load_state_dict(best_state)
            for pg in opt.param_groups: pg["lr"]*=0.5
            if nan_c>=5: break
            continue
        nan_c=0
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        if not torch.isfinite(gn):
            n_gskip += 1
            if n_gskip in (1, 10, 100) or n_gskip % 500 == 0:
                print(f"    ! epoch {epoch}: loss is finite but the "
                      f"gradient is not; update skipped "
                      f"({n_gskip} so far)")
            opt.zero_grad(set_to_none=True)
            for pg in opt.param_groups: pg["lr"] *= 0.7
            sched.step()
            continue
        opt.step(); sched.step()
        history.append({"epoch":epoch,"total":ep,"data":ep_d,
                        "data_misfit":ep_d,
                        "fd":ep_f,"cons":ep_c,"anch":ep_a})
        unw=ep_d+ep_f
        if np.isfinite(unw) and unw<best_l:
            best_l=unw; best_state={k:v.clone() for k,v in model.state_dict().items()}
        if epoch%100==0 or epoch<=3:
            print(f"  {epoch:>5}  {ep:>8.5f}  {ep_d:>8.5f}  {ep_f:>8.5f}  {ep_c:>8.5f}  {ep_a:>8.5f}")
    if best_state: model.load_state_dict(best_state)
    print(f"  Best={best_l:.5f}   logged {len(history)}/{epochs} epochs"
          + (f"   ({n_gskip} skipped on non-finite gradients)" if n_gskip else ""))
    if not history:
        print("    *** train_proposed logged NOTHING: every epoch was "
              "skipped. The proposed method will be absent from every "
              "figure. Do not report this run. ***")
    return history


# ── PLOTS ─────────────────────────────────────────────────────────────

def plot_solution_accuracy_combined(all_res, system, cfg, t_arr, out_dir):
    try:
        import scipy.stats as st
        has_scipy = True
    except ImportError:
        has_scipy = False

    n_m = len(METHODS)
    fig, axes = plt.subplots(3, n_m, figsize=(5.5 * n_m, 14))
    fig.suptitle(f"{cfg['name']} — Solution Accuracy Comparison: All Methods",
                 fontsize=14, fontweight="bold")

    Nt1 = len(t_arr)
    quart_labels = ["t∈[0,T/4]", "t∈[T/4,T/2]", "t∈[T/2,3T/4]", "t∈[3T/4,T]"]

    for col, m in enumerate(METHODS):
        d = all_res[m]
        color = STYLES[m]["color"]
        u_pred = d.get("u_pred_traj")
        u_ref  = d.get("u_ref_traj")
        is_best = (m == "Proposed")

        if u_pred is None or u_ref is None:
            for row in range(3):
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=14)
            axes[0, col].set_title(m, fontsize=12,
                                   fontweight="bold" if is_best else "normal",
                                   color=color)
            continue

        u_pred = np.array(u_pred); u_ref = np.array(u_ref)
        abs_err = np.abs(u_pred - u_ref)        # (Nt+1, Nx)
        rel_l2_t = (np.sqrt(np.mean(abs_err**2, axis=1)) /
                    (np.sqrt(np.mean(u_ref**2,  axis=1)) + 1e-8))  # (Nt+1,)

        def _highlight(ax):
            if is_best:
                for sp in ax.spines.values():
                    sp.set_edgecolor("#C0392B"); sp.set_linewidth(1.8)

        # ── Row 0: Violin by time quartile ──────────────────────────
        ax = axes[0, col]
        quarts = [(0, Nt1//4), (Nt1//4, Nt1//2),
                  (Nt1//2, 3*Nt1//4), (3*Nt1//4, Nt1)]
        data_q = []
        for (a, b) in quarts:
            vals = abs_err[a:b, :].ravel()
            vals = vals[vals > 1e-14]
            data_q.append(vals if len(vals) > 0 else np.array([1e-14]))
        parts = ax.violinplot(data_q, showmedians=False, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.65)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts: parts[key].set_color(color)
        for i, vals in enumerate(data_q):
            ax.hlines(np.median(vals), i+0.75, i+1.25, color="red", lw=2.0, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(quart_labels, fontsize=8)
        ax.set_title(m, fontsize=12, fontweight="bold" if is_best else "normal",
                     color=color)
        if col == 0: ax.set_ylabel("|u_pred − u_ref|", fontsize=10)
        ax.grid(True, alpha=0.25); _highlight(ax)

        # ── Row 1: Temporal error evolution ─────────────────────────
        ax = axes[1, col]
        mean_err = abs_err.mean(axis=1)
        ci_lo = np.percentile(abs_err, 2.5,  axis=1)
        ci_hi = np.percentile(abs_err, 97.5, axis=1)
        ax.plot(t_arr, mean_err, color=color, lw=2.0, label="Mean |error|")
        ax.fill_between(t_arr, ci_lo, ci_hi, color=color, alpha=0.20, label="95% CI")
        ax.set_xlabel("t", fontsize=10)
        if col == 0: ax.set_ylabel("Mean Absolute Error", fontsize=10)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.25)
        _highlight(ax)

        # ── Row 2: Relative-L2 KDE ───────────────────────────────────
        ax = axes[2, col]
        rl2 = rel_l2_t[(rel_l2_t > 0) & np.isfinite(rel_l2_t)]
        mean_l2 = float(np.mean(rl2)) if len(rl2) > 0 else float("nan")
        if has_scipy and len(rl2) > 3:
            log_v = np.log10(rl2 + 1e-14)
            kde = st.gaussian_kde(log_v)
            xs = np.linspace(log_v.min(), log_v.max(), 400)
            ax.fill_between(10**xs, kde(xs), color=color, alpha=0.55, label="Rel. L²")
            ax.axvline(mean_l2, color="navy", lw=1.8, ls="--",
                       label=f"Mean={mean_l2:.4f}")
        else:
            ax.text(0.5, 0.5, f"Mean={mean_l2:.4f}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
        ax.set_xscale("log")
        ax.set_xlabel("Relative $L^2$ Error", fontsize=10)
        if col == 0: ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Mean $L^2$={mean_l2:.4f}", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25); _highlight(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, f"cmp_solution_accuracy_{system}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_reactions(all_res,system,cfg,out_dir):
    fig,axes=plt.subplots(1,5,figsize=(27,5))
    fig.suptitle(f"{cfg['name']} — Reaction Identification R_theta(u,v)",fontsize=13,fontweight="bold")
    d0=all_res["Proposed"]; vmax_global=np.abs(d0["Rt"]).max()*0.9
    im0=axes[0].contourf(d0["UU"],d0["VV"],d0["Rt"],levels=40,cmap="RdBu_r",vmin=-vmax_global,vmax=vmax_global)
    plt.colorbar(im0,ax=axes[0]); axes[0].set_title("R_true (Ground Truth)",fontsize=9,fontweight="bold")
    axes[0].set_xlabel("u"); axes[0].set_ylabel("v")
    for sp in axes[0].spines.values(): sp.set_edgecolor("black"); sp.set_linewidth(2)
    for ax,m in zip(axes[1:],METHODS):
        d=all_res[m]; UU=d["UU"]; VV=d["VV"]; Rp=d["Rp"]
        im=ax.contourf(UU,VV,Rp,levels=40,cmap="RdBu_r",vmin=-vmax_global,vmax=vmax_global)
        plt.colorbar(im,ax=ax)
        is_best=(m=="Proposed")
        ax.set_title(f"{m}\nL2={d['l2']:.4f} | L2_in={d['l2_in']:.4f}",fontsize=9,
                     fontweight="bold" if is_best else "normal")
        ax.set_xlabel("u"); ax.set_ylabel("")
        if is_best:
            for sp in ax.spines.values(): sp.set_edgecolor("#C0392B"); sp.set_linewidth(2.5)
        elif m=="FDPINN":
            for sp in ax.spines.values(): sp.set_edgecolor("#8E44AD"); sp.set_linewidth(1.5)
    plt.tight_layout()
    p=os.path.join(out_dir,f"cmp_reaction_{system}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); print(f"  Saved: {p}")

    for m_plot,tag in [("Proposed","proposed"),("FDPINN","fdpinn")]:
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        fig.suptitle(f"{cfg['name']} — R_true vs {m_plot}",fontsize=12,fontweight="bold")
        d=all_res[m_plot]; UU=d["UU"]; VV=d["VV"]; vmax=np.abs(d["Rt"]).max()*0.9
        for ax,Z,title,cmap in [(axes[0],d["Rt"],"R_true","RdBu_r"),
                                 (axes[1],d["Rp"],f"R_theta ({m_plot})","RdBu_r"),
                                 (axes[2],np.abs(d["Rp"]-d["Rt"]),"|Error|","YlOrRd")]:
            im=ax.contourf(UU,VV,Z,levels=40,cmap=cmap); plt.colorbar(im,ax=ax)
            ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_title(title,fontsize=10)
        plt.tight_layout()
        p=os.path.join(out_dir,f"cmp_reaction_{tag}_{system}.png")
        plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); print(f"  Saved: {p}")

def plot_trajectories(all_res,system,cfg,x_np,t_arr,out_dir):
    fig,ax=plt.subplots(figsize=(10,6))
    ur_ref=all_res["Proposed"]["u_ref_final"]
    ax.plot(x_np,ur_ref,**REF_STYLE,label="Reference")
    for m in METHODS:
        s=STYLES[m]
        ax.plot(x_np,all_res[m]["u_pred_final"],color=s["color"],ls=s["ls"],lw=s["lw"],label=m)
    ax.set_xlabel("x",fontsize=12); ax.set_ylabel(f"u(x, T={cfg['T']})",fontsize=12)
    ax.set_title(f"{cfg['name']} — Final-Time Trajectory Comparison",fontsize=12,fontweight="bold")
    ax.legend(fontsize=10,framealpha=0.92); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    p=os.path.join(out_dir,f"cmp_trajectory_{system}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); print(f"  Saved: {p}")

def _get_loss_curve(history):
    """Returns (epochs, weighted_total) -- the raw stored loss, ramp-weighted."""
    if not history: return [],[]
    if isinstance(history[0],dict):
        if "epoch" in history[0]: ep=[d["epoch"] for d in history]; tot=[max(d["total"],1e-10) for d in history]
        else: ep=list(range(1,len(history)+1)); tot=[max(d.get("total",d.get("loss",1e-10)),1e-10) for d in history]
    else: ep=list(range(1,len(history)+1)); tot=[max(float(v),1e-10) for v in history]
    return ep,tot

def _get_unweighted_curve(res):
    h=res["history"]
    if not h: return [],[]

    def _scalarise(seq, keys):
        """PRE-EXISTING BUG: this helper assumed PINN/FDPINN histories were
        lists of floats, but train_pinn_family_system appends dicts, so
        float(v) raised TypeError and every coupled run died at the
        plotting stage.  Accept either shape."""
        out=[]
        for v in seq:
            if isinstance(v, dict):
                x=None
                for k in keys:
                    if k in v: x=v[k]; break
                if x is None:
                    x=v.get("total", 1e-10)
            else:
                x=v
            out.append(max(float(x),1e-10))
        return out

    if res["method"]=="PINN":
        ep=[d["epoch"] if isinstance(d,dict) else i+1 for i,d in enumerate(h)]
        return ep,_scalarise(h,["data_misfit","data"])
    if res["method"]=="FDPINN":
        hu=res.get("history_unweighted",h)
        ep=[d["epoch"] if isinstance(d,dict) else i+1 for i,d in enumerate(hu)]
        return ep,_scalarise(hu,["data_misfit","data"])
    # PI-RNN / Proposed: history is a list of dicts with 'data' and 'fd' keys
    if isinstance(h[0],dict) and "data" in h[0] and "fd" in h[0]:
        ep=[d["epoch"] for d in h]
        vals=[max(d["data"]+d["fd"],1e-10) for d in h]
        return ep,vals
    return _get_loss_curve(h)

def _misfit_curve(res):
    h = res.get("history") or []
    out = [(d["epoch"], d["data_misfit"]) for d in h
           if isinstance(d, dict) and "data_misfit" in d
           and np.isfinite(d["data_misfit"])]
    if not out:
        return [], []
    return [a for a, _ in out], [max(float(b), 1e-12) for _, b in out]


def plot_misfit(all_res, system, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    missing = []
    for m in METHODS:
        s = MISFIT_STYLES[m]
        ep, vals = _misfit_curve(all_res[m])
        if not ep:
            missing.append(m)
            print(f"  ! {m} logged no 'data_misfit'; omitted from the "
                  f"misfit figure")
            continue
        ax.semilogy(ep, vals, color=s["color"], ls=s["ls"], lw=s["lw"],
                    label=m, zorder=5 if m == "Proposed" else 3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Normalised data misfit", fontsize=12)
    ax.set_title(f"{cfg['name']} — common data misfit",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.92)
    ax.grid(True, alpha=0.3, which="both")
    if missing:
        ax.text(0.02, 0.02, "no history logged: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=8, color="#C0392B")
    plt.tight_layout()
    p = os.path.join(out_dir, f"cmp_misfit_{system}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_losses(all_res,system,cfg,out_dir):
    fig,axes=plt.subplots(1,2,figsize=(16,5.5))
    fig.suptitle(f"{cfg['name']} — Training Loss Convergence",fontsize=12,fontweight="bold")

    ax=axes[0]
    missing=[]
    for m in METHODS:
        s=STYLES[m]; ep,vals=_get_unweighted_curve(all_res[m])
        if ep:
            first=max(vals[0],1e-10); norm=[max(v/first,1e-12) for v in vals]
            ax.semilogy(ep,norm,color=s["color"],ls=s["ls"],lw=s["lw"],
                        label=m,zorder=5 if m=="Proposed" else 3)
        else:
            # A method with no curve used to vanish without a word, which
            # is how a four-method figure quietly became a three-method
            # one. Say so, on the console and on the axes.
            missing.append(m)
            print(f"  ! {m} produced no loss curve (empty history) -- "
                  f"omitted from the loss figure")
    ax.set_xlabel("Epoch",fontsize=11); ax.set_ylabel("Normalised Loss (log)",fontsize=11)
    ax.set_title("All Methods — Normalised Unweighted Loss (each starts at 1.0)",fontsize=10)
    ax.legend(fontsize=9,framealpha=0.92,loc="upper right"); ax.grid(True,alpha=0.3)

    ax=axes[1]
    for m in METHODS:
        s=STYLES[m]; ep,vals=_get_unweighted_curve(all_res[m])
        if ep:
            ax.semilogy(ep,vals,color=s["color"],ls=s["ls"],lw=s["lw"],
                        label=m,zorder=5 if m=="Proposed" else 3)
    ax.set_xlabel("Epoch",fontsize=11); ax.set_ylabel("Unweighted Loss (log scale)",fontsize=11)
    ax.set_title("All Methods — Absolute Unweighted Loss",fontsize=10)
    ax.legend(fontsize=9,framealpha=0.92,loc="upper right"); ax.grid(True,alpha=0.3)
    if missing:
        for a in axes:
            a.text(0.02,0.02,"no history logged: "+", ".join(missing),
                   transform=a.transAxes,fontsize=8,color="#C0392B")

    plt.tight_layout()
    p=os.path.join(out_dir,f"cmp_loss_{system}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); print(f"  Saved: {p}")


# ── RUN ONE SYSTEM ────────────────────────────────────────────────────

def run_one_system(system, epochs, device):
    cfg = copy.deepcopy(SYSTEM_CFGS[system])
    cfg["name_key"] = system

    print(f"\n{'='*70}")
    print(f"  System: {cfg['name']}")
    print(f"  Epochs per method: {epochs}")
    print(f"  PINN/FDPINN: corrected scalar-baseline formulation")
    print(f"  PI-RNN: plain RNN + FD residual, no BRIDGE losses")
    print(f"{'='*70}")

    x, L_np = build_laplacian(cfg)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=device)

    xl, xr = cfg["domain"]
    phi_t = layer_envelope(x_t, xl, xr, cfg["eps_diff"])

    D_u = cfg["eps_diff"] if system == "fhn_partial" else cfg["D1"]
    A_u_t, _, _ = make_imex(
        D_u, L_np, cfg, neumann=(cfg["bc_type"] == "neumann"),
        device=device          # FIX 2: keep the IMEX matrix on the run device
    )

    print("Generating reference trajectories...")
    refs, t_arr = generate_reference(system, x, cfg, L_np)

    print("Extracting FD pairs for the proposed method...")
    u_fd, v_fd, R_fd = extract_fd(refs, L_np, cfg, system)
    u_t, v_t, R_t = aggregate_fd(u_fd, v_fd, R_fd, cfg)
    u_fd_t = torch.tensor(u_t, dtype=torch.float32, device=device)
    v_fd_t = torch.tensor(v_t, dtype=torch.float32, device=device)
    R_fd_t = torch.tensor(R_t, dtype=torch.float32, device=device)

    # Common state scale over all observed trajectories.
    state_scale = float(np.mean(
        np.abs(np.concatenate([r[0].flatten() for r in refs]))
    ) + 1e-8)
    all_traj_indices = list(range(len(refs)))
    print(f"  Shared state scale: {state_scale:.6f}")
    print(f"  Fixed trajectory IDs: {[j/(len(refs)-1) if len(refs)>1 else 0.0 for j in all_traj_indices]}")

    ur_ref, vr_ref = refs[0]
    all_res = {}

    # ------------------------------------------------------------------
    # PINN: scalar-baseline formulation
    # ------------------------------------------------------------------
    t0 = time.time()
    pinn_u, pinn_R, hist, npar = run_pinn_system(
        cfg, refs, t_arr, x, L_t, epochs,
        device=device, state_scale=state_scale,
        traj_indices=all_traj_indices, n_refs_total=len(refs)
    )
    l2, l2in, UU, VV, Rp, Rt = reaction_errors(
        pinn_R, system, cfg, u_fd, v_fd, device
    )
    u_pred = rollout_reaction_model(
        system, pinn_R, ur_ref[0], vr_ref, x_t, L_t, A_u_t, cfg
    )
    l2u = float(np.sqrt(np.mean((u_pred-ur_ref)**2)) /
                (np.sqrt(np.mean(ur_ref**2)) + 1e-8))
    all_res["PINN"] = dict(
        l2=l2, l2_in=l2in, UU=UU, VV=VV, Rp=Rp, Rt=Rt,
        u_pred_final=u_pred[-1], u_pred_traj=u_pred,
        u_ref_traj=ur_ref, u_ref_final=ur_ref[-1],
        history=hist, method="PINN", n_params=npar
    )
    print(f"  PINN done: eps_R={l2:.4f}, eps_u={l2u:.4e}, "
          f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # FDPINN: same scalar-baseline formulation, FD residual
    # ------------------------------------------------------------------
    t0 = time.time()
    fdp_u, fdp_R, hist, npar = run_fdpinn_system(
        cfg, refs, t_arr, x, L_t, epochs,
        device=device, state_scale=state_scale,
        traj_indices=all_traj_indices, n_refs_total=len(refs)
    )
    l2, l2in, UU, VV, Rp, Rt = reaction_errors(
        fdp_R, system, cfg, u_fd, v_fd, device
    )
    u_pred = rollout_reaction_model(
        system, fdp_R, ur_ref[0], vr_ref, x_t, L_t, A_u_t, cfg
    )
    l2u = float(np.sqrt(np.mean((u_pred-ur_ref)**2)) /
                (np.sqrt(np.mean(ur_ref**2)) + 1e-8))
    all_res["FDPINN"] = dict(
        l2=l2, l2_in=l2in, UU=UU, VV=VV, Rp=Rp, Rt=Rt,
        u_pred_final=u_pred[-1], u_pred_traj=u_pred,
        u_ref_traj=ur_ref, u_ref_final=ur_ref[-1],
        history=hist, method="FDPINN", n_params=npar
    )
    print(f"  FDPINN done: eps_R={l2:.4f}, eps_u={l2u:.4e}, "
          f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # PI-RNN: same corrected scalar-baseline methodology, coupled
    # ------------------------------------------------------------------
    t0 = time.time()
    pirnn, pirnn_R, hist, npar = train_pirnn_system(
        cfg, refs, t_arr, x, L_t, epochs, L_np_res=L_np,
        device=device, state_scale=state_scale,
        window=cfg["tbptt"]
    )
    l2, l2in, UU, VV, Rp, Rt = reaction_errors(
        pirnn_R, system, cfg, u_fd, v_fd, device
    )
    u_pred = pirnn_system_rollout(
        pirnn, ur_ref[0], vr_ref, cfg, device
    )
    l2u = float(np.sqrt(np.mean((u_pred-ur_ref)**2)) /
                (np.sqrt(np.mean(ur_ref**2)) + 1e-8))
    all_res["PI-RNN"] = dict(
        l2=l2, l2_in=l2in, UU=UU, VV=VV, Rp=Rp, Rt=Rt,
        u_pred_final=u_pred[-1], u_pred_traj=u_pred,
        u_ref_traj=ur_ref, u_ref_final=ur_ref[-1],
        history=hist, method="PI-RNN", n_params=npar
    )
    print(f"  PI-RNN done: eps_R={l2:.4f}, eps_u={l2u:.4e}, "
          f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Proposed BRIDGE: unchanged proposed methodology
    # ------------------------------------------------------------------
    model_prop = ProposedModel(Lambda=cfg["Lambda"]).to(device)
    t0 = time.time()
    hist = train_proposed(
        system, model_prop, cfg, refs, A_u_t, L_t, x_t, phi_t,
        u_fd_t, v_fd_t, R_fd_t, epochs=epochs,
        state_scale=state_scale        # FIX B: the shared normalisation
    )

    l2, l2in, UU, VV, Rp, Rt = reaction_errors(
        model_prop, system, cfg, u_fd, v_fd, device
    )

    model_prop.eval()
    model_prop._Ho, model_prop._Hl = model_prop.init_hidden(x_t.shape[0], device=device)

    def prop_react_fn(u, v, z_out, z_lay, rho):
        R, Ho, Hl = model_prop.forward_dual(
            u, v, z_out, z_lay, rho, model_prop._Ho, model_prop._Hl
        )
        model_prop._Ho = Ho
        model_prop._Hl = Hl
        return R

    # PRE-EXISTING BUG: rollout_nograd_shared indexes .detach() on its
    # state arguments, but ur_ref/vr_ref arrive here as numpy arrays.
    _u0 = torch.as_tensor(ur_ref[0], dtype=torch.float32, device=device)
    _ur = torch.as_tensor(ur_ref, dtype=torch.float32, device=device)
    _vr = torch.as_tensor(vr_ref, dtype=torch.float32, device=device)
    u_pred = rollout_nograd_shared(
        system, prop_react_fn, _u0, _ur, _vr,
        x_t, phi_t, A_u_t, L_t, cfg
    )
    u_pred = np.asarray(u_pred)
    l2u = float(np.sqrt(np.mean((u_pred-ur_ref)**2)) /
                (np.sqrt(np.mean(ur_ref**2)) + 1e-8))

    npar = count_parameters(model_prop)
    all_res["Proposed"] = dict(
        l2=l2, l2_in=l2in, UU=UU, VV=VV, Rp=Rp, Rt=Rt,
        u_pred_final=u_pred[-1], u_pred_traj=u_pred,
        u_ref_traj=ur_ref, u_ref_final=ur_ref[-1],
        history=hist, method="Proposed", n_params=npar
    )
    print(f"  Proposed done: eps_R={l2:.4f}, eps_u={l2u:.4e}, "
          f"({time.time()-t0:.0f}s)")

    # Existing figures
    plot_reactions(all_res, system, cfg, OUT_DIR)
    plot_trajectories(all_res, system, cfg, x, t_arr, OUT_DIR)
    plot_misfit(all_res, system, cfg, OUT_DIR)   # FIX C: Figure 16 quantity
    plot_losses(all_res, system, cfg, OUT_DIR)
    plot_solution_accuracy_combined(all_res, system, cfg, t_arr, OUT_DIR)

    print(f"\n{'='*66}")
    print(f"  RESULTS: {cfg['name']}")
    print(f"{'='*66}")
    print(f"  {'Method':<12} {'eps_R(full)':>12} {'eps_R(in)':>12} {'eps_u':>12}")
    print("  " + "-"*52)
    for m in METHODS:
        d = all_res[m]
        uerr = float(np.sqrt(np.mean(
            (d["u_pred_traj"]-d["u_ref_traj"])**2
        )) / (np.sqrt(np.mean(d["u_ref_traj"]**2))+1e-8))
        print(f"  {m:<12} {d['l2']:>12.4f} {d['l2_in']:>12.4f} {uerr:>12.4e}")

    save = {
        m: {
            "eps_R_full": float(d["l2"]),
            "eps_R_in": float(d["l2_in"]),
            "eps_u": float(np.sqrt(np.mean(
                (d["u_pred_traj"]-d["u_ref_traj"])**2
            )) / (np.sqrt(np.mean(d["u_ref_traj"]**2))+1e-8)),
            "n_params": int(d["n_params"])
        } for m, d in all_res.items()
    }
    with open(
        os.path.join(OUT_DIR, f"cmp_results_{system}.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(save, f, indent=2)
    return all_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", default="all",
        choices=["fhn_partial", "predator_prey", "all"]
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    systems = (
        ["fhn_partial", "predator_prey"]
        if args.system == "all" else [args.system]
    )
    print(f"\nComparison: {systems}")
    print(f"Epochs (all methods): {args.epochs}")

    all_results = {}
    for s in systems:
        all_results[s] = run_one_system(s, args.epochs, device)
    print(f"\n{'='*66}")
    print("  FINAL SUMMARY")
    print(f"{'='*66}")
    for s in systems:
        print(f"\n  {SYSTEM_CFGS[s]['name']}")
        for m in METHODS:
            d = all_results[s][m]
            uerr = float(np.sqrt(np.mean(
                (d["u_pred_traj"]-d["u_ref_traj"])**2
            )) / (np.sqrt(np.mean(d["u_ref_traj"]**2))+1e-8))
            print(f"    {m:<12} eps_R={d['l2']:.4f}  "
                  f"eps_u={uerr:.4e}  params={d['n_params']}")


if __name__ == '__main__':
    main()
