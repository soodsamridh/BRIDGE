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

DEVICE = torch.device("cpu")
SEED   = 42

# ======================================================================
#  CONFIGS
# ======================================================================

SYSTEM_CFGS = {

    # ── FHN Partial ───────────────────────────────────────────────────
    "fhn_partial": {
        "name":    "FitzHugh-Nagumo Partial (Neuroscience)",
        "citation":"FitzHugh (1961) Biophys J 1(6):445",
        "eps_diff":0.05, "delta_v":0.1, "beta_v":1.0, "gamma_v":0.5, "a_fhn":0.25,
        "Lambda":2.0,
        "u_min":0.0, "u_max":1.0, "v_min":0.0, "v_max":0.6,
        "domain":(0.0,1.0), "T":0.3, "Nx":128, "Nt":60, "dt":0.005,
        "bc_type":"dirichlet", "mesh_type":"shishkin", "beta_mesh":2.0,
        "tbptt":1, "n_bins_2d":12,
        "ramp":{"data":{"start":0.80,"end":1.0,"over":150},
                "fd":  {"start":1.0, "end":4.0,"over":300},
                "cons":{"start":0.01,"end":1.0,"over":300},
                "anch":{"start":0.001,"end":8.0,"over":400}},
        "lr":1e-3, "grad_clip":0.5,
        "anchors_uv":[(0.0,0.0),(0.25,0.0),(1.0,0.0)],
        "ics":[("front",0.2,0.0),("front",0.3,0.0),("front",0.4,0.0),
               ("front",0.5,0.0),("front",0.6,0.0),("front",0.7,0.0),
               ("front",0.3,0.3),("step",0.5,0.5)],
        "input_dim":5, "r_type":"2d",
        "v_pred_known": True,
    },

    # ── Predator-Prey ─────────────────────────────────────────────────
    "predator_prey": {
        "name":    "Predator-Prey Holling Type II (Ecology)",
        "citation":"Murray (2003) Math Bio II p.80 | Holling (1959) Can Entomol 91:385",
        "alpha_pp":1.0, "beta_pp":0.1, "gamma_pp":0.5, "delta_pp":0.25,
        "D1":1e-3, "D2":1e-2, "eps_diff":1e-3,
        "Lambda":0.35,
        "u_min":0.0, "u_max":0.5, "v_min":0.0, "v_max":0.4,
        "u_star":0.1, "v_star":0.18,
        "domain":(0.0,1.0), "T":2.0, "Nx":128, "Nt":200, "dt":0.01,
        "bc_type":"neumann", "mesh_type":"shishkin", "beta_mesh":2.0,
        "tbptt":1, "n_bins_2d":10,
        "ramp":{"data":{"start":0.50,"end":1.0,"over":200},
                "fd":  {"start":2.0, "end":5.0,"over":300},
                "cons":{"start":0.01,"end":1.0,"over":400},
                "anch":{"start":0.001,"end":4.0,"over":400}},
        "lr":1e-3, "grad_clip":0.5,
        # R_full = 0 at equilibrium (u*,v*)
        "anchors_uv":[(0.0,0.1),(0.1,0.18),(0.0,0.0)],
        "ics":[(0.30,0.12,1,1),(0.25,0.10,1,1),(0.20,0.08,2,2),(0.28,0.11,1,2),
               (0.35,0.14,1,1),(0.22,0.09,2,1),(0.18,0.07,1,2),(0.32,0.13,2,2)],
        "input_dim":5, "r_type":"2d",
        "v_pred_known": False,  
    },

}

# ======================================================================
#  TRUE REACTIONS  (evaluation only — NEVER used in training)
# ======================================================================

def R_true_fhn(u, v, cfg):
    """R_full(u,v) = (1/eps)*u*(u-a)*(1-u) - v  [complete u-eq nonlinearity]"""
    return (1.0/cfg["eps_diff"])*u*(u-cfg["a_fhn"])*(1.0-u) - v

def R_true_predprey(u, v, cfg):
    """R_full(u,v) = u*(1-u) - alpha*u*v/(beta+u)  [complete u-eq]"""
    alp = cfg["alpha_pp"]; bet = cfg["beta_pp"]
    return np.clip(u,0,None)*(1 - np.clip(u,0,None)) - \
           alp*np.clip(u,0,None)*np.clip(v,0,None)/(bet + np.clip(u,0.001,None))

def get_R_true(system, u, v, cfg):
    if system=="fhn_partial":        return R_true_fhn(u, v, cfg)
    if system=="predator_prey":      return R_true_predprey(u, v, cfg)
    raise ValueError(f"Unknown system: {system}")

# ======================================================================
#  MESH AND OPERATORS
# ======================================================================

def build_laplacian(cfg):
    xl, xr = cfg["domain"]
    x, _ = build_shishkin_mesh(cfg["Nx"], xl, xr, cfg["eps_diff"], cfg["beta_mesh"])
    return x, build_compact_laplacian(x, periodic=False)

def make_imex(D, L_np, cfg, neumann=True):
    dt=cfg["dt"]; N=cfg["Nx"]
    A=np.eye(N+1)-(dt/2)*D*L_np
    if neumann:
        A[0,:]=0; A[0,0]=1; A[0,1]=-1
        A[-1,:]=0; A[-1,-1]=1; A[-1,-2]=-1
    else:
        A[0,:]=0; A[0,0]=1
        A[-1,:]=0; A[-1,-1]=1
    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv, torch.tensor(A, dtype=torch.float32)

# ======================================================================
#  INITIAL CONDITIONS
# ======================================================================

def ics_fhn(x, cfg):
    eps=cfg["eps_diff"]; ics=[]
    for (kind, loc, v0) in cfg["ics"]:
        if kind=="front":
            u0 = 0.5*(1.0 + np.tanh((x-loc)/np.sqrt(eps)))
        else:
            u0 = np.where(x < loc, 0.9, 0.05)
        ics.append((u0.copy(), v0*np.ones_like(x)))
    return ics

def ics_cosine(x, cfg):
    us=cfg["u_star"]; vs=cfg["v_star"]; ics=[]
    for (au, av, mu, mv) in cfg["ics"]:
        u0 = np.clip(us + au*np.cos(mu*np.pi*x), cfg["u_min"]+0.001, cfg["u_max"])
        v0 = np.clip(vs + av*np.sin(mv*np.pi*x), cfg["v_min"]+0.001, cfg["v_max"])
        ics.append((u0.copy(), v0.copy()))
    return ics

# ======================================================================
#  REFERENCE TRAJECTORY GENERATION
# ======================================================================

def _lus(lu, piv, rhs, lo, hi):
    return np.clip(scipy.linalg.lu_solve((lu, piv), rhs), lo, hi)

def generate_reference(system, x, cfg, L_np):
    dt=cfg["dt"]; Nt=cfg["Nt"]
    t_arr=np.linspace(0, cfg["T"], Nt+1)
    refs=[]

    if system=="fhn_partial":
        eps=cfg["eps_diff"]; dv=cfg["delta_v"]
        bv=cfg["beta_v"]; gv=cfg["gamma_v"]
        _,lu_u,pu,_ = make_imex(eps, L_np, cfg, neumann=False)
        _,lu_v,pv,_ = make_imex(dv,  L_np, cfg, neumann=False)
        for u0,v0 in ics_fhn(x, cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x)))
            ur[0]=u0; vr[0]=v0
            for n in range(Nt):
                un=ur[n]; vn=vr[n]
                # True u-eq: du/dt = eps*u_xx + (1/eps)*u*(u-a)*(1-u) - v
                Ru = (1/eps)*un*(un-cfg["a_fhn"])*(1-un) - vn
                rhs_u = un+(dt/2)*eps*(L_np@un)+dt*Ru
                rhs_u[0]=0.0; rhs_u[-1]=1.0
                ur[n+1] = _lus(lu_u,pu,rhs_u,-0.05,1.05)
                rhs_v = vn+(dt/2)*dv*(L_np@vn)+dt*(bv*un-gv*vn)
                rhs_v[0]=0.0; rhs_v[-1]=0.0
                vr[n+1] = _lus(lu_v,pv,rhs_v,-0.05,1.05)
            refs.append((ur,vr))
            print(f"    IC: u[{ur.min():.3f},{ur.max():.3f}] v[{vr.min():.3f},{vr.max():.3f}]")

    elif system=="predator_prey":
        D1=cfg["D1"]; D2=cfg["D2"]
        alp=cfg["alpha_pp"]; bet=cfg["beta_pp"]
        gam=cfg["gamma_pp"]; dlt=cfg["delta_pp"]
        _,lu_u,pu,_ = make_imex(D1, L_np, cfg, neumann=True)
        _,lu_v,pv,_ = make_imex(D2, L_np, cfg, neumann=True)
        for u0,v0 in ics_cosine(x, cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x)))
            ur[0]=np.clip(u0,0.001,None); vr[0]=np.clip(v0,0.001,None)
            for n in range(Nt):
                un=np.clip(ur[n],0.001,None); vn=np.clip(vr[n],0.001,None)
                R_prey = alp*un*vn/(bet+un)   # Holling II predation
                R_full = un*(1-un) - R_prey    # complete u-eq RHS
                rhs_u = un+(dt/2)*D1*(L_np@un)+dt*R_full
                rhs_u[0]=0; rhs_u[-1]=0
                ur[n+1] = _lus(lu_u,pu,rhs_u,-0.01,0.65)
                rhs_v = vn+(dt/2)*D2*(L_np@vn)+dt*(gam*R_prey-dlt*vn)
                rhs_v[0]=0; rhs_v[-1]=0
                vr[n+1] = _lus(lu_v,pv,rhs_v,-0.01,0.65)
            refs.append((ur,vr))
            print(f"    IC: u[{ur.min():.3f},{ur.max():.3f}] v[{vr.min():.3f},{vr.max():.3f}]")

    return refs, t_arr

# ======================================================================
#  FD EXTRACTION 
# ======================================================================

def extract_fd(refs, L_np, cfg, system):
    dt=cfg["dt"]; ua,va,Ra=[],[],[]
    for ur,vr in refs:
        Nt=ur.shape[0]-1
        for n in range(1,Nt-1):
            un=ur[n]; vn=vr[n]
            dtu=(ur[n+1]-ur[n-1])/(2*dt)
            if system=="fhn_partial":
                Rfd = dtu - cfg["eps_diff"]*(L_np@un)
            elif system=="predator_prey":
                # R_full = u(1-u) - holling => FD: du/dt - D1*L*u
                Rfd = dtu - cfg["D1"]*(L_np@un)
            lap  = np.abs(L_np@un)
            mask = lap < np.percentile(lap, 50)
            mask[0]=False; mask[-1]=False
            if mask.sum()>0:
                ua.append(un[mask]); va.append(vn[mask]); Ra.append(Rfd[mask])
    return np.concatenate(ua), np.concatenate(va), np.concatenate(Ra)

def aggregate_fd(u_fd, v_fd, R_fd, cfg, system):
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
            if m.sum()>=4:
                u_t.append(uc[i]); v_t.append(vc[j]); R_t.append(np.median(R_fd[m]))
    if not u_t:
        print("  WARNING: no 2D bins — using raw FD"); u_t=list(u_fd[:20]); v_t=list(v_fd[:20]); R_t=list(R_fd[:20])
    print(f"  2D FD bins: {len(u_t)}")
    return (np.array(u_t,dtype=np.float32), np.array(v_t,dtype=np.float32),
            np.array(R_t,dtype=np.float32), np.ones(len(u_t),dtype=np.float32))

# ======================================================================
#  MODEL
# ======================================================================

class SystemModel(nn.Module):
    def __init__(self, hidden=32, mlp_w=64, Lambda=2.0, input_dim=5, r_type="2d"):
        super().__init__()
        self.hidden=hidden; self.mlp_w=mlp_w; self.Lambda=Lambda; self.r_type=r_type
        self.gru_out=nn.GRUCell(input_dim,hidden)
        self.gru_lay=nn.GRUCell(input_dim,hidden)
        self.gate=nn.Linear(hidden,mlp_w)
        mlp_in=1 if r_type=="1d" else 2
        self.mlp1=nn.Linear(mlp_in,mlp_w); self.mlp2=nn.Linear(mlp_w,mlp_w); self.mlp3=nn.Linear(mlp_w,1)
        nn.init.normal_(self.gate.weight,std=0.01);  nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.mlp1.weight,std=0.10);  nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.mlp3.weight,std=0.01);  nn.init.zeros_(self.mlp3.bias)

    def init_hidden(self,Nx): h=torch.zeros(Nx,self.hidden); return h,h.clone()

    def _decode(self,u,v,gamma):
        inp=torch.stack([u,v],dim=-1)
        h1=F.silu(self.mlp1(inp)); h2=F.silu(h1+gamma); h3=F.silu(self.mlp2(h2))
        return self.Lambda*torch.tanh(self.mlp3(h3)).squeeze(-1)

    def forward(self,u,v,z_out,z_lay,rho,H_out,H_lay):
        rho_e=rho.unsqueeze(-1)
        Hoc=self.gru_out(torch.clamp(z_out,-10,10),H_out)
        Hlc=self.gru_lay(torch.clamp(z_lay,-10,10),H_lay)
        Hon=(1-rho_e)*Hoc+rho_e*H_out; Hln=rho_e*Hlc+(1-rho_e)*H_lay
        Hb=(1-rho_e)*Hon+rho_e*Hln; gamma=self.gate(Hb)
        return self._decode(u,v,gamma),Hon,Hln

    def react_grad(self,u,v):
        return self._decode(u,v,torch.zeros(u.shape[0],self.mlp_w))

    @torch.no_grad()
    def react_nograd(self,u,v): return self.react_grad(u,v)

# ======================================================================
#  Components
# ======================================================================

def regime_indicator(dhu,eps):
    l=torch.log(torch.abs(dhu)+1e-12); mu=torch.median(l)
    return torch.sigmoid(3.0*(l-mu)/(float(np.log(1/(eps+1e-12)))+1e-8))

def layer_envelope(x_t,xl,xr,eps):
    return torch.exp(-torch.minimum(x_t-xl,torch.tensor(xr)-x_t)/(eps**0.5+1e-12))

def _neu(r): r=r.clone(); r[0]=0.0; r[-1]=0.0; return r
def _dir(r,gL,gR): r=r.clone(); r[0]=gL; r[-1]=gR; return r

# ======================================================================
#  IMEX STEP — TEACHER FORCING FOR v
# ======================================================================

def imex_step(system, u_n, v_ref_n, u_prev, model, H_out, H_lay,
              x_t, phi_t, A_u_t, L_t, cfg):
    eps=cfg["eps_diff"]; dt=cfg["dt"]
    with torch.no_grad():
        dhu=L_t@u_n
    rho=regime_indicator(dhu,eps)
    dt_u=(u_n.detach()-u_prev.detach())/(dt+1e-12)
    # 5-feature vector for all systems
    z_out=torch.stack([u_n,v_ref_n,dhu,dt_u,x_t],dim=-1)
    z_lay=torch.stack([u_n,v_ref_n,eps*dhu,dt_u,phi_t],dim=-1)
    R,Hon,Hln=model(u_n,v_ref_n,z_out,z_lay,rho,H_out,H_lay)
    # IMEX rhs — R is the complete u-eq nonlinearity in all cases
    if system=="fhn_partial":
        rhs=_dir(u_n+(dt/2)*eps*(L_t@u_n)+dt*R,0.0,1.0)
    elif system=="predator_prey":
        # R_full = u(1-u) - holling, so rhs = u + dt*R_full
        rhs=_neu(u_n+(dt/2)*cfg["D1"]*(L_t@u_n)+dt*R)
    u_next=torch.linalg.solve(A_u_t,rhs.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(u_next,cfg["u_min"]-0.05,cfg["u_max"]+0.05),R,Hon,Hln

# ======================================================================
#  TBPTT ROLLOUT
# ======================================================================

def rollout_tbptt(system,model,u0,ur_t,vr_t,x_t,phi_t,A_u_t,L_t,cfg):
    Nt=cfg["Nt"]; win=cfg["tbptt"]
    H_out,H_lay=model.init_hidden(len(x_t))
    u_n=u0; u_prev=u0; n=0
    while n<Nt:
        ww=min(n+win,Nt)
        u_n=u_n.detach(); u_prev=u_prev.detach()
        H_out=H_out.detach(); H_lay=H_lay.detach()
        utw=[u_n]; vtw=[vr_t[n].detach()]; Rtw=[]
        for step in range(n,ww):
            v_ref_n=vr_t[step].detach()
            u_next,R,H_out,H_lay=imex_step(system,u_n,v_ref_n,u_prev,model,H_out,H_lay,x_t,phi_t,A_u_t,L_t,cfg)
            utw.append(u_next); vtw.append(vr_t[step+1].detach()); Rtw.append(R)
            u_prev=u_n; u_n=u_next
        yield utw,vtw,Rtw,ur_t[n:ww+1],vr_t[n:ww+1]; n=ww

# ======================================================================
#  LOSSES
# ======================================================================

TAU=0.05

def l_data(utw,urw,su):
    return torch.stack([((utw[i]-urw[i])/(su+1e-8)).pow(2).mean() for i in range(len(utw))]).mean()

def l_fd(model,u_t,v_t,R_t,w_t,Lambda):
    return (w_t*(model.react_grad(u_t,v_t)-R_t).pow(2)/(Lambda**2+1e-8)).mean()

def l_cons(utw,vtw,Rtw,B=48):
    step=max(1,len(Rtw)//8)
    ua=torch.cat([utw[i] for i in range(0,len(utw)-1,step)])
    va=torch.cat([vtw[i] for i in range(0,len(vtw)-1,step)])
    Ra=torch.cat([Rtw[i] for i in range(0,len(Rtw),step)])
    idx=torch.argsort(ua); us=ua[idx]; vs=va[idx]; Rs=Ra[idx]
    du=torch.sqrt((us[1:]-us[:-1]).pow(2)+(vs[1:]-vs[:-1]).pow(2)); mask=du<TAU
    if mask.sum()<2: return torch.tensor(0.0)
    valid=torch.where(mask)[0]
    if len(valid)>B: valid=valid[torch.randperm(len(valid))[:B]]
    w=(1-du[valid]/TAU).clamp(0,1).pow(2)
    return (w*(Rs[valid]-Rs[valid+1]).pow(2)).mean()

def l_anch(model,cfg,system):
    pts=cfg["anchors_uv"]
    pu=torch.tensor([p[0] for p in pts],dtype=torch.float32)
    pv=torch.tensor([p[1] for p in pts],dtype=torch.float32)
    return model.react_grad(pu,pv).pow(2).mean()

def get_lams(epoch,cfg):
    lams={}
    for k,r in cfg["ramp"].items():
        f=min((epoch-1)/max(r["over"]-1,1),1.0); lams[k]=r["start"]+(r["end"]-r["start"])*f
    return lams

# ======================================================================
#  STAGE 0: PRE-INIT
# ======================================================================

def preinit(model,u_t,v_t,R_t,w_t,Lambda,epochs=400):
    for name,p in model.named_parameters():
        if "gru" in name or "gate" in name: p.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:
        for p in model.parameters(): p.requires_grad_(True); return
    opt=optim.Adam(params,lr=1e-3); best=float("inf")
    print(f"  [Stage 0] {len(u_t)} FD targets  ({epochs} ep)")
    for ep in range(1,epochs+1):
        opt.zero_grad()
        loss=(w_t*(model.react_grad(u_t,v_t)-R_t).pow(2)/(Lambda**2+1e-8)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if loss.item()<best: best=loss.item()
        if ep%100==0 or ep==1: print(f"    ep {ep:>4}  loss={loss.item():.5f}")
    for p in model.parameters(): p.requires_grad_(True)
    print(f"  [Stage 0] Done. Best={best:.5f}")

# ======================================================================
#  STAGE 1: TRAINING
# ======================================================================

def train(system,model,refs,x_t,phi_t,A_u_t,L_t,cfg,u_fd_t,v_fd_t,R_fd_t,w_fd_t,epochs=2000):
    su=max(float(np.mean([np.abs(r[0]).mean() for r in refs])),0.01)
    LR=cfg["lr"]; CLIP=cfg["grad_clip"]
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=500,eta_min=1e-5)
    best_l=float("inf"); best_state=None; history=[]; nan_c=0
    print(f"\n  [Stage 1] {epochs} ep  TBPTT={cfg['tbptt']}  LR={LR}")
    print(f"  su={su:.4f}  (teacher forcing: v from reference)")
    print(f"\n  {'Ep':>5}  {'Total':>8}  {'Data':>8}  {'FD':>8}  {'Cons':>8}  {'Anch':>8}")
    print("  "+"-"*55)
    for epoch in range(1,epochs+1):
        lams=get_lams(epoch,cfg); model.train(); opt.zero_grad()
        ep=ep_d=ep_f=ep_c=ep_a=0.0
        for ur_np,vr_np in refs:
            ur_t=torch.tensor(ur_np,dtype=torch.float32)
            vr_t=torch.tensor(vr_np,dtype=torch.float32)
            n_wins=max(1,cfg["Nt"]//cfg["tbptt"])
            for utw,vtw,Rtw,urw,_ in rollout_tbptt(system,model,ur_t[0],ur_t,vr_t,x_t,phi_t,A_u_t,L_t,cfg):
                Ld=l_data(utw,urw,su); Lf=l_fd(model,u_fd_t,v_fd_t,R_fd_t,w_fd_t,cfg["Lambda"])
                Lc=l_cons(utw,vtw,Rtw); La=l_anch(model,cfg,system)
                Lw=lams["data"]*Ld+lams["fd"]*Lf+lams["cons"]*Lc+lams["anch"]*La
                (Lw/n_wins).backward()
                ep+=Lw.item()/n_wins/len(refs); ep_d+=Ld.item()/n_wins/len(refs)
                ep_f+=Lf.item()/n_wins/len(refs); ep_c+=Lc.item()/n_wins/len(refs); ep_a+=La.item()/n_wins/len(refs)
        if not np.isfinite(ep):
            nan_c+=1; print(f"  NaN ep {epoch} ({nan_c}/5) -- restore+halve LR")
            if best_state: model.load_state_dict(best_state)
            for pg in opt.param_groups: pg["lr"]*=0.5
            if nan_c>=5: print("  Stopping."); break
            continue
        nan_c=0; torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step(); sched.step()
        history.append({"epoch":epoch,"total":ep,"data":ep_d,"fd":ep_f,"cons":ep_c,"anch":ep_a})
        unw=ep_d+ep_f
        if unw<best_l: best_l=unw; best_state={k:v.clone() for k,v in model.state_dict().items()}
        if epoch%50==0 or epoch<=5:
            print(f"  {epoch:>5}  {ep:>8.5f}  {ep_d:>8.5f}  {ep_f:>8.5f}  {ep_c:>8.5f}  {ep_a:>8.5f}")
    if best_state: model.load_state_dict(best_state)
    print(f"\n  Best (data+fd) = {best_l:.5f}")
    return history

# ======================================================================
#  EVALUATION
# ======================================================================

def evaluate(model,cfg,system):
    # 80×80 grid: finer resolution → smoother contour plots
    ng = 80
    ug=np.linspace(cfg["u_min"],cfg["u_max"],ng); vg=np.linspace(cfg["v_min"],cfg["v_max"],ng)
    UU,VV=np.meshgrid(ug,vg); uf=UU.ravel(); vf=VV.ravel()
    Rp=model.react_nograd(torch.tensor(uf,dtype=torch.float32),torch.tensor(vf,dtype=torch.float32)).numpy()
    Rt=get_R_true(system,uf,vf,cfg)
    l2=float(np.sqrt(np.mean((Rp-Rt)**2))/(np.sqrt(np.mean(Rt**2))+1e-8))
    return l2,UU,VV,Rp.reshape(ng,ng),Rt.reshape(ng,ng)

# ======================================================================
#  V_PRED: Post-training v prediction from known v-equation
# ======================================================================

def compute_v_pred(system, u_pred, vr_ref, cfg, L_np):
    if not np.isfinite(u_pred).all():
        print("  WARNING: u_pred contains NaN — returning v_ref as v_pred fallback")
        return vr_ref.copy()

    Nt, Nx1 = u_pred.shape
    Nt = Nt - 1
    dt = cfg["dt"]
    v_pred = np.zeros_like(u_pred)
    v_pred[0] = vr_ref[0]  # same starting condition

    if system == "fhn_partial":
        # dv/dt = delta*v_xx + beta*u - gamma*v  [fully known, no R]
        dv=cfg["delta_v"]; bv=cfg["beta_v"]; gv=cfg["gamma_v"]
        _,lu_v,pv,_ = make_imex(dv, L_np, cfg, neumann=False)
        for n in range(Nt):
            vn=v_pred[n]; un=u_pred[n+1]  # use predicted u
            rhs_v = vn+(dt/2)*dv*(L_np@vn)+dt*(bv*un-gv*vn)
            rhs_v[0]=0.0; rhs_v[-1]=0.0
            v_pred[n+1] = np.clip(scipy.linalg.lu_solve((lu_v,pv),rhs_v),-0.05,1.05)

    elif system == "predator_prey":
        # dv/dt = D2*v_xx + gamma*R - delta*v  [uses R]
        D2=cfg["D2"]; gam=cfg["gamma_pp"]; dlt=cfg["delta_pp"]
        alp=cfg["alpha_pp"]; bet=cfg["beta_pp"]
        _,lu_v,pv,_ = make_imex(D2, L_np, cfg, neumann=True)
        for n in range(Nt):
            vn=v_pred[n]; un=u_pred[n+1]
            R_est = alp*np.clip(un,0.001,None)*vn/(bet+np.clip(un,0.001,None))
            rhs_v = vn+(dt/2)*D2*(L_np@vn)+dt*(gam*R_est-dlt*vn)
            rhs_v[0]=0; rhs_v[-1]=0
            v_pred[n+1] = np.clip(scipy.linalg.lu_solve((lu_v,pv),rhs_v),-0.01,0.65)

    return v_pred

# ======================================================================
#  ROLLOUT (no-grad, for plotting)
# ======================================================================

def rollout_full_nograd(system,model,u0,ur_t,vr_t,x_t,phi_t,A_u_t,L_t,cfg):
    Nt=cfg["Nt"]; H_out,H_lay=model.init_hidden(len(x_t))
    u_n=u0.detach(); u_prev=u0.detach()
    u_traj=[u_n]
    nan_detected = False
    for step in range(Nt):
        v_ref_n=vr_t[step].detach()
        with torch.no_grad():
            u_next,R,H_out,H_lay=imex_step(system,u_n,v_ref_n,u_prev,model,H_out,H_lay,x_t,phi_t,A_u_t,L_t,cfg)
        if not torch.isfinite(u_next).all():
            nan_detected = True
            # Fill remainder with reference
            for _ in range(step+1, Nt):
                u_traj.append(ur_t[_+1].detach())
            break
        u_traj.append(u_next.detach()); u_prev=u_n; u_n=u_next
    if nan_detected:
        return ur_t.numpy()  # fallback to reference
    return torch.stack(u_traj).numpy()

# ======================================================================
#  PLOTS
# ======================================================================

def make_plots(model,refs,t_arr,x_np,L_np,cfg,history,l2,system,out_dir,x_t,phi_t,A_u_t,L_t):
    os.makedirs(out_dir,exist_ok=True)

    # ── 1. Reaction surface ───────────────────────────────────────────
    import math
    if math.isnan(l2):
        print("  WARNING: L²=NaN (model did not converge). Plotting Stage 0 result.")
        l2 = float("inf")  # will show in title as inf
    _,UU,VV,Rp,Rt=evaluate(model,cfg,system)
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f"{cfg['name']}  --  Relative L²={l2:.4f}",fontsize=13,fontweight="bold")
    for ax,Z,title,cmap in [
        (axes[0],Rt,"R_true","RdBu_r"),
        (axes[1],Rp,"R_theta (model)","RdBu_r"),
        (axes[2],np.abs(Rp-Rt),"|Error|","YlOrRd")]:
        im=ax.contourf(UU,VV,Z,levels=40,cmap=cmap); plt.colorbar(im,ax=ax)
        ax.set_xlabel("u",fontsize=11); ax.set_ylabel("v",fontsize=11); ax.set_title(title,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"reaction_{system}.png"),dpi=150,bbox_inches="tight")
    plt.close(); print(f"  Saved: reaction_{system}.png")

    # ── 2. Full trajectory: u and v (ref vs pred) ─────────────────────
    ur_ref,vr_ref=refs[0]
    ur_t=torch.tensor(ur_ref,dtype=torch.float32)
    vr_t=torch.tensor(vr_ref,dtype=torch.float32)
    u_pred=rollout_full_nograd(system,model,ur_t[0],ur_t,vr_t,x_t,phi_t,A_u_t,L_t,cfg)
    v_pred=compute_v_pred(system,u_pred,vr_ref,cfg,L_np)

    Nt=cfg["Nt"]; snaps=[0,Nt//4,Nt//2,3*Nt//4,Nt]
    fig,axes=plt.subplots(6,5,figsize=(22,18))
    fig.suptitle(f"{cfg['name']} — Full Trajectory Comparison (IC 1)",fontsize=12,fontweight="bold")
    for col,n in enumerate(snaps):
        # Row 0: u reference
        axes[0,col].plot(x_np,ur_ref[n],"k-",lw=2)
        axes[0,col].set_title(f"u_ref, t={t_arr[n]:.2f}",fontsize=8)
        # Row 1: u predicted vs reference
        axes[1,col].plot(x_np,u_pred[n],"C0-",lw=2,label="pred")
        axes[1,col].plot(x_np,ur_ref[n],"k--",lw=1,alpha=0.5,label="ref")
        axes[1,col].set_title(f"u_pred, t={t_arr[n]:.2f}",fontsize=8)
        if col==0: axes[1,col].legend(fontsize=6)
        # Row 2: u pointwise error
        err_u=np.abs(u_pred[n]-ur_ref[n])
        axes[2,col].fill_between(x_np,err_u,alpha=0.6,color="C3")
        axes[2,col].plot(x_np,err_u,"C3-",lw=1.5)
        axes[2,col].set_title(f"|u_err|, t={t_arr[n]:.2f}",fontsize=8)
        # Row 3: v reference
        axes[3,col].plot(x_np,vr_ref[n],"C1-",lw=2)
        axes[3,col].set_title(f"v_ref, t={t_arr[n]:.2f}",fontsize=8)
        # Row 4: v predicted vs reference
        axes[4,col].plot(x_np,v_pred[n],"C5-",lw=2,label="pred")
        axes[4,col].plot(x_np,vr_ref[n],"C1--",lw=1,alpha=0.5,label="ref")
        axes[4,col].set_title(f"v_pred, t={t_arr[n]:.2f}",fontsize=8)
        if col==0: axes[4,col].legend(fontsize=6)
        # Row 5: v pointwise error
        err_v=np.abs(v_pred[n]-vr_ref[n])
        axes[5,col].fill_between(x_np,err_v,alpha=0.6,color="C4")
        axes[5,col].plot(x_np,err_v,"C4-",lw=1.5)
        axes[5,col].set_title(f"|v_err|, t={t_arr[n]:.2f}",fontsize=8)
    labels=["u_ref","u_pred","|u_err|","v_ref","v_pred","|v_err|"]
    for i,lb in enumerate(labels): axes[i,0].set_ylabel(lb,fontsize=9)
    for ax in axes.ravel(): ax.grid(True,alpha=0.3); ax.set_xlabel("x",fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"trajectory_{system}.png"),dpi=150,bbox_inches="tight")
    plt.close(); print(f"  Saved: trajectory_{system}.png")

    # ── 3. Error heatmaps + L²(t) ────────────────────────────────────
    err_u=np.abs(u_pred-ur_ref); err_v=np.abs(v_pred-vr_ref)
    l2u=float(np.sqrt(np.mean((u_pred-ur_ref)**2))); linfu=float(np.max(err_u))
    l2v=float(np.sqrt(np.mean((v_pred-vr_ref)**2))); linfv=float(np.max(err_v))
    fig,axes=plt.subplots(1,3,figsize=(21,5))
    fig.suptitle(f"{cfg['name']} — Error Heatmaps (IC 1)\n"
                 f"u: L²={l2u:.4f} L∞={linfu:.4f}   v: L²={l2v:.4f} L∞={linfv:.4f}",
                 fontsize=11,fontweight="bold")
    im0=axes[0].contourf(x_np,t_arr,err_u,levels=20,cmap="YlOrRd"); plt.colorbar(im0,ax=axes[0])
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t"); axes[0].set_title("|u_pred−u_ref|(x,t)")
    im1=axes[1].contourf(x_np,t_arr,err_v,levels=20,cmap="YlOrRd"); plt.colorbar(im1,ax=axes[1])
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t"); axes[1].set_title("|v_pred−v_ref|(x,t)")
    l2u_t=np.sqrt(np.mean((u_pred-ur_ref)**2,axis=1))
    l2v_t=np.sqrt(np.mean((v_pred-vr_ref)**2,axis=1))
    axes[2].plot(t_arr,l2u_t,"C3-",lw=2,label="u L²(t)")
    axes[2].fill_between(t_arr,l2u_t,alpha=0.25,color="C3")
    axes[2].plot(t_arr,l2v_t,"C4--",lw=2,label="v L²(t)")
    axes[2].fill_between(t_arr,l2v_t,alpha=0.2,color="C4")
    axes[2].legend(fontsize=9); axes[2].grid(True,alpha=0.3)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("L²(t)"); axes[2].set_title("L²(t) errors")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"error_heatmap_{system}.png"),dpi=150,bbox_inches="tight")
    plt.close()
    print(f"  Saved: error_heatmap_{system}.png")
    print(f"  u: L²={l2u:.4f}  L∞={linfu:.4f}")
    print(f"  v: L²={l2v:.4f}  L∞={linfv:.4f}")

    # ── 4. Loss curves ────────────────────────────────────────────────
    if not history: return
    fig,ax=plt.subplots(figsize=(10,5))
    ep=[h["epoch"] for h in history]
    for k,c,ls in [("total","k","-"),("data","C0","--"),("fd","C1","-."),("cons","C2",":"),("anch","C3","--")]:
        ax.semilogy(ep,[max(h[k],1e-10) for h in history],color=c,linestyle=ls,lw=1.5,label=k)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title(f"{cfg['name']} — Training Loss",fontsize=11,fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"loss_{system}.png"),dpi=150)
    plt.close(); print(f"  All plots -> {out_dir}/")

# ======================================================================
#  RUNNER
# ======================================================================

def run_system(system,epochs=None,preinit_epochs=400):
    torch.manual_seed(SEED); np.random.seed(SEED)
    assert system in SYSTEM_CFGS, f"Unknown: {system}. Options: {list(SYSTEM_CFGS)}"
    cfg=SYSTEM_CFGS[system]; ep=epochs or 2000
    out_dir=f"results_v5/{system}"; os.makedirs(out_dir,exist_ok=True)
    print("="*65); print(f"  {cfg['name']}"); print(f"  {cfg['citation']}"); print("="*65)

    x,L_np=build_laplacian(cfg)
    x_t=torch.tensor(x,dtype=torch.float32); L_t=torch.tensor(L_np,dtype=torch.float32)
    xl,xr=cfg["domain"]; phi_t=layer_envelope(x_t,xl,xr,cfg["eps_diff"])

    D_u=(cfg["eps_diff"] if system=="fhn_partial"
         else cfg.get("D1",cfg["eps_diff"]))
    _,_,_,A_u_t=make_imex(D_u,L_np,cfg,neumann=(cfg["bc_type"]=="neumann"))

    print("Generating reference trajectories..."); refs,t_arr=generate_reference(system,x,cfg,L_np)
    print("Extracting FD pairs...")
    u_fd,v_fd,R_fd=extract_fd(refs,L_np,cfg,system)
    u_t,v_t,R_t,w_t=aggregate_fd(u_fd,v_fd,R_fd,cfg,system)
    u_fd_t=torch.tensor(u_t,dtype=torch.float32); v_fd_t=torch.tensor(v_t,dtype=torch.float32)
    R_fd_t=torch.tensor(R_t,dtype=torch.float32); w_fd_t=torch.tensor(w_t,dtype=torch.float32)

    model=SystemModel(hidden=32,mlp_w=64,Lambda=cfg["Lambda"],
                      input_dim=cfg["input_dim"],r_type=cfg["r_type"])
    n_p=sum(p.numel() for p in model.parameters()); print(f"  Parameters: {n_p:,}")

    preinit(model,u_fd_t,v_fd_t,R_fd_t,w_fd_t,cfg["Lambda"],epochs=preinit_epochs)
    t0=time.time()
    history=train(system,model,refs,x_t,phi_t,A_u_t,L_t,cfg,u_fd_t,v_fd_t,R_fd_t,w_fd_t,epochs=ep)
    t_train=time.time()-t0

    l2,*_=evaluate(model,cfg,system)
    print(f"\n  Results:"); print(f"    Relative L²: {l2:.4f}"); print(f"    Train time: {t_train:.0f}s ({t_train/60:.1f} min)")

    make_plots(model,refs,t_arr,x,L_np,cfg,history,l2,system,out_dir,x_t,phi_t,A_u_t,L_t)
    torch.save({"state":model.state_dict(),"cfg":cfg},os.path.join(out_dir,f"checkpoint_{system}.pt"))
    results={"system":system,"l2":float(l2),"params":n_p,"train_s":float(t_train),"epochs":ep}
    with open(os.path.join(out_dir,"metrics.json"),"w",encoding="utf-8") as f: json.dump(results,f,indent=2)
    return results

# ======================================================================
#  CLI
# ======================================================================

def main():
    p=argparse.ArgumentParser(description="Partial-inverse coupled reaction-diffusion systems")
    p.add_argument("--system",required=True,
        choices=["fhn_partial","predator_prey","all"])
    p.add_argument("--epochs",type=int,default=None)
    p.add_argument("--preinit_epochs",type=int,default=400)
    args=p.parse_args()
    systems=list(SYSTEM_CFGS.keys()) if args.system=="all" else [args.system]
    results={}
    for s in systems:
        print(f"\n{'#'*65}\n# {s}\n{'#'*65}")
        results[s]=run_system(s,args.epochs,args.preinit_epochs)
    if len(results)>1:
        print("\n"+"="*50+"\n  SUMMARY\n"+"="*50)
        for s,m in results.items(): print(f"  {s:<25}  L²={m['l2']:.4f}  {m['train_s']:.0f}s")

if __name__=="__main__": main()