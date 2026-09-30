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

SEED    = 42
EPOCHS  = 300
OUT_DIR = "arch_grid_results_systems"
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_DEPTHS = [3, 4, 5]
DEFAULT_WIDTHS = [32, 64, 128]
DEFAULT_HIDDEN = 32
INPUT_DIM      = 5

# ======================================================================
#  SYSTEM CONFIGS 
# ======================================================================

SYSTEM_CFGS = {
    "fhn_partial": {
        "name":    "FitzHugh-Nagumo Partial (Neuroscience)",
        "eps_diff":0.05, "delta_v":0.1, "beta_v":1.0, "gamma_v":0.5, "a_fhn":0.25,
        "Lambda":2.0, "u_min":0.0, "u_max":1.0, "v_min":0.0, "v_max":0.6,
        "domain":(0.0,1.0), "T":0.3, "Nx":52, "Nt":60, "dt":0.005,
        "bc_type":"dirichlet", "beta_mesh":2.0, "n_bins_2d":12,
        "anchors_uv":[(0.0,0.0),(0.25,0.0),(1.0,0.0)],
        "ics":[("front",0.3,0.0),("front",0.5,0.0),("front",0.6,0.0),("step",0.5,0.5)],
        "input_dim":5,
    },
    "predator_prey": {
        "name":    "Predator-Prey Holling Type II (Ecology)",
        "alpha_pp":1.0, "beta_pp":0.1, "gamma_pp":0.5, "delta_pp":0.25,
        "D1":1e-3, "D2":1e-2, "eps_diff":1e-3, "Lambda":0.35,
        "u_min":0.0, "u_max":0.5, "v_min":0.0, "v_max":0.4,
        "u_star":0.1, "v_star":0.18,
        "domain":(0.0,1.0), "T":2.0, "Nx":128, "Nt":200, "dt":0.01,
        "bc_type":"neumann", "beta_mesh":2.0, "n_bins_2d":10,
        "anchors_uv":[(0.0,0.1),(0.1,0.18),(0.0,0.0)],
        "ics":[(0.30,0.12,1,1),(0.25,0.10,1,1),(0.20,0.08,2,2),(0.28,0.11,1,2)],
        "input_dim":5,
    },
}

# ======================================================================
#  TRUE REACTIONS
# ======================================================================

def get_R_true(system, u, v, cfg):
    if system == "fhn_partial":
        return (1.0/cfg["eps_diff"])*u*(u-cfg["a_fhn"])*(1.0-u) - v
    if system == "predator_prey":
        a=cfg["alpha_pp"]; b=cfg["beta_pp"]
        return np.clip(u,0,None)*(1-np.clip(u,0,None)) - \
               a*np.clip(u,0,None)*np.clip(v,0,None)/(b+np.clip(u,0.001,None))

# ======================================================================
#  MESH, IMEX, ICs, REFERENCE, FD  (identical to systems_arch_study.py)
# ======================================================================

def build_laplacian(cfg):
    xl,xr=cfg["domain"]
    x,_=build_shishkin_mesh(cfg["Nx"],xl,xr,cfg["eps_diff"],cfg["beta_mesh"])
    return x, build_compact_laplacian(x,periodic=False)

def make_imex(D,L_np,cfg,neumann=True):
    dt=cfg["dt"]; N=cfg["Nx"]
    A=np.eye(N+1)-(dt/2)*D*L_np
    if neumann: A[0,:]=0;A[0,0]=1;A[0,1]=-1;A[-1,:]=0;A[-1,-1]=1;A[-1,-2]=-1
    else: A[0,:]=0;A[0,0]=1;A[-1,:]=0;A[-1,-1]=1
    lu,piv=scipy.linalg.lu_factor(A)
    return A,lu,piv,torch.tensor(A,dtype=torch.float32)

def layer_envelope(x_t,xl,xr,eps):
    return torch.exp(-torch.minimum(x_t-xl,torch.tensor(xr)-x_t)/(eps**0.5+1e-12))

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

def generate_reference(system, x, cfg, L_np):
    dt=cfg["dt"]; Nt=cfg["Nt"]; t_arr=np.linspace(0,cfg["T"],Nt+1); refs=[]
    if system=="fhn_partial":
        eps=cfg["eps_diff"]; dv=cfg["delta_v"]; bv=cfg["beta_v"]; gv=cfg["gamma_v"]
        _,lu_u,pu,_=make_imex(eps,L_np,cfg,neumann=False)
        _,lu_v,pv,_=make_imex(dv,L_np,cfg,neumann=False)
        for u0,v0 in make_ics(system,x,cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x))); ur[0]=u0; vr[0]=v0
            for n in range(Nt):
                un=ur[n]; vn=vr[n]
                Ru=(1/eps)*un*(un-cfg["a_fhn"])*(1-un)-vn
                rhs_u=un+(dt/2)*eps*(L_np@un)+dt*Ru; rhs_u[0]=0.0; rhs_u[-1]=1.0
                ur[n+1]=_lus(lu_u,pu,rhs_u,-0.05,1.05)
                rhs_v=vn+(dt/2)*dv*(L_np@vn)+dt*(bv*un-gv*vn); rhs_v[0]=0.0; rhs_v[-1]=0.0
                vr[n+1]=_lus(lu_v,pv,rhs_v,-0.05,1.05)
            refs.append((ur,vr))
    elif system=="predator_prey":
        D1=cfg["D1"]; D2=cfg["D2"]; alp=cfg["alpha_pp"]; bet=cfg["beta_pp"]
        gam=cfg["gamma_pp"]; dlt=cfg["delta_pp"]
        _,lu_u,pu,_=make_imex(D1,L_np,cfg,neumann=True)
        _,lu_v,pv,_=make_imex(D2,L_np,cfg,neumann=True)
        for u0,v0 in make_ics(system,x,cfg):
            ur=np.zeros((Nt+1,len(x))); vr=np.zeros((Nt+1,len(x)))
            ur[0]=np.clip(u0,0.001,None); vr[0]=np.clip(v0,0.001,None)
            for n in range(Nt):
                un=np.clip(ur[n],0.001,None); vn=np.clip(vr[n],0.001,None)
                Rp=alp*un*vn/(bet+un); Rf=un*(1-un)-Rp
                rhs_u=un+(dt/2)*D1*(L_np@un)+dt*Rf; rhs_u[0]=0; rhs_u[-1]=0
                ur[n+1]=_lus(lu_u,pu,rhs_u,-0.01,0.65)
                rhs_v=vn+(dt/2)*D2*(L_np@vn)+dt*(gam*Rp-dlt*vn); rhs_v[0]=0; rhs_v[-1]=0
                vr[n+1]=_lus(lu_v,pv,rhs_v,-0.01,0.65)
            refs.append((ur,vr))
    return refs, t_arr

def extract_fd(refs, L_np, cfg, system):
    dt=cfg["dt"]; ua,va,Ra=[],[],[]
    for ur,vr in refs:
        Nt=ur.shape[0]-1
        for n in range(1,Nt-1):
            un=ur[n]; vn=vr[n]; dtu=(ur[n+1]-ur[n-1])/(2*dt)
            Rfd=dtu-cfg["eps_diff"]*(L_np@un) if system=="fhn_partial" else dtu-cfg["D1"]*(L_np@un)
            lap=np.abs(L_np@un); mask=lap<np.percentile(lap,50); mask[0]=False; mask[-1]=False
            if mask.sum()>0: ua.append(un[mask]); va.append(vn[mask]); Ra.append(Rfd[mask])
    return np.concatenate(ua),np.concatenate(va),np.concatenate(Ra)

def aggregate_fd(u_fd, v_fd, R_fd, cfg):
    nb=cfg["n_bins_2d"]
    ue=np.linspace(cfg["u_min"],cfg["u_max"],nb+1); ve=np.linspace(cfg["v_min"],cfg["v_max"],nb+1)
    uc=0.5*(ue[:-1]+ue[1:]); vc=0.5*(ve[:-1]+ve[1:])
    ui=np.clip(np.digitize(u_fd,ue)-1,0,nb-1); vi=np.clip(np.digitize(v_fd,ve)-1,0,nb-1)
    u_t,v_t,R_t=[],[],[]
    for i in range(nb):
        for j in range(nb):
            m=(ui==i)&(vi==j)
            if m.sum()>=4: u_t.append(uc[i]); v_t.append(vc[j]); R_t.append(np.median(R_fd[m]))
    return (np.array(u_t,dtype=np.float32),np.array(v_t,dtype=np.float32),
            np.array(R_t,dtype=np.float32))

# ======================================================================
#  FLEXIBLE MODEL 
# ======================================================================

class FlexSystemModel(nn.Module):
    """eps-US-GRU-S, configurable GRU hidden / decoder depth / decoder width.
    Fixed: dual GRU regime encoder, additive gate, SiLU, teacher forcing.
    """
    def __init__(self, hidden=32, mlp_width=64, mlp_depth=3, Lambda=2.0, input_dim=5):
        super().__init__()
        self.hidden=hidden; self.mlp_width=mlp_width; self.mlp_depth=mlp_depth
        self.Lambda=Lambda
        self.gru_out=nn.GRUCell(input_dim,hidden)
        self.gru_lay=nn.GRUCell(input_dim,hidden)
        self.gate=nn.Linear(hidden,mlp_width)
        layers=[nn.Linear(2,mlp_width)]
        for _ in range(mlp_depth-2):
            layers.append(nn.Linear(mlp_width,mlp_width))
        layers.append(nn.Linear(mlp_width,1))
        self.mlp_layers=nn.ModuleList(layers)
        nn.init.normal_(self.gate.weight,std=0.01); nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.mlp_layers[0].weight,std=0.10); nn.init.zeros_(self.mlp_layers[0].bias)
        nn.init.normal_(self.mlp_layers[-1].weight,std=0.01); nn.init.zeros_(self.mlp_layers[-1].bias)

    def init_hidden(self,Nx):
        h=torch.zeros(Nx,self.hidden); return h,h.clone()

    def _decode(self,u,v,gamma):
        inp=torch.stack([u,v],dim=-1)
        h=F.silu(self.mlp_layers[0](inp))
        h=F.silu(h+gamma)
        for layer in self.mlp_layers[1:-1]:
            h=F.silu(layer(h))
        out=self.mlp_layers[-1](h)
        return self.Lambda*torch.tanh(out).squeeze(-1)

    def forward(self,u,v,z_out,z_lay,rho,H_out,H_lay):
        rho_e=rho.unsqueeze(-1)
        Hoc=self.gru_out(torch.clamp(z_out,-10,10),H_out)
        Hlc=self.gru_lay(torch.clamp(z_lay,-10,10),H_lay)
        Hon=(1-rho_e)*Hoc+rho_e*H_out; Hln=rho_e*Hlc+(1-rho_e)*H_lay
        Hb=(1-rho_e)*Hon+rho_e*Hln; gamma=self.gate(Hb)
        return self._decode(u,v,gamma),Hon,Hln

    def react_grad(self,u,v):
        gamma=torch.zeros(u.shape[0],self.mlp_width)
        return self._decode(u,v,gamma)

    @torch.no_grad()
    def react_nograd(self,u,v): return self.react_grad(u,v)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def count_params(hidden, mlp_depth, mlp_width, input_dim=INPUT_DIM):
    gru_p  = 3*hidden*(input_dim+hidden+2)
    gate_p = hidden*mlp_width + mlp_width
    mlp_p  = 3*mlp_width
    for _ in range(mlp_depth-2):
        mlp_p += mlp_width*mlp_width + mlp_width
    mlp_p += mlp_width*1 + 1
    return 2*gru_p + gate_p + mlp_p

# ======================================================================
#  IMEX STEP, EVALUATE  
# ======================================================================

def regime_indicator(dhu,eps):
    l=torch.log(torch.abs(dhu)+1e-12); mu=torch.median(l)
    return torch.sigmoid(3.0*(l-mu)/(float(np.log(1/(eps+1e-12)))+1e-8))

def _neu(r): r=r.clone(); r[0]=0.0; r[-1]=0.0; return r
def _dir(r,gL,gR): r=r.clone(); r[0]=gL; r[-1]=gR; return r

TAU=0.05

def imex_step(system,u_n,v_ref_n,u_prev,model,H_out,H_lay,x_t,phi_t,A_u_t,L_t,cfg):
    eps=cfg["eps_diff"]; dt=cfg["dt"]
    with torch.no_grad(): dhu=L_t@u_n
    rho=regime_indicator(dhu,eps)
    dt_u=(u_n.detach()-u_prev.detach())/(dt+1e-12)
    z_out=torch.stack([u_n,v_ref_n,dhu,dt_u,x_t],dim=-1)
    z_lay=torch.stack([u_n,v_ref_n,eps*dhu,dt_u,phi_t],dim=-1)
    R,Hon,Hln=model(u_n,v_ref_n,z_out,z_lay,rho,H_out,H_lay)
    if system=="fhn_partial": rhs=_dir(u_n+(dt/2)*eps*(L_t@u_n)+dt*R,0.0,1.0)
    else: rhs=_neu(u_n+(dt/2)*cfg["D1"]*(L_t@u_n)+dt*R)
    u_next=torch.linalg.solve(A_u_t,rhs.unsqueeze(-1)).squeeze(-1)
    return torch.clamp(u_next,cfg["u_min"]-0.05,cfg["u_max"]+0.05),R,Hon,Hln

def evaluate(model, system, cfg, u_fd=None, v_fd=None):
    ng=60
    ug=np.linspace(cfg["u_min"],cfg["u_max"],ng); vg=np.linspace(cfg["v_min"],cfg["v_max"],ng)
    UU,VV=np.meshgrid(ug,vg); uf=UU.ravel(); vf=VV.ravel()
    Rp=model.react_nograd(torch.tensor(uf,dtype=torch.float32),
                           torch.tensor(vf,dtype=torch.float32)).numpy()
    Rt=get_R_true(system,uf,vf,cfg)
    l2=float(np.sqrt(np.mean((Rp-Rt)**2))/(np.sqrt(np.mean(Rt**2))+1e-8))
    l2_in=l2
    if u_fd is not None and len(u_fd)>0:
        Rp2=model.react_nograd(torch.tensor(u_fd.astype(np.float32)),
                                torch.tensor(v_fd.astype(np.float32))).numpy()
        Rt2=get_R_true(system,u_fd,v_fd,cfg)
        l2_in=float(np.sqrt(np.mean((Rp2-Rt2)**2))/(np.sqrt(np.mean(Rt2**2))+1e-8))
    return l2,l2_in

# ======================================================================
#  TRAINING 
# ======================================================================

def train_arch(system, model, refs, cfg, x_t, phi_t, A_u_t, L_t,
               u_fd_t, v_fd_t, R_fd_t, epochs):
    su=max(float(np.mean([np.abs(r[0]).mean() for r in refs])),0.01)
    Lambda=cfg["Lambda"]

    for n,p in model.named_parameters():
        if "gru" in n: p.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    if params:
        pi_opt=optim.Adam(params,lr=1e-3); best_s0=float("inf")
        for _ in range(1000):
            pi_opt.zero_grad()
            loss=(model.react_grad(u_fd_t,v_fd_t)-R_fd_t).pow(2).mean()/(Lambda**2+1e-8)
            loss.backward(); pi_opt.step()
            if loss.item()<best_s0: best_s0=loss.item()
    for p in model.parameters(): p.requires_grad_(True)

    opt=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=400,eta_min=1e-6)
    history=[]; best_l=float("inf"); best_state=None; nan_c=0

    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad()
        f=min((epoch-1)/(epochs-1+1e-8),1.0)
        lam_d=1.0; lam_f=3.0+2.0*f; lam_a=2.0+3.0*f; lam_c=0.01+0.49*f
        ep=0.0

        for ur_np,vr_np in refs:
            ur_t=torch.tensor(ur_np,dtype=torch.float32)
            vr_t=torch.tensor(vr_np,dtype=torch.float32)
            H_out,H_lay=model.init_hidden(x_t.shape[0])
            u_n=ur_t[0]; u_prev=ur_t[0]
            ua_buf=[]; va_buf=[]
            for step in range(cfg["Nt"]):
                H_out=H_out.detach(); H_lay=H_lay.detach()
                u_n=u_n.detach(); u_prev=u_prev.detach()
                v_ref_n=vr_t[step].detach()
                u_next,R,H_out,H_lay=imex_step(system,u_n,v_ref_n,u_prev,
                                                model,H_out,H_lay,x_t,phi_t,A_u_t,L_t,cfg)
                Ld=((u_next-ur_t[step+1])/(su+1e-8)).pow(2).mean()
                Lf=(model.react_grad(u_fd_t,v_fd_t)-R_fd_t).pow(2).mean()/(Lambda**2+1e-8)
                pts=cfg["anchors_uv"]
                pu=torch.tensor([p[0] for p in pts],dtype=torch.float32)
                pv=torch.tensor([p[1] for p in pts],dtype=torch.float32)
                La=model.react_grad(pu,pv).pow(2).mean()
                ua_buf.append(u_n.detach()); va_buf.append(vr_t[step].detach())
                Lw=lam_d*Ld+lam_f*Lf+lam_a*La
                (Lw/cfg["Nt"]/len(refs)).backward()
                ep+=Lw.item()/cfg["Nt"]/len(refs)
                u_prev=u_n; u_n=u_next
            if len(ua_buf)>2:
                ua_c=torch.cat(ua_buf); va_c=torch.cat(va_buf)
                if len(ua_c)>512:
                    perm=torch.randperm(len(ua_c))[:512]
                    ua_c=ua_c[perm]; va_c=va_c[perm]
                Rs_fresh=model.react_grad(ua_c,va_c)
                sort_idx=torch.argsort(ua_c)
                us2=ua_c[sort_idx]; Rs2=Rs_fresh[sort_idx]
                du2=torch.abs(us2[1:]-us2[:-1]); mask2=du2<TAU
                if mask2.sum()>=2:
                    valid2=torch.where(mask2)[0]
                    if len(valid2)>64: valid2=valid2[torch.randperm(len(valid2))[:64]]
                    w2=(1-du2[valid2]/TAU).clamp(0,1).pow(2)
                    Lc=(w2*(Rs2[valid2]-Rs2[valid2+1]).pow(2)).mean()
                    (lam_c*Lc/len(refs)).backward()
                    ep+=lam_c*Lc.item()/len(refs)

        if not np.isfinite(ep):
            nan_c+=1
            if best_state: model.load_state_dict(best_state)
            for pg in opt.param_groups: pg["lr"]*=0.5
            if nan_c>=5: break
            continue
        nan_c=0; torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        opt.step(); sched.step()
        history.append(ep)
        if ep<best_l: best_l=ep; best_state={k:v.clone() for k,v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return history

# ======================================================================
#  RUN ONE (system, depth, width) AT FIXED HIDDEN
# ======================================================================

def run_one(system, hidden, mlp_depth, mlp_width, shared, epochs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg=shared["cfg"]; x_t=shared["x_t"]; phi_t=shared["phi_t"]
    A_u_t=shared["A_u_t"]; L_t=shared["L_t"]; refs=shared["refs"]
    u_fd_t=shared["u_fd_t"]; v_fd_t=shared["v_fd_t"]; R_fd_t=shared["R_fd_t"]
    u_fd=shared["u_fd"]; v_fd=shared["v_fd"]

    model=FlexSystemModel(hidden=hidden,mlp_width=mlp_width,mlp_depth=mlp_depth,
                          Lambda=cfg["Lambda"],input_dim=cfg["input_dim"])
    n_params=model.count_params()
    t0=time.time()
    history=train_arch(system,model,refs,cfg,x_t,phi_t,A_u_t,L_t,
                       u_fd_t,v_fd_t,R_fd_t,epochs)
    t_train=time.time()-t0
    l2,l2_in=evaluate(model,system,cfg,u_fd,v_fd)
    return dict(l2=l2,l2_in=l2_in,params=n_params,train_s=t_train)

def precompute_shared(system):
    cfg=copy.deepcopy(SYSTEM_CFGS[system])
    x,L_np=build_laplacian(cfg)
    x_t=torch.tensor(x,dtype=torch.float32); L_t=torch.tensor(L_np,dtype=torch.float32)
    xl,xr=cfg["domain"]; phi_t=layer_envelope(x_t,xl,xr,cfg["eps_diff"])
    D_u=cfg["eps_diff"] if system=="fhn_partial" else cfg["D1"]
    _,_,_,A_u_t=make_imex(D_u,L_np,cfg,neumann=(cfg["bc_type"]=="neumann"))
    refs,t_arr=generate_reference(system,x,cfg,L_np)
    u_fd,v_fd,R_fd=extract_fd(refs,L_np,cfg,system)
    u_t,v_t,R_t=aggregate_fd(u_fd,v_fd,R_fd,cfg)
    print(f"    FD bins: {len(u_t)}   ICs: {len(refs)}")
    return dict(cfg=cfg,x=x,L_np=L_np,x_t=x_t,L_t=L_t,phi_t=phi_t,A_u_t=A_u_t,
                refs=refs,t_arr=t_arr,u_fd=u_fd,v_fd=v_fd,
                u_fd_t=torch.tensor(u_t,dtype=torch.float32),
                v_fd_t=torch.tensor(v_t,dtype=torch.float32),
                R_fd_t=torch.tensor(R_t,dtype=torch.float32))

# ======================================================================
#  PLOTS 
# ======================================================================

def plot_system_heatmap(system, depths, widths, hidden, grid_l2, grid_l2in, out_dir):
    """1x2 viridis heatmap: rows=depth D, cols=W{val}_err, like the uploaded image."""
    sname=SYSTEM_CFGS[system]["name"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    fig.suptitle(f"{sname}  --  Architecture Grid (GRU hidden H={hidden})",
                 fontsize=12,fontweight="bold")

    col_labels=[f"W{w}_err" for w in widths]
    row_labels=[str(d) for d in depths]

    for ax,Z,title in [(axes[0],grid_l2,"Relative L² error (full grid)"),
                       (axes[1],grid_l2in,"Relative L² error (in-support)")]:
        im=ax.imshow(Z,cmap="viridis",aspect="auto")
        ax.set_xticks(range(len(widths))); ax.set_xticklabels(col_labels,fontsize=10)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels(row_labels,fontsize=10)
        ax.set_ylabel("Decoder Depth D",fontsize=10)
        ax.set_title(title,fontsize=10)
        # Mark the best (minimum) cell
        bi,bj=np.unravel_index(np.argmin(Z),Z.shape)
        ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,
                    edgecolor="white",linewidth=3))
        ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,
                    edgecolor="#C0392B",linewidth=1.5,linestyle="--"))
        cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        cbar.set_label("Relative error",fontsize=9)
    plt.tight_layout()
    p=os.path.join(out_dir,f"arch_grid_{system}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_joint_heatmap(depths, widths, hidden, joint_score, eff_score,
                       per_system_l2, systems, out_dir):
    """Joint normalised score across systems + efficiency-adjusted score."""
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    fig.suptitle(f"Joint Architecture Selection (GRU hidden H={hidden})",
                 fontsize=12,fontweight="bold")
    col_labels=[f"W{w}_err" for w in widths]
    row_labels=[str(d) for d in depths]

    for ax,Z,title in [(axes[0],joint_score,"Joint normalised L² (mean over systems)"),
                       (axes[1],eff_score,"Efficiency score (+ 0.25 x normalised params)")]:
        im=ax.imshow(Z,cmap="viridis",aspect="auto")
        ax.set_xticks(range(len(widths))); ax.set_xticklabels(col_labels,fontsize=10)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels(row_labels,fontsize=10)
        ax.set_ylabel("Decoder Depth D",fontsize=10)
        ax.set_title(title,fontsize=10)
        bi,bj=np.unravel_index(np.argmin(Z),Z.shape)
        ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor="white",linewidth=3))
        ax.add_patch(plt.Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor="#C0392B",
                    linewidth=1.5,linestyle="--"))
        # Annotate D,W of best cell
        ax.text(bj,bi-0.65,f"D={depths[bi]},W={widths[bj]}",ha="center",
               fontsize=9,color="#C0392B",fontweight="bold")
        cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        cbar.set_label("score (lower=better)",fontsize=9)
    plt.tight_layout()
    p=os.path.join(out_dir,"arch_grid_joint.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

# ======================================================================
#  TABLES
# ======================================================================

def save_tables(systems, depths, widths, hidden, all_l2, all_l2in, all_params,
                joint_score, eff_score, out_dir):
    lines=[]
    bi_acc,bj_acc=np.unravel_index(np.argmin(joint_score),joint_score.shape)
    bi_eff,bj_eff=np.unravel_index(np.argmin(eff_score),eff_score.shape)
    lines.append(f"Architecture Grid Results (GRU hidden H={hidden})")
    lines.append("="*70)
    for system in systems:
        sname=SYSTEM_CFGS[system]["name"]
        lines.append(f"\n{sname}")
        dw_label = "D" + chr(92) + "W"
        lines.append(f"  {dw_label:<6}"+"".join(f"{'W='+str(w):>12}" for w in widths))
        for i,d in enumerate(depths):
            row=f"  {'D='+str(d):<6}"
            for j,w in enumerate(widths):
                mk="*" if (system==systems[0] and i==bi_acc and j==bj_acc) else " "
                row+=f"{all_l2[system][i,j]:>10.4f}{mk} "
            lines.append(row)
        lines.append(f"  Params: "+", ".join(f"D={d}: "+", ".join(
            f"W{w}={count_params(hidden,d,w):,}" for w in widths) for d in depths))

    lines.append(f"\n{'='*70}")
    lines.append("RECOMMENDED CONFIGURATIONS (computed from results)")
    lines.append(f"  Best accuracy (joint):  D={depths[bi_acc]}, W={widths[bj_acc]}, H={hidden}")
    lines.append(f"    params={count_params(hidden,depths[bi_acc],widths[bj_acc]):,}")
    for system in systems:
        lines.append(f"    {system}: L2(full)={all_l2[system][bi_acc,bj_acc]:.4f}  "
                     f"L2(in-support)={all_l2in[system][bi_acc,bj_acc]:.4f}")
    lines.append(f"\n  Parameter-efficient alternative: D={depths[bi_eff]}, W={widths[bj_eff]}, H={hidden}")
    lines.append(f"    params={count_params(hidden,depths[bi_eff],widths[bj_eff]):,}")
    for system in systems:
        lines.append(f"    {system}: L2(full)={all_l2[system][bi_eff,bj_eff]:.4f}  "
                     f"L2(in-support)={all_l2in[system][bi_eff,bj_eff]:.4f}")

    txt="\n".join(lines)
    print("\n"+txt)
    with open(os.path.join(out_dir,"arch_grid_table.txt"),"w",encoding="utf-8") as f:
        f.write(txt+"\n")

    # LaTeX
    n_sys=len(systems)
    latex=[
        r"\begin{table}[ht!]",
        r"\centering",
        f"\\caption{{Architecture grid (Depth $D$ x Width $W$, GRU hidden $H={hidden}$). "
        f"Relative $L^2$ error (full grid). Bold = recommended configuration.}}",
        r"\begin{tabular}{l" + "c"*(len(widths)*n_sys) + "}",
        r"\toprule",
        "Depth $D$ & " + " & ".join(
            f"$W={w}$ ({SYSTEM_CFGS[s]['name'].split('(')[0].strip()})"
            for s in systems for w in widths) + r" \\",
        r"\midrule",
    ]
    for i,d in enumerate(depths):
        row=f"$D={d}$"
        for s in systems:
            for j,w in enumerate(widths):
                v=all_l2[s][i,j]
                cell=f"\\textbf{{{v:.4f}}}" if (i==bi_acc and j==bj_acc) else f"{v:.4f}"
                row+=f" & {cell}"
        latex.append(row+r" \\")
    latex+=[r"\bottomrule",r"\end{tabular}",r"\label{tab:arch_grid_systems}",r"\end{table}"]
    with open(os.path.join(out_dir,"arch_grid_table.tex"),"w",encoding="utf-8") as f:
        f.write("\n".join(latex)+"\n")
    print(f"  Tables saved: arch_grid_table.[txt|tex]")

    return (depths[bi_acc],widths[bj_acc]),(depths[bi_eff],widths[bj_eff])

# ======================================================================
#  MAIN
# ======================================================================

def main():
    parser=argparse.ArgumentParser(description="Depth x Width architecture grid for eps-US-GRU-S systems")
    parser.add_argument("--system",default="all",choices=["fhn_partial","predator_prey","all"])
    parser.add_argument("--epochs",type=int,default=EPOCHS)
    parser.add_argument("--hidden",type=int,default=DEFAULT_HIDDEN)
    parser.add_argument("--depths",type=str,default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--widths",type=str,default=",".join(str(w) for w in DEFAULT_WIDTHS))
    args=parser.parse_args()

    systems=["fhn_partial","predator_prey"] if args.system=="all" else [args.system]
    depths=[int(d) for d in args.depths.split(",")]
    widths=[int(w) for w in args.widths.split(",")]
    epochs=args.epochs; hidden=args.hidden

    n_runs=len(depths)*len(widths)*len(systems)
    t_per={"fhn_partial":1.8,"predator_prey":6.2}  # sec/epoch (4 ICs), observed
    t_est=sum(t_per.get(s,3)*epochs for s in systems)*len(depths)*len(widths)
    print(f"\nArchitecture grid: D in {depths}  W in {widths}  H={hidden}")
    print(f"{n_runs} runs x {epochs} epochs  ~{t_est/3600:.1f}h total")
    print("Tip: --system fhn_partial or smaller --depths/--widths to reduce scope.\n")

    print("Pre-computing shared data per system...")
    shared={}
    for s in systems:
        print(f"  {SYSTEM_CFGS[s]['name']}...")
        shared[s]=precompute_shared(s)

    all_l2={s:np.zeros((len(depths),len(widths))) for s in systems}
    all_l2in={s:np.zeros((len(depths),len(widths))) for s in systems}
    all_params=np.zeros((len(depths),len(widths)))

    for i,d in enumerate(depths):
        for j,w in enumerate(widths):
            n_p=count_params(hidden,d,w); all_params[i,j]=n_p
            print(f"\n  [D={d}, W={w}]  params={n_p:,}")
            for s in systems:
                print(f"    {SYSTEM_CFGS[s]['name']}...",end="  ",flush=True)
                t0=time.time()
                try:
                    r=run_one(s,hidden,d,w,shared[s],epochs)
                    print(f"L2={r['l2']:.4f}  L2_in={r['l2_in']:.4f}  ({time.time()-t0:.0f}s)")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    r=dict(l2=float("nan"),l2_in=float("nan"),params=n_p,train_s=0)
                all_l2[s][i,j]=r["l2"]; all_l2in[s][i,j]=r["l2_in"]

    # Per-system heatmaps
    for s in systems:
        plot_system_heatmap(s,depths,widths,hidden,all_l2[s],all_l2in[s],OUT_DIR)

    # Joint normalised score (min-max per system, then average)
    norm_l2={}
    for s in systems:
        Z=all_l2[s]; mn,mx=Z.min(),Z.max()
        norm_l2[s]=(Z-mn)/(mx-mn+1e-12)
    joint_score=np.mean([norm_l2[s] for s in systems],axis=0)

    # Efficiency score: joint + 0.25 * normalised(params)
    pmn,pmx=all_params.min(),all_params.max()
    norm_params=(all_params-pmn)/(pmx-pmn+1e-12)
    eff_score=joint_score+0.25*norm_params

    if len(systems)>1:
        plot_joint_heatmap(depths,widths,hidden,joint_score,eff_score,all_l2,systems,OUT_DIR)
    else:
        plot_joint_heatmap(depths,widths,hidden,joint_score,eff_score,all_l2,systems,OUT_DIR)

    rec_acc,rec_eff=save_tables(systems,depths,widths,hidden,all_l2,all_l2in,
                                all_params,joint_score,eff_score,OUT_DIR)

    # Save JSON
    save={
        "hidden":hidden,"depths":depths,"widths":widths,"epochs":epochs,
        "l2":{s:all_l2[s].tolist() for s in systems},
        "l2_in":{s:all_l2in[s].tolist() for s in systems},
        "params":all_params.tolist(),
        "joint_score":joint_score.tolist(),
        "efficiency_score":eff_score.tolist(),
        "recommended_accuracy":{"D":rec_acc[0],"W":rec_acc[1]},
        "recommended_efficient":{"D":rec_eff[0],"W":rec_eff[1]},
    }
    with open(os.path.join(OUT_DIR,"arch_grid_results.json"),"w") as f:
        json.dump(save,f,indent=2)

    print(f"\n{'='*65}\n  SUMMARY\n{'='*65}")
    print(f"  Most-effective configuration (joint, both systems): D={rec_acc[0]}, W={rec_acc[1]}, H={hidden}")
    print(f"    params = {count_params(hidden,rec_acc[0],rec_acc[1]):,}")
    print(f"  Parameter-efficient alternative:                     D={rec_eff[0]}, W={rec_eff[1]}, H={hidden}")
    print(f"    params = {count_params(hidden,rec_eff[0],rec_eff[1]):,}")

if __name__=="__main__": main()