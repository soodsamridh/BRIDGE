"""
loss.py  —  v5 loss functions (final).

L_res is REMOVED. Reason:
  The IMEX truncation error gives mean residual ~0.9 even with correct R.
  L_res with s_res=1 gives L_res=5.1 at R_true → explodes.
  L_res and L_data have opposing gradients → training diverges.
  L_res is mathematically redundant: the IMEX scheme already enforces
  û^{n+1} = A^{-1}(û^n + dt*R_n). L_data penalises û != u_ref.
  Together they force R_theta to be correct. L_res adds nothing.

Active loss terms:
  L_data  — trajectory matching (PRIMARY — R is learned here)
  L_cons  — scalar law identifiability
  L_anch  — zero-crossings from PDE equilibria
  L_fd    — soft FD shape guidance (auxiliary)
  L_pos   — reaction positivity (Fisher only)
"""

import torch
import config as C


def loss_data(u_traj, u_ref_t, s_u):
    Nt = len(u_traj)-1
    return torch.stack([
        ((u_traj[n]-u_ref_t[n])/(s_u+1e-8)).pow(2).mean()
        for n in range(Nt+1)
    ]).mean()


def loss_consistency(u_traj, R_traj, tau, B=64):
    Nt   = len(R_traj)
    step = max(1, Nt//20)
    u_all = torch.cat([u_traj[n] for n in range(0,Nt,step)])
    R_all = torch.cat([R_traj[n] for n in range(0,Nt,step)])
    idx   = torch.argsort(u_all)
    u_s   = u_all[idx]; R_s = R_all[idx]
    du    = torch.abs(u_s[1:]-u_s[:-1])
    mask  = du < tau
    if mask.sum() < 2:
        return torch.tensor(0.0, device=u_all.device)
    valid = torch.where(mask)[0]
    if len(valid) > B:
        valid = valid[torch.randperm(len(valid),device=u_all.device)[:B]]
    du_v  = du[valid]
    dR_sq = (R_s[valid]-R_s[valid+1]).pow(2)
    w     = (1.0-du_v/tau).clamp(0,1).pow(2)
    return (w*dR_sq).mean()


def loss_anchor(model, anchors, device):
    if not anchors:
        return torch.tensor(0.0, device=device)
    u_a = torch.tensor([a[0] for a in anchors],
                        dtype=torch.float32, device=device)
    R_a = torch.tensor([a[1] for a in anchors],
                        dtype=torch.float32, device=device)
    return (model.reaction_from_u_only(u_a)-R_a).pow(2).mean()


def loss_fd(model, u_fd_t, R_fd_t, weights_t, Lambda):
    """Soft FD auxiliary — normalised by Lambda^2."""
    if u_fd_t is None or len(u_fd_t) == 0:
        return torch.tensor(0.0)
    R_pred = model.reaction_from_u_only(u_fd_t)
    sq_err = (R_pred-R_fd_t).pow(2)/(Lambda**2+1e-8)
    return (weights_t*sq_err).mean()


def loss_positivity(R_traj):
    return torch.stack([
        torch.relu(-R_traj[n]).pow(2).mean()
        for n in range(len(R_traj))
    ]).mean()


def total_loss(u_traj, R_traj, u_ref_t, L_op_t, cfg,
               s_u, epoch, anchors, model, device,
               u_fd_t=None, R_fd_t=None, weights_t=None):
    Lambda = cfg["Lambda"]
    tau    = C.TAU_CONS

    # Per-problem lambda_fd override (e.g. FHN needs stronger FD anchoring)
    lambda_fd = cfg.get("lambda_fd", C.LAMBDA_FD)

    Ld   = C.LAMBDA_DATA * loss_data(u_traj, u_ref_t, s_u)
    Lcon = C.LAMBDA_CONS * loss_consistency(u_traj, R_traj, tau)
    Lanc = C.LAMBDA_ANCH * loss_anchor(model, anchors, device)
    Lfd  = lambda_fd     * loss_fd(model, u_fd_t, R_fd_t, weights_t, Lambda)

    Lpos = torch.tensor(0.0, device=device)
    if cfg.get("positivity", False):
        Lpos = C.LAMBDA_POS * loss_positivity(R_traj)

    L = Ld + Lcon + Lanc + Lfd + Lpos

    return L, {
        "L_data":  Ld.item(),
        "L_cons":  Lcon.item(),
        "L_anch":  Lanc.item(),
        "L_fd":    Lfd.item(),
        "L_pos":   Lpos.item(),
        "total":   L.item(),
    }
