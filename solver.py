import torch
import numpy as np
from regime import regime_indicator, layer_envelope
from equations import get_bc_values


# ══════════════════════════════════════════════════════════
#  Scalar IMEX step
# ══════════════════════════════════════════════════════════

def imex_step(u_n, u_prev, model, H_out, H_lay,
              x_t, phi_t, A_t, L_op_t, cfg, problem, t_next, device):
    eps_diff = cfg["eps_diff"]
    dt       = cfg["dt"]
    periodic = cfg["bc_type"] == "periodic"
    u_min    = cfg["u_min"]
    u_max    = cfg["u_max"]
    sqrt_eps = eps_diff**0.5

    with torch.no_grad():
        delta_h = L_op_t @ u_n
    rho = regime_indicator(delta_h, eps_diff)

    dt_u  = (u_n.detach() - u_prev.detach()) / (dt + 1e-12)
    z_out = torch.stack([u_n, delta_h, dt_u, x_t], dim=-1)
    z_lay = torch.stack([u_n, eps_diff*delta_h, phi_t,
                          sqrt_eps*torch.abs(delta_h)], dim=-1)

    R_n, H_out_n, H_lay_n = model(u_n, z_out, z_lay, rho, H_out, H_lay)

    rhs = u_n + (dt/2)*eps_diff*(L_op_t@u_n) + dt*R_n

    if not periodic:
        gL, gR = get_bc_values(problem, t_next, cfg)
        rhs = rhs.clone()
        rhs[0]  = torch.tensor(gL, dtype=torch.float32, device=device)
        rhs[-1] = torch.tensor(gR, dtype=torch.float32, device=device)

    u_next = torch.linalg.solve(A_t, rhs.unsqueeze(-1)).squeeze(-1)
    u_next = torch.clamp(u_next, u_min, u_max)
    return u_next, R_n, H_out_n, H_lay_n


def rollout_tbptt(model, u0_t, u_ref_t, x_t, phi_t, A_t, L_op_t,
                  cfg, problem, t_arr, device, window):
    Nt = cfg["Nt"]
    Nx = len(x_t)
    H_out, H_lay = model.init_hidden(Nx, device)
    u_n = u0_t; u_prev = u0_t
    n = 0
    while n < Nt:
        w_start = n; w_end = min(n + window, Nt)
        u_n = u_n.detach(); u_prev = u_prev.detach()
        H_out = H_out.detach(); H_lay = H_lay.detach()
        u_traj_w = [u_n]; R_traj_w = []
        for step in range(w_start, w_end):
            u_next, R_n, H_out, H_lay = imex_step(
                u_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, A_t, L_op_t, cfg, problem,
                float(t_arr[step+1]), device)
            u_traj_w.append(u_next); R_traj_w.append(R_n)
            u_prev = u_n; u_n = u_next
        u_ref_w = u_ref_t[w_start:w_end+1]
        yield u_traj_w, R_traj_w, u_ref_w
        n = w_end


def rollout_full_nograd(model, u0_t, u_ref_t, x_t, phi_t, A_t, L_op_t,
                        cfg, problem, t_arr, device):
    Nt = cfg["Nt"]; Nx = len(x_t)
    H_out, H_lay = model.init_hidden(Nx, device)
    u_traj = [u0_t]; R_traj = []
    u_n = u0_t; u_prev = u0_t
    for n in range(Nt):
        u_next, R_n, H_out, H_lay = imex_step(
            u_n, u_prev, model, H_out, H_lay,
            x_t, phi_t, A_t, L_op_t, cfg, problem,
            float(t_arr[n+1]), device)
        u_traj.append(u_next); R_traj.append(R_n)
        u_prev = u_n; u_n = u_next
    return u_traj, R_traj


# ══════════════════════════════════════════════════════════
#  FHN BVP — IMEX step for u, explicit update for v
# ══════════════════════════════════════════════════════════

def _apply_neumann_rhs(rhs):
    rhs = rhs.clone()
    rhs[0]  = 0.0
    rhs[-1] = 0.0
    return rhs


def imex_step_fhn_bvp(u_n, v_n, u_prev, model, H_out, H_lay,
                       x_t, phi_t, A_u_t, L_op_t, cfg, device):
    D   = cfg["D"]
    dt  = cfg["dt"]
    eps = cfg["eps_diff"]  

    with torch.no_grad():
        delta_h = L_op_t @ u_n          

    rho = regime_indicator(delta_h, eps) 

    # Time derivative feature
    dt_u = (u_n.detach() - u_prev.detach()) / (dt + 1e-12)

    # GRU features
    z_out = torch.stack([u_n, v_n, delta_h, dt_u, x_t], dim=-1)
    z_lay = torch.stack([u_n, v_n, D*delta_h, dt_u, phi_t], dim=-1)

    R_u, H_out_n, H_lay_n = model(u_n, v_n, z_out, z_lay, rho,
                                    H_out, H_lay,
                                    detach_gru=cfg.get("_detach_gru", False))

    # ── IMEX solve for u — DIFFERENTIABLE ─────────────────
    rhs_u = _apply_neumann_rhs(u_n + (dt/2)*D*(L_op_t@u_n) + dt*R_u)
    u_next = torch.linalg.solve(A_u_t, rhs_u.unsqueeze(-1)).squeeze(-1)
    u_next = torch.clamp(u_next, cfg["u_min"], cfg["u_max"])

   
    with torch.no_grad():
        u_det  = u_n.detach()
        v_det  = v_n.detach()
        eps_v  = cfg["eps_v"]
        beta_v = cfg["beta_v"]
        v_next = v_det + dt * eps_v * (-beta_v * v_det + u_det)
        v_next = torch.clamp(v_next, cfg["v_min"], cfg["v_max"])

    return u_next, v_next, R_u, H_out_n, H_lay_n


def rollout_tbptt_fhn_bvp(model, u0, v0, u_ref_t, v_ref_t,
                            x_t, phi_t, A_u_t, L_op_t, cfg, device, window):
    Nt = cfg["Nt"]
    Nx = len(x_t)
    H_out, H_lay = model.init_hidden(Nx, device)
    u_n = u0; v_n = v0; u_prev = u0
    n = 0
    while n < Nt:
        w_start = n; w_end = min(n + window, Nt)
        u_n    = u_n.detach(); v_n = v_n.detach()
        u_prev = u_prev.detach()
        H_out  = H_out.detach(); H_lay = H_lay.detach()
        utw = [u_n]; vtw = [v_n]; Rtuw = []
        for step in range(w_start, w_end):
            u_next, v_next, R_u, H_out, H_lay = imex_step_fhn_bvp(
                u_n, v_n, u_prev, model, H_out, H_lay,
                x_t, phi_t, A_u_t, L_op_t, cfg, device)
            utw.append(u_next); vtw.append(v_next); Rtuw.append(R_u)
            u_prev = u_n; u_n = u_next; v_n = v_next
        yield utw, vtw, Rtuw, u_ref_t[w_start:w_end+1], v_ref_t[w_start:w_end+1]
        n = w_end


@torch.no_grad()
def rollout_nograd_fhn_bvp(model, u0, v0, x_t, phi_t, A_u_t, L_op_t, cfg, device):
    Nt = cfg["Nt"]; Nx = len(x_t)
    H_out, H_lay = model.init_hidden(Nx, device)
    u_traj = [u0]; v_traj = [v0]; R_traj = []
    u_n = u0; v_n = v0; u_prev = u0
    for _ in range(Nt):
        u_next, v_next, R_u, H_out, H_lay = imex_step_fhn_bvp(
            u_n, v_n, u_prev, model, H_out, H_lay,
            x_t, phi_t, A_u_t, L_op_t, cfg, device)
        u_traj.append(u_next); v_traj.append(v_next); R_traj.append(R_u)
        u_prev = u_n; u_n = u_next; v_n = v_next
    return u_traj, v_traj, R_traj