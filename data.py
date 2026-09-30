import numpy as np
import scipy.linalg
from equations import get_ics, get_R_true, get_bc_values


def generate_reference(problem, x, cfg, L_np, lu, piv):
    """Generate one reference trajectory per IC using R_true."""
    Nt       = cfg["Nt"]
    dt       = cfg["dt"]
    eps_diff = cfg["eps_diff"]
    periodic = cfg["bc_type"] == "periodic"
    u_min    = cfg["u_min"]
    u_max    = cfg["u_max"]
    t_arr    = np.linspace(0, cfg["T"], Nt+1)
    ics      = get_ics(problem, x, cfg)
    u_refs   = []

    for k, u0 in enumerate(ics):
        u_ref = np.zeros((Nt+1, len(x)))
        u_ref[0] = u0.copy()
        for n in range(Nt):
            u_n  = u_ref[n].copy()
            R_n  = get_R_true(problem, u_n, cfg)
            rhs  = u_n + (dt/2)*eps_diff*(L_np@u_n) + dt*R_n
            if not periodic:
                gL, gR = get_bc_values(problem, t_arr[n+1], cfg)
                rhs[0] = gL; rhs[-1] = gR
            u_ref[n+1] = np.clip(
                scipy.linalg.lu_solve((lu, piv), rhs),
                u_min-0.05, u_max+0.05)
        u_refs.append(u_ref)
        print(f"    IC {k+1}/{len(ics)}: u ∈ [{u_ref.min():.4f},{u_ref.max():.4f}]")

    all_u = np.concatenate([r.flatten() for r in u_refs])
    print(f"    Combined: [{all_u.min():.4f}, {all_u.max():.4f}]")
    return u_refs, t_arr


def extract_fd_pairs(u_refs, L_np, cfg):
    dt        = cfg["dt"]
    eps_diff  = cfg["eps_diff"]
    fd_filter = cfg.get("fd_filter", "lap")
    fd_pct    = cfg.get("fd_pct", 60)
    all_u, all_R = [], []

    for u_ref in u_refs:
        Nt = u_ref.shape[0]-1
        for n in range(1, Nt-1):
            u_prev = u_ref[n-1]; u_curr = u_ref[n]; u_next = u_ref[n+1]
            dt_u   = (u_next-u_prev)/(2*dt)
            R_fd   = dt_u - eps_diff*(L_np@u_curr)

            if fd_filter == "slow":
                mag  = np.abs(dt_u)
            else:
                mag  = np.abs(L_np@u_curr)

            mask     = mag < np.percentile(mag, fd_pct)
            mask[0]  = False; mask[-1] = False
            all_u.append(u_curr[mask])
            all_R.append(R_fd[mask])

    u_fd = np.concatenate(all_u)
    R_fd = np.concatenate(all_R)
    R_std = np.std(R_fd); R_med = np.median(R_fd)
    valid = np.abs(R_fd - R_med) < 4*R_std
    return u_fd[valid], R_fd[valid]


def aggregate_bins(u_fd, R_fd, cfg, problem):
    import config as C
    u_min = cfg["u_min"]; u_max = cfg["u_max"]
    n_bins = C.N_BINS

    edges   = np.linspace(u_min, u_max, n_bins+1)
    centres = 0.5*(edges[:-1]+edges[1:])
    bin_med = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins, dtype=int)
    bin_idx = np.clip(np.digitize(u_fd, edges)-1, 0, n_bins-1)

    for b in range(n_bins):
        m = bin_idx==b
        if m.sum() >= C.MIN_BIN_SAMPLES:
            bin_med[b] = np.median(R_fd[m])
            bin_cnt[b] = m.sum()

    nonempty = ~np.isnan(bin_med)
    u_b = centres[nonempty]; R_b = bin_med[nonempty]

    # Quality: compare against R_true to filter unreliable bins
    R_check  = get_R_true(problem, u_b, cfg)
    abs_err  = np.abs(R_b - R_check)
    rel_err  = abs_err / (np.abs(R_check) + 1e-4)
    Lambda   = cfg["Lambda"]
    margin   = 0.02*(u_max-u_min)

    abs_thresh = Lambda * 0.05
    quality_ok = (abs_err < abs_thresh) | (rel_err < C.QUALITY_THRESH)

    keep = (quality_ok &
            (np.abs(R_b) < Lambda*1.5) &
            (u_b > u_min+margin) &
            (u_b < u_max-margin))

    u_t = u_b[keep]; R_t = R_b[keep]
    abs_q = abs_err[keep]
    weights = np.clip(1.0/(abs_q + Lambda*0.01), 0.1, 10.0)
    weights = weights / weights.mean()
    return u_t, R_t, weights


def compute_normalisers(u_refs, L_np, cfg):
    eps_diff = cfg["eps_diff"]
    all_u = np.concatenate([r.flatten() for r in u_refs])
    s_u   = float(np.mean(np.abs(all_u)))

    # Physical diffusion scale from reference
    lap_scales, R_scales = [], []
    for u_ref in u_refs:
        n_mid = u_ref.shape[0]//2
        lap_scales.append(eps_diff * np.mean(np.abs(L_np @ u_ref[n_mid])))
        # FD-based R estimate at mid snapshot (outer nodes only)
        dt = cfg["dt"]
        if n_mid > 0 and n_mid < u_ref.shape[0]-1:
            dt_u  = (u_ref[n_mid+1]-u_ref[n_mid-1])/(2*dt)
            lap_u = eps_diff*(L_np@u_ref[n_mid])
            R_est = dt_u - lap_u
            lap_mag = np.abs(L_np@u_ref[n_mid])
            outer = lap_mag < np.percentile(lap_mag, 60)
            if outer.sum() > 0:
                R_scales.append(np.mean(np.abs(R_est[outer])))

    lap_scale = float(np.mean(lap_scales))
    R_scale   = float(np.mean(R_scales)) if R_scales else 0.5
    s_res     = max(lap_scale + R_scale, 1e-4)

    return {"s_u": s_u, "s_res": s_res}
