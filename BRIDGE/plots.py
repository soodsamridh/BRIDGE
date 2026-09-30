import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, torch
from equations import get_R_true

PLOT_DIR = "BRIDGE_output"


def _d(p):
    d = os.path.join(PLOT_DIR, p)
    os.makedirs(d, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════
#  Reaction identification
# ══════════════════════════════════════════════════════════

def plot_reaction(model, u_bins, R_bins, problem, cfg, device):
    d      = _d(problem)
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    u_t    = torch.tensor(u_grid, dtype=torch.float32, device=device)
    R_pred = model.reaction_from_u_only_nograd(u_t).cpu().numpy()
    R_true = get_R_true(problem, u_grid, cfg)
    l2     = (np.sqrt(np.mean((R_pred - R_true) ** 2)) /
              (np.sqrt(np.mean(R_true ** 2)) + 1e-8))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(u_grid, R_true, "k-",  lw=2.5,
            label="$R_{\\mathrm{true}}(u)$", zorder=5)
    ax.plot(u_grid, R_pred, "r--", lw=2.0,
            label="$R_\\theta$ (model)", zorder=4)
    if u_bins is not None and len(u_bins) > 0:
        ax.scatter(u_bins, R_bins, s=22, color="steelblue", alpha=0.75,
                   label="FD targets (auxiliary)", zorder=6)
    ax.set_xlabel("$u$", fontsize=12)
    ax.set_ylabel("$R(u)$", fontsize=12)
    ax.set_title(f"{cfg['name']} — Reaction Identification  "
                 f"($\\varepsilon_R = {l2:.4f}$)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(d, f"reaction_{problem}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}  (L2={l2:.4f})")
    return l2


# ══════════════════════════════════════════════════════════
#  Trajectory 
# ══════════════════════════════════════════════════════════

def plot_trajectory(x_np, u_ref_np, u_traj, t_arr_np, problem, cfg):
    d     = _d(problem)
    Nt    = cfg["Nt"]
    snaps = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
    fig.suptitle(f"{cfg['name']} — Predicted vs Reference Trajectory",
                 fontsize=13, fontweight="bold")

    for ax, n in zip(axes, snaps):
        u_r = u_ref_np[n]
        u_p = (u_traj[n].detach().cpu().numpy()
               if hasattr(u_traj[n], "detach")
               else np.array(u_traj[n]))
        ax.plot(x_np, u_r, "k-",  lw=1.8, label="Reference", zorder=4)
        ax.plot(x_np, u_p, "r--", lw=2.5, label="Predicted",  zorder=6)
        ax.set_title(f"$t = {t_arr_np[n]:.3f}$", fontsize=10)
        ax.set_xlabel("$x$", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("$u(x,t)$", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    path = os.path.join(d, f"trajectory_{problem}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  Pointwise error analysis (NEW)
# ══════════════════════════════════════════════════════════

def plot_pointwise_error(x_np, u_ref_np, u_traj, t_arr_np, problem, cfg):
    d  = _d(problem)
    Nt = cfg["Nt"]

    # Build error matrix  shape = (Nt+1, Nx)
    err_mat = np.zeros((Nt + 1, len(x_np)))
    for n in range(Nt + 1):
        u_r = u_ref_np[n]
        u_p = (u_traj[n].detach().cpu().numpy()
               if hasattr(u_traj[n], "detach")
               else np.array(u_traj[n]))
        err_mat[n] = np.abs(u_r - u_p)

    err_flat = err_mat.flatten()
    l2_traj  = float(np.sqrt(np.mean(err_flat ** 2)))
    mean_err = float(err_flat.mean())
    max_err  = float(err_flat.max())
    T        = float(t_arr_np[-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{cfg['name']} — Pointwise Error Analysis  "
                 f"(trajectory $L^2 = {l2_traj:.2e}$)",
                 fontsize=13, fontweight="bold")

    # ── A: Space-time heatmap (t on x-axis, x on y-axis) ──
    ax = axes[0]
    vmin = max(err_flat[err_flat > 0].min(), 1e-10) if (err_flat > 0).any() else 1e-10
    vmax = max_err + 1e-12
    im   = ax.imshow(
        err_mat.T,         
        origin="lower",
        aspect="auto",
        extent=[0, T, x_np[0], x_np[-1]],
        cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$|u_{\\mathrm{pred}} - u_{\\mathrm{ref}}|$", fontsize=9)
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel("$x$", fontsize=11)
    ax.set_title("Space-Time Error Heatmap (log scale)", fontsize=10)

    # ── B: Final-time error profile ────────────────────────
    ax = axes[1]
    err_T = err_mat[-1]
    ax.plot(x_np, err_T, color="#C0392B", lw=2.0)
    ax.fill_between(x_np, 0, err_T, alpha=0.25, color="#C0392B")
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$|u_{\\mathrm{pred}} - u_{\\mathrm{ref}}|$", fontsize=11)
    ax.set_title(f"Error Profile at Final Time $t = {T:.3f}$", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ── C: Error histogram ─────────────────────────────────
    ax = axes[2]
    pos_err = err_flat[err_flat > 1e-16]
    if len(pos_err) > 10:
        ax.hist(np.log10(pos_err), bins=50,
                color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_xlabel("$\\log_{10}$(pointwise error)", fontsize=11)
    else:
        ax.hist(err_flat, bins=30,
                color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Pointwise error", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Error Distribution\n"
                 f"mean = {mean_err:.2e},   max = {max_err:.2e}", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(d, f"pointwise_error_{problem}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return l2_traj, mean_err, max_err


# ══════════════════════════════════════════════════════════
#  Training loss
# ══════════════════════════════════════════════════════════

def plot_loss(history, problem, cfg):
    d  = _d(problem)
    ep = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{cfg['name']} — Training Loss History",
                 fontsize=12, fontweight="bold")

    axes[0].semilogy(ep, [h["total"] for h in history], "k-", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total loss (log)")
    axes[0].set_title("Total Loss"); axes[0].grid(True, alpha=0.3)

    keys = ["L_data", "L_cons", "L_anch", "L_fd", "L_pos"]
    cols = ["steelblue", "green", "red", "purple", "orange"]
    for key, c in zip(keys, cols):
        vals = [max(h.get(key, 1e-12), 1e-12) for h in history]
        if max(vals) > 1e-11:
            axes[1].semilogy(ep, vals, lw=1.2, label=key, color=c)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Component loss (log)")
    axes[1].set_title("Loss Components")

    plt.tight_layout()
    path = os.path.join(d, f"loss_{problem}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  Identifiability scatter
# ══════════════════════════════════════════════════════════

def plot_scatter(all_u_trajs, all_R_trajs, problem, cfg, device, step=5):
    d      = _d(problem)
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    R_true = get_R_true(problem, u_grid, cfg)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_grid, R_true, "k-", lw=2.5,
            label="$R_{\\mathrm{true}}(u)$", zorder=10)
    cols = plt.cm.tab10(np.linspace(0, 0.7, max(len(all_u_trajs), 1)))
    for ic_idx, (u_traj, R_traj) in enumerate(zip(all_u_trajs, all_R_trajs)):
        ul, Rl = [], []
        for n in range(0, len(R_traj), step):
            ul.append(u_traj[n].detach().cpu().numpy())
            Rl.append(R_traj[n].detach().cpu().numpy())
        ax.scatter(np.concatenate(ul), np.concatenate(Rl),
                   s=2, alpha=0.2, color=cols[ic_idx], label=f"IC{ic_idx+1}")
    ax.set_xlabel("$u$", fontsize=12)
    ax.set_ylabel("$R(u)$", fontsize=12)
    ax.set_title(f"{cfg['name']} — Identifiability Scatter", fontsize=12)
    ax.legend(fontsize=9, markerscale=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(d, f"scatter_{problem}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  Epsilon sensitivity
# ══════════════════════════════════════════════════════════

def plot_eps_sensitivity(eps_results, problem, pname):
    os.makedirs("results_v5", exist_ok=True)
    eps_vals = [r[0] for r in eps_results]
    l2_model = [r[1] for r in eps_results]
    l2_dist  = [r[2] for r in eps_results]
    labels   = [f"{e:.0e}" for e in eps_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{pname} — Relative $L^2$ Error vs Perturbation Parameter $\\varepsilon$",
        fontsize=13, fontweight="bold")

    # ── Line plot ──────────────────────────────────────────
    ax = axes[0]
    ax.semilogx(eps_vals, l2_model, "o-", color="#3498DB",
                lw=2.0, ms=7, label="Model $\\varepsilon_R$")
    ax.semilogx(eps_vals, l2_dist,  "s--", color="#C0392B",
                lw=2.0, ms=7, label="Distilled $\\varepsilon_R$")
    ax.invert_xaxis()   # large ε on left = less singular
    ax.set_xlabel("$\\varepsilon$  (perturbation parameter)", fontsize=11)
    ax.set_ylabel("Relative $L^2$ Error $\\varepsilon_R$", fontsize=11)
    ax.set_title("$\\varepsilon_R$ vs $\\varepsilon$\n"
                 "(right = more singular)", fontsize=10)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which="both")

    # ── Bar chart ──────────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(eps_vals)); w = 0.35
    b1 = ax.bar(x - w/2, l2_model, w, color="#3498DB",
                edgecolor="black", linewidth=0.8,
                label="Model $\\varepsilon_R$")
    b2 = ax.bar(x + w/2, l2_dist,  w, color="#C0392B",
                edgecolor="black", linewidth=0.8, alpha=0.85,
                label="Distilled $\\varepsilon_R$")
    for bar, val in zip(list(b1) + list(b2), l2_model + l2_dist):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\varepsilon$={lb}" for lb in labels], fontsize=9)
    ax.set_ylabel("Relative $L^2$ Error $\\varepsilon_R$", fontsize=11)
    ax.set_title("$\\varepsilon_R$ per Epsilon Value", fontsize=10)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_ylim(0, max(l2_model + l2_dist) * 1.30)

    plt.tight_layout()
    path = f"results_v5/eps_sensitivity_{problem}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
