import sys, os, time, json, copy, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from numerics  import build_all_operators
from data      import generate_reference, extract_fd_pairs, aggregate_bins
from equations import get_R_true, get_bc_values
from model     import EpsUSGRU
from trainer   import train
from baselines import (run_pinn_baseline, run_fdpinn_baseline,
                       train_pirnn, pirnn_rollout,
                       sweep_weight, count_parameters)

SEED     = 42
EPOCHS   = 1000
HIDDEN   = 32
OUT_DIR  = "comparison_results"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")       
SWEEP_WEIGHTS = (0.01, 0.1, 1.0, 10.0)

METHODS = ["PINN", "FDPINN", "PIRNN", "Proposed"]

STYLES = {
    "PINN":     {"color": "#E07B54", "ls": "--", "lw": 1.8},
    "FDPINN":   {"color": "#2980B9", "ls": "-.", "lw": 2.0},
    "PIRNN":    {"color": "#27AE60", "ls": ":",  "lw": 2.0},
    "Proposed": {"color": "#C0392B", "ls": "-",  "lw": 2.5},
}
REF_STYLE = {"color": "black", "ls": "-", "lw": 2.5}

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
#  Shared evaluation
# ══════════════════════════════════════════════════════════

def rollout_reaction_only(R_net, u0_np, x_t, A_t, L_t, cfg, problem,
                          t_arr_np):
    Nt = cfg["Nt"]; dt = cfg["dt"]; eps = cfg["eps_diff"]
    periodic = cfg["bc_type"] == "periodic"
    dev = x_t.device
    u_n = torch.tensor(u0_np, dtype=torch.float32, device=dev)
    traj = [u_n.cpu().numpy()]
    with torch.no_grad():
        for n in range(Nt):
            R = R_net.reaction_nograd(u_n)
            rhs = u_n + (dt / 2) * eps * (L_t @ u_n) + dt * R
            if not periodic:
                gL, gR = get_bc_values(problem, float(t_arr_np[n + 1]), cfg)
                rhs = rhs.clone()
                rhs[0] = torch.as_tensor(gL, dtype=torch.float32, device=dev)
                rhs[-1] = torch.as_tensor(gR, dtype=torch.float32, device=dev)
            u_n = torch.linalg.solve(A_t, rhs.unsqueeze(-1)).squeeze(-1)
            u_n = torch.clamp(u_n, cfg["u_min"], cfg["u_max"])
            traj.append(u_n.cpu().numpy())
    return np.stack(traj)


def traj_error(u_pred, u_ref):
    n = min(len(u_pred), len(u_ref))
    num = np.sqrt(np.mean((u_pred[:n] - u_ref[:n]) ** 2))
    den = np.sqrt(np.mean(u_ref[:n] ** 2)) + 1e-8
    return float(num / den)


def eps_R_of_net(R_net, u_grid, R_true, device):
    Rp = R_net.reaction_nograd(
        torch.tensor(u_grid, dtype=torch.float32,
                     device=device)).cpu().numpy()
    return (float(np.sqrt(np.mean((Rp - R_true) ** 2)) /
                  (np.sqrt(np.mean(R_true ** 2)) + 1e-8)), Rp)


# ══════════════════════════════════════════════════════════
#  Proposed
# ══════════════════════════════════════════════════════════

def run_ours(problem, cfg, u_refs_np, t_arr_np, L_np, A_t, L_t,
             u_tgt, R_tgt, weights, x_np, epochs=EPOCHS):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = DEVICE
    model = EpsUSGRU(hidden_dim=HIDDEN, mlp_width=32,
                     Lambda=cfg["Lambda"], input_dim=4).to(dev)
    n_params = count_parameters(model)
    print(f"    Proposed  ({n_params} params, {epochs} ep, {dev})",
          flush=True)
    history, all_ut, _ = train(
        model, problem, cfg, x_np, u_refs_np, t_arr_np,
        L_np, A_t, L_t, cfg["anchors"], dev,
        u_fd_np=u_tgt, R_fd_np=R_tgt, w_fd_np=weights, epochs=epochs)
    model.eval()
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    R_pred = model.reaction_from_u_only_nograd(
        torch.tensor(u_grid, dtype=torch.float32, device=dev)).cpu().numpy()
    R_true = get_R_true(problem, u_grid, cfg)
    l2 = float(np.sqrt(np.mean((R_pred - R_true) ** 2)) /
               (np.sqrt(np.mean(R_true ** 2)) + 1e-8))
    # LAMBDA_DATA = 1 in config.py, so the logged L_data is the raw
    # normalised misfit and is comparable with the other methods.
    hist = [{"epoch": k + 1, "total": h["total"], "data_misfit": h["L_data"]}
            for k, h in enumerate(history)]
    traj = np.stack([u.detach().cpu().numpy() for u in all_ut[0]])
    return l2, R_pred, u_grid, R_true, hist, traj, n_params


# ══════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════

def plot_reaction(results, problem, cfg, pname):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    u_g = np.array(results["Proposed"]["u_grid"])
    ax.plot(u_g, get_R_true(problem, u_g, cfg), **REF_STYLE,
            label=r"$R_{\mathrm{true}}(u)$", zorder=10)
    for m in METHODS:
        s = STYLES[m]; d = results[m]
        ax.plot(np.array(d["u_grid"]), np.array(d["R_pred"]),
                color=s["color"], linestyle=s["ls"], lw=s["lw"],
                label=f"{m}  ($\\varepsilon_R$={d['l2']:.4f})", zorder=5)
    ax.set_xlabel("$u$", fontsize=12); ax.set_ylabel("$R(u)$", fontsize=12)
    ax.set_title(f"{pname} — identified reaction law",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.92); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_trajectory(results, problem, cfg, pname, x_np):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_np, np.array(results["Proposed"]["u_final_ref"]), **REF_STYLE,
            label=f"Reference $u(x,T={cfg['T']})$", zorder=10)
    for m in METHODS:
        s = STYLES[m]
        ax.plot(x_np, np.array(results[m]["u_final_pred"]),
                color=s["color"], linestyle=s["ls"], lw=s["lw"],
                label=f"{m}  ($\\varepsilon_u$={results[m]['l2_u']:.2e})",
                zorder=5)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel(f"$u(x,T={cfg['T']})$", fontsize=12)
    ax.set_title(f"{pname} — final-time trajectory",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.92); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_misfit(results, pname):
    """Cross-method figure -- this is the Figure 16 quantity."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in METHODS:
        s = STYLES[m]; h = results[m]["history"]
        ax.semilogy([d["epoch"] for d in h],
                    [max(d["data_misfit"], 1e-12) for d in h],
                    color=s["color"], linestyle=s["ls"], lw=s["lw"], label=m)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Normalised data misfit", fontsize=12)
    ax.set_title(f"{pname} — common data misfit",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.92)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    global DEVICE
    ap = argparse.ArgumentParser(description="Method comparison for BRIDGE")
    ap.add_argument("--problem", default="all",
                    choices=["fisher", "allen_cahn", "all"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the physics-loss weight for each baseline "
                         "using a VALIDATION trajectory (not eps_R) "
                         "and keep the best by validation error")
    ap.add_argument("--sweep-epochs", type=int, default=300,
                    help="epochs per sweep point (default 300)")
    args = ap.parse_args()

    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {DEVICE}")

    problems = (["fisher", "allen_cahn"] if args.problem == "all"
                else [args.problem])
    all_results = {}
    sweeps = {}

    print(f"Epochs per method: {args.epochs}")
    print("Baselines are in their own papers' formulations; no BRIDGE "
          "loss term is used by any of them.\n")

    for problem in problems:
        cfg = copy.deepcopy(PROBLEMS[problem])
        pname = cfg["name"]
        print(f"\n{'='*64}\n  Problem: {pname}\n{'='*64}")

        x, L_np, _, lu, piv, L_t, A_t = build_all_operators(problem, cfg,
                                                            DEVICE)
        print("  Generating trajectories...")
        u_refs, t_arr = generate_reference(problem, x, cfg, L_np, lu, piv)
        # FD pairs are used ONLY by the proposed method (its L_fd term).
        print("  Extracting FD pairs (proposed method only)...")
        u_fd_raw, R_fd_raw = extract_fd_pairs(u_refs, L_np, cfg)
        u_tgt, R_tgt, weights = aggregate_bins(u_fd_raw, R_fd_raw, cfg,
                                               problem)

        # ONE state_scale for every method
        n_roll = min(cfg["n_rollout_ics"], len(u_refs))
        su = float(np.mean(np.abs(
            np.concatenate([r.flatten() for r in u_refs[:n_roll]]))))
        print(f"  state_scale = {su:.4f}  (shared by all methods)")

        u_ref_traj = u_refs[0]
        u0_np = u_refs[0][0]
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
        R_true = get_R_true(problem, u_grid, cfg)
        pres = {}; sweeps[problem] = {}

        def eps_R_of(R_net):
            return eps_R_of_net(R_net, u_grid, R_true, DEVICE)[0]

        def val_error(R_net):
            val_idx = min(1, len(u_refs) - 1)
            u0_val = u_refs[val_idx][0]
            traj = rollout_reaction_only(R_net, u0_val, x_t, A_t, L_t,
                                         cfg, problem, t_arr)
            return traj_error(traj, u_refs[val_idx])

        shared = dict(cfg=cfg, refs=u_refs, t_arr=t_arr, x=x, L_t=L_t,
                      seed=SEED, device=DEVICE, state_scale=su)

        # ---- PINN and FDPINN ------------------------------------------
        for tag, runner in (("PINN", run_pinn_baseline),
                            ("FDPINN", run_fdpinn_baseline)):
            print(f"\n  [{tag}]")
            t0 = time.time()
            if args.sweep:
                print(f"    sweeping w_f over {SWEEP_WEIGHTS} "
                      f"at {args.sweep_epochs} epochs")
                best_w, recs = sweep_weight(
                    runner, val_error, SWEEP_WEIGHTS, key="w_f",
                    epochs=args.sweep_epochs, coupled=False,
                    verbose=False, **shared)
                sweeps[problem][tag] = recs
                print(f"    best w_f = {best_w:g}; retraining at "
                      f"{args.epochs} epochs")
            else:
                best_w = 1.0
                sweeps[problem][tag] = None
            u_net, R_net, hist, npar = runner(
                epochs=args.epochs, coupled=False, w_f=best_w, **shared)
            l2, Rp = eps_R_of_net(R_net, u_grid, R_true, DEVICE)
            traj = rollout_reaction_only(R_net, u0_np, x_t, A_t, L_t,
                                         cfg, problem, t_arr)
            l2u = traj_error(traj, u_ref_traj)
            print(f"  eps_R = {l2:.4f}   eps_u = {l2u:.4e}   "
                  f"({time.time()-t0:.0f}s, {npar} params)")
            pres[tag] = dict(l2=l2, l2_u=l2u, weight=float(best_w),
                             R_pred=Rp.tolist(), u_grid=u_grid.tolist(),
                             R_true=R_true.tolist(), history=hist,
                             n_params=npar,
                             u_final_pred=traj[-1].tolist(),
                             u_final_ref=u_ref_traj[-1].tolist())

        # ---- PIRNN ----------------------------------------------------
        print("\n  [PIRNN]")
        t0 = time.time()
        if args.sweep:
            print(f"    sweeping w_G over {SWEEP_WEIGHTS} "
                  f"at {args.sweep_epochs} epochs")
            best_w, recs = sweep_weight(
                train_pirnn, val_error, SWEEP_WEIGHTS, key="w_G",
                epochs=args.sweep_epochs, verbose=False, **shared)
            sweeps[problem]["PIRNN"] = recs
            print(f"    best w_G = {best_w:g}; retraining at "
                  f"{args.epochs} epochs")
        else:
            best_w = 1.0
            sweeps[problem]["PIRNN"] = None
        rnn, R_net, hist, npar = train_pirnn(
            epochs=args.epochs, w_G=best_w, **shared)
        l2, Rp = eps_R_of_net(R_net, u_grid, R_true, DEVICE)
        # PIRNN predicts the state itself, so its trajectory is its own
        # autonomous rollout, not an IMEX solve.
        traj = pirnn_rollout(rnn, u0_np, cfg, DEVICE)
        l2u = traj_error(traj, u_ref_traj)
        print(f"  eps_R = {l2:.4f}   eps_u = {l2u:.4e}   "
              f"({time.time()-t0:.0f}s, {npar} params)")
        pres["PIRNN"] = dict(l2=l2, l2_u=l2u, weight=float(best_w),
                             R_pred=Rp.tolist(), u_grid=u_grid.tolist(),
                             R_true=R_true.tolist(), history=hist,
                             n_params=npar,
                             u_final_pred=traj[-1].tolist(),
                             u_final_ref=u_ref_traj[-1].tolist())

        # ---- Proposed -------------------------------------------------
        print("\n  [Proposed — BRIDGE]")
        t0 = time.time()
        l2, Rp, ug, Rt, hist, traj, npar = run_ours(
            problem, cfg, u_refs, t_arr, L_np, A_t, L_t,
            u_tgt, R_tgt, weights, x, epochs=args.epochs)
        l2u = traj_error(traj, u_ref_traj)
        print(f"  eps_R = {l2:.4f}   eps_u = {l2u:.4e}   "
              f"({time.time()-t0:.0f}s, {npar} params)")
        pres["Proposed"] = dict(l2=l2, l2_u=l2u, R_pred=Rp.tolist(),
                                u_grid=ug.tolist(), R_true=Rt.tolist(),
                                history=hist, n_params=npar,
                                u_final_pred=traj[-1].tolist(),
                                u_final_ref=u_ref_traj[-1].tolist())

        all_results[problem] = pres

        for maker, stem, note in (
                (lambda: plot_reaction(pres, problem, cfg, pname),
                 "reaction", ""),
                (lambda: plot_trajectory(pres, problem, cfg, pname, x),
                 "trajectory", ""),
                (lambda: plot_misfit(pres, pname),
                 "misfit", "   <- Figure 16 quantity")):
            fig = maker()
            p = os.path.join(OUT_DIR, f"comparison_{stem}_{problem}.png")
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  Saved: {p}{note}")

    save = {p: {m: {k: v for k, v in d.items() if k != "history"}
                for m, d in all_results[p].items()} for p in all_results}
    with open(os.path.join(OUT_DIR, "comparison_table.json"), "w") as f:
        json.dump(save, f, indent=2)
    with open(os.path.join(OUT_DIR, "comparison_histories.json"), "w") as f:
        json.dump({p: {m: all_results[p][m]["history"] for m in all_results[p]}
                   for p in all_results}, f)
    with open(os.path.join(OUT_DIR, "comparison_sweeps.json"), "w") as f:
        json.dump(sweeps, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  {'Method':<12}" + "".join(
        f"{PROBLEMS[p]['name'] + ' eps_R':>22}" for p in problems))
    print("  " + "-" * 62)
    for m in METHODS:
        row = f"  {m:<12}"
        for p in problems:
            d = all_results[p][m]
            row += f"{d['l2']:>13.4f} ({d['l2_u']:.1e})"
        print(row + ("   <- ours" if m == "Proposed" else ""))
    print("\n  (eps_u in parentheses)")

    print(f"\n  {'Method':<12}{'trainable parameters':>22}")
    print("  " + "-" * 34)
    for m in METHODS:
        print(f"  {m:<12}{all_results[problems[0]][m]['n_params']:>22d}")
    print()


if __name__ == "__main__":
    main()