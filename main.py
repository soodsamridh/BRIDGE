import argparse, os, copy, time, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")

import config as C
from config    import PROBLEMS
from numerics  import build_all_operators
from data      import generate_reference, extract_fd_pairs, aggregate_bins
from model     import EpsUSGRU
from trainer   import preinit_mlp, train
from distill   import distil
from plots     import (plot_reaction, plot_trajectory, plot_loss,
                       plot_scatter, plot_eps_sensitivity, plot_pointwise_error)
from equations import get_R_true
from solver    import rollout_full_nograd


# ══════════════════════════════════════════════════════════
#  SOLVER REGISTRY
#  Comment out any line to skip that problem.
#  Add new problems by importing their runner and registering here.
# ══════════════════════════════════════════════════════════

# Each entry: "key": callable that accepts (epochs=None) and returns metrics dict
def _import_system(sysname):
    def _run(epochs=None, preinit_epochs=400):
        from systems import run_system
        return run_system(sysname, epochs=epochs,
                          preinit_epochs=preinit_epochs)
    return _run

SOLVERS = {
    # ── Scalar singularly perturbed RDEs ──────────────────
    "fisher":            None,   # Fisher-KPP     R(u)=6u(1-u)
    "allen_cahn":        None,   # Allen-Cahn     R(u)=5u(1-u^2)
    # ── inverse system RDEs ───────────────────────
    "fhn_partial":          lambda: _import_system("fhn_partial"),
    "predator_prey":        lambda: _import_system("predator_prey"),
}


SCALAR_PROBLEMS = {"fisher", "allen_cahn"}  # routes to run_scalar()


# ══════════════════════════════════════════════════════════
#  Hyperparameter table
# ══════════════════════════════════════════════════════════

def print_hyperparameter_table():
    H   = C.GRU_HIDDEN
    W   = C.MLP_WIDTH
    inp = 4
    gru_p  = 3*(inp+H+1)*H
    film_p = H*(2*W)+2*W
    mlp_p  = (1*W+W)+(W*W+W)+(W*1+1)
    total  = 2*gru_p+film_p+mlp_p
    arch   = f"Dual-GRU({H})+FiLM+MLP({W}x3)"

    print()
    print("="*80)
    print("  HYPERPARAMETER TABLE — eps-US-GRU v5")
    print("="*80)
    hdr = (f"{'Problem':<26} {'Mesh':<14} {'Nt':>5} {'dt':>7} "
           f"{'Architecture':<26} {'Params':>8} {'Epochs':>7}")
    print(hdr); print("-"*80)
    for key, cfg in PROBLEMS.items():
        if key not in SCALAR_PROBLEMS: continue   # skip system problems
        Nx   = cfg["Nx"]; dt=cfg["dt"]; Nt=cfg["Nt"]
        mesh = ("Shishkin" if cfg["mesh_type"]=="shishkin"
                else "Uniform ") + f" N={Nx}"
        print(f"{cfg['name']:<26} {mesh:<14} {Nt:>5} {dt:>7.4f} "
              f"{arch:<26} {total:>8,} {C.EPOCHS:>7}")

    print("-"*80)

    print(f"\n  Common (scalar): Adam LR={C.LR:.0e}, clip={C.GRAD_CLIP}, "
          f"WD={C.WEIGHT_DECAY:.0e}, tau={C.TAU_CONS}\n")


# ══════════════════════════════════════════════════════════
#  Scalar problem runner
# ══════════════════════════════════════════════════════════

def run_scalar(problem, cfg, epochs, device):
    print("\n" + "="*65)
    print(f"  Problem: {cfg['name']}")
    print("="*65)

    t_wall = time.time()

    x, L_np, A_np, lu, piv, L_t, A_t = build_all_operators(
        problem, cfg, device)
    print(f"  Mesh: {cfg['mesh_type']}  Nx={cfg['Nx']}")

    t0 = time.time()
    u_refs, t_arr = generate_reference(problem, x, cfg, L_np, lu, piv)
    print(f"  Generated {len(u_refs)} IC(s) in {time.time()-t0:.1f}s")

    u_fd, R_fd = extract_fd_pairs(u_refs, L_np, cfg)
    u_tgt, R_tgt, weights = aggregate_bins(u_fd, R_fd, cfg, problem)
    print(f"  FD targets: {len(u_tgt)}")

    anchors   = cfg["anchors"]
    n_rollout = cfg.get("n_rollout_ics", len(u_refs))
    su        = float(np.mean(np.abs(
        np.concatenate([r.flatten() for r in u_refs[:n_rollout]]))))

    torch.manual_seed(42)
    model = EpsUSGRU(
        hidden_dim=C.GRU_HIDDEN, mlp_width=C.MLP_WIDTH,
        Lambda=cfg["Lambda"], input_dim=4).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    preinit_mlp(model, u_tgt, R_tgt, weights, cfg, device)

    print(f"\n  Training: {epochs} epochs  TBPTT={cfg.get('tbptt', C.TBPTT_WINDOW)}")
    t_train = time.time()
    history, all_u_trajs, all_R_trajs = train(
        model, problem, cfg, x, u_refs, t_arr,
        L_np, A_t, L_t, anchors, device,
        u_fd_np=u_tgt, R_fd_np=R_tgt, w_fd_np=weights,
        epochs=epochs)
    t_train = time.time()-t_train
    print(f"  Training done in {t_train:.1f}s")

    # Distillation
    scalar_mlp, u_grid_d, R_dist = distil(model, cfg, anchors, device)
    R_true_d = get_R_true(problem, u_grid_d, cfg)
    l2_dist  = (np.sqrt(np.mean((R_dist-R_true_d)**2)) /
                (np.sqrt(np.mean(R_true_d**2))+1e-8))

    # Reaction L2
    u_g   = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    u_ev  = torch.tensor(u_g, dtype=torch.float32, device=device)
    R_pred = model.reaction_from_u_only_nograd(u_ev).cpu().numpy()
    R_true = get_R_true(problem, u_g, cfg)
    l2_model = (np.sqrt(np.mean((R_pred-R_true)**2)) /
                (np.sqrt(np.mean(R_true**2))+1e-8))

    t_wall = time.time()-t_wall

    # Plots
    print("  Saving plots...")
    import regime as reg
    x_t   = torch.tensor(x, dtype=torch.float32, device=device)
    xl,xr = cfg["domain"]; eps = cfg["eps_diff"]
    phi_t = reg.layer_envelope(x_t, xl, xr, eps)
    t_arr_t = torch.tensor(t_arr, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_traj_plot, _ = rollout_full_nograd(
            model,
            torch.tensor(u_refs[0][0], dtype=torch.float32, device=device),
            torch.tensor(u_refs[0],    dtype=torch.float32, device=device),
            x_t, phi_t, A_t, L_t, cfg, problem, t_arr_t, device)

    plot_reaction(model, u_tgt, R_tgt, problem, cfg, device)
    plot_trajectory(x, u_refs[0], u_traj_plot, t_arr, problem, cfg)
    l2_traj, mean_err, max_err = plot_pointwise_error(
        x, u_refs[0], u_traj_plot, t_arr, problem, cfg)
    plot_loss(history, problem, cfg)
    plot_scatter(all_u_trajs, all_R_trajs, problem, cfg, device)

    print(f"\n  {'='*50}")
    print(f"  RESULTS — {cfg['name']}")
    print(f"  {'='*50}")
    print(f"  Reaction L2 (model):    {l2_model:.4f}")
    print(f"  Reaction L2 (distilled):{l2_dist:.4f}")
    print(f"  Trajectory L2 (rollout):{l2_traj:.2e}")
    print(f"  Max pointwise error:    {max_err:.2e}")
    print(f"  Training time:          {t_train:.1f}s ({t_train/60:.1f} min)")
    print(f"  Wall-clock time:        {t_wall:.1f}s ({t_wall/60:.1f} min)")

    out_dir = os.path.join("results_v5", problem)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "problem": problem},
               os.path.join(out_dir, f"checkpoint_{problem}.pt"))

    metrics = {
        "problem":      problem,
        "l2_model":     float(l2_model),
        "l2_distilled": float(l2_dist),
        "l2_traj":      float(l2_traj),
        "max_err_traj": float(max_err),
        "train_time_s": float(t_train),
        "wall_time_s":  float(t_wall),
        "final_loss":   history[-1]["total"],
        "epochs":       epochs,
    }
    with open(os.path.join(out_dir,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ══════════════════════════════════════════════════════════
#  Epsilon sensitivity  (scalar problems only)
# ══════════════════════════════════════════════════════════

EPS_STUDY = {
    "fisher":     [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
    "allen_cahn": [1e-4, 1e-5, 1e-6],
}

def run_eps_sensitivity(problem, device, epochs_eps):
    if problem not in EPS_STUDY:
        print(f"  No eps study defined for {problem}.")
        return
    eps_list = EPS_STUDY[problem]
    base_cfg = copy.deepcopy(PROBLEMS[problem])
    pname    = base_cfg["name"]
    print(f"\n{'='*65}")
    print(f"  EPSILON SENSITIVITY — {pname}")
    print(f"  eps values: {eps_list}")
    print(f"{'='*65}")
    res = []
    for eps in eps_list:
        cfg = copy.deepcopy(base_cfg); cfg["eps_diff"] = eps
        print(f"\n  eps={eps:.1e}", end="  ", flush=True)
        try:
            m = run_scalar(problem, cfg, epochs_eps, device)
            res.append((eps, m["l2_model"], m["l2_distilled"]))
            print(f"  L2={m['l2_model']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            res.append((eps, float("nan"), float("nan")))
    plot_eps_sensitivity(res, problem, pname)
    print(f"\n  eps sensitivity done. Check results_v5/")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    valid_keys = list(SOLVERS.keys())

    parser = argparse.ArgumentParser(
        description="eps-US-GRU v5 — run ONE problem at a time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Registered problems (edit SOLVERS dict to enable/disable):
  fisher         Fisher-KPP        R(u) = 6u(1-u)
  allen_cahn     Allen-Cahn        R(u) = 5u(1-u^2)
  fhn_partial    FHN system        identify R(u,v) = (1/eps)u(u-a)(1-u) - v
  predator_prey  Predator-Prey     identify R(u,v) = u(1-u) - alpha*u*v/(beta+u)

Examples:
  python main.py --problem fisher
  python main.py --problem allen_cahn --epochs 3000
  python main.py --problem fhn_partial
  python main.py --problem fisher --eps_study
  python main.py --show_table
""")

    parser.add_argument(
        "--problem",
        required=True,
        nargs="+",          # accept one or more problem names
        choices=valid_keys,
        metavar="PROBLEM",
        help="One or more problems to run in sequence. Choices: " + ", ".join(valid_keys))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default: from config)")
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu or cuda (default: auto-detect)")
    parser.add_argument(
        "--eps_study",
        action="store_true",
        help="Run epsilon sensitivity study after training")
    parser.add_argument(
        "--eps_epochs",
        type=int,
        default=500,
        help="Epochs per run in eps sensitivity study (default: 500)")
    parser.add_argument(
        "--show_table",
        action="store_true",
        help="Print hyperparameter table and exit")
    args = parser.parse_args()

    if args.show_table:
        print_hyperparameter_table()
        return

    device = C.DEVICE if args.device == "auto" else torch.device(args.device)
    problems = args.problem   # now a list
    all_metrics = {}

    for problem in problems:
        print(f"\nDevice: {device}   Problem: {problem}")
        np.random.seed(42); torch.manual_seed(42)

        solver_entry = SOLVERS.get(problem)

        # ── System problems: use dedicated runner ──────────
        if problem not in SCALAR_PROBLEMS:
            runner = solver_entry()          # lazy import
            epochs = args.epochs
            m = runner(epochs=epochs)
            print(f"\n  Done — {problem}")
            for k, v in m.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}" if v > 1e-4 else f"    {k}: {v:.2e}")
            all_metrics[problem] = m

        # ── Scalar problems ────────────────────────────────
        else:
            cfg    = PROBLEMS[problem].copy()
            epochs = args.epochs if args.epochs is not None else C.EPOCHS
            m      = run_scalar(problem, cfg, epochs, device)
            all_metrics[problem] = m

            if args.eps_study:
                run_eps_sensitivity(problem, device, args.eps_epochs)

    # ── Summary when multiple problems were run ────────────
    if len(problems) > 1:
        print("\n" + "="*55)
        print("  SUMMARY")
        print("="*55)
        for prob, m in all_metrics.items():
            l2 = m.get("l2", m.get("l2_model", float("nan")))
            t  = m.get("train_s", m.get("train_time_s", 0))
            print(f"  {prob:<25}  L²={l2:.4f}  time={t:.0f}s")


if __name__ == "__main__":
    main()