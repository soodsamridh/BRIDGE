import sys, os, copy, time, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config as C
from config    import PROBLEMS
from numerics  import build_all_operators
from data      import generate_reference, extract_fd_pairs, aggregate_bins
from model     import EpsUSGRU
from trainer   import preinit_mlp, train
from distill   import distil
from equations import get_R_true

OUT_DIR = "eps_study_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  Epsilon values to study
# ══════════════════════════════════════════════════════════

EPS_VALUES = {
    "fisher": [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
    "allen_cahn": [1e-4, 1e-5, 1e-6],
}

# The "standard" ε used in the main experiments (for reference line)
STANDARD_EPS = {
    "fisher":     1.0,
    "allen_cahn": 1e-4,
}


# ══════════════════════════════════════════════════════════
#  Build ε-modified config
# ══════════════════════════════════════════════════════════

def build_eps_cfg(problem, eps, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["eps_diff"] = eps

    if problem == "fisher":
        pass

    elif problem == "allen_cahn":
        pass

    return cfg


# ══════════════════════════════════════════════════════════
#  Run model for one (problem, eps) pair
# ══════════════════════════════════════════════════════════

def run_one(problem, eps, cfg, epochs, device):
    x, L_np, A_np, lu, piv, L_t, A_t = build_all_operators(problem, cfg, device)

    u_refs, t_arr = generate_reference(problem, x, cfg, L_np, lu, piv)
    u_fd, R_fd   = extract_fd_pairs(u_refs, L_np, cfg)
    u_tgt, R_tgt, weights = aggregate_bins(u_fd, R_fd, cfg, problem)

    anchors = cfg["anchors"]
    torch.manual_seed(42); np.random.seed(42)
    model = EpsUSGRU(
        hidden_dim=C.GRU_HIDDEN, mlp_width=C.MLP_WIDTH,
        Lambda=cfg["Lambda"], input_dim=4).to(device)

    preinit_mlp(model, u_tgt, R_tgt, weights, cfg, device)

    t0 = time.time()
    history, all_u_trajs, all_R_trajs = train(
        model, problem, cfg, x, u_refs, t_arr,
        L_np, A_t, L_t, anchors, device,
        u_fd_np=u_tgt, R_fd_np=R_tgt, w_fd_np=weights,
        epochs=epochs)
    t_train = time.time() - t0

    # Reaction L²
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    u_ev   = torch.tensor(u_grid, dtype=torch.float32, device=device)
    R_pred = model.reaction_from_u_only_nograd(u_ev).cpu().numpy()
    R_true = get_R_true(problem, u_grid, cfg)
    l2_mdl = (np.sqrt(np.mean((R_pred - R_true)**2)) /
               (np.sqrt(np.mean(R_true**2)) + 1e-8))

    # Distillation L²
    _, u_grid_d, R_dist = distil(model, cfg, anchors, device)
    R_true_d = get_R_true(problem, u_grid_d, cfg)
    l2_dist  = (np.sqrt(np.mean((R_dist - R_true_d)**2)) /
                (np.sqrt(np.mean(R_true_d**2)) + 1e-8))

    return {
        "eps":        float(eps),
        "l2_model":   float(l2_mdl),
        "l2_distil":  float(l2_dist),
        "train_s":    float(t_train),
        "R_pred":     R_pred.tolist(),
        "R_true":     R_true.tolist(),
        "u_grid":     u_grid.tolist(),
        "n_fd_tgts":  int(len(u_tgt)),
    }


# ══════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════

def plot_reaction_all_eps(results, problem, pname):
    """All R_theta curves for every ε on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    u_g  = np.array(results[0]["u_grid"])
    R_tr = np.array(results[0]["R_true"])
    ax.plot(u_g, R_tr, "k-", lw=3.0, label="$R_{\\mathrm{true}}(u)$",
            zorder=10)

    cmap   = plt.cm.plasma
    n_eps  = len(results)
    colors = [cmap(0.1 + 0.75 * i / (n_eps - 1)) for i in range(n_eps)]

    for i, r in enumerate(results):
        eps = r["eps"]
        l2  = r["l2_model"]
        lbl = f"$\\varepsilon={eps:.0e}$  ($\\varepsilon_R={l2:.3f}$)"
        is_std = (eps == STANDARD_EPS[problem])
        ax.plot(np.array(r["u_grid"]), np.array(r["R_pred"]),
                lw=2.2 if is_std else 1.5,
                linestyle="--" if is_std else ":",
                color=colors[i],
                label=lbl + ("  ✦" if is_std else ""),
                zorder=5)

    ax.set_xlabel("$u$", fontsize=12)
    ax.set_ylabel("$R(u)$", fontsize=12)
    ax.set_title(f"{pname} — Reaction Identification Across $\\varepsilon$ Values",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"eps_reaction_{problem}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="eps-US-GRU Epsilon Sensitivity Study")
    parser.add_argument("--problem",
                        default="all",
                        choices=["fisher", "allen_cahn", "all"],
                        help="Which problem to study")
    parser.add_argument("--epochs",
                        type=int, default=500,
                        help="Training epochs per (problem, ε) run "
                             "(default 500; use 1000 for paper-quality results)")
    parser.add_argument("--device",
                        default="auto",
                        help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    device = C.DEVICE if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")
    print(f"Epochs per run: {args.epochs}")

    problems = (["fisher", "allen_cahn"] if args.problem == "all"
                else [args.problem])

    all_summary = {}

    for problem in problems:
        pname    = PROBLEMS[problem]["name"]
        base_cfg = copy.deepcopy(PROBLEMS[problem])
        eps_list = EPS_VALUES[problem]

        print(f"\n{'='*65}")
        print(f"  Problem: {pname}")
        print(f"  ε values: {eps_list}")
        print(f"{'='*65}")

        results = []
        for eps in eps_list:
            cfg = build_eps_cfg(problem, eps, base_cfg)
            print(f"\n  ε = {eps:.1e}", end="  ", flush=True)
            try:
                r = run_one(problem, eps, cfg, args.epochs, device)
                results.append(r)
                print(f"→  Model L²={r['l2_model']:.4f}  "
                      f"Distil L²={r['l2_distil']:.4f}  "
                      f"FD bins={r['n_fd_tgts']}  "
                      f"({r['train_s']:.0f}s)")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                results.append({
                    "eps":float(eps), "l2_model":float("nan"),
                    "l2_distil":float("nan"), "train_s":0.0,
                    "R_pred":[], "R_true":[], "u_grid":[],
                    "n_fd_tgts":0,
                })

        all_summary[problem] = results


        # ── Plots ─────────────────────────────────────────
        valid = [r for r in results if not np.isnan(r["l2_model"])]
        if len(valid) >= 2:
            plot_reaction_all_eps(valid, problem, pname)

    # ── Combined JSON ─────────────────────────────────────
    save = {p: [{k: v for k, v in r.items()
                 if k not in ("R_pred","R_true","u_grid")}
                for r in all_summary[p]]
            for p in all_summary}
    jpath = os.path.join(OUT_DIR, "eps_summary.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Summary → {jpath}")

    # ── Final console summary ─────────────────────────────
    print(f"\n{'='*65}")
    print("  EPSILON SENSITIVITY SUMMARY")
    print(f"{'='*65}")
    for problem in problems:
        pname = PROBLEMS[problem]["name"]
        print(f"\n  {pname}:")
        print(f"  {'ε':>10}  {'Model L²':>10}  {'Distil L²':>10}")
        print("  " + "-"*36)
        for r in all_summary[problem]:
            mk = "  ← standard" if r["eps"]==STANDARD_EPS[problem] else ""
            l2m = f"{r['l2_model']:.4f}" if not np.isnan(r["l2_model"]) else " FAILED"
            l2d = f"{r['l2_distil']:.4f}" if not np.isnan(r["l2_distil"]) else " FAILED"
            print(f"  {r['eps']:>10.1e}  {l2m:>10}  {l2d:>10}{mk}")


if __name__ == "__main__":
    main()