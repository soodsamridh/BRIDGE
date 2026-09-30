import sys, os, copy, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config as C
from config    import PROBLEMS
from numerics  import build_all_operators
from data      import generate_reference, extract_fd_pairs, aggregate_bins
from trainer   import preinit_mlp, train
from equations import get_R_true

DEVICE  = torch.device("cpu")
SEED    = 42
OUT_DIR = "arch_study_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  Base configuration
# ══════════════════════════════════════════════════════════

BASE_HIDDEN    = 32
BASE_MLP_DEPTH = 3
BASE_MLP_WIDTH = 32
INPUT_DIM      = 4

# ══════════════════════════════════════════════════════════
#  Architecture sweep definitions
# ══════════════════════════════════════════════════════════

SWEEPS = {
    "gru_hidden": {
        "title":  "GRU hidden dimension H",
        "xlabel": "GRU hidden dim H",
        "values": [8, 16, 32, 64],
        "base":   BASE_HIDDEN,
        "desc":   "Encoder capacity (Dual GRU hidden size)",
    },
    "mlp_depth": {
        "title":  "Scalar MLP depth D",
        "xlabel": "MLP depth D (hidden layers)",
        "values": [1, 2, 3, 4],
        "base":   BASE_MLP_DEPTH,
        "desc":   "Number of hidden layers in R(u) decoder",
    },
    "mlp_width": {
        "title":  "Scalar MLP width W",
        "xlabel": "MLP width W (neurons per layer)",
        "values": [8, 16, 32, 64],
        "base":   BASE_MLP_WIDTH,
        "desc":   "Width of each hidden layer in R(u) decoder",
    },
}

PROBLEMS_TO_RUN = ["fisher", "allen_cahn"]


# ══════════════════════════════════════════════════════════
#  Flexible EpsUSGRU with variable hidden/depth/width
# ══════════════════════════════════════════════════════════

class FlexEpsUSGRU(nn.Module):
    def __init__(self, hidden_dim=32, mlp_depth=3, mlp_width=32,
                 Lambda=2.0, input_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_width  = mlp_width
        self.mlp_depth  = mlp_depth
        self.Lambda     = Lambda

        # Dual GRU encoder 
        self.gru_out   = nn.GRUCell(input_dim, hidden_dim)
        self.gru_lay   = nn.GRUCell(input_dim, hidden_dim)

        self.film_proj = nn.Linear(hidden_dim, 2*mlp_width)

        # Scalar MLP decoder
        layers = []
        if mlp_depth == 1:
            layers.append(nn.Linear(1, 1))
        else:
            layers.append(nn.Linear(1, mlp_width))
            for _ in range(mlp_depth - 2):
                layers.append(nn.Linear(mlp_width, mlp_width))
            layers.append(nn.Linear(mlp_width, 1))

        self.mlp_layers = nn.ModuleList(layers)

        nn.init.normal_(self.mlp_layers[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp_layers[-1].bias)

    @property
    def _eff_width(self):
        return 1 if self.mlp_depth == 1 else self.mlp_width

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden_dim, device=device)
        return h, h.clone()

    def _run_mlp(self, x, alpha=None, beta=None):
        """
        Run the scalar MLP with optional FiLM conditioning.
        x shape: (Nx, 1)
        alpha, beta shape: (Nx, mlp_width) or None
        Returns: (Nx,) tensor
        """
        if self.mlp_depth == 1:
            out = self.mlp_layers[0](x)
        else:
            h = x
            for i, layer in enumerate(self.mlp_layers):
                h = layer(h)
                is_last = (i == len(self.mlp_layers) - 1)
                if not is_last:
                    if i == 0 and alpha is not None:
                        # FiLM modulation on first hidden layer only
                        h = F.relu((1 + alpha) * h + beta)
                    else:
                        h = F.relu(h)
            out = h
        return self.Lambda * torch.tanh(out).squeeze(-1)

    def forward(self, u, z_out, z_lay, rho, H_out, H_lay):
        rho_e   = rho.unsqueeze(-1)
        H_out_c = self.gru_out(z_out, H_out)
        H_lay_c = self.gru_lay(z_lay, H_lay)
        H_out_n = (1 - rho_e) * H_out_c + rho_e * H_out
        H_lay_n =      rho_e  * H_lay_c + (1 - rho_e) * H_lay
        H_blend = (1 - rho_e) * H_out_n + rho_e * H_lay_n

        if self.mlp_depth > 1:
            film  = self.film_proj(H_blend)
            alpha = film[:, :self.mlp_width]
            beta  = film[:, self.mlp_width:]
        else:
            alpha = beta = None

        R_pred = self._run_mlp(u.unsqueeze(-1), alpha, beta)
        return R_pred, H_out_n, H_lay_n

    def reaction_from_u_only(self, u_vals):
        """Neutral FiLM (alpha=0, beta=0) — gradient transparent."""
        Nx = u_vals.shape[0]
        dev = u_vals.device
        if self.mlp_depth > 1:
            alpha = torch.zeros(Nx, self.mlp_width, device=dev)
            beta  = torch.zeros(Nx, self.mlp_width, device=dev)
        else:
            alpha = beta = None
        return self._run_mlp(u_vals.unsqueeze(-1), alpha, beta)

    @torch.no_grad()
    def reaction_from_u_only_nograd(self, u_vals):
        return self.reaction_from_u_only(u_vals)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════
#  Parameter counter
# ══════════════════════════════════════════════════════════

def count_params(hidden, mlp_depth, mlp_width, input_dim=INPUT_DIM):
    gru_p  = 3 * (input_dim + hidden + 1) * hidden  # one GRUCell
    film_p = hidden * (2*mlp_width) + 2*mlp_width   # FiLM Linear
    if mlp_depth == 1:
        mlp_p = 1 * 1 + 1   # Linear(1,1)
    else:
        mlp_p = 1*mlp_width + mlp_width             # first layer
        for _ in range(mlp_depth - 2):
            mlp_p += mlp_width*mlp_width + mlp_width
        mlp_p += mlp_width*1 + 1                    # output layer
    return 2*gru_p + film_p + mlp_p


# ══════════════════════════════════════════════════════════
#  Run one (problem, architecture) combination
# ══════════════════════════════════════════════════════════

def run_one(problem, cfg, hidden, mlp_depth, mlp_width,
            u_refs, t_arr, L_np, A_t, L_t,
            u_tgt, R_tgt, weights, x, epochs, device):

    torch.manual_seed(SEED); np.random.seed(SEED)

    model = FlexEpsUSGRU(
        hidden_dim=hidden,
        mlp_depth=mlp_depth,
        mlp_width=mlp_width,
        Lambda=cfg["Lambda"],
        input_dim=INPUT_DIM,
    ).to(device)

    # Pre-init scalar MLP from FD bin-medians
    # Adapt preinit for FlexEpsUSGRU (same logic as preinit_mlp)
    for name, p in model.named_parameters():
        if "gru" in name or "film" in name:
            p.requires_grad_(False)
    mlp_params = [p for p in model.parameters() if p.requires_grad]
    if mlp_params:
        pi_opt = torch.optim.Adam(mlp_params, lr=3e-3)
        u_t = torch.tensor(u_tgt, dtype=torch.float32, device=device)
        R_t = torch.tensor(R_tgt, dtype=torch.float32, device=device)
        w_t = torch.tensor(weights, dtype=torch.float32, device=device)
        for _ in range(300):
            pi_opt.zero_grad()
            Rp = model.reaction_from_u_only(u_t)
            loss = (w_t * (Rp - R_t).pow(2) / (cfg["Lambda"]**2 + 1e-8)).mean()
            loss.backward(); pi_opt.step()
    for p in model.parameters(): p.requires_grad_(True)

    # Stage 1 — coupled IMEX training
    t0 = time.time()
    history, _, _ = train(
        model, problem, cfg, x, u_refs, t_arr,
        L_np, A_t, L_t, cfg["anchors"], device,
        u_fd_np=u_tgt, R_fd_np=R_tgt, w_fd_np=weights,
        epochs=epochs)
    t_train = time.time() - t0

    # Evaluate L² error on R
    model.eval()
    u_grid = np.linspace(cfg["u_min"], cfg["u_max"], 400)
    u_ev   = torch.tensor(u_grid, dtype=torch.float32, device=device)
    R_pred = model.reaction_from_u_only_nograd(u_ev).cpu().numpy()
    R_true = get_R_true(problem, u_grid, cfg)
    l2 = (np.sqrt(np.mean((R_pred - R_true)**2)) /
          (np.sqrt(np.mean(R_true**2)) + 1e-8))

    n_params = model.count_params()
    final_loss = history[-1]["total"]

    return {
        "l2":        float(l2),
        "params":    n_params,
        "train_s":   float(t_train),
        "final_loss":float(final_loss),
        "R_pred":    R_pred.tolist(),
        "R_true":    R_true.tolist(),
        "u_grid":    u_grid.tolist(),
    }


# ══════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════

def plot_sweep(sweep_name, sweep_cfg, results_by_problem, problems):
    """
    Two-row figure:
      Row 1: L² error bar charts (one per problem)
      Row 2: Param count line + L² line (dual y-axis)
    """
    values   = sweep_cfg["values"]
    base_val = sweep_cfg["base"]
    xlabel   = sweep_cfg["xlabel"]
    title    = sweep_cfg["title"]

    n_probs = len(problems)
    fig, axes = plt.subplots(2, n_probs, figsize=(6*n_probs, 10))
    if n_probs == 1: axes = axes.reshape(2, 1)

    fig.suptitle(f"Architecture Study — {title}",
                 fontsize=14, fontweight="bold")

    COLORS = ["#AED6F1", "#3498DB", "#1A5276", "#C0392B", "#884EA0"]

    for col, problem in enumerate(problems):
        pname = PROBLEMS[problem]["name"]
        res   = results_by_problem[problem]

        l2s    = [res[v]["l2"]     for v in values]
        params = [res[v]["params"] for v in values]
        labels = [str(v) for v in values]

        # Assign colors — base value gets distinct red border
        colors = [COLORS[min(i, len(COLORS)-1)] for i in range(len(values))]
        base_idx = values.index(base_val) if base_val in values else -1

        # ── Row 0: Bar chart of L² ──────────────────────────
        ax = axes[0, col]
        bars = ax.bar(labels, l2s, color=colors,
                      edgecolor="black", linewidth=0.8, width=0.55)
        if base_idx >= 0:
            bars[base_idx].set_edgecolor("#C0392B")
            bars[base_idx].set_linewidth(2.5)
        for bar, val in zip(bars, l2s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Relative $L^2$ Error $\\varepsilon_R$", fontsize=10)
        ax.set_title(f"{pname}  —  $L^2$ Error", fontsize=11)
        ax.set_ylim(0, max(l2s) * 1.3)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        if base_idx >= 0:
            ax.axvline(base_idx, color="#C0392B", linewidth=1.0,
                       linestyle=":", alpha=0.5, label="Base config")

        # ── Row 1: Dual-axis — params + L² vs value ────────
        ax2 = axes[1, col]
        x_pos = np.arange(len(values))

        color_l2    = "#2980B9"
        color_param = "#27AE60"

        l1, = ax2.plot(x_pos, l2s, "o-", color=color_l2,
                       lw=2.0, ms=7, label="$L^2$ error")
        ax2.set_ylabel("Relative $L^2$ Error $\\varepsilon_R$",
                       fontsize=10, color=color_l2)
        ax2.tick_params(axis="y", labelcolor=color_l2)

        ax3 = ax2.twinx()
        l2p, = ax3.plot(x_pos, params, "s--", color=color_param,
                        lw=1.8, ms=7, label="Parameters")
        ax3.set_ylabel("Parameter count", fontsize=10, color=color_param)
        ax3.tick_params(axis="y", labelcolor=color_param)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_xlabel(xlabel, fontsize=10)
        ax2.set_title(f"{pname}  —  $L^2$ vs Params", fontsize=11)

        if base_idx >= 0:
            ax2.axvline(base_idx, color="#C0392B", linewidth=1.2,
                        linestyle=":", alpha=0.6, label="Base config")

        lines = [l1, l2p]
        labs  = [l.get_label() for l in lines]
        ax2.legend(lines, labs, fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             "Red dashed line = proposed base configuration",
             ha="center", fontsize=10, style="italic", color="#C0392B")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = os.path.join(OUT_DIR, f"arch_sweep_{sweep_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def plot_summary_heatmap(all_results, problems, epochs):
    """
    Heatmap of L² over (sweep, value, problem) — quick visual overview.
    """
    sweep_names = list(SWEEPS.keys())
    n_rows = len(sweep_names)
    n_cols = len(problems)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 3.5*n_rows))
    if n_cols == 1: axes = axes.reshape(n_rows, 1)
    if n_rows == 1: axes = axes.reshape(1, n_cols)

    fig.suptitle(f"Architecture Study — $L^2$ Error Heatmap ({epochs} epochs)",
                 fontsize=13, fontweight="bold")

    for row, sweep_name in enumerate(sweep_names):
        sw = SWEEPS[sweep_name]
        values = sw["values"]; base = sw["base"]

        for col, problem in enumerate(problems):
            pname = PROBLEMS[problem]["name"]
            ax    = axes[row, col]

            l2s    = [all_results[sweep_name][problem][v]["l2"] for v in values]
            labels = [str(v) for v in values]
            base_i = values.index(base) if base in values else -1

            im = ax.imshow(np.array(l2s).reshape(1, -1),
                           aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=max(l2s)*1.1)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticks([])
            ax.set_title(f"{pname}  [{sw['xlabel']}]", fontsize=9)

            for i, v in enumerate(l2s):
                color = "white" if v > max(l2s)*0.6 else "black"
                ax.text(i, 0, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
            if base_i >= 0:
                ax.add_patch(plt.Rectangle(
                    (base_i-0.5, -0.5), 1, 1,
                    fill=False, edgecolor="#2980B9", linewidth=2.5))

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "arch_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Architecture sensitivity study for eps-US-GRU")
    parser.add_argument("--problem", default="all",
                        choices=["fisher", "allen_cahn", "all"],
                        help="Which problem to run (default: all)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs per configuration (default: 500)")
    parser.add_argument("--sweep", default="all",
                        choices=["gru_hidden", "mlp_depth", "mlp_width", "all"],
                        help="Which sweep to run (default: all)")
    args = parser.parse_args()

    problems = (["fisher", "allen_cahn"] if args.problem == "all"
                else [args.problem])
    sweeps   = (list(SWEEPS.keys()) if args.sweep == "all"
                else [args.sweep])
    epochs   = args.epochs

    # Time estimate
    n_runs = sum(len(SWEEPS[s]["values"]) for s in sweeps) * len(problems)
    t_per  = {"fisher": 0.95, "allen_cahn": 15.5}  # sec/epoch per GRU method
    t_est  = sum(t_per.get(p, 5) * epochs
                 for s in sweeps
                 for _ in SWEEPS[s]["values"]
                 for p in problems)
    print(f"\nArchitecture study: {n_runs} runs × {epochs} epochs")
    print(f"Estimated time: ~{t_est/60:.0f} min  ({t_est/3600:.1f} h)")
    print(f"Tip: --problem fisher or --sweep gru_hidden to reduce scope.\n")

    # Pre-compute shared data for each problem (reference trajectories, FD pairs)
    shared_data = {}
    for problem in problems:
        cfg = copy.deepcopy(PROBLEMS[problem])
        print(f"\nPre-computing data for {cfg['name']}...")
        x, L_np, _, lu, piv, L_t, A_t = build_all_operators(problem, cfg, DEVICE)
        u_refs, t_arr = generate_reference(problem, x, cfg, L_np, lu, piv)
        u_fd, R_fd    = extract_fd_pairs(u_refs, L_np, cfg)
        u_tgt, R_tgt, weights = aggregate_bins(u_fd, R_fd, cfg, problem)
        print(f"  FD targets: {len(u_tgt)}   ICs: {len(u_refs)}")
        shared_data[problem] = dict(
            cfg=cfg, x=x, L_np=L_np, A_t=A_t, L_t=L_t,
            u_refs=u_refs, t_arr=t_arr,
            u_tgt=u_tgt, R_tgt=R_tgt, weights=weights)

    all_results = {}   # all_results[sweep_name][problem][value] = result dict

    for sweep_name in sweeps:
        sw = SWEEPS[sweep_name]
        print(f"\n{'='*65}")
        print(f"  SWEEP: {sw['title']}")
        print(f"  Values: {sw['values']}   Base: {sw['base']}")
        print(f"{'='*65}")

        all_results[sweep_name] = {p: {} for p in problems}

        for v in sw["values"]:
            # Build architecture config for this value
            if sweep_name == "gru_hidden":
                h, d, w = v, BASE_MLP_DEPTH, BASE_MLP_WIDTH
            elif sweep_name == "mlp_depth":
                h, d, w = BASE_HIDDEN, v, BASE_MLP_WIDTH
            else:
                h, d, w = BASE_HIDDEN, BASE_MLP_DEPTH, v

            n_p    = count_params(h, d, w)
            is_base = (v == sw["base"])
            mark   = "  ← BASE" if is_base else ""

            print(f"\n  [{sw['xlabel']} = {v}]  params={n_p:,}{mark}")

            for problem in problems:
                pname = PROBLEMS[problem]["name"]
                sd    = shared_data[problem]
                cfg   = sd["cfg"]

                print(f"    {pname}...", end="  ", flush=True)
                t0 = time.time()
                try:
                    result = run_one(
                        problem, cfg, h, d, w,
                        sd["u_refs"], sd["t_arr"], sd["L_np"],
                        sd["A_t"], sd["L_t"],
                        sd["u_tgt"], sd["R_tgt"], sd["weights"],
                        sd["x"], epochs, DEVICE)
                    print(f"L²={result['l2']:.4f}  "
                          f"({time.time()-t0:.0f}s)")
                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    result = dict(l2=float("nan"), params=n_p,
                                  train_s=0.0, final_loss=float("nan"),
                                  R_pred=[], R_true=[], u_grid=[])

                all_results[sweep_name][problem][v] = result

        # Per-sweep plots
        results_by_problem = {
            problem: all_results[sweep_name][problem]
            for problem in problems}
        plot_sweep(sweep_name, sw, results_by_problem, problems)

    # Summary plots and tables
    print("\n\nSaving summary plots and tables...")
    plot_summary_heatmap(all_results, problems, epochs)

    # Save full results JSON
    def strip(d):
        """Remove large arrays from JSON to keep file size small."""
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items()
                    if k not in ("R_pred", "R_true", "u_grid")}
        return d

    json_path = os.path.join(OUT_DIR, "arch_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # JSON keys must be strings
        save_dict = {}
        for sn in all_results:
            save_dict[sn] = {}
            for prob in all_results[sn]:
                save_dict[sn][prob] = {
                    str(v): strip(r)
                    for v, r in all_results[sn][prob].items()}
        json.dump(save_dict, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Final console summary ──────────────────────────────
    print(f"\n{'='*65}")
    print("  ARCHITECTURE STUDY SUMMARY")
    print(f"{'='*65}")
    for sweep_name in sweeps:
        sw = SWEEPS[sweep_name]
        print(f"\n  {sw['title']}:")
        print(f"  {'Value':<8} " +
              "  ".join(f"{PROBLEMS[p]['name']:>14}" for p in problems))
        print("  " + "-"*50)
        for v in sw["values"]:
            is_base = (v == sw["base"])
            l2_strs = []
            for problem in problems:
                l2 = all_results[sweep_name][problem][v]["l2"]
                l2_strs.append(f"{l2:.4f}" if not np.isnan(l2) else "FAILED")
            mk = "  *" if is_base else ""
            print(f"  {str(v):<8} " +
                  "  ".join(f"{s:>14}" for s in l2_strs) + mk)
        print("  * = proposed base configuration")


if __name__ == "__main__":
    main()