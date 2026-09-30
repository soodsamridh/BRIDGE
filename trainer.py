import torch
import torch.optim as optim
import numpy as np

import config as C
from solver import rollout_tbptt, rollout_full_nograd
from loss import total_loss
from regime import layer_envelope


def preinit_mlp(model, u_targets, R_targets, weights, cfg, device):
    Lambda = cfg["Lambda"]
    for name, p in model.named_parameters():
        if "gru" in name or "film" in name:
            p.requires_grad_(False)
    mlp_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(mlp_params, lr=C.LR_PREINIT)

    u_t = torch.tensor(u_targets, dtype=torch.float32, device=device)
    R_t = torch.tensor(R_targets, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights,   dtype=torch.float32, device=device)

    print(f"\n  [Stage 0] MLP pre-init: {len(u_t)} FD targets  "
          f"u∈[{u_targets.min():.3f},{u_targets.max():.3f}]")
    best = float("inf")
    for ep in range(1, C.EPOCHS_PREINIT+1):
        opt.zero_grad()
        R_pred = model.reaction_from_u_only(u_t)
        loss   = (w_t*(R_pred-R_t).pow(2)/(Lambda**2+1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_params, 1.0)
        opt.step()
        if loss.item() < best: best = loss.item()
        if ep % 100 == 0 or ep == 1:
            print(f"    ep {ep:>3}/{C.EPOCHS_PREINIT}  loss={loss.item():.5f}")

    for p in model.parameters(): p.requires_grad_(True)
    print(f"  [Stage 0] Done. Best = {best:.5f}")


def train(model, problem, cfg, x_np, u_refs_np, t_arr_np,
          L_op_np, A_t, L_op_t, anchors, device,
          u_fd_np=None, R_fd_np=None, w_fd_np=None, epochs=None):

    if epochs is None: epochs = C.EPOCHS

    grad_clip    = cfg.get("grad_clip", C.GRAD_CLIP)
    tbptt_window = cfg.get("tbptt", C.TBPTT_WINDOW)
    n_rollout    = min(cfg.get("n_rollout_ics", len(u_refs_np)), len(u_refs_np))

    u_refs_rollout = u_refs_np[:n_rollout]
    n_ics          = len(u_refs_rollout)

    all_u_rollout = np.concatenate([r.flatten() for r in u_refs_rollout])
    s_u = float(np.mean(np.abs(all_u_rollout)))

    u_refs_t = [torch.tensor(u, dtype=torch.float32, device=device)
                for u in u_refs_rollout]
    t_arr    = torch.tensor(t_arr_np, dtype=torch.float32, device=device)
    x_t      = torch.tensor(x_np, dtype=torch.float32, device=device)
    x_left, x_right = cfg["domain"]
    phi_t    = layer_envelope(x_t, x_left, x_right, cfg["eps_diff"])

    u_fd_t = (torch.tensor(u_fd_np, dtype=torch.float32, device=device)
              if u_fd_np is not None and len(u_fd_np) > 0 else None)
    R_fd_t = (torch.tensor(R_fd_np, dtype=torch.float32, device=device)
              if R_fd_np is not None and len(R_fd_np) > 0 else None)
    w_fd_t = (torch.tensor(w_fd_np, dtype=torch.float32, device=device)
              if w_fd_np is not None and len(w_fd_np) > 0 else None)

    print(f"  s_u={s_u:.4f}  TBPTT={tbptt_window}")
    print(f"  Rollout ICs: {n_ics}/{len(u_refs_np)}  "
          f"FD targets: {len(u_fd_t) if u_fd_t is not None else 0}")

    optimizer = optim.Adam(model.parameters(),
                            lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=1, eta_min=1e-5)

    history    = []
    best_loss  = float("inf")
    best_state = None

    print(f"\n{'Ep':>5}  {'Total':>8}  {'L_data':>8}  "
          f"{'L_cons':>8}  {'L_anch':>8}  {'L_fd':>8}  LR")
    print("-"*65)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        epoch_loss  = torch.tensor(0.0, device=device)
        epoch_terms = {k: 0.0 for k in
                       ["L_data","L_cons","L_anch","L_fd","L_pos","total"]}

        for u_ref_t in u_refs_t:
            # Count windows for this IC
            Nt = cfg["Nt"]
            n_wins = max(1, Nt // tbptt_window)

            # Accumulate loss over all TBPTT windows
            ic_loss = torch.tensor(0.0, device=device)
            ic_terms = {k: 0.0 for k in epoch_terms}

            for u_traj_w, R_traj_w, u_ref_w in rollout_tbptt(
                    model, u_ref_t[0], u_ref_t, x_t, phi_t, A_t, L_op_t,
                    cfg, problem, t_arr, device, window=tbptt_window):

                L_w, terms_w = total_loss(
                    u_traj_w, R_traj_w, u_ref_w, L_op_t, cfg,
                    s_u, epoch, anchors, model, device,
                    u_fd_t=u_fd_t, R_fd_t=R_fd_t, weights_t=w_fd_t)

                # Backprop through this window immediately
                (L_w / n_wins).backward()

                ic_loss = ic_loss + L_w.detach() / n_wins
                for k in ic_terms:
                    ic_terms[k] += terms_w.get(k, 0.0) / n_wins

            epoch_loss = epoch_loss + ic_loss / n_ics
            for k in epoch_terms:
                epoch_terms[k] += ic_terms[k] / n_ics

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        epoch_terms["total"] = epoch_loss.item()
        epoch_terms["epoch"] = epoch
        epoch_terms["lr"]    = scheduler.get_last_lr()[0]
        history.append(epoch_terms)

        if epoch_terms["total"] < best_loss:
            best_loss  = epoch_terms["total"]
            best_state = {k: v.clone() for k,v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"{epoch:>5}  {epoch_terms['total']:>8.4f}  "
                  f"{epoch_terms['L_data']:>8.5f}  "
                  f"{epoch_terms['L_cons']:>8.5f}  "
                  f"{epoch_terms['L_anch']:>8.5f}  "
                  f"{epoch_terms['L_fd']:>8.5f}  "
                  f"{epoch_terms['lr']:.2e}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Best model restored (loss={best_loss:.5f})")

    # Final no-grad rollout for distillation
    model.eval()
    all_u_trajs, all_R_trajs = [], []
    with torch.no_grad():
        for u_ref_t in u_refs_t:
            u_tf, R_tf = rollout_full_nograd(
                model, u_ref_t[0], u_ref_t, x_t, phi_t, A_t, L_op_t,
                cfg, problem, t_arr, device)
            all_u_trajs.append(u_tf)
            all_R_trajs.append(R_tf)

    return history, all_u_trajs, all_R_trajs
