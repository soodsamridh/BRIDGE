import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def count_parameters(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


# ======================================================================
#  Networks
# ======================================================================

class SolutionNet(nn.Module):
    """u_theta(x,t), or u_theta(x,t,v) for the partial-inverse setting.

    Raissi's solution network: fully connected, tanh, moderate depth.
    Width 56 / depth 5 puts the PINN and FDPINN parameter count within a
    few percent of the BRIDGE scalar model.
    """

    def __init__(self, coupled=False, depth=5, width=56):
        super().__init__()
        self.coupled = coupled
        in_dim = 3 if coupled else 2
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, v=None):
        inp = (torch.stack([x, t, v], -1) if self.coupled
               else torch.stack([x, t], -1))
        return self.net(inp).squeeze(-1)


class ReactionNet(nn.Module):
    """R_psi(u) or R_psi(u,v): the unknown reaction law.

    Replaces Raissi's parameter vector lambda. Depth, width and the
    bounded output Lambda*tanh(.) match the BRIDGE decoder so that
    reaction-network capacity is not a confound.
    """

    def __init__(self, coupled=False, width=None, Lambda=2.0, depth=3):
        super().__init__()
        self.coupled = coupled
        self.Lambda = Lambda
        in_dim = 2 if coupled else 1
        if width is None:
            width = 64 if coupled else 32
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u, v=None):
        inp = (torch.stack([u, v], -1) if self.coupled else u.unsqueeze(-1))
        return self.Lambda * torch.tanh(self.net(inp)).squeeze(-1)

 
    def react_grad(self, u, v=None):
        return self.forward(u, v)

    @torch.no_grad()
    def react_nograd(self, u, v=None):
        return self.forward(u, v)

    def reaction_from_u_only(self, u):
        return self.forward(u)

    @torch.no_grad()
    def reaction_from_u_only_nograd(self, u):
        return self.forward(u)

    @torch.no_grad()
    def reaction_nograd(self, u):
        return self.forward(u)


class ZhengRNN(nn.Module):
    def __init__(self, n_state, hidden=32):
        super().__init__()
        self.n_state = n_state
        self.hidden = hidden
        self.cell = nn.RNNCell(n_state, hidden, nonlinearity="tanh")
        self.dec = nn.Linear(hidden, n_state)
        nn.init.normal_(self.dec.weight, std=0.01)
        nn.init.zeros_(self.dec.bias)

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden, device=device)

    def step(self, u_n, h, dt):
        h = self.cell(u_n.unsqueeze(0), h)
        return u_n + dt * self.dec(h).squeeze(0), h


# ======================================================================
#  Residuals
# ======================================================================

def residual_autodiff(u_net, R_net, x, t, eps, v=None):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_net(x, t, v) if u_net.coupled else u_net(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                               create_graph=True)[0]
    R = R_net(u, v) if R_net.coupled else R_net(u)
    return u_t - eps * u_xx - R


def residual_fd(u_net, R_net, x_t, t_levels, L_t, dt, eps, v_levels=None):
    res = []
    for n in range(1, len(t_levels) - 1):
        tp = torch.full_like(x_t, float(t_levels[n - 1]))
        tc = torch.full_like(x_t, float(t_levels[n]))
        tn = torch.full_like(x_t, float(t_levels[n + 1]))
        if u_net.coupled:
            vp, vc, vn = v_levels[n - 1], v_levels[n], v_levels[n + 1]
            up, uc, un = (u_net(x_t, tp, vp), u_net(x_t, tc, vc),
                          u_net(x_t, tn, vn))
            R = R_net(uc, vc)
        else:
            up, uc, un = u_net(x_t, tp), u_net(x_t, tc), u_net(x_t, tn)
            R = R_net(uc)
        res.append(((un - up) / (2.0 * dt) - eps * (L_t @ uc) - R)[1:-1])
    return torch.cat(res)


# ======================================================================
#  PINN / FD-PINN
# ======================================================================

def _assemble(refs, t_arr, x, Nt, coupled, device, max_snaps=60,
              state_scale=None):
    snap = np.linspace(0, Nt, min(Nt + 1, max_snaps), dtype=int)
    xs, ts, us, vs = [], [], [], []
    for r in refs:
        ur, vr = r if coupled else (r, None)
        for n in snap:
            xs.append(x); ts.append(np.full_like(x, t_arr[n])); us.append(ur[n])
            if coupled:
                vs.append(vr[n])

    def _t(a):
        return torch.tensor(np.concatenate(a), dtype=torch.float32,
                            device=device)

    out = dict(xd=_t(xs), td=_t(ts), ud=_t(us),
               vd=_t(vs) if coupled else None)
    if state_scale is None:
        all_u = np.concatenate([(r[0] if coupled else r).flatten()
                                for r in refs])
        state_scale = float(np.mean(np.abs(all_u)))
    out["state_scale"] = float(state_scale)
    return out


def train_pinn_family(method, cfg, refs, t_arr, x, L_t, epochs,
                      coupled=False, seed=42, device=None,
                      w_f=1.0, lr=1e-3, wd=0.0, state_scale=None,
                      verbose=True, track_best=True):
    device = device or torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    eps = cfg.get("D1", cfg["eps_diff"])
    dt = cfg["dt"]; Nt = cfg["Nt"]
    res_scale = float(cfg["Lambda"])

    S = _assemble(refs, t_arr, x, Nt, coupled, device,
                  state_scale=state_scale)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    u_net = SolutionNet(coupled=coupled).to(device)
    R_net = ReactionNet(coupled=coupled, Lambda=cfg["Lambda"]).to(device)
    params = list(u_net.parameters()) + list(R_net.parameters())
    n_params = count_parameters(u_net, R_net)

    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                 eta_min=1e-5)

    stride = max(1, Nt // 40)
    t_lv = t_arr[::stride]
    v_lv = None
    if coupled:
        vr0 = refs[0][1]
        v_lv = [torch.tensor(vr0[k], dtype=torch.float32, device=device)
                for k in range(0, Nt + 1, stride)]

    hist = []; best = float("inf"); best_state = None
    if verbose:
        print(f"    {method.upper()}  ({n_params} params, {epochs} ep, "
              f"w_f={w_f:g}, {device})", flush=True)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        u_pred = (u_net(S["xd"], S["td"], S["vd"]) if coupled
                  else u_net(S["xd"], S["td"]))
        mse_u = ((u_pred - S["ud"]) / (S["state_scale"] + 1e-8)).pow(2).mean()
        if method == "pinn":
            f = residual_autodiff(u_net, R_net, S["xd"], S["td"], eps,
                                  v=S["vd"] if coupled else None)
        else:
            f = residual_fd(u_net, R_net, x_t, t_lv, L_t, dt * stride, eps,
                            v_levels=v_lv)
        mse_f = (f / res_scale).pow(2).mean()

        loss = mse_u + w_f * mse_f
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        m = float(mse_u.item())
        hist.append({"epoch": ep, "total": float(loss.item()),
                     "data_misfit": m, "residual": float(mse_f.item())})
        if track_best and m < best:
            best = m
            best_state = ({k: v.detach().clone()
                           for k, v in u_net.state_dict().items()},
                          {k: v.detach().clone()
                           for k, v in R_net.state_dict().items()})
        if verbose and ep % 250 == 0:
            print(f"      ep {ep}/{epochs}  MSE_u={m:.5f}  "
                  f"MSE_f={mse_f.item():.5f}")

    if track_best and best_state is not None:
        u_net.load_state_dict(best_state[0])
        R_net.load_state_dict(best_state[1])
    return u_net, R_net, hist, n_params


def run_pinn_baseline(cfg, refs, t_arr, x, L_t, epochs, **kw):
    return train_pinn_family("pinn", cfg, refs, t_arr, x, L_t, epochs, **kw)


def run_fdpinn_baseline(cfg, refs, t_arr, x, L_t, epochs, **kw):
    return train_pinn_family("fdpinn", cfg, refs, t_arr, x, L_t, epochs, **kw)


# ======================================================================
#  PIRNN 
# ======================================================================

def train_pirnn(cfg, refs, t_arr, x, L_t, epochs, seed=42, device=None,
                w_G=1.0, lr=1e-3, wd=0.0, state_scale=None, window=None,
                verbose=True, track_best=True):
    device = device or torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    eps = cfg["eps_diff"]; dt = cfg["dt"]; Nt = cfg["Nt"]
    Nx1 = len(x)
    res_scale = float(cfg["Lambda"])
    window = window or cfg.get("tbptt", 15)

    n_roll = min(cfg.get("n_rollout_ics", len(refs)), len(refs))
    u_refs = [torch.tensor(r, dtype=torch.float32, device=device)
              for r in refs[:n_roll]]
    if state_scale is None:
        state_scale = float(np.mean(np.abs(
            np.concatenate([r.flatten() for r in refs[:n_roll]]))))

    rnn = ZhengRNN(Nx1, hidden=32).to(device)
    R_net = ReactionNet(coupled=False, Lambda=cfg["Lambda"]).to(device)
    params = list(rnn.parameters()) + list(R_net.parameters())
    n_params = count_parameters(rnn, R_net)

    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                 eta_min=1e-5)

    hist = []; best = float("inf"); best_state = None
    if verbose:
        print(f"    PIRNN  ({n_params} params, {epochs} ep, "
              f"w_G={w_G:g}, {device})", flush=True)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        # Accumulate squared errors and counts for exact common-misfit computation
        total_se = 0.0
        total_count = 0
        tot_loss = 0.0
        n_windows = 0

        for u_ref in u_refs:
            h = rnn.init_hidden(device)
            u_n = u_ref[0]
            n = 0
            while n < Nt:
                w_end = min(n + window, Nt)
                u_n = u_n.detach(); h = h.detach()
                seq = [u_n]
                for s in range(n, w_end):
                    u_n, h = rnn.step(u_n, h, dt)
                    seq.append(u_n)
                U = torch.stack(seq)                      # (w+1, Nx1)

                # MSE_X -- fit to the observed sequence
                diff = U - u_ref[n:w_end + 1]
                se = (diff / (state_scale + 1e-8)).pow(2).sum()
                total_se += se.item()
                total_count += U.numel()
                mse_X = se / U.numel()

                # MSE_G -- governing residual on the PREDICTED outputs
                if U.shape[0] >= 3:
                    Uc = U[1:-1]
                    dU = (U[2:] - U[:-2]) / (2.0 * dt)
                    lap = Uc @ L_t.T
                    g = dU - eps * lap - R_net(Uc.reshape(-1)) \
                        .reshape(Uc.shape)
                    mse_G = (g[:, 1:-1] / res_scale).pow(2).mean()
                else:
                    mse_G = torch.zeros((), device=device)

                Lw = mse_X + w_G * mse_G
                Lw.backward()
                tot_loss += float(Lw.detach())
                n_windows += 1
                n = w_end

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        # Common data misfit computed identically to PINN/FDPINN
        common_misfit = (total_se / total_count) if total_count > 0 else 0.0
        avg_loss = tot_loss / max(n_windows, 1)

        hist.append({"epoch": ep, "total": avg_loss,
                     "data_misfit": common_misfit})
        if track_best and common_misfit < best:
            best = common_misfit
            best_state = ({k: v.detach().clone()
                           for k, v in rnn.state_dict().items()},
                          {k: v.detach().clone()
                           for k, v in R_net.state_dict().items()})
        if verbose and ep % 250 == 0:
            print(f"      ep {ep}/{epochs}  MSE_X={common_misfit:.5f}  total={avg_loss:.5f}")

    if track_best and best_state is not None:
        rnn.load_state_dict(best_state[0])
        R_net.load_state_dict(best_state[1])
    return rnn, R_net, hist, n_params


@torch.no_grad()
def pirnn_rollout(rnn, u0_np, cfg, device):
    """Autonomous rollout of the PIRNN state predictor."""
    dt = cfg["dt"]; Nt = cfg["Nt"]
    u_n = torch.tensor(u0_np, dtype=torch.float32, device=device)
    h = rnn.init_hidden(device)
    traj = [u_n.cpu().numpy()]
    for _ in range(Nt):
        u_n, h = rnn.step(u_n, h, dt)
        traj.append(u_n.cpu().numpy())
    return np.stack(traj)


# ======================================================================
#  Weight sweep
# ======================================================================

def sweep_weight(train_fn, eval_fn, weights, key="w_f", **kw):
    """Sweep physics-loss weight, selecting by a VALIDATION metric.

    The evaluation function eval_fn should compute a validation error
    (e.g., trajectory error on a held-out initial condition) that does
    NOT use the ground truth reaction law. This avoids data leakage.
    """
    records = []; best = None
    for w in weights:
        out = train_fn(**{key: w}, **kw)
        # out is (model_or_net, R_net, hist, n_params); R_net is index 1
        R_net = out[1]
        e = float(eval_fn(R_net))
        records.append({key: float(w), "val_error": e,
                        "final_misfit": hist[-1]["data_misfit"]})
        print(f"      {key}={w:<7g} val_error={e:.4f}  "
              f"misfit={hist[-1]['data_misfit']:.5f}")
        if best is None or e < best[0]:
            best = (e, w)
    return best[1], records