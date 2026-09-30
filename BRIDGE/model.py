import torch
import torch.nn as nn
import torch.nn.functional as F

STREAMS = ("both", "outer", "layer")


# ======================================================================
#  Scalar and semi-known architectures
# ======================================================================

class EpsUSGRU(nn.Module):
    def __init__(self, hidden_dim=32, mlp_width=32, Lambda=2.0, input_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_width  = mlp_width
        self.Lambda     = Lambda

        self.gru_out   = nn.GRUCell(input_dim, hidden_dim)
        self.gru_lay   = nn.GRUCell(input_dim, hidden_dim)
        self.film_proj = nn.Linear(hidden_dim, 2*mlp_width)

        self.mlp1 = nn.Linear(1, mlp_width)
        self.mlp2 = nn.Linear(mlp_width, mlp_width)
        self.mlp3 = nn.Linear(mlp_width, 1)

        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden_dim, device=device)
        return h, h.clone()

    def forward(self, u, z_out, z_lay, rho, H_out, H_lay):
        rho_e   = rho.unsqueeze(-1)
        H_out_c = self.gru_out(z_out, H_out)
        H_lay_c = self.gru_lay(z_lay, H_lay)
        H_out_n = (1-rho_e)*H_out_c + rho_e*H_out
        H_lay_n =    rho_e *H_lay_c + (1-rho_e)*H_lay
        H_blend = (1-rho_e)*H_out_n + rho_e*H_lay_n

        film  = self.film_proj(H_blend)
        alpha = film[:, :self.mlp_width]
        beta  = film[:, self.mlp_width:]

        h1     = F.relu(self.mlp1(u.unsqueeze(-1)))
        h2     = F.relu((1+alpha)*h1 + beta)
        h3     = F.relu(self.mlp2(h2))
        R_pred = self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

        return R_pred, H_out_n, H_lay_n

    def reaction_from_u_only(self, u_vals):
        Nx     = u_vals.shape[0]
        device = u_vals.device
        alpha  = torch.zeros(Nx, self.mlp_width, device=device)
        beta   = torch.zeros(Nx, self.mlp_width, device=device)
        h1 = F.relu(self.mlp1(u_vals.unsqueeze(-1)))
        h2 = F.relu((1+alpha)*h1 + beta)
        h3 = F.relu(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    def reaction_from_u_only_nograd(self, u_vals):
        with torch.no_grad():
            return self.reaction_from_u_only(u_vals)


class EpsUSGRU_SemiKnown(nn.Module):
    def __init__(self, hidden_dim=32, mlp_width=32, Lambda=6.0, input_dim=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_width  = mlp_width
        self.Lambda     = Lambda

        self.gru_out   = nn.GRUCell(input_dim, hidden_dim)
        self.gru_lay   = nn.GRUCell(input_dim, hidden_dim)
        self.film_proj = nn.Linear(hidden_dim, 2*mlp_width)

        self.mlp1 = nn.Linear(2, mlp_width)   # [u, v] → hidden
        self.mlp2 = nn.Linear(mlp_width, mlp_width)
        self.mlp3 = nn.Linear(mlp_width, 1)

        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden_dim, device=device)
        return h, h.clone()

    def _decode(self, u, v, alpha, beta):
        uv  = torch.stack([u, v], dim=-1)          # (Nx, 2)
        h1  = F.relu(self.mlp1(uv))                # (Nx, W)
        h2  = F.relu((1 + alpha) * h1 + beta)      # FiLM on first hidden
        h3  = F.relu(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    def forward(self, u, v, z_out, z_lay, rho, H_out, H_lay,
                detach_gru=False):
        rho_e   = rho.unsqueeze(-1)
        H_out_c = self.gru_out(torch.clamp(z_out, -10, 10), H_out)
        H_lay_c = self.gru_lay(torch.clamp(z_lay, -10, 10), H_lay)
        H_out_n = (1-rho_e)*H_out_c + rho_e*H_out
        H_lay_n =    rho_e *H_lay_c + (1-rho_e)*H_lay
        H_blend = (1-rho_e)*H_out_n + rho_e*H_lay_n

        if detach_gru:
            H_blend = H_blend.detach()

        film  = self.film_proj(H_blend)
        alpha = film[:, :self.mlp_width]
        beta  = film[:, self.mlp_width:]

        R_u = self._decode(u, v, alpha, beta)
        return R_u, H_out_n, H_lay_n

    def reaction_grad(self, u_vals, v_vals):
        Nx     = u_vals.shape[0]
        device = u_vals.device
        alpha  = torch.zeros(Nx, self.mlp_width, device=device)
        beta_  = torch.zeros(Nx, self.mlp_width, device=device)
        return self._decode(u_vals, v_vals, alpha, beta_)

    @torch.no_grad()
    def reaction_nograd(self, u_vals, v_vals):
        return self.reaction_grad(u_vals, v_vals)


# ======================================================================
#  Switchable encoder (capacity and gate-alignment study)
# ======================================================================

def count_parameters(*modules):
    return int(sum(p.numel() for m in modules for p in m.parameters()))


class EncoderReactionNet(nn.Module):
    def __init__(self, mode="dual", hidden=32, mlp_w=32, Lambda=1.0,
                 input_dim=4, state_dim=1, gate="film", act="relu"):
        super().__init__()
        assert mode in ("dual", "single"), mode
        assert gate in ("film", "additive"), gate
        assert act in ("relu", "silu"), act
        self.mode = mode
        self.hidden = hidden
        self.mlp_w = mlp_w
        self.Lambda = float(Lambda)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.gate_kind = gate
        self._act = F.silu if act == "silu" else F.relu

        self.gru_out = nn.GRUCell(input_dim, hidden)
        if mode == "dual":
            self.gru_lay = nn.GRUCell(input_dim, hidden)

        out_dim = 2 * mlp_w if gate == "film" else mlp_w
        self.proj = nn.Linear(hidden, out_dim)
        self.mlp1 = nn.Linear(state_dim, mlp_w)
        self.mlp2 = nn.Linear(mlp_w, mlp_w)
        self.mlp3 = nn.Linear(mlp_w, 1)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.mlp1.weight, std=0.10)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)

    # -- hidden state ---------------------------------------------------
    def init_hidden(self, n_nodes, device):
        h = torch.zeros(int(n_nodes), self.hidden, device=device)
        return h, h.clone()

    # -- decoder --------------------------------------------------------
    def _decode(self, state, ctx):
        act = self._act
        h1 = act(self.mlp1(state))
        if self.gate_kind == "film":
            if ctx is None:
                h2 = act(h1)
            else:
                a = ctx[..., :self.mlp_w]
                b = ctx[..., self.mlp_w:]
                h2 = act((1 + a) * h1 + b)
        else:
            h2 = act(h1) if ctx is None else act(h1 + ctx)
        h3 = act(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    @staticmethod
    def _stack(u, v=None):
        return u.unsqueeze(-1) if v is None else torch.stack([u, v], dim=-1)

    def forward(self, u, z_out, z_lay, rho, H_out, H_lay, v=None,
                stream="both"):
        assert stream in STREAMS, stream
        shp = u.shape
        flat = (-1, self.input_dim)

        if self.mode == "dual":
            rho_e = rho.reshape(-1, 1)
            Ho = self.gru_out(z_out.reshape(flat), H_out)
            Hl = self.gru_lay(z_lay.reshape(flat), H_lay)
            H_out_n = (1 - rho_e) * Ho + rho_e * H_out
            H_lay_n = rho_e * Hl + (1 - rho_e) * H_lay
            if stream == "outer":
                H_use = H_out_n
            elif stream == "layer":
                H_use = H_lay_n
            else:
                H_use = (1 - rho_e) * H_out_n + rho_e * H_lay_n
        else:
            H_out_n = self.gru_out(z_out.reshape(flat), H_out)
            H_lay_n = H_out_n
            H_use = H_out_n

        ctx = self.proj(H_use).reshape(*shp, -1)
        return self._decode(self._stack(u, v), ctx), H_out_n, H_lay_n

    # -- identified law (gamma = 0) --------------------------------------
    def react_id(self, u, v=None):
        return self._decode(self._stack(u, v), None)

    @torch.no_grad()
    def react_id_nograd(self, u, v=None):
        return self.react_id(u, v)



# ======================================================================
#  Parameter matching
# ======================================================================

def _count_for(mode, hidden, geom):
    return count_parameters(EncoderReactionNet(mode=mode, hidden=hidden,
                                               **geom))


def matched_hidden(dual_hidden=32, geom=None, hi=512):
    geom = geom or {}
    target = _count_for("dual", dual_hidden, geom)
    lo, high = 2, hi
    while lo < high:
        mid = (lo + high) // 2
        if _count_for("single", mid, geom) < target:
            lo = mid + 1
        else:
            high = mid
    cands = [h for h in (lo - 1, lo) if h >= 2]
    best = min(cands, key=lambda h: abs(_count_for("single", h, geom) - target))
    return best, target


def build_pair(geom, dual_hidden=32, tol=0.02, device=None, verbose=True):
    device = device or torch.device("cpu")
    h_single, target = matched_hidden(dual_hidden, geom)
    dual = EncoderReactionNet(mode="dual", hidden=dual_hidden,
                              **geom).to(device)
    single = EncoderReactionNet(mode="single", hidden=h_single,
                                **geom).to(device)
    n_d = count_parameters(dual); n_s = count_parameters(single)
    rel = (n_s - n_d) / n_d
    if abs(rel) > tol:
        raise RuntimeError(
            f"parameter matching failed: dual={n_d}, single={n_s} "
            f"({rel:+.2%}), tolerance {tol:.0%}. Adjust mlp_w or the "
            f"search range in matched_hidden().")
    info = {"dual_hidden": dual_hidden, "single_hidden": h_single,
            "dual_params": n_d, "single_params": n_s,
            "rel_gap": float(rel), "geom": {k: v for k, v in geom.items()
                                            if k != "Lambda"}}
    if verbose:
        print(f"    dual   : hidden={dual_hidden:<4} params={n_d}")
        print(f"    single : hidden={h_single:<4} params={n_s} ({rel:+.2%})")
    return dual, single, info


