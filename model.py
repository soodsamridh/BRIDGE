"""
model.py  —  ε-US-GRU v5 architectures.

EpsUSGRU          — scalar problems (Fisher, AC, FHN scalar, Bistable)
                    Input dim=4: [u, Δ_h u, ∂_t u, x]
                    MLP input:   u only (1D)

EpsUSGRU_SemiKnown — FHN Bonhoeffer-Van der Pol semi-known system
                    Input dim=5: [u, v, Δ_h u, ∂_t u, x]
                    MLP input:   (u, v) jointly (2D)
                    Single head: outputs R_u(u,v) only
                    v enters as known conditioning input (not predicted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
#  Scalar model (unchanged)
# ══════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════
#  Semi-known system model for FHN Bonhoeffer-Van der Pol
# ══════════════════════════════════════════════════════════

class EpsUSGRU_SemiKnown(nn.Module):
    """
    ε-US-GRU for the FHN semi-known system.

    Identifies: R_u(u,v) = -v + u - u³/3   (activator nonlinearity)
    Uses:       v known from explicit ODE solve at each step

    Architecture differences from scalar EpsUSGRU:
      - GRU input dim = 5: [u, v, Δ_h u, ∂_t u, x]
        (v included as known feature; no Δ_h v since v has no diffusion)
      - LayerNorm on GRU inputs (prevents O(1/D) feature blow-up)
      - MLP input = 2D: (u, v) instead of u only
      - Single decoder head (R_u only)
      - FiLM modulates based on H_blend from both GRU cells

    Parameters: ~10,593 (comparable to scalar 10,369)
    """
    def __init__(self, hidden_dim=32, mlp_width=32, Lambda=6.0, input_dim=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_width  = mlp_width
        self.Lambda     = Lambda

        # No LayerNorm — clamping is used instead (no trainable params to destabilise)
        # Clamp features to [-10, 10] before GRU to prevent blow-up
        self.gru_out   = nn.GRUCell(input_dim, hidden_dim)
        self.gru_lay   = nn.GRUCell(input_dim, hidden_dim)
        self.film_proj = nn.Linear(hidden_dim, 2*mlp_width)

        # 2D input MLP: (u, v) → R_u
        self.mlp1 = nn.Linear(2, mlp_width)   # [u, v] → hidden
        self.mlp2 = nn.Linear(mlp_width, mlp_width)
        self.mlp3 = nn.Linear(mlp_width, 1)

        nn.init.normal_(self.mlp3.weight, std=0.01)
        nn.init.zeros_(self.mlp3.bias)

    def init_hidden(self, Nx, device):
        h = torch.zeros(Nx, self.hidden_dim, device=device)
        return h, h.clone()

    def _decode(self, u, v, alpha, beta):
        """Apply FiLM-modulated 2D MLP to predict R_u(u,v)."""
        uv  = torch.stack([u, v], dim=-1)          # (Nx, 2)
        h1  = F.relu(self.mlp1(uv))                # (Nx, W)
        h2  = F.relu((1 + alpha) * h1 + beta)      # FiLM on first hidden
        h3  = F.relu(self.mlp2(h2))
        return self.Lambda * torch.tanh(self.mlp3(h3)).squeeze(-1)

    def forward(self, u, v, z_out, z_lay, rho, H_out, H_lay,
                detach_gru=False):
        """
        u, v        — current field values (Nx,)
        z_out       — outer-regime features (Nx, input_dim)
        z_lay       — layer-regime features (Nx, input_dim)
        rho         — regime weights (Nx,)
        detach_gru  — if True, detach H_blend before film_proj.
                      Use this during GRU-freeze phase so that the
                      gradient from film_proj/MLP does NOT flow backward
                      through the GRU forward computation.
                      Freezing GRU weights alone does NOT stop this —
                      requires explicit detach of the hidden state tensor.
        """
        rho_e   = rho.unsqueeze(-1)
        H_out_c = self.gru_out(torch.clamp(z_out, -10, 10), H_out)
        H_lay_c = self.gru_lay(torch.clamp(z_lay, -10, 10), H_lay)
        H_out_n = (1-rho_e)*H_out_c + rho_e*H_out
        H_lay_n =    rho_e *H_lay_c + (1-rho_e)*H_lay
        H_blend = (1-rho_e)*H_out_n + rho_e*H_lay_n

        # Detach H_blend during GRU-freeze phase — severs gradient path
        # through GRU cell equations (tanh/sigmoid) that causes NaN
        if detach_gru:
            H_blend = H_blend.detach()

        film  = self.film_proj(H_blend)
        alpha = film[:, :self.mlp_width]
        beta  = film[:, self.mlp_width:]

        R_u = self._decode(u, v, alpha, beta)
        return R_u, H_out_n, H_lay_n

    def reaction_grad(self, u_vals, v_vals):
        """Neutral FiLM (α=0, β=0) — gradient-transparent."""
        Nx     = u_vals.shape[0]
        device = u_vals.device
        alpha  = torch.zeros(Nx, self.mlp_width, device=device)
        beta_  = torch.zeros(Nx, self.mlp_width, device=device)
        return self._decode(u_vals, v_vals, alpha, beta_)

    @torch.no_grad()
    def reaction_nograd(self, u_vals, v_vals):
        return self.reaction_grad(u_vals, v_vals)