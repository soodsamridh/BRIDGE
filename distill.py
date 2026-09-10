"""
distill.py — Post-training distillation of pure scalar R_scalar(u).

After Stage 1, R_θ may still have residual FiLM spatial dependence.
Distillation gives a hard guarantee: a pure MLP f(u) with no hidden state.

Process:
  1. Evaluate R_θ on a dense u-grid using neutral FiLM
  2. Add anchor points at known zeros
  3. Fit a smooth 3-layer Tanh MLP to the combined data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ScalarMLP(nn.Module):
    """Pure u → R function. Tanh activations for smooth output."""
    def __init__(self, hidden=32, Lambda=2.0):
        super().__init__()
        self.Lambda = Lambda
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u):
        return self.Lambda * torch.tanh(
            self.net(u.unsqueeze(-1))).squeeze(-1)


def distil(model, cfg, anchors, device, epochs=1000):
    """
    Distil R_θ into a pure scalar MLP.

    Returns scalar_mlp, u_grid, R_distilled arrays.
    """
    u_min  = cfg["u_min"]; u_max = cfg["u_max"]
    Lambda = cfg["Lambda"]

    # Evaluate model on dense grid with neutral FiLM
    u_grid = np.linspace(u_min, u_max, 200)
    u_t    = torch.tensor(u_grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        R_model = model.reaction_from_u_only(u_t).cpu().numpy()

    # Build training set: dense grid + anchor points
    u_train = u_grid.tolist()
    R_train = R_model.tolist()
    w_train = [1.0] * len(u_grid)

    for (u_a, R_a) in anchors:
        u_train.append(u_a); R_train.append(R_a); w_train.append(20.0)

    u_tt = torch.tensor(u_train, dtype=torch.float32, device=device)
    R_tt = torch.tensor(R_train, dtype=torch.float32, device=device)
    w_tt = torch.tensor(w_train, dtype=torch.float32, device=device)
    w_tt = w_tt / w_tt.mean()

    scalar_mlp = ScalarMLP(hidden=32, Lambda=Lambda).to(device)
    opt   = optim.Adam(scalar_mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss  = float("inf")
    best_state = None
    for ep in range(epochs):
        opt.zero_grad()
        R_pred = scalar_mlp(u_tt)
        loss   = (w_tt * (R_pred - R_tt).pow(2)).mean()
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k,v in scalar_mlp.state_dict().items()}

    if best_state: scalar_mlp.load_state_dict(best_state)
    scalar_mlp.eval()

    with torch.no_grad():
        R_dist = scalar_mlp(u_t).cpu().numpy()

    print(f"  Distillation done. Best loss = {best_loss:.6f}")
    return scalar_mlp, u_grid, R_dist
