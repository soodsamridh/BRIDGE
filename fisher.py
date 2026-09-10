"""
epsilon-US-GRU for Fisher-KPP equation
u_t = ε u_xx + β u (1-u), ε=0.01, β=6
Exact travelling wave: u(x,t) = 1/(1 + exp((x - ct)/√ε))^2, c = 2√(εβ)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- Configuration --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Problem parameters
epsilon = 0.01          # diffusion coefficient (ε_diff = ε)
beta = 6.0              # reaction strength
Lambda = 1.0            # output scale (R = O(1))
g_L = 1.0               # u(0,t)
g_R = 0.0               # u(1,t)
T = 0.3                 # final time
dt = 0.005              # time step
Nt = int(T / dt) + 1    # number of time points

# Spatial discretization (Shishkin mesh)
N = 128                 # number of intervals (must be even)
beta_shishkin = 2.449   # for Fisher
sigma = min(0.25, (2 * np.sqrt(epsilon) / beta_shishkin) * np.log(N))
sigma = max(sigma, 4.0 / N)   # ensure at least 4 points in layer
n_fine = N // 4
n_coarse = N // 2

x_left = np.linspace(0, sigma, n_fine + 1, endpoint=False)
x_mid = np.linspace(sigma, 1 - sigma, n_coarse + 1)
x_right = np.linspace(1 - sigma, 1, n_fine + 1)
x = np.concatenate([x_left, x_mid[1:], x_right[1:]])
x = torch.tensor(x, dtype=torch.float32, device=device)
Nx = len(x)
print(f"Mesh: Nx={Nx}, sigma={sigma:.4f}, h1={sigma/(n_fine):.4f}, h2={(1-2*sigma)/n_coarse:.4f}")

# ------------------------- Compact Laplacian -------------------------
def compact_laplacian(x):
    N = len(x)
    M = torch.zeros(N, N, device=x.device)
    D = torch.zeros(N, N, device=x.device)
    for i in range(1, N-1):
        hm = x[i] - x[i-1]
        hp = x[i+1] - x[i]
        S = hm + hp
        Q = hm*hm + 3*hm*hp + hp*hp
        M[i,i-1] = hp * (hm*hm + hm*hp - hp*hp) / (S * Q)
        M[i,i] = 1.0
        M[i,i+1] = hm * (-hm*hm + hm*hp + hp*hp) / (S * Q)
        D[i,i-1] = 12 * hp / (S * Q)
        D[i,i] = -12 / Q
        D[i,i+1] = 12 * hm / (S * Q)
    # Boundary: simple 2nd order (can be improved)
    h0 = x[1] - x[0]
    M[0,0] = 1.0
    D[0,0] = -2.0/(h0*h0)
    D[0,1] = 2.0/(h0*h0)
    hN = x[-1] - x[-2]
    M[-1,-1] = 1.0
    D[-1,-2] = 2.0/(hN*hN)
    D[-1,-1] = -2.0/(hN*hN)
    L_op = torch.linalg.solve(M, D)
    return L_op

L_op = compact_laplacian(x)                     # discrete Laplacian (Nx x Nx)
A_imex = torch.eye(Nx, device=device) - (dt/2) * epsilon * L_op   # IMEX matrix

# Precompute gradient operator (for |∇u|) and boundary-layer envelope φ
def gradient(u):
    grad = torch.zeros_like(u)
    grad[1:-1] = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
    grad[0] = (u[1] - u[0]) / (x[1] - x[0])
    grad[-1] = (u[-1] - u[-2]) / (x[-1] - x[-2])
    return torch.abs(grad)

phi = torch.exp(-torch.min(x, 1 - x) / np.sqrt(epsilon))

# ------------------------- Exact initial condition -------------------------
c_exact = 2 * np.sqrt(epsilon * beta)   # wave speed = 0.4899
def exact_solution(x, t):
    return 1.0 / (1.0 + torch.exp((x - c_exact * t) / np.sqrt(epsilon)))**2

u0 = exact_solution(x, 0.0)             # initial condition

# ------------------------- Reference trajectory (using true reaction) -------------------------
def true_R(u):
    return beta * u * (1 - u)

def imex_step(u, R):
    rhs = u + (dt/2) * epsilon * (L_op @ u) + dt * R
    rhs[0] = g_L
    rhs[-1] = g_R
    return torch.linalg.solve(A_imex, rhs)

u_ref = [u0.clone()]
for n in range(Nt-1):
    R_true = true_R(u_ref[-1])
    u_next = imex_step(u_ref[-1], R_true)
    u_ref.append(u_next)
u_ref = torch.stack(u_ref)   # shape (Nt, Nx)

# ------------------------- Neural Operator Model (ε-US-GRU) -------------------------
class EpsUSGRU(nn.Module):
    def __init__(self, Nx, d_hidden=32, mlp_width=32, Lambda=1.0):
        super().__init__()
        self.Nx = Nx
        self.d_hidden = d_hidden
        self.Lambda = Lambda
        # Dual GRU cells (input size 4)
        self.gru_out = nn.GRUCell(4, d_hidden)
        self.gru_lay = nn.GRUCell(4, d_hidden)
        # FiLM: hidden state -> (alpha, beta) each of size mlp_width
        self.film = nn.Linear(d_hidden, 2 * mlp_width)
        # Scalar MLP: u -> h1 -> FiLM modulation -> output
        self.mlp1 = nn.Linear(1, mlp_width)
        self.mlp2 = nn.Linear(mlp_width, 1)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u, dt_u, H_out, H_lay):
        """
        u: (Nx,) current solution
        dt_u: (Nx,) time derivative (e.g., backward difference)
        H_out, H_lay: (Nx, d_hidden) previous hidden states
        Returns: Rθ (Nx,), H_out_new, H_lay_new, rho (Nx,)
        """
        # ---- Regime indicator (log-median) ----
        Lu = L_op @ u
        log_abs = torch.log(torch.abs(Lu) + 1e-8)
        median_log = torch.median(log_abs)
        denom = torch.log(torch.tensor(1.0 / epsilon))
        rho = torch.sigmoid(3.0 * (log_abs - median_log) / denom)   # gamma=3

        # ---- Input features ----
        # Outer: [u, Δu, ∂_t u, x]
        outer_feat = torch.stack([u, Lu, dt_u, x], dim=1)   # (Nx,4)
        # Layer: [u, εΔu, φ, √ε|∇u|]
        layer_feat = torch.stack([u, epsilon * Lu, phi, (epsilon**0.5) * gradient(u)], dim=1)

        # ---- GRU updates ----
        H_out_new = self.gru_out(outer_feat, H_out)
        H_lay_new = self.gru_lay(layer_feat, H_lay)

        # ---- Regime gating (freeze inappropriate GRU) ----
        rho_exp = rho.unsqueeze(1)   # (Nx,1)
        H_out_new = (1 - rho_exp) * H_out_new + rho_exp * H_out
        H_lay_new = rho_exp * H_lay_new + (1 - rho_exp) * H_lay

        # ---- Blended hidden state for FiLM ----
        H = (1 - rho_exp) * H_out_new + rho_exp * H_lay_new

        # ---- FiLM modulation ----
        film_params = self.film(H)          # (Nx, 2*mlp_width)
        alpha, beta = film_params.chunk(2, dim=1)

        # ---- Scalar MLP (input only u) ----
        h1 = self.tanh(self.mlp1(u.unsqueeze(1)))          # (Nx, mlp_width)
        h2 = self.tanh((1 + alpha) * h1 + beta)            # FiLM modulation
        R = self.Lambda * self.tanh(self.mlp2(h2)).squeeze(1)

        return R, H_out_new, H_lay_new, rho

# ------------------------- Consistency Loss (identifiability) -------------------------
def consistency_loss(u_traj, R_traj, tau=0.02, n_samples=1000):
    """
    Enforces that R(u) is same for all points with similar u.
    u_traj: (Nt, Nx) predicted solution
    R_traj: (Nt, Nx) predicted reaction
    """
    u_flat = u_traj.flatten()
    R_flat = R_traj.flatten()
    # Randomly sample a subset for efficiency
    n_total = len(u_flat)
    idx = torch.randperm(n_total)[:n_samples]
    u_samp = u_flat[idx]
    R_samp = R_flat[idx]
    loss = 0.0
    count = 0
    # For each sampled point, find neighbours within tau and penalise difference
    for i in range(len(u_samp)):
        mask = torch.abs(u_samp - u_samp[i]) < tau
        if mask.sum() > 1:
            loss += ((R_samp[mask] - R_samp[i])**2).mean()
            count += 1
    return loss / max(count, 1)

# ------------------------- Training -------------------------
model = EpsUSGRU(Nx, d_hidden=32, mlp_width=32, Lambda=Lambda).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

epochs = 500
print("Starting training...")
for epoch in range(epochs):
    # ---- Rollout with BPTT ----
    u_pred = [u0.clone()]
    H_out = torch.zeros(Nx, model.d_hidden, device=device)
    H_lay = torch.zeros(Nx, model.d_hidden, device=device)
    # We'll store predictions and reactions for loss
    R_pred_list = []
    dt_u_prev = torch.zeros(Nx, device=device)   # dummy for n=0, will use reference
    for n in range(Nt-1):
        # Time derivative: backward difference from previous predicted step
        if n == 0:
            # Use reference for first step derivative to avoid cold start
            dt_u = (u_ref[1] - u0) / dt
        else:
            dt_u = (u_pred[-1] - u_pred[-2]) / dt
        R_pred, H_out, H_lay, _ = model(u_pred[-1], dt_u, H_out, H_lay)
        u_next = imex_step(u_pred[-1], R_pred)
        u_pred.append(u_next)
        R_pred_list.append(R_pred)
    u_pred = torch.stack(u_pred)          # (Nt, Nx)
    R_pred_traj = torch.stack(R_pred_list)  # (Nt-1, Nx)

    # ---- Loss computation ----
    # Data loss (trajectory matching)
    data_loss = torch.mean((u_pred - u_ref)**2)
    # PDE residual loss (optional, helps regularisation)
    res_loss = 0.0
    for n in range(Nt-1):
        du_dt = (u_pred[n+1] - u_pred[n]) / dt
        Lu = L_op @ u_pred[n]
        # Recompute R at step n (without updating hidden states, use stored)
        # For simplicity, recompute using a temporary forward (no grad to hidden)
        R_n, _, _, _ = model(u_pred[n], du_dt, H_out, H_lay)
        residual = du_dt - epsilon * Lu - R_n
        res_loss += torch.mean(residual**2)
    res_loss = res_loss / (Nt-1)
    # Consistency loss (scalar reaction enforcement)
    cons_loss = consistency_loss(u_pred, R_pred_traj, tau=0.02, n_samples=1000)

    total_loss = (1.0 * data_loss +
                  0.1 * res_loss +
                  10.0 * cons_loss)

    # ---- Backpropagation ----
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4e} | Data: {data_loss.item():.4e} | Cons: {cons_loss.item():.4e}")

# ------------------------- Post‑training: Extract scalar R(u) -------------------------
# Collect all (u, R) pairs from the final predicted trajectory
with torch.no_grad():
    u_pred_final = [u0.clone()]
    H_out = torch.zeros(Nx, model.d_hidden, device=device)
    H_lay = torch.zeros(Nx, model.d_hidden, device=device)
    all_u = []
    all_R = []
    for n in range(Nt-1):
        if n == 0:
            dt_u = (u_ref[1] - u0) / dt
        else:
            dt_u = (u_pred_final[-1] - u_pred_final[-2]) / dt
        R_pred, H_out, H_lay, _ = model(u_pred_final[-1], dt_u, H_out, H_lay)
        u_next = imex_step(u_pred_final[-1], R_pred)
        u_pred_final.append(u_next)
        all_u.append(u_pred_final[-1].cpu().numpy())
        all_R.append(R_pred.cpu().numpy())
    u_all = np.concatenate(all_u)
    R_all = np.concatenate(all_R)

# Bin averaging
bins = np.linspace(0, 1, 100)
bin_centers = (bins[:-1] + bins[1:]) / 2
R_binned = []
for i in range(len(bin_centers)):
    mask = (u_all >= bins[i]) & (u_all < bins[i+1])
    if np.any(mask):
        R_binned.append(np.mean(R_all[mask]))
    else:
        R_binned.append(np.nan)

# Fit a simple MLP to the binned data (optional)
from sklearn.neural_network import MLPRegressor
valid = ~np.isnan(R_binned)
X = bin_centers[valid].reshape(-1,1)
y = R_binned[valid]
mlp = MLPRegressor(hidden_layer_sizes=(16,16), activation='tanh', max_iter=1000)
mlp.fit(X, y)
u_dense = np.linspace(0,1,200)
R_scalar = mlp.predict(u_dense.reshape(-1,1))

# ------------------------- Plotting -------------------------
plt.figure(figsize=(12,5))

# Plot 1: Identified reaction vs true reaction
plt.subplot(1,2,1)
plt.plot(u_dense, R_scalar, 'b-', linewidth=2, label='Identified R(u) (distilled)')
u_true = np.linspace(0,1,200)
R_true = beta * u_true * (1 - u_true)
plt.plot(u_true, R_true, 'r--', linewidth=2, label='True R(u) = 6u(1-u)')
plt.xlabel('u', fontsize=12)
plt.ylabel('R(u)', fontsize=12)
plt.title('Fisher-KPP: Reaction Identification', fontsize=14)
plt.legend()
plt.grid(True)

# Plot 2: Final predicted trajectory vs reference at selected times
plt.subplot(1,2,2)
times = [0, 50, 100, 150]   # indices
for t_idx in times:
    if t_idx < Nt:
        plt.plot(x.cpu(), u_pred_final[t_idx].cpu(), '--', label=f'Pred t={t_idx*dt:.2f}')
        plt.plot(x.cpu(), u_ref[t_idx].cpu(), ':', alpha=0.7)
plt.plot(x.cpu(), u_ref[0].cpu(), 'k-', label='Initial')
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x,t)', fontsize=12)
plt.title('Trajectory Comparison (dashed=pred, dotted=ref)', fontsize=10)
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

plt.tight_layout()
plt.savefig('fisher_epsilon_US_GRU_results.png', dpi=150)
plt.show()

# Optional: print error
err = np.sqrt(np.mean((R_scalar - beta*u_dense*(1-u_dense))**2))
print(f"RMS error in identified reaction: {err:.4f}")