"""
numerics.py — Mesh construction and finite difference operators.

Shishkin mesh: concentrates nodes in O(sqrt(eps)) boundary layers.
4th-order compact Laplacian: L_op = M^{-1} @ D.
IMEX matrix A: used only in Stage 2 verification.
"""

import numpy as np
import scipy.linalg
import torch


def build_shishkin_mesh(N, x_left, x_right, eps_diff, beta=2.0):
    L      = x_right - x_left
    sigma  = min(0.25, (2*np.sqrt(eps_diff)/beta)*np.log(N))
    n_lay  = N // 4
    n_out  = N // 2
    x1     = x_left  + sigma * L
    x2     = x_right - sigma * L
    seg1   = np.linspace(x_left, x1, n_lay+1)
    seg2   = np.linspace(x1,    x2,  n_out+1)
    seg3   = np.linspace(x2, x_right, n_lay+1)
    x      = np.concatenate([seg1, seg2[1:], seg3[1:]])
    assert len(x) == N+1
    h1 = seg1[1]-seg1[0]; h2 = seg2[1]-seg2[0]
    assert h1 <= h2+1e-12
    return x, sigma


def build_uniform_mesh(N, x_left, x_right):
    return np.linspace(x_left, x_right, N+1), None


def build_mesh(problem, cfg):
    Nx = cfg["Nx"]
    x_left, x_right = cfg["domain"]
    if cfg["mesh_type"] == "uniform":
        x, _ = build_uniform_mesh(Nx, x_left, x_right)
    else:
        x, sigma = build_shishkin_mesh(Nx, x_left, x_right,
                                        cfg["eps_diff"], cfg.get("beta",2.0))
    return x


def build_compact_laplacian(x, periodic=False):
    N  = len(x)-1; Nx = N+1
    M  = np.eye(Nx); D = np.zeros((Nx,Nx))
    for i in range(1,N):
        hm = x[i]-x[i-1]; hp = x[i+1]-x[i]
        S  = hm+hp; Q = hm**2+3*hm*hp+hp**2
        M[i,i-1] =  hp*(hm**2+hm*hp-hp**2)/(S*Q)
        M[i,i]   =  1.0
        M[i,i+1] =  hm*(-hm**2+hm*hp+hp**2)/(S*Q)
        D[i,i-1] =  12*hp/(S*Q)
        D[i,i]   = -12/Q
        D[i,i+1] =  12*hm/(S*Q)
    if periodic:
        h = x[1]-x[0]; S=2*h; Q=4*h**2
        for i in [0,N]:
            l=(i-1)%Nx; r=(i+1)%Nx
            M[i,l]=hp*(hm**2+hm*hp-hp**2)/(S*Q)
            M[i,i]=1
            M[i,r]=hm*(-hm**2+hm*hp+hp**2)/(S*Q)
            D[i,l]=12*hp/(S*Q); D[i,i]=-12/Q; D[i,r]=12*hm/(S*Q)
    return np.linalg.solve(M, D)


def build_imex_matrix(L_op, dt, eps_diff, periodic=False):
    Nx = L_op.shape[0]
    A  = np.eye(Nx) - (dt/2)*eps_diff*L_op
    if not periodic:
        A[0,:]=0; A[0,0]=1; A[-1,:]=0; A[-1,-1]=1
    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv


def build_all_operators(problem, cfg, device):
    """Build mesh, Laplacian, IMEX matrix — all in one call."""
    x       = build_mesh(problem, cfg)
    periodic = cfg["bc_type"] == "periodic"
    L_np    = build_compact_laplacian(x, periodic=periodic)
    A_np, lu, piv = build_imex_matrix(
        L_np, cfg["dt"], cfg["eps_diff"], periodic=periodic)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=device)
    A_t = torch.tensor(A_np, dtype=torch.float32, device=device)
    return x, L_np, A_np, lu, piv, L_t, A_t
