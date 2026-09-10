"""
fd.py — 4th-order compact Laplacian (Lele 1992) and IMEX matrix A.

L_op is precomputed once as a dense numpy matrix, then converted to
a torch tensor for use in the training rollout.  The LU factorisation
of A = I - (dt/2)*eps_diff*L_op is also precomputed once.
"""

import numpy as np
import torch
import scipy.linalg


# ══════════════════════════════════════════════════════════
#  4th-order compact Laplacian
# ══════════════════════════════════════════════════════════

def build_compact_laplacian(x, periodic=False):
    """
    Build the 4th-order compact (Pade) Laplacian on node array x.

    For non-periodic BCs: rows 0 and N are left as identity (zero
    second derivative at boundaries — overwritten by BC enforcement
    in the time stepper).

    For periodic BCs (Allen-Cahn): circulant wrapping is applied.

    Returns L_op as a numpy array of shape (N+1, N+1).
    """
    N  = len(x) - 1
    Nx = N + 1

    M_mat = np.eye(Nx)
    D_mat = np.zeros((Nx, Nx))

    for i in range(1, N):
        h_m = x[i]   - x[i-1]   # h_minus
        h_p = x[i+1] - x[i]     # h_plus
        S   = h_m + h_p
        Q   = h_m**2 + 3.0*h_m*h_p + h_p**2

        # Mass-matrix row
        M_mat[i, i-1] =  h_p * (h_m**2 + h_m*h_p - h_p**2) / (S * Q)
        M_mat[i, i]   =  1.0
        M_mat[i, i+1] =  h_m * (-h_m**2 + h_m*h_p + h_p**2) / (S * Q)

        # Stiffness-matrix row
        D_mat[i, i-1] =  12.0 * h_p           / (S * Q)
        D_mat[i, i]   = -12.0                  / Q
        D_mat[i, i+1] =  12.0 * h_m           / (S * Q)

    if periodic:
        # Wrap boundary rows
        h_m = x[1]  - x[0]          # left gap (uniform for AC)
        h_p = x[1]  - x[0]
        S   = h_m + h_p
        Q   = h_m**2 + 3.0*h_m*h_p + h_p**2

        for i in [0, N]:
            left  = (i - 1) % Nx
            right = (i + 1) % Nx
            M_mat[i, left]  =  h_p * (h_m**2 + h_m*h_p - h_p**2) / (S * Q)
            M_mat[i, i]     =  1.0
            M_mat[i, right] =  h_m * (-h_m**2 + h_m*h_p + h_p**2) / (S * Q)
            D_mat[i, left]  =  12.0 * h_p / (S * Q)
            D_mat[i, i]     = -12.0        / Q
            D_mat[i, right] =  12.0 * h_m / (S * Q)

    # L_op = M^{-1} @ D
    L_op = np.linalg.solve(M_mat, D_mat)
    return L_op


# ══════════════════════════════════════════════════════════
#  IMEX matrix A and LU factorisation
# ══════════════════════════════════════════════════════════

def build_imex_matrix(L_op, dt, eps_diff, periodic=False):
    """
    A = I - (dt/2)*eps_diff*L_op

    For non-periodic BCs: overwrite rows 0 and N with identity (BCs
    are enforced by setting rhs[0], rhs[-1] directly in the stepper).

    Returns (A_np, lu_piv) where lu_piv is scipy's LU factorisation
    tuple for fast repeated solves.
    """
    Nx    = L_op.shape[0]
    A     = np.eye(Nx) - (dt / 2.0) * eps_diff * L_op

    if not periodic:
        # Boundary rows → identity  (BCs enforced via rhs)
        A[0,  :] = 0.0;  A[0,  0]  = 1.0
        A[-1, :] = 0.0;  A[-1, -1] = 1.0

    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv


# ══════════════════════════════════════════════════════════
#  Torch versions for differentiable rollout
# ══════════════════════════════════════════════════════════

def numpy_to_torch(L_op_np, A_np, device):
    """Convert precomputed numpy matrices to torch tensors."""
    L_op = torch.tensor(L_op_np, dtype=torch.float32, device=device)
    A    = torch.tensor(A_np,    dtype=torch.float32, device=device)
    return L_op, A


# ══════════════════════════════════════════════════════════
#  Full build pipeline
# ══════════════════════════════════════════════════════════

def build_fd_operators(x, cfg, device):
    """
    Convenience function: build L_op, A, LU in one call.

    Returns:
        L_op_np  : numpy (Nx, Nx)
        A_np     : numpy (Nx, Nx)
        lu, piv  : scipy LU factors for numpy solves
        L_op_t   : torch tensor (Nx, Nx)
        A_t      : torch tensor (Nx, Nx)
    """
    periodic = cfg.get("bc_left", "dirichlet") == "periodic"
    dt       = cfg["dt"]
    eps_diff = cfg["eps_diff"]

    L_op_np        = build_compact_laplacian(x, periodic=periodic)
    A_np, lu, piv  = build_imex_matrix(L_op_np, dt, eps_diff, periodic=periodic)
    L_op_t, A_t    = numpy_to_torch(L_op_np, A_np, device)

    return L_op_np, A_np, lu, piv, L_op_t, A_t