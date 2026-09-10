"""
mesh.py — Shishkin mesh (sigma ≤ 0.25) and uniform mesh.
"""

import numpy as np


def shishkin_mesh(N, x_left, x_right, eps_diff, beta=2.0):
    """
    Build a 1-D Shishkin mesh on [x_left, x_right] with N intervals
    (N+1 nodes).

    Transition parameter:
        sigma = min(0.25, (2*sqrt(eps_diff)/beta) * ln(N))

    The cap at 0.25 is critical: it guarantees h1 <= h2 (layer step
    finer than outer step).

    Layout (for a domain with layers at BOTH ends):
        [x_left, x_left+sigma*L] — fine   (N/4 intervals)
        [x_left+sigma*L, x_right-sigma*L] — coarse (N/2 intervals)
        [x_right-sigma*L, x_right] — fine  (N/4 intervals)
    where L = x_right - x_left.
    """
    L = x_right - x_left
    sigma = min(0.25, (2.0 * np.sqrt(eps_diff) / beta) * np.log(N))

    n_layer = N // 4       # nodes in each layer sub-interval
    n_outer = N // 2       # nodes in outer sub-interval

    x1 = x_left + sigma * L
    x2 = x_right - sigma * L

    seg1 = np.linspace(x_left, x1, n_layer + 1)
    seg2 = np.linspace(x1, x2,   n_outer + 1)
    seg3 = np.linspace(x2, x_right, n_layer + 1)

    x = np.concatenate([seg1, seg2[1:], seg3[1:]])
    assert len(x) == N + 1, f"Expected {N+1} nodes, got {len(x)}"

    h1 = seg1[1] - seg1[0]   # layer step
    h2 = seg2[1] - seg2[0]   # outer step
    assert h1 <= h2 + 1e-14, \
        f"Shishkin mesh failure: h1={h1:.4e} > h2={h2:.4e}. Increase sigma cap."

    return x, sigma, h1, h2


def uniform_mesh(N, x_left, x_right):
    """
    Uniform mesh — used for Allen-Cahn (periodic BCs, phase separation
    across whole domain).
    """
    x = np.linspace(x_left, x_right, N + 1)
    h = x[1] - x[0]
    return x, h


def build_mesh(problem, cfg):
    """
    Returns node array x (shape [Nx+1]) and a dict of mesh info.
    """
    Nx        = cfg["Nx"]
    x_left, x_right = cfg["domain"]
    eps_diff  = cfg["eps_diff"]
    mesh_type = cfg.get("mesh_type", "shishkin")

    if mesh_type == "uniform":
        x, h = uniform_mesh(Nx, x_left, x_right)
        info = {"type": "uniform", "h": h}
    else:  # shishkin
        beta = cfg.get("beta", 2.0)
        x, sigma, h1, h2 = shishkin_mesh(Nx, x_left, x_right, eps_diff, beta)
        info = {"type": "shishkin", "sigma": sigma, "h1": h1, "h2": h2}

    return x, info