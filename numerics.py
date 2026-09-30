import numpy as np
import scipy.linalg
import torch


# ======================================================================
#  MESH
# ======================================================================

def shishkin_sigma(N, eps, beta=2.0, scale="sqrt"):
    s = np.sqrt(eps) if scale == "sqrt" else eps
    return float(min(0.25, (2.0 * s / beta) * np.log(N)))


def build_shishkin_mesh(N, x_left, x_right, eps, beta=2.0, scale="sqrt"):
    L = x_right - x_left
    sigma = shishkin_sigma(N, eps, beta, scale)
    n_lay = N // 4
    n_out = N - 2 * n_lay
    x1 = x_left + sigma * L
    x2 = x_right - sigma * L
    seg1 = np.linspace(x_left, x1, n_lay + 1)
    seg2 = np.linspace(x1, x2, n_out + 1)
    seg3 = np.linspace(x2, x_right, n_lay + 1)
    x = np.concatenate([seg1, seg2[1:], seg3[1:]])
    assert len(x) == N + 1, (len(x), N + 1)
    assert (x[1] - x[0]) <= (seg2[1] - seg2[0]) + 1e-12
    return x, sigma


def build_uniform_mesh(N, x_left, x_right):
    return np.linspace(x_left, x_right, N + 1), None


def build_mesh(problem, cfg):
    Nx = cfg["Nx"]
    x_left, x_right = cfg["domain"]
    if cfg["mesh_type"] == "uniform":
        x, _ = build_uniform_mesh(Nx, x_left, x_right)
    else:
        x, sigma = build_shishkin_mesh(Nx, x_left, x_right,
                                        cfg["eps_diff"], cfg.get("beta",2.0))
    return x



# ======================================================================
#  COMPACT LAPLACIAN
# ======================================================================

def _interior_rows(M, D, x, N):
    for i in range(1, N):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        S = hm + hp
        Q = hm ** 2 + 3 * hm * hp + hp ** 2
        M[i, i - 1] = hp * (hm ** 2 + hm * hp - hp ** 2) / (S * Q)
        M[i, i] = 1.0
        M[i, i + 1] = hm * (-hm ** 2 + hm * hp + hp ** 2) / (S * Q)
        D[i, i - 1] = 12 * hp / (S * Q)
        D[i, i] = -12 / Q
        D[i, i + 1] = 12 * hm / (S * Q)


def _periodic_rows(M, D, x, N):
    hm = x[N] - x[N - 1]          # spacing to the left of node 0
    hp = x[1] - x[0]              # spacing to the right of node 0
    S = hm + hp
    Q = hm ** 2 + 3 * hm * hp + hp ** 2

    M[0, N - 1] = hp * (hm ** 2 + hm * hp - hp ** 2) / (S * Q)
    M[0, 0] = 1.0
    M[0, 1] = hm * (-hm ** 2 + hm * hp + hp ** 2) / (S * Q)
    D[0, N - 1] = 12 * hp / (S * Q)
    D[0, 0] = -12 / Q
    D[0, 1] = 12 * hm / (S * Q)

    M[N, :] = 0.0
    M[N, N] = 1.0
    M[N, 0] = -1.0
    D[N, :] = 0.0


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


def build_compact_laplacian_corrected(x, periodic=False):
    N = len(x) - 1
    Nx = N + 1
    M = np.eye(Nx)
    D = np.zeros((Nx, Nx))
    _interior_rows(M, D, x, N)
    if periodic:
        _periodic_rows(M, D, x, N)
    return np.linalg.solve(M, D)


def build_imex_matrix(L_op, dt, diff, periodic=False):
    Nx = L_op.shape[0]
    A = np.eye(Nx) - (dt / 2) * diff * L_op
    if not periodic:
        A[0, :] = 0; A[0, 0] = 1
        A[-1, :] = 0; A[-1, -1] = 1
    lu, piv = scipy.linalg.lu_factor(A)
    return A, lu, piv


def build_all_operators(problem, cfg, device):
    x       = build_mesh(problem, cfg)
    periodic = cfg["bc_type"] == "periodic"
    L_np    = build_compact_laplacian(x, periodic=periodic)
    A_np, lu, piv = build_imex_matrix(
        L_np, cfg["dt"], cfg["eps_diff"], periodic=periodic)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=device)
    A_t = torch.tensor(A_np, dtype=torch.float32, device=device)
    return x, L_np, A_np, lu, piv, L_t, A_t



# ======================================================================
#  REGION MASKS AND RESOLUTION REPORTING
# ======================================================================

def layer_mask_boundary(x, sigma, domain):
    xl, xr = domain
    L = xr - xl
    d = np.minimum(x - xl, xr - x)
    return d <= sigma * L + 1e-12


def layer_mask_curvature(u_ref, L_np, pct=90.0):
    lap = np.abs(u_ref @ L_np.T)                       # (Nt+1, Nx+1)
    lap[:, 0] = 0.0; lap[:, -1] = 0.0
    thr = np.percentile(lap[:, 1:-1], pct, axis=1, keepdims=True)
    mask = lap >= thr
    mask[:, 0] = False; mask[:, -1] = False
    return mask


def ac_nx_for_resolution(eps, nodes=9.0, domain=(-1.0, 1.0), scale=1.0):
    w = 2.0 * np.arctanh(0.8) * np.sqrt(2.0 * eps / scale)
    span = domain[1] - domain[0]
    return int(np.ceil(span / (w / nodes)))


def nx_for_width(width, nodes, domain):
    span = float(domain[1] - domain[0])
    if not np.isfinite(width) or width <= 0:
        return 128
    return int(np.ceil(span / (width / float(nodes))))


def fhn_front_width(eps):
    return 2.0 * np.sqrt(2.0) * float(eps)


def measured_layer_width(u_ref, x):
    d = np.diff(x)
    g = np.abs(np.gradient(u_ref, x, axis=1))
    gmax = float(g.max())
    rng = float(u_ref.max() - u_ref.min())
    if gmax <= 0 or rng <= 0 or not np.isfinite(gmax):
        n = float("nan")
        return n, n, n, n
    w = rng / gmax
    j = int(np.unravel_index(int(np.argmax(g)), g.shape)[1])
    h_local = float(d[min(max(j - 1, 0), len(d) - 1)])
    if j < len(d):
        h_local = float(min(h_local, d[j]))
    return w, w / h_local, float(x[j]), h_local


def layer_on_refined_region(x_peak, sigma, domain, tol=1.0):
    if sigma is None:
        return True
    xl, xr = domain
    L = xr - xl
    dist = min(x_peak - xl, xr - x_peak)
    return bool(dist <= tol * sigma * L + 1e-12)


def boundary_skip(tol=1e-3, weight=0.1):
    k = 1
    while weight ** k > tol and k < 10:
        k += 1
    return k


def layer_budget_report(u_ref, L_np, mask, skip):
    m = (np.broadcast_to(mask, u_ref.shape).copy()
         if np.ndim(mask) == 1 else mask[:u_ref.shape[0]].copy())
    drop = np.zeros(u_ref.shape[1], dtype=bool)
    if skip > 0:
        drop[:skip] = True; drop[-skip:] = True
    lay = m
    lay_drop = m & drop
    lap2 = (u_ref @ L_np.T) ** 2
    u2 = u_ref ** 2

    def frac(field):
        d = field[lay].sum()
        return float(field[lay_drop].sum() / d) if d > 0 else 0.0

    return {"nodes": frac(np.ones_like(u2)),
            "u2": frac(u2), "curvature": frac(lap2),
            "layer_nodes": int(lay.sum()),
            "dropped_layer_nodes": int(lay_drop.sum())}


def interface_width(problem, eps, scale=1.0):
    k = 2.0 * np.arctanh(0.8)
    if problem == "fisher":
        return k * np.sqrt(eps / scale)
    if problem == "allen_cahn":
        return k * np.sqrt(2.0 * eps / scale)
    raise ValueError(problem)



# ======================================================================
#  SELF-TEST
# ======================================================================

def _order(errs, Ns):
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(Ns[i + 1] / Ns[i]))
            for i in range(len(errs) - 1)]


def self_test():
    print("=" * 70)
    print("  encoder_numerics self-test")
    print("=" * 70)

    print("\n[1] Non-periodic interior, Shishkin mesh, closure-consistent")
    print("    u = sin(pi x) on [0,1]  (u'' vanishes at both ends).")
    Ns = [64, 128, 256]; errs = []
    for N in Ns:
        x, _ = build_shishkin_mesh(N, 0.0, 1.0, 1e-2, 2.0)
        L = build_compact_laplacian(x, periodic=False)
        u = np.sin(np.pi * x); ex = -np.pi ** 2 * np.sin(np.pi * x)
        errs.append(np.abs((L @ u - ex)[1:-1]).max())
        print(f"      N={N:<5} max err = {errs[-1]:.3e}")
    print(f"      observed order: "
          + ", ".join(f"{o:.2f}" for o in _order(errs, Ns)))

    print("\n[1b] The endpoint closure, quantified (endpoint closure).")
    print("     u = exp(sin 3x), u''(0) = 9. The closure forces u''_0 = 0,")
    print("     so node 1 inherits |M[1,0]|*|u''(0)| and it does NOT")
    print("     converge. This is a modelling choice, not a bug.")
    for N in (64, 256):
        x = np.linspace(0, 1, N + 1)
        L = build_compact_laplacian(x, periodic=False)
        u = np.exp(np.sin(3 * x))
        ex = (9 * np.cos(3 * x) ** 2 - 9 * np.sin(3 * x)) * np.exp(np.sin(3 * x))
        e = np.abs(L @ u - ex)
        print(f"      N={N:<5} node1={e[1]:.4f}  node2={e[2]:.4f}  "
              f"node3={e[3]:.4f}  node5={e[5]:.2e}")
    print(f"      boundary_skip() = {boundary_skip()} nodes dropped per end")

    print("\n[2] Periodic, uniform mesh.  u = sin(pi x) on [-1,1].")
    print("    ALL nodes, including the wrap-around rows.")
    Ns = [64, 128, 256, 512]; errs = []
    for N in Ns:
        x = np.linspace(-1, 1, N + 1)
        L = build_compact_laplacian(x, periodic=True)
        u = np.sin(np.pi * x); ex = -np.pi ** 2 * np.sin(np.pi * x)
        errs.append(np.abs(L @ u - ex).max())
        print(f"      N={N:<5} max err = {errs[-1]:.3e}")
    print(f"      observed order: "
          + ", ".join(f"{o:.2f}" for o in _order(errs, Ns)))

    print("\n[3] Periodic closure consistency: row N must equal row 0.")
    x = np.linspace(-1, 1, 129)
    L = build_compact_laplacian(x, periodic=True)
    print(f"      max |L[N,:] - L[0,:]| = {np.abs(L[-1] - L[0]).max():.3e}")

    print("\n[4] Periodic operator on a constant: L_h(1) must be 0.")
    print(f"      max |L @ ones| = {np.abs(L @ np.ones(129)).max():.3e}")

    print("\n[5] Layer resolution (manuscript reaction laws, scale = 1)")
    print(f"      {'problem':<12}{'eps':>9}{'mesh':>10}{'Nx':>7}"
          f"{'width':>10}{'fine h':>11}{'nodes':>8}")
    for eps in (1.0, 1e-2, 1e-3, 1e-4, 1e-5):
        x, sig = build_shishkin_mesh(128, 0.0, 1.0, eps, 2.0)
        w = interface_width("fisher", eps, 1.0); h = x[1] - x[0]
        print(f"      {'fisher':<12}{eps:>9.0e}{'shishkin':>10}{128:>7}"
              f"{w:>10.5f}{h:>11.6f}{w/h:>8.1f}")
    for eps in (1e-2, 1e-3, 1e-4):
        Nx = ac_nx_for_resolution(eps, scale=1.0)
        w = interface_width("allen_cahn", eps, 1.0); h = 2.0 / Nx
        print(f"      {'allen_cahn':<12}{eps:>9.0e}{'uniform':>10}{Nx:>7}"
              f"{w:>10.5f}{h:>11.6f}{w/h:>8.1f}")
    print("\n      For reference, Allen-Cahn at the configured Nx=128:")
    for eps in (1e-2, 1e-3, 1e-4):
        w = interface_width("allen_cahn", eps, 1.0); h = 2.0 / 128
        flag = "  <-- UNRESOLVED" if w / h < 2 else ""
        print(f"      {'allen_cahn':<12}{eps:>9.0e}{'uniform':>10}{128:>7}"
              f"{w:>10.5f}{h:>11.6f}{w/h:>8.1f}{flag}")

    print("\n  Expected: orders near 4 in [1] and [2]; [3] and [4] at")
    print("  round-off. If [2] is not fourth order the patch did not take.")
    print("=" * 70)



if __name__ == "__main__":
    self_test()
