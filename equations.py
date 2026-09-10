"""
equations.py  —  ICs, BCs, and R_true for all problems.

R_true is used ONLY for:
  1. Generating the reference trajectory (observed data)
  2. Post-training evaluation of identification quality
It is NEVER used during Stage 1 training.

Problems:
  fisher       — Fisher-KPP  R(u)=6u(1-u)
  allen_cahn   — Allen-Cahn  R(u)=5u(1-u²)
  fhn          — FHN scalar  R(u)=(1/eps)u(1-u)(u-0.25)
  bistable     — Bistable    R(u)=u(1-u²)
  fhn_bvp      — Bonhoeffer-Van der Pol / FHN semi-known system
                  IDENTIFY: R_u(u,v) = -v + u - u³/3
                  KNOWN:    ∂v/∂t = ε(-βv + u)  [explicit ODE, no diffusion]
"""

import numpy as np


# ══════════════════════════════════════════════════════════
#  R_true (evaluation only — never used in training)
# ══════════════════════════════════════════════════════════

def R_fisher(u):       return 6.0 * u * (1.0 - u)
def R_allen_cahn(u):   return 5.0 * u * (1.0 - u**2)
def R_fhn_scalar(u, eps=0.05, a=0.25):
    return (1.0/eps) * u * (1.0-u) * (u-a)
def R_bistable(u):     return u * (1.0 - u**2)

# FHN_SEMI: activator reaction (the unknown to identify)
def R_u_fhn_semi(u, v, a=0.10):
    """True activator reaction: R_u(u,v) = u(1-u)(u-a) - v"""
    return u * (1.0 - u) * (u - a) - v

# FHN_SEMI: recovery equation (KNOWN — used in forward solve, no diffusion)
def R_v_fhn_semi(u, v, cfg):
    """Known recovery ODE: dv/dt = delta*(u - v).  Pure ODE."""
    return cfg["delta_v"] * (u - v)


def get_R_true(problem, u, cfg):
    if problem == "fisher":      return R_fisher(u)
    if problem == "allen_cahn":  return R_allen_cahn(u)
    if problem == "fhn":         return R_fhn_scalar(u, cfg["eps_diff"],
                                                      cfg.get("a_fhn", 0.25))
    if problem == "bistable":    return R_bistable(u)
    if problem == "fhn_semi":
        raise ValueError("fhn_semi: call R_u_fhn_semi(u, v) directly")
    raise ValueError(f"Unknown problem: {problem}")


# ══════════════════════════════════════════════════════════
#  Initial conditions
# ══════════════════════════════════════════════════════════

def get_ics(problem, x, cfg):
    """Return list of IC arrays (one per IC name in cfg['ics'])."""
    eps     = cfg.get("eps_diff", 0.01)
    ic_names = cfg["ics"]
    ics = []

    for name in ic_names:
        if problem == "fisher":
            if name == "original":
                ics.append(1/(1+np.exp(x/np.sqrt(0.01)))**2)
            elif name == "shifted_05":
                ics.append(1/(1+np.exp((x-0.5)/np.sqrt(0.01)))**2)
            elif name == "shifted_08":
                ics.append(1/(1+np.exp((x-0.8)/np.sqrt(0.01)))**2)
            elif name == "shifted_095":
                ics.append(1/(1+np.exp((x-0.95)/np.sqrt(0.01)))**2)

        elif problem == "allen_cahn":
            if name == "x2cosx":
                ics.append(x**2 * np.cos(np.pi*x))
            elif name == "tanh0":
                ics.append(np.tanh(x/0.1))
            elif name == "tanh_neg05":
                ics.append(np.tanh((x+0.5)/0.1))

        elif problem == "fhn":
            if name == "front03":
                ics.append(0.5*(1+np.tanh((x-0.3)/np.sqrt(eps))))
            elif name == "front06":
                ics.append(0.5*(1+np.tanh((x-0.6)/np.sqrt(eps))))

        elif problem == "bistable":
            if name == "tanh03":  ics.append(np.tanh((x-0.3)/np.sqrt(eps)))
            elif name == "tanh06": ics.append(np.tanh((x-0.6)/np.sqrt(eps)))
            elif name == "tanh04": ics.append(np.tanh((x-0.4)/np.sqrt(eps)))
            elif name == "tanh05": ics.append(np.tanh((x-0.5)/np.sqrt(eps)))

        elif problem == "fhn_semi":
            # Returns (u0, v0) tuples.
            # v0 chosen INDEPENDENTLY of u0 for diverse (u,v) coverage.
            # NOT constrained to nullcline — key for 2D identifiability.
            eps = cfg.get("eps_diff", 0.01)
            if name == "front_v0":
                u0 = 0.5*(1 + np.tanh((x - 0.3)/np.sqrt(eps)))
                v0 = np.zeros_like(x)           # v starts at 0
                ics.append((u0, v0))
            elif name == "step_v05":
                u0 = np.where(x < 0.5, 0.9, 0.05)
                v0 = 0.5*np.ones_like(x)        # v starts at 0.5
                ics.append((u0, v0))
            elif name == "bump_v02":
                u0 = np.exp(-50*(x-0.3)**2)*0.9
                v0 = 0.2*np.ones_like(x)        # v starts at 0.2
                ics.append((u0, v0))
            elif name == "front2_v08":
                u0 = 0.5*(1 + np.tanh((x - 0.6)/np.sqrt(eps)))
                v0 = 0.8*np.ones_like(x)        # v starts at 0.8
                ics.append((u0, v0))
    return ics


# ══════════════════════════════════════════════════════════
#  Boundary conditions
# ══════════════════════════════════════════════════════════

def get_bc_values(problem, t, cfg):
    """Returns (g_L, g_R) or (None, None) for periodic/Neumann."""
    if problem == "fisher":
        return (1/(1+np.exp(-5*t))**2,
                1/(1+np.exp(1/np.sqrt(0.01)-5*t))**2)
    elif problem == "allen_cahn":
        return None, None      # periodic
    elif problem == "fhn":
        return 0.0, 1.0
    elif problem == "bistable":
        return -1.0, 1.0
    elif problem == "fhn_semi":
        return None, None      # Neumann — handled in IMEX rhs directly
    raise ValueError(f"Unknown: {problem}")