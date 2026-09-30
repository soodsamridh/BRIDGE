import numpy as np


# ======================================================================
#  R_true (evaluation only -- never used in training)
# ======================================================================

def R_fisher(u):       return 6.0 * u * (1.0 - u)
def R_allen_cahn(u):   return 5.0 * u * (1.0 - u**2)


def get_R_true(problem, u, cfg):
    if problem == "fisher":      return R_fisher(u)
    if problem == "allen_cahn":  return R_allen_cahn(u)
    raise ValueError(f"Unknown problem: {problem}")


# ======================================================================
#  Initial conditions
# ======================================================================

def get_ics(problem, x, cfg):
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
    return ics


# ======================================================================
#  Boundary conditions
# ======================================================================

def get_bc_values(problem, t, cfg):
    if problem == "fisher":
        return (1/(1+np.exp(-5*t))**2,
                1/(1+np.exp(1/np.sqrt(0.01)-5*t))**2)
    elif problem == "allen_cahn":
        return None, None      # periodic
    raise ValueError(f"Unknown: {problem}")
