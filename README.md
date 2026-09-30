# BRIDGE

Reference implementation for:

**A Boundary-Layer-Aware Dual-GRU Framework for Reaction Identification and
Solution of Singularly Perturbed Reaction-Diffusion Problems**

Samridh Sood, Subit Kumar Jain

Department of Mathematics and Scientific Computing, National Institute of
Technology Hamirpur, India



# Abstract
Inverse identification of unknown reaction laws in singularly perturbed reaction-diffusion systems is fundamentally challenging due to the coexistence of sharp boundary layers, multiscale dynamics, and severe gradient imbalances. Conventional physics-informed and recurrent learning approaches treat all spatial nodes identically, allowing the O(1/ε) activations in the boundary layer to dominate parameter updates and obscure the informative outer-region reaction signal. This paper presents BRIDGE (boundary-layer reaction identification via dual-GRU encoders), a regime-aware dual-GRU framework that integrates high-order numerical discretization with regime-specific recurrent learning to identify the reaction law and predict the resulting solution dynamics. The method employs a fourth-order compact finite-difference scheme on a Shishkin mesh with an unconditionally stable IMEX time integration to generate ε-uniform reference trajectories. A  finite-difference residual extraction with bin-median aggregation provides direct pointwise supervision, while a continuous regime-gating mechanism routes boundary-layer and outer-region features through specialised GRU cells, preventing the gradient dominance that defeats single-encoder architectures. The identified reaction law
is embedded back into the governing equation via a differentiable implicit-explicit (IMEX) rollout, ensuring consistency between reaction recovery and trajectory evolution. The framework is validated on four benchmark reaction-diffusion problems spanning scalar equations and coupled systems. Comparative, ablation, and sensitivity studies demonstrate that Bridge consistently achieves accurate reaction identification, lower trajectory prediction errors, and improved robustness.

# Requirements
Python 3.9 or newer. 
numpy
scipy
torch
matplotlib


# Problems

Four problems are considered. Two are scalar inverse problems, where the
whole reaction term R(u) is identified; two are coupled systems, where the
full nonlinearity R(u,v) of the u`-equation is identified while the
`v`-equation is treated as known.

| key | problem | reaction identified |
|---|---|---|
| `fisher` | Fisher-KPP | `R(u) = 6u(1-u)` |
| `allen_cahn` | Allen-Cahn | `R(u) = 5u(1-u²)` |
| `fhn_partial` | FitzHugh-Nagumo | `R(u,v) = (1/ε)u(u-a)(1-u) - v` |
| `predator_prey` | Predator-Prey, Holling type II | `R(u,v) = u(1-u) - αuv/(β+u)` |

## Running

```bash
python main.py --problem fisher
python main.py --problem allen_cahn
python main.py --problem fhn_partial
python main.py --problem predator_prey
```

Coupled systems can also be run through their own runner, which is what
`main.py` dispatches to:

```bash
python systems.py --system fhn_partial
python systems.py --system predator_prey
```
