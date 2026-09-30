# BRIDGE

Reference implementation for:

A Boundary-Layer-Aware Dual-GRU Framework for Reaction Identification and
Solution of Singularly Perturbed Reaction-Diffusion Problems

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

# Configuration

Numerical and training settings live in `config.py` (scalar problems) and in
`SYSTEM_CFGS` inside `systems.py` (coupled systems). The defaults reproduce
the reported runs.

Network. Two GRU encoders of hidden size 32, one over the outer region
and one over the layer, blended by the continuous regime gate; FiLM
conditioning into a 3-layer MLP of width 32 for the scalar problems, and an
additive gate with SiLU for the coupled systems. 10,561 parameters for the
scalar dual-encoder model.

Training. Adam, learning rate 2e-3, weight decay 1e-5, gradient clipping
1.0, 1000 epochs for the scalar problems and 2000 for the coupled systems,
preceded by an MLP pre-initialisation stage. Loss is the sum of data, FD
residual, consistency and equilibrium-anchor terms with weights 1.0, 3.0,
1.0 and 5.0. TBPTT window 15 for the scalar problems and 1 for the coupled
systems

The spatial operator is the fourth-order compact Padé Laplacian in all four
of its roles, reference generation, FD residual extraction, regime
indicator and IMEX rollout, with Crank-Nicolson time stepping.

The true reaction is used only to generate the reference trajectory and to
score the identified law after training. It is never seen by the model
during training.
