"""
regime.py — Regime indicator ρ and feature vectors z_out / z_lay.

The regime indicator is derived from matched asymptotic expansion:
  In the outer region: |L@u| ~ O(1)
  In the boundary layer: |L@u| ~ O(1/ε)

Taking log-ratio and normalising by ln(1/ε) gives ρ ∈ (0,1).
"""

import torch
import torch.nn.functional as F


def regime_indicator(delta_h_u, eps_diff, gamma=3.0, delta=1e-8):
    """
    ρ_i = sigmoid(γ · (ln|Δu_i| - median(ln|Δu|)) / ln(1/ε))

    In layer:       ρ ≈ 0.95
    In outer region: ρ ≈ 0.05
    """
    l_i    = torch.log(torch.abs(delta_h_u) + delta)
    mu_bar = torch.median(l_i)
    ln_inv = torch.log(torch.tensor(1.0/(eps_diff+1e-12)))
    return torch.sigmoid(gamma * (l_i - mu_bar) / (ln_inv + 1e-8))


def layer_envelope(x_t, x_left, x_right, eps_diff):
    """φ_i = exp(-dist(x_i, boundary)/sqrt(ε))"""
    dist = torch.minimum(x_t - x_left, x_right - x_t)
    return torch.exp(-dist / (eps_diff**0.5 + 1e-12))


def build_features(u, delta_h_u, x_t, phi, eps_diff):
    """
    z_out : [u, Δu, x]         O(1) in outer region  (dim 3)
    z_lay : [u, ε·Δu, φ]       O(1) in boundary layer (dim 3)

    Note: ∂_t u is excluded in Stage 1 (no time rollout).
    """
    z_out = torch.stack([u, delta_h_u, x_t], dim=-1)
    z_lay = torch.stack([u, eps_diff*delta_h_u, phi], dim=-1)
    return z_out, z_lay
