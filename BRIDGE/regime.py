import torch
import torch.nn.functional as F


def regime_indicator(delta_h_u, eps_diff, gamma=3.0, delta=1e-8):
    l_i    = torch.log(torch.abs(delta_h_u) + delta)
    mu_bar = torch.median(l_i)
    ln_inv = torch.log(torch.tensor(1.0/(eps_diff+1e-12)))
    return torch.sigmoid(gamma * (l_i - mu_bar) / (ln_inv + 1e-8))


def layer_envelope(x_t, x_left, x_right, eps_diff):
    dist = torch.minimum(x_t - x_left, x_right - x_t)
    return torch.exp(-dist / (eps_diff**0.5 + 1e-12))


def build_features(u, delta_h_u, x_t, phi, eps_diff):
    z_out = torch.stack([u, delta_h_u, x_t], dim=-1)
    z_lay = torch.stack([u, eps_diff*delta_h_u, phi], dim=-1)
    return z_out, z_lay
