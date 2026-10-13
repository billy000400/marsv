"""Attributable verbatim copy of `slerp_rescale` for equivalence tests (path control, PLAN addendum).

Source: https://github.com/MShinkle/activation_plateau_mechanisms/blob/main/scripts/utils.py
Repository HEAD when copied: 4222c3efb77fe7983676cf96b3254dc000e8a0bc
Last commit touching scripts/utils.py: 9e844beda338ff7a036116cdc92ff915524df577 (2025-10-03)
MIT License, Copyright (c) 2025 MShinkle. The function body below is unmodified (lines 104-130 of utils.py).
Note: the angle is acos(abs(dot_product)), not the usual signed-dot SLERP angle.
"""
import torch


def slerp_rescale(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation with norm rescaling.
    Interpolates angle evenly, interpolates magnitude linearly.
    """
    # Get norms for rescaling
    norm_v0 = torch.norm(v0, dim=-1, keepdim=True)
    norm_v1 = torch.norm(v1, dim=-1, keepdim=True)

    # Normalize vectors
    v0_norm = v0 / norm_v0
    v1_norm = v1 / norm_v1

    # Compute angle between normalized vectors
    dot_product = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(torch.abs(dot_product))

    # SLERP computation
    sin_theta = torch.sin(theta)
    weight_v0 = torch.sin((1 - t) * theta) / sin_theta
    weight_v1 = torch.sin(t * theta) / sin_theta
    slerp_result = weight_v0 * v0_norm + weight_v1 * v1_norm

    # Rescale to linearly interpolated norm
    target_norm = (1 - t) * norm_v0 + t * norm_v1
    return slerp_result * target_norm
