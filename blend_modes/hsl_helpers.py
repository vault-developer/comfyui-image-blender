import torch
from ..helpers import set_ilvy, add_ilvy

def get_lightness(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1)
    min_vals, _ = torch.min(image, dim=-1)
    lightness = (max_vals + min_vals) * 0.5
    return lightness

def set_lightness(image: torch.Tensor, new_lightness: torch.Tensor) -> torch.Tensor:
    result = set_ilvy(image, new_lightness, get_lightness)
    return result.clamp(0, 1)

def add_lightness(image: torch.Tensor, new_lightness: torch.Tensor) -> torch.Tensor:
    result = add_ilvy(image, new_lightness, get_lightness)
    return result.clamp(0, 1)

def get_saturation_hsl(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1, keepdim=True)
    min_vals, _ = torch.min(image, dim=-1, keepdim=True)

    chroma = max_vals - min_vals
    lightness = get_lightness(image)
    divisor = 1.0 - torch.abs(2.0 * lightness.unsqueeze(-1) - 1.0)

    saturation = torch.where(
        divisor > 1e-8,
        chroma / divisor,
        torch.zeros_like(chroma)
    )

    return saturation.squeeze(-1).clamp(0, 1)