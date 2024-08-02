import torch
from ..helpers import set_ilvy, add_ilvy

def get_value(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1)
    return max_vals

def set_value(image: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
    result = set_ilvy(image, new_value, get_value)
    return result.clamp(0, 1)

def add_value(image: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
    result = add_ilvy(image, new_value, get_value)
    return result.clamp(0, 1)

def get_saturation_hsv(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1, keepdim=True)
    min_vals, _ = torch.min(image, dim=-1, keepdim=True)

    chroma = max_vals - min_vals

    saturation = torch.where(
        chroma > 1e-8,
        chroma / max_vals,
        torch.zeros_like(chroma)
    )

    return saturation.squeeze(-1).clamp(0, 1)