import torch
from ..helpers import set_ilvy, add_ilvy

def get_luminosity(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    luminosity = 0.299 * r + 0.587 * g + 0.114 * b
    return luminosity

def set_luminosity(image: torch.Tensor, new_luminosity: torch.Tensor) -> torch.Tensor:
    result = set_ilvy(image, new_luminosity, get_luminosity)
    return result.clamp(0, 1)

def add_luminosity(image: torch.Tensor, new_luminosity: torch.Tensor) -> torch.Tensor:
    result = add_ilvy(image, new_luminosity, get_luminosity)
    return result.clamp(0, 1)

def get_saturation_hsy(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1, keepdim=True)
    min_vals, _ = torch.min(image, dim=-1, keepdim=True)

    saturation = max_vals - min_vals

    return saturation.squeeze(-1).clamp(0, 1)