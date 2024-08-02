import torch
from ..helpers import set_ilvy, add_ilvy

def get_intensity(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(image, dim=-1)

def set_intensity(image: torch.Tensor, new_intensity: torch.Tensor) -> torch.Tensor:
    result = set_ilvy(image, new_intensity, get_intensity)
    return result.clamp(0, 1)

def add_intensity(image: torch.Tensor, new_intensity: torch.Tensor) -> torch.Tensor:
    result = add_ilvy(image, new_intensity, get_intensity)
    return result.clamp(0, 1)

def get_saturation_hsi(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1, keepdim=True)
    min_vals, _ = torch.min(image, dim=-1, keepdim=True)

    chroma = max_vals - min_vals
    intensity = get_intensity(image)

    saturation = torch.where(
        chroma > 1e-8,
        1.0 - min_vals / intensity.unsqueeze(-1),
        torch.zeros_like(chroma)
    )

    return saturation.squeeze(-1).clamp(0, 1)
