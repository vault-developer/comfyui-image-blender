import torch
from typing import Callable

def replace_zeros(tensor: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    return torch.where(tensor == 0, torch.tensor(epsilon, device=tensor.device), tensor)

def float_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 256).to(torch.uint8)

def uint8_to_float(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor/256, 0.0, 1.0)

def set_saturation(image: torch.Tensor, new_saturation: torch.Tensor) -> torch.Tensor:
    result = image.clone()

    min_idx = torch.argmin(result, dim=-1, keepdim=True)
    max_idx = torch.argmax(result, dim=-1, keepdim=True)
    mid_idx = torch.where(min_idx == max_idx, min_idx, 3 - min_idx - max_idx)

    min_val = torch.gather(result, -1, min_idx)
    mid_val = torch.gather(result, -1, mid_idx)
    max_val = torch.gather(result, -1, max_idx)

    result.scatter_(-1, min_idx, torch.zeros_like(min_val))
    result.scatter_(-1, mid_idx, ((mid_val - min_val) * new_saturation.unsqueeze(-1)) / replace_zeros(max_val - min_val))
    result.scatter_(-1, max_idx, new_saturation.unsqueeze(-1))

    # manage zeros
    result = torch.where(
        (max_val - min_val) > 0,
        result,
        torch.zeros_like(result)
    )

    return result.clamp(0, 1)

def set_ilvy(image: torch.Tensor, new_ilvy: torch.Tensor, get_ilvy: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    result = add_ilvy(image, new_ilvy - get_ilvy(image), get_ilvy)
    return result.clamp(0, 1)

def add_ilvy(image: torch.Tensor, new_intensity: torch.Tensor, get_ilvy: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    image = image + new_intensity.unsqueeze(-1)

    intensity = get_ilvy(image)
    min = torch.min(image, dim=-1).values
    max = torch.max(image, dim=-1).values

    # adjust overflows
    mask_min = min < 0.0
    iln = torch.where(mask_min, 1.0 / (intensity - min + 1e-8), torch.zeros_like(intensity))
    image = torch.where(mask_min.unsqueeze(-1), intensity.unsqueeze(-1) + ((image - intensity.unsqueeze(-1)) * intensity.unsqueeze(-1)) * iln.unsqueeze(-1), image)

    mask_max = (max > 1.0) & ((max - intensity) > torch.finfo(max.dtype).eps)
    il = torch.where(mask_max, 1.0 - intensity, torch.zeros_like(intensity))
    ixl = torch.where(mask_max, 1.0 / (max - intensity + 1e-8), torch.zeros_like(intensity))
    image = torch.where(mask_max.unsqueeze(-1), intensity.unsqueeze(-1) + ((image - intensity.unsqueeze(-1)) * il.unsqueeze(-1)) * ixl.unsqueeze(-1), image)

    return image.clamp(0, 1)