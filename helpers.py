import torch

def replace_zeros(tensor: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    return torch.where(tensor == 0, torch.tensor(epsilon, device=tensor.device), tensor)

def float_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 256).to(torch.uint8)

def uint8_to_float(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor/256, 0.0, 1.0)

def get_intensity(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(image, dim=-1)

def add_intensity(image: torch.Tensor, new_intensity: torch.Tensor) -> torch.Tensor:
    image = image + new_intensity.unsqueeze(-1)

    intensity = get_intensity(image)
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