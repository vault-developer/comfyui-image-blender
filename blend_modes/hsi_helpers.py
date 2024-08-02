import torch

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

def set_intensity(image: torch.Tensor, new_intensity: torch.Tensor) -> torch.Tensor:
    result = add_intensity(image, new_intensity - get_intensity(image))
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
