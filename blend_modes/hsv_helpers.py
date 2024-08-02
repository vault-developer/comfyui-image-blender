import torch

def get_value(image: torch.Tensor) -> torch.Tensor:
    max_vals, _ = torch.max(image, dim=-1)
    return max_vals

def set_value(image: torch.Tensor, new_lightness: torch.Tensor) -> torch.Tensor:
    result = add_value(image, new_lightness - get_value(image))
    return result.clamp(0, 1)

def add_value(image: torch.Tensor, new_intensity: torch.Tensor) -> torch.Tensor:
    image = image + new_intensity.unsqueeze(-1)

    intensity = get_value(image)
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