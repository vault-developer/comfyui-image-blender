from ..blend_modes_enum import BlendModes
import torch

def hsi_increase_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
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

    assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
    assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

    blend_image_lightness = get_intensity(blend_image)
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

hsi_blend_functions = {
    #BlendModes.HSI_COLOR: hsi_color,
    # BlendModes.HSI_HUE: hsi_hue,
    # BlendModes.HSI_SATURATION: hsi_saturation,
    # BlendModes.HSI_INTENSITY: hsi_intensity,
    # BlendModes.HSI_DECREASE_SATURATION: hsi_decrease_saturation,
    # BlendModes.HSI_INCREASE_SATURATION: hsi_increase_saturation,
    # BlendModes.HSI_DECREASE_INTENSITY: hsi_decrease_intensity,
    BlendModes.HSI_INCREASE_INTENSITY: hsi_increase_intensity,
}