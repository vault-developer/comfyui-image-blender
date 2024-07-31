from ..helpers import get_intensity, add_intensity
from ..blend_modes_enum import BlendModes
import torch

def hsi_increase_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
    assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

    blend_image_lightness = get_intensity(blend_image)
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsi_decrease_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
    assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

    blend_image_lightness = get_intensity(blend_image) - 1
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

hsi_blend_functions = {
    #BlendModes.HSI_COLOR: hsi_color,
    # BlendModes.HSI_HUE: hsi_hue,
    # BlendModes.HSI_SATURATION: hsi_saturation,
    # BlendModes.HSI_INTENSITY: hsi_intensity,
    # BlendModes.HSI_DECREASE_SATURATION: hsi_decrease_saturation,
    # BlendModes.HSI_INCREASE_SATURATION: hsi_increase_saturation,
    BlendModes.HSI_DECREASE_INTENSITY: hsi_decrease_intensity,
    BlendModes.HSI_INCREASE_INTENSITY: hsi_increase_intensity,
}