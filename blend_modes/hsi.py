from .hsi_helpers import get_intensity, add_intensity, get_saturation_hsi, set_intensity
from ..blend_modes_enum import BlendModes
from ..helpers import set_saturation
import torch

def hsi_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_intensity(blend_image)
    result = set_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsi_increase_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_intensity(blend_image)
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsi_decrease_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_intensity(blend_image) - 1
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsi_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    intensity = get_intensity(base_image)
    saturation = get_saturation_hsi(blend_image)
    result = set_saturation(base_image, saturation)
    result = set_intensity(result, intensity)
    return result.clamp(0, 1)

def hsi_increase_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsi(base_image)
    blend_image_saturation = get_saturation_hsi(blend_image)

    new_saturation = torch.lerp(base_image_saturation, torch.ones_like(base_image_saturation), blend_image_saturation)

    base_image_lightness = get_intensity(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_intensity(result, base_image_lightness)

    return result.clamp(0, 1)

def hsi_decrease_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsi(base_image)
    blend_image_saturation = get_saturation_hsi(blend_image)

    new_saturation = torch.lerp(torch.zeros_like(base_image_saturation), base_image_saturation, blend_image_saturation)

    base_image_lightness = get_intensity(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_intensity(result, base_image_lightness)

    return result.clamp(0, 1)

def hsi_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_intensity = get_intensity(base_image)
    result = set_intensity(blend_image, base_image_intensity)
    return result.clamp(0, 1)

def hsi_hue(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsi(base_image)
    base_image_lightness = get_intensity(base_image)

    result = blend_image.clone()
    result = set_saturation(result, base_image_saturation)
    result = set_intensity(result, base_image_lightness)
    return result.clamp(0, 1)

hsi_blend_functions = {
    BlendModes.HSI_COLOR: hsi_color,
    BlendModes.HSI_HUE: hsi_hue,
    BlendModes.HSI_SATURATION: hsi_saturation,
    BlendModes.HSI_INTENSITY: hsi_intensity,
    BlendModes.HSI_DECREASE_SATURATION: hsi_decrease_saturation,
    BlendModes.HSI_INCREASE_SATURATION: hsi_increase_saturation,
    BlendModes.HSI_DECREASE_INTENSITY: hsi_decrease_intensity,
    BlendModes.HSI_INCREASE_INTENSITY: hsi_increase_intensity,
}