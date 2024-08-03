from .hsl_helpers import get_lightness, set_lightness, get_saturation_hsl, add_lightness
from ..blend_modes_enum import BlendModes
from ..helpers import set_saturation
import torch

def hsl_lightness(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_lightness(blend_image)
    result = set_lightness(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsl_increase_lightness(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_lightness(blend_image)
    result = add_lightness(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsl_decrease_lightness(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_lightness(blend_image) - 1
    result = add_lightness(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsl_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    intensity = get_lightness(base_image)
    saturation = get_saturation_hsl(blend_image)
    result = set_saturation(base_image, saturation)
    result = set_lightness(result, intensity)
    return result.clamp(0, 1)

def hsl_increase_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsl(base_image)
    blend_image_saturation = get_saturation_hsl(blend_image)

    new_saturation = torch.lerp(base_image_saturation, torch.ones_like(base_image_saturation), blend_image_saturation)

    base_image_lightness = get_lightness(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_lightness(result, base_image_lightness)

    return result.clamp(0, 1)

def hsl_decrease_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsl(base_image)
    blend_image_saturation = get_saturation_hsl(blend_image)

    new_saturation = torch.lerp(torch.zeros_like(base_image_saturation), base_image_saturation, blend_image_saturation)

    base_image_lightness = get_lightness(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_lightness(result, base_image_lightness)

    return result.clamp(0, 1)

def hsl_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_intensity = get_lightness(base_image)
    result = set_lightness(blend_image, base_image_intensity)
    return result.clamp(0, 1)

def hsl_hue(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsl(base_image)
    base_image_lightness = get_lightness(base_image)

    result = blend_image.clone()
    result = set_saturation(result, base_image_saturation)
    result = set_lightness(result, base_image_lightness)
    return result.clamp(0, 1)

hsl_blend_functions = {
    BlendModes.HSL_COLOR: hsl_color,
    BlendModes.HSL_HUE: hsl_hue,
    BlendModes.HSL_SATURATION: hsl_saturation,
    BlendModes.HSL_LIGHTNESS: hsl_lightness,
    BlendModes.HSL_DECREASE_SATURATION: hsl_decrease_saturation,
    BlendModes.HSL_INCREASE_SATURATION: hsl_increase_saturation,
    BlendModes.HSL_DECREASE_LIGHTNESS: hsl_decrease_lightness,
    BlendModes.HSL_INCREASE_LIGHTNESS: hsl_increase_lightness,
}