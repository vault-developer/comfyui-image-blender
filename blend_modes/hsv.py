from .hsv_helpers import get_value, set_value, get_saturation_hsv, add_value
from ..blend_modes_enum import BlendModes
from ..helpers import set_saturation
import torch

def hsv_value(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_value(blend_image)
    result = set_value(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsv_increase_value(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_value(blend_image)
    result = add_value(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsv_decrease_value(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_lightness = get_value(blend_image) - 1
    result = add_value(base_image, blend_image_lightness)
    return result.clamp(0, 1)

def hsv_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    intensity = get_value(base_image)
    saturation = get_saturation_hsv(blend_image)
    result = set_saturation(base_image, saturation)
    result = set_value(result, intensity)
    return result.clamp(0, 1)

def hsv_increase_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsv(base_image)
    blend_image_saturation = get_saturation_hsv(blend_image)

    new_saturation = torch.lerp(base_image_saturation, torch.ones_like(base_image_saturation), blend_image_saturation)

    base_image_lightness = get_value(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_value(result, base_image_lightness)

    return result.clamp(0, 1)

def hsv_decrease_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsv(base_image)
    blend_image_saturation = get_saturation_hsv(blend_image)

    new_saturation = torch.lerp(torch.zeros_like(base_image_saturation), base_image_saturation, blend_image_saturation)

    base_image_lightness = get_value(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_value(result, base_image_lightness)

    return result.clamp(0, 1)

def hsv_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_intensity = get_value(base_image)
    result = set_value(blend_image, base_image_intensity)
    return result.clamp(0, 1)

def hsv_hue(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsv(base_image)
    base_image_lightness = get_value(base_image)

    result = blend_image.clone()
    result = set_saturation(result, base_image_saturation)
    result = set_value(result, base_image_lightness)
    return result.clamp(0, 1)

hsv_blend_functions = {
    BlendModes.HSV_COLOR: hsv_color,
    BlendModes.HSV_HUE: hsv_hue,
    BlendModes.HSV_SATURATION: hsv_saturation,
    BlendModes.HSV_VALUE: hsv_value,
    BlendModes.HSV_DECREASE_SATURATION: hsv_decrease_saturation,
    BlendModes.HSV_INCREASE_SATURATION: hsv_increase_saturation,
    BlendModes.HSV_DECREASE_VALUE: hsv_decrease_value,
    BlendModes.HSV_INCREASE_VALUE: hsv_increase_value,
}