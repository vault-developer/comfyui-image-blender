from ..helpers import get_lightness, set_lightness, get_saturation_hsl, set_saturation_hsl
from ..blend_modes_enum import BlendModes
import torch

def hsl_lightness(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
    assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

    blend_image_lightness = get_lightness(blend_image)
    result = set_lightness(base_image, blend_image_lightness)
    return result.clamp(0, 1)

# def hsi_increase_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
#     assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
#     assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"
#
#     blend_image_lightness = get_intensity(blend_image)
#     result = add_intensity(base_image, blend_image_lightness)
#     return result.clamp(0, 1)
#
# def hsi_decrease_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
#     assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
#     assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"
#
#     blend_image_lightness = get_intensity(blend_image) - 1
#     result = add_intensity(base_image, blend_image_lightness)
#     return result.clamp(0, 1)

def hsl_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    intensity = get_lightness(base_image)
    saturation = get_saturation_hsl(blend_image)
    result = set_saturation_hsl(base_image, saturation)
    result = set_lightness(result, intensity)
    return result.clamp(0, 1)

# def hsi_increase_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
#     base_image_saturation = get_saturation_hsi(base_image)
#     blend_image_saturation = get_saturation_hsi(blend_image)
#
#     new_saturation = torch.lerp(base_image_saturation, torch.ones_like(base_image_saturation), blend_image_saturation)
#
#     base_image_lightness = get_intensity(base_image)
#
#     result = set_saturation_hsi(base_image, new_saturation)
#     result = set_intensity(result, base_image_lightness)
#
#     return result.clamp(0, 1)
#
# def hsi_decrease_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
#     base_image_saturation = get_saturation_hsi(base_image)
#     blend_image_saturation = get_saturation_hsi(blend_image)
#
#     new_saturation = torch.lerp(torch.zeros_like(base_image_saturation), base_image_saturation, blend_image_saturation)
#
#     base_image_lightness = get_intensity(base_image)
#
#     result = set_saturation_hsi(base_image, new_saturation)
#     result = set_intensity(result, base_image_lightness)
#
#     return result.clamp(0, 1)

def hsl_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_intensity = get_lightness(base_image)
    result = set_lightness(blend_image, base_image_intensity)
    return result.clamp(0, 1)

def hsl_hue(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsl(base_image)
    base_image_lightness = get_lightness(base_image)

    result = blend_image.clone()
    result = set_saturation_hsl(result, base_image_saturation)
    result = set_lightness(result, base_image_lightness)
    return result.clamp(0, 1)

hsl_blend_functions = {
    BlendModes.HSL_COLOR: hsl_color,
    BlendModes.HSL_HUE: hsl_hue,
    BlendModes.HSL_SATURATION: hsl_saturation,
    BlendModes.HSL_LIGHTNESS: hsl_lightness,
    # BlendModes.HSL_DECREASE_SATURATION: hsi_decrease_saturation,
    # BlendModes.HSL_INCREASE_SATURATION: hsi_increase_saturation,
    # BlendModes.HSL_DECREASE_LIGHTNESS: hsi_decrease_intensity,
    # BlendModes.HSL_INCREASE_LIGHTNESS: hsi_increase_intensity,
}