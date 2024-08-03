from .hsy_helpers import get_luminosity, set_luminosity, get_saturation_hsy, add_luminosity
from ..helpers import set_saturation
from ..blend_modes_enum import BlendModes
import torch

def hsy_luminosity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_luminosity = get_luminosity(blend_image)
    result = set_luminosity(base_image, blend_image_luminosity)
    return result.clamp(0, 1)

def hsy_increase_luminosity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_luminosity = get_luminosity(blend_image)
    result = add_luminosity(base_image, blend_image_luminosity)
    return result.clamp(0, 1)

def hsy_decrease_luminosity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image_luminosity = get_luminosity(blend_image) - 1
    result = add_luminosity(base_image, blend_image_luminosity)
    return result.clamp(0, 1)

def hsy_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    luminosity = get_luminosity(base_image)
    saturation = get_saturation_hsy(blend_image)
    result = set_saturation(base_image, saturation)
    result = set_luminosity(result, luminosity)
    return result.clamp(0, 1)

def hsy_increase_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsy(base_image)
    blend_image_saturation = get_saturation_hsy(blend_image)

    new_saturation = torch.lerp(base_image_saturation, torch.ones_like(base_image_saturation), blend_image_saturation)

    base_image_luminosity = get_luminosity(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_luminosity(result, base_image_luminosity)

    return result.clamp(0, 1)

def hsy_decrease_saturation(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsy(base_image)
    blend_image_saturation = get_saturation_hsy(blend_image)

    new_saturation = torch.lerp(torch.zeros_like(base_image_saturation), base_image_saturation, blend_image_saturation)

    base_image_luminosity = get_luminosity(base_image)

    result = set_saturation(base_image, new_saturation)
    result = set_luminosity(result, base_image_luminosity)

    return result.clamp(0, 1)

def hsy_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_luminosity = get_luminosity(base_image)
    result = set_luminosity(blend_image, base_image_luminosity)
    return result.clamp(0, 1)

def hsy_hue(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_image_saturation = get_saturation_hsy(base_image)
    base_image_luminosity = get_luminosity(base_image)

    result = blend_image.clone()
    result = set_saturation(result, base_image_saturation)
    result = set_luminosity(result, base_image_luminosity)
    return result.clamp(0, 1)

hsy_blend_functions = {
    BlendModes.HSY_COLOR: hsy_color,
    BlendModes.HSY_HUE: hsy_hue,
    BlendModes.HSY_SATURATION: hsy_saturation,
    BlendModes.HSY_LUMINOSITY: hsy_luminosity,
    BlendModes.HSY_DECREASE_SATURATION: hsy_decrease_saturation,
    BlendModes.HSY_INCREASE_SATURATION: hsy_increase_saturation,
    BlendModes.HSY_DECREASE_LUMINOSITY: hsy_decrease_luminosity,
    BlendModes.HSY_INCREASE_LUMINOSITY: hsy_increase_luminosity,
}