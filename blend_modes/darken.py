from ..helpers import replace_zeros
from ..blend_modes_enum import BlendModes
import torch

def darken_burn(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    safe_base_image = replace_zeros(base_image)
    result = 1 - (1 - blend_image) / safe_base_image
    return torch.clamp(result, 0.0, 1.0)

def darken_darken(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.min(base_image, blend_image)
    return torch.clamp(result, 0.0, 1.0)

def darken_darker_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_sum = torch.sum(base_image, dim=-1, keepdim=True)
    blend_sum = torch.sum(blend_image, dim=-1, keepdim=True)
    return torch.where(base_sum < blend_sum, base_image, blend_image)

def darken_easy_burn(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    inverted_base_image = 1.0 - base_image
    safe_inverted_base_image = replace_zeros(inverted_base_image)
    result = 1.0 - torch.pow(safe_inverted_base_image, blend_image * (15 / 13))
    return torch.clamp(result, 0.0, 1.0)

def darken_fog_darken(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result_low = (1 - base_image) * base_image + base_image * blend_image
    result_high = base_image * blend_image + base_image - torch.pow(base_image, 2)
    result = torch.where(base_image < 0.5, result_low, result_high)
    return torch.clamp(result, 0.0, 1.0)

def darken_gamma_dark(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    safe_base_image =replace_zeros(base_image)
    result = torch.pow(blend_image, 1.0 / safe_base_image)
    result = torch.where(base_image == 0.0, torch.zeros_like(result), result)
    return torch.clamp(result, 0.0, 1.0)

def darken_linear_burn(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image + blend_image - 1
    return torch.clamp(result, 0.0, 1.0)

def darken_shade(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = 1.0 - ((1.0 - blend_image) * base_image + torch.sqrt(1.0 - base_image))
    return torch.clamp(result, 0.0, 1.0)

darken_blend_functions = {
    BlendModes.DARKEN_BURN: darken_burn,
    BlendModes.DARKEN_DARKEN: darken_darken,
    BlendModes.DARKEN_DARKER_COLOR: darken_darker_color,
    BlendModes.DARKEN_EASY_BURN: darken_easy_burn,
    BlendModes.DARKEN_FOG_DARKEN: darken_fog_darken,
    BlendModes.DARKEN_GAMMA_DARK: darken_gamma_dark,
    BlendModes.DARKEN_LINEAR_BURN: darken_linear_burn,
    BlendModes.DARKEN_SHADE: darken_shade,
}