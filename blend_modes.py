from .blend_modes_enum import BlendModes
import torch

def normal(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def dissolve(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    random_mask = torch.rand_like(base_image) > 0.5
    return torch.where(random_mask, blend_image, base_image)

def darken(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.minimum(base_image, blend_image)

def multiply(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return base_image * blend_image

def color_burn(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return 1 - (1 - base_image) / blend_image

def linear_burn(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return base_image + blend_image - 1

def darker_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_sum = torch.sum(base_image, dim=-1, keepdim=True)
    blend_sum = torch.sum(blend_image, dim=-1, keepdim=True)
    return torch.where(base_sum < blend_sum, base_image, blend_image)

def lighten(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.maximum(base_image, blend_image)

def screen(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return 1 - (1 - base_image) * (1 - blend_image)

def color_dodge(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(base_image / (1 - blend_image), 0, 1)

def linear_dodge(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return base_image + blend_image

def lighten_color(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    base_sum = torch.sum(base_image, dim=-1, keepdim=True)
    blend_sum = torch.sum(blend_image, dim=-1, keepdim=True)
    return torch.where(base_sum > blend_sum, base_image, blend_image)

def overlay(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(base_image < 0.5, 2 * base_image * blend_image, 1 - 2 * (1 - base_image) * (1 - blend_image))

def soft_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        blend_image < 0.5,
        2 * base_image * blend_image + base_image**2 * (1 - 2 * blend_image),
        2 * base_image * (1 - blend_image) + torch.sqrt(base_image) * (2 * blend_image - 1)
    )

def hard_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(blend_image < 0.5, 2 * base_image * blend_image, 1 - 2 * (1 - base_image) * (1 - blend_image))

def vivid_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        blend_image < 0.5,
        1 - (1 - base_image) / (2 * blend_image),
        base_image / (2 * (1 - blend_image))
    )

def linear_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return base_image + 2 * blend_image - 1

def pin_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(blend_image < 0.5, torch.minimum(base_image, 2 * blend_image), torch.maximum(base_image, 2 * blend_image - 1))

def hard_mix(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.where(base_image + blend_image < 1, 0, 1)

def difference(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.abs(base_image - blend_image)

def exclusion(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return base_image + blend_image - 2 * base_image * blend_image

def subtract(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(base_image - blend_image, 0, 1)

def divide(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(base_image / blend_image, 0, 1)

blend_functions = {
    # Normal
    BlendModes.NORMAL: normal,
    BlendModes.DISSOLVE: dissolve,
    # Darken
    BlendModes.DARKEN: darken,
    BlendModes.MULTIPLY: multiply,
    BlendModes.COLOR_BURN: color_burn,
    BlendModes.LINEAR_BURN: linear_burn,
    BlendModes.DARKER_COLOR: darker_color,
    # Lighten
    BlendModes.LIGHTEN: lighten,
    BlendModes.SCREEN: screen,
    BlendModes.COLOR_DODGE: color_dodge,
    BlendModes.LINEAR_DODGE: linear_dodge,
    BlendModes.LIGHTEN_COLOR: lighten_color,
    # Overlay
    BlendModes.OVERLAY: overlay,
    BlendModes.SOFT_LIGHT: soft_light,
    BlendModes.HARD_LIGHT: hard_light,
    BlendModes.VIVID_LIGHT: vivid_light,
    BlendModes.LINEAR_LIGHT: linear_light,
    BlendModes.PIN_LIGHT: pin_light,
    BlendModes.HARD_MIX: hard_mix,
    # Inversion
    BlendModes.DIFFERENCE: difference,
    BlendModes.EXCLUSION: exclusion,
    BlendModes.SUBTRACT: subtract,
    BlendModes.DIVIDE: divide,
}