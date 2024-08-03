from .lighten import lighten_hard_light
from ..blend_modes_enum import BlendModes
import torch

def mix_normal(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def mix_overlay(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return lighten_hard_light(blend_image, base_image)

mix_blend_functions = {
    BlendModes.MIX_NORMAL: mix_normal,
    BlendModes.MIX_OVERLAY: mix_overlay,
}