from ..blend_modes_enum import BlendModes
import torch

def normal(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

normal_blend_functions = {
    BlendModes.NORMAL: normal,
}