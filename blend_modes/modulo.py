from ..blend_modes_enum import BlendModes
import torch

def modulo_modulo(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.fmod(blend_image, base_image)
    return result

def modulo_divisive_modulo(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.where(
        base_image == 0,
        torch.fmod((1.0 / 1e-10) * blend_image, 1.0),
        torch.fmod((1.0 / base_image) * blend_image, 1.0)
    )
    return result

modulo_blend_functions = {
    BlendModes.MODULO_MODULO: modulo_modulo,
    BlendModes.MODULO_DIVISIVE_MODULO: modulo_divisive_modulo,
}