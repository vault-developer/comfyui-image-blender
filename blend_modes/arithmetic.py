from ..blend_modes_enum import BlendModes
import torch

def arithmetic_addition(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def arithmetic_divide(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def arithmetic_inverse_subtract(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def arithmetic_multiply(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

def arithmetic_subtract(_, blend_image: torch.Tensor) -> torch.Tensor:
    return blend_image

arithmetic_blend_functions = {
    BlendModes.ARITHMETIC_ADDITION: arithmetic_addition,
    BlendModes.ARITHMETIC_DIVIDE: arithmetic_divide,
    BlendModes.ARITHMETIC_INVERSE_SUBTRACT: arithmetic_inverse_subtract,
    BlendModes.ARITHMETIC_MULTIPLY: arithmetic_multiply,
    BlendModes.ARITHMETIC_SUBTRACT: arithmetic_subtract,
}