from ..helpers import replace_zeros
from ..blend_modes_enum import BlendModes
import torch

def arithmetic_addition(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image + blend_image
    return torch.clamp(result, 0.0, 1.0)

def arithmetic_divide(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    safe_blend_image = replace_zeros(blend_image)
    safe_result = base_image / safe_blend_image
    return torch.clamp(safe_result, 0.0, 1.0)

def arithmetic_inverse_subtract(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    inverted_base_image = 1 - base_image
    result = blend_image - inverted_base_image
    return torch.clamp(result, 0.0, 1.0)

def arithmetic_multiply(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image * blend_image
    return torch.clamp(result, 0.0, 1.0)

def arithmetic_subtract(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image - blend_image
    return torch.clamp(result, 0.0, 1.0)

arithmetic_blend_functions = {
    BlendModes.ARITHMETIC_ADDITION: arithmetic_addition,
    BlendModes.ARITHMETIC_DIVIDE: arithmetic_divide,
    BlendModes.ARITHMETIC_INVERSE_SUBTRACT: arithmetic_inverse_subtract,
    BlendModes.ARITHMETIC_MULTIPLY: arithmetic_multiply,
    BlendModes.ARITHMETIC_SUBTRACT: arithmetic_subtract,
}