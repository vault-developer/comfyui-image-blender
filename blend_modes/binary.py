from ..helpers import replace_zeros, float_to_uint8, uint8_to_float
from ..blend_modes_enum import BlendModes
import torch

def binary_and(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = float_to_uint8(base_image) & float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_converse(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = ~float_to_uint8(base_image) | float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_implication(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = float_to_uint8(base_image) | ~float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_nand(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = ~(float_to_uint8(base_image) & float_to_uint8(blend_image))
    return uint8_to_float(result)

def binary_nor(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = ~(float_to_uint8(base_image) | float_to_uint8(blend_image))
    return uint8_to_float(result)

def binary_not_converse(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = float_to_uint8(base_image) & ~float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_not_implication(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = ~float_to_uint8(base_image) & float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_or(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = float_to_uint8(base_image) | float_to_uint8(blend_image)
    return uint8_to_float(result)

def binary_xnor(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = ~(float_to_uint8(base_image) ^ float_to_uint8(blend_image))
    return uint8_to_float(result)

def binary_xor(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = float_to_uint8(base_image) ^ float_to_uint8(blend_image)
    return uint8_to_float(result)

binary_blend_functions = {
    BlendModes.BINARY_AND: binary_and,
    BlendModes.BINARY_CONVERSE: binary_converse,
    BlendModes.BINARY_IMPLICATION: binary_implication,
    BlendModes.BINARY_NAND: binary_nand,
    BlendModes.BINARY_NOR: binary_nor,
    BlendModes.BINARY_NOT_CONVERSE: binary_not_converse,
    BlendModes.BINARY_NOT_IMPLICATION: binary_not_implication,
    BlendModes.BINARY_OR: binary_or,
    BlendModes.BINARY_XNOR: binary_xnor,
    BlendModes.BINARY_XOR: binary_xor,
}