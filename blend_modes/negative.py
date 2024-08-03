from ..blend_modes_enum import BlendModes
import torch

def negative_difference(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(base_image, blend_image)
    min_val = torch.min(base_image, blend_image)
    result = max_val - min_val
    return result.clamp(0, 1)

# This method behaves differently in Krita
def negative_equivalence(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    diff = blend_image - base_image
    abs_diff = torch.abs(diff)
    result = 1 - abs_diff
    return result.clamp(0, 1)

def negative_additive_subtractive(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    sqrt_base = torch.sqrt(base_image)
    sqrt_blend = torch.sqrt(blend_image)
    diff = sqrt_blend - sqrt_base
    result = torch.abs(diff)
    return result.clamp(0, 1)

def negative_exclusion(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    product = base_image * blend_image
    result = blend_image + base_image - 2 * product
    return result.clamp(0, 1)

def negative_arcus_tangent(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    zero_tensor = torch.zeros_like(base_image)
    unit_tensor = torch.ones_like(base_image)

    result = torch.where(
        blend_image == 0,
        torch.where(base_image == 0, zero_tensor, unit_tensor),
        2.0 * torch.atan(base_image / blend_image) / torch.pi
    )

    return result.clamp(0, 1)

def negative_negation(blend_image: torch.Tensor, base_image: torch.Tensor) -> torch.Tensor:
    unit_tensor = torch.ones_like(base_image)
    difference = unit_tensor - base_image - blend_image
    abs_difference = torch.abs(difference)
    result = unit_tensor - abs_difference
    return result.clamp(0, 1)

negative_blend_functions = {
    BlendModes.NEGATIVE_DIFFERENCE: negative_difference,
    BlendModes.NEGATIVE_EQUIVALENCE: negative_equivalence,
    BlendModes.NEGATIVE_ADDITIVE_SUBTRACTIVE: negative_additive_subtractive,
    BlendModes.NEGATIVE_EXCLUSION: negative_exclusion,
    BlendModes.NEGATIVE_ARCUS_TANGENT: negative_arcus_tangent,
    BlendModes.NEGATIVE_NEGATION: negative_negation,
}