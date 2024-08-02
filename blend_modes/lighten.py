from .lighten_helpers import penumbra
from ..blend_modes_enum import BlendModes
import torch

def lighten_color_dodge(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    max_val = torch.tensor(1.0, device=blend_image.device, dtype=blend_image.dtype)
    zero_val = torch.tensor(0.0, device=blend_image.device, dtype=blend_image.dtype)

    result = torch.where(
        blend_image == max_val,
        torch.where(base_image == zero_val, zero_val, max_val),
        torch.clamp(base_image / (max_val - blend_image), 0, 1)
    )

    return result

def lighten_linear_dodge(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image + blend_image
    return result.clamp( 0, 1)

def lighten_lighten(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    return torch.max(base_image, blend_image)

def lighten_linear_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = (blend_image + blend_image + base_image) - 1.0
    return torch.clamp(result, 0, 1)

def lighten_screen(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = 1 - (1 - blend_image) * (1 - base_image)
    return torch.clamp(result, 0, 1)

def lighten_pin_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    blend_image2 = 2 * blend_image
    min = torch.min(base_image, blend_image2)
    result = torch.max(blend_image2 - 1.0, min)
    return result.clamp(0, 1)

def lighten_vivid_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    half_val = 0.5
    one_val = 1.0
    zero_val = 0.0

    result = torch.where(
        blend_image < half_val,
        torch.where(
            blend_image == zero_val,
            torch.where(base_image == one_val, one_val, zero_val),
            torch.clamp(one_val - (1 - base_image) / (2 * blend_image), 0, 1)
        ),
        torch.where(
            blend_image == one_val,
            torch.where(base_image == zero_val, zero_val, one_val),
            torch.clamp(base_image / (2 * (1 - blend_image)), 0, 1)
        )
    )

    return result

def lighten_flat_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    mask_src_zero = torch.isclose(blend_image, torch.tensor(0.0))
    result = torch.zeros_like(base_image)

    mask_src_nonzero = ~mask_src_zero
    inv_src = 1 - blend_image[mask_src_nonzero]

    sum_tensor = inv_src + base_image[mask_src_nonzero]
    hard_mix_result = torch.where(sum_tensor > 1.0, torch.ones_like(sum_tensor), torch.zeros_like(sum_tensor))
    penumbra_a_result = penumbra(base_image[mask_src_nonzero], blend_image[mask_src_nonzero], lighten_color_dodge)
    penumbra_b_result = penumbra(blend_image[mask_src_nonzero], base_image[mask_src_nonzero], lighten_color_dodge)

    result[mask_src_nonzero] = torch.where(
        torch.isclose(hard_mix_result, torch.tensor(1.0)),
        penumbra_b_result,
        penumbra_a_result
    )

    return result

lighten_blend_functions = {
    BlendModes.LIGHTEN_COLOR_DODGE: lighten_color_dodge,
    BlendModes.LIGHTEN_LINEAR_DODGE: lighten_linear_dodge,
    BlendModes.LIGHTEN_LIGHTEN: lighten_lighten,
    BlendModes.LIGHTEN_LINEAR_LIGHT: lighten_linear_light,
    BlendModes.LIGHTEN_SCREEN: lighten_screen,
    BlendModes.LIGHTEN_PIN_LIGHT: lighten_pin_light,
    BlendModes.LIGHTEN_VIVID_LIGHT: lighten_vivid_light,
    BlendModes.LIGHTEN_FLAT_LIGHT: lighten_flat_light,
    # BlendModes.LIGHTEN_HARD_LIGHT: lighten_color_dodge,
    # BlendModes.LIGHTEN_SOFT_LIGHT_IFS_ILLUSIONS: lighten_color_dodge,
    # BlendModes.LIGHTEN_SOFT_LIGHT_PEGTOP_DELPHI: lighten_color_dodge,
    # BlendModes.LIGHTEN_SOFT_LIGHT_PHOTOSHOP: lighten_color_dodge,
    # BlendModes.LIGHTEN_SOFT_LIGHT_SVG: lighten_color_dodge,
    # BlendModes.LIGHTEN_GAMMA_LIGHT: lighten_color_dodge,
    # BlendModes.LIGHTEN_GAMMA_ILLUMINATION: lighten_color_dodge,
    # BlendModes.LIGHTEN_LIGHTER_COLOR: lighten_color_dodge,
    # BlendModes.LIGHTEN_PNORM_A: lighten_color_dodge,
    # BlendModes.LIGHTEN_PNORM_B: lighten_color_dodge,
    # BlendModes.LIGHTEN_SUPER_LIGHT: lighten_color_dodge,
    # BlendModes.LIGHTEN_TINT_IFS_ILLUSIONS: lighten_color_dodge,
    # BlendModes.LIGHTEN_FOG_LIGHTEN_IFS_ILLUSIONS: lighten_color_dodge,
    # BlendModes.LIGHTEN_EASY_DODGE: lighten_color_dodge,
    # BlendModes.LIGHTEN_LUMINOSITY_SAI: lighten_color_dodge,
}