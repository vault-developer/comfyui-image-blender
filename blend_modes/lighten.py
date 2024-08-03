from ..helpers import inv
from .darken import darken_gamma_dark
from .hsy_helpers import get_luminosity
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

def lighten_hard_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(blend_image)
    sum = blend_image + blend_image

    screen_result = torch.where(
        blend_image > 0.5,
        base_image + sum - ones - base_image * (sum - ones),
        torch.zeros_like(base_image)
    )

    multiply_result = torch.where(
        blend_image <= 0.5,
        base_image * sum,
        torch.zeros_like(base_image)
    )

    return torch.clamp(screen_result + multiply_result, 0.0, 1.0)

def lighten_soft_light_ifs_illusions(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    exponent = torch.pow(2.0, 2.0 * (0.5 - blend_image))
    result = torch.pow(base_image, exponent)

    return torch.clamp(result, 0.0, 1.0)

def lighten_soft_light_pegtop_delphi(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    term1 = base_image * lighten_screen(blend_image, base_image)
    term2 = blend_image * base_image * (1 - base_image)

    result = lighten_linear_dodge(term1, term2)
    return torch.clamp(result, 0.0, 1.0)

def lighten_soft_light_ps(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    condition = blend_image > 0.5
    high_result = base_image + (2 * blend_image - 1) * (torch.sqrt(base_image) - base_image)
    low_result = base_image - (1 - 2 * blend_image) * base_image * (1 - base_image)
    result = torch.where(condition, high_result, low_result)
    return torch.clamp(result, 0.0, 1.0)

def lighten_soft_light_svg(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    condition_src = blend_image > 0.5
    condition_dst = base_image > 0.25

    D_high = torch.sqrt(base_image)
    D_low = ((16.0 * base_image - 12.0) * base_image + 4.0) * base_image
    D = torch.where(condition_dst, D_high, D_low)

    high_result = base_image + (2.0 * blend_image - 1.0) * (D - base_image)
    low_result = base_image - (1.0 - 2.0 * blend_image) * base_image * (1.0 - base_image)

    result = torch.where(condition_src, high_result, low_result)
    return torch.clamp(result, 0.0, 1.0)

def lighten_gamma_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.pow(base_image, blend_image)
    return torch.clamp(result, 0.0, 1.0)

def lighten_gamma_illumination(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    inv_base = 1.0 - base_image
    inv_blend = 1.0 - blend_image

    result = 1 - darken_gamma_dark(inv_blend, inv_base)
    return torch.clamp(result, 0.0, 1.0)

def lighten_lighter_color(blend_image: torch.Tensor, base_image: torch.Tensor) -> torch.Tensor:
    lum_base = get_luminosity(base_image)
    lum_blend = get_luminosity(blend_image)

    mask = lum_blend > lum_base
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, 3)

    result = torch.where(mask, blend_image, base_image)
    return torch.clamp(result, 0.0, 1.0)

def lighten_pnorm_a(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    power_factor = 2.3333333333333333
    inverse_power_factor = 0.428571428571434

    src_power = torch.pow(blend_image, power_factor)
    dst_power = torch.pow(base_image, power_factor)

    sum_power = src_power + dst_power
    result = torch.pow(sum_power, inverse_power_factor)

    return result.clamp(0, 1)

def lighten_pnorm_b(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    power_factor = 4.0
    inverse_power_factor = 0.25

    src_power = torch.pow(blend_image, power_factor)
    dst_power = torch.pow(base_image, power_factor)

    sum_power = src_power + dst_power
    result = torch.pow(sum_power, inverse_power_factor)

    return result.clamp(0, 1)


def lighten_super_light(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    def pow_tensor(tensor, power):
        return torch.pow(tensor, power)

    def inv(tensor):
        return 1.0 - tensor

    power_val = 2.875

    result = torch.where(
        blend_image < 0.5,
        inv(pow_tensor(pow_tensor(inv(base_image), power_val) + pow_tensor(inv(2.0 * blend_image), power_val), 1.0 / power_val)),
        pow_tensor(pow_tensor(base_image, power_val) + pow_tensor(2.0 * blend_image - 1.0, power_val), 1.0 / power_val)
    )

    return result.clamp(0, 1)

def lighten_tint_ifs_illusions(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = base_image * (1 - blend_image) + torch.sqrt(blend_image)
    return result.clamp(0, 1)

def lighten_fog_lighten_ifs_illusions(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.where(
        blend_image < 0.5,
        inv(inv(blend_image) * blend_image) - inv(base_image) * inv(blend_image),
        blend_image - inv(base_image) * inv(blend_image) + torch.pow(inv(blend_image), 2)
    )

    return result.clamp(0, 1)

def lighten_easy_dodge(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    result = torch.where(
        blend_image == 1.0,
        torch.tensor(1.0, device=blend_image.device, dtype=blend_image.dtype),
        torch.pow(base_image, inv(torch.where(blend_image != 1.0, blend_image, torch.tensor(0.999999999999, device=blend_image.device, dtype=blend_image.dtype))) * 1.039999999)
    )

    return result.clamp(0, 1)

def lighten_luminosity_sai(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # Check if alpha channel is present
    has_alpha = src.shape[-1] == 4

    if has_alpha:
        src_rgb, src_a = src[..., :3], src[..., 3:]
        dst_rgb, dst_a = dst[..., :3], dst[..., 3:]
    else:
        src_rgb, src_a = src, torch.ones_like(src[..., :1])
        dst_rgb, dst_a = dst, torch.ones_like(dst[..., :1])

    result_rgb = src_rgb * src_a + dst_rgb
    result_rgb = torch.clamp(result_rgb, 0, 1)
    if has_alpha:
        result = torch.cat([result_rgb, dst_a], dim=-1)
    else:
        result = result_rgb

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
    BlendModes.LIGHTEN_HARD_LIGHT: lighten_hard_light,
    BlendModes.LIGHTEN_SOFT_LIGHT_IFS_ILLUSIONS: lighten_soft_light_ifs_illusions,
    BlendModes.LIGHTEN_SOFT_LIGHT_PEGTOP_DELPHI: lighten_soft_light_pegtop_delphi,
    BlendModes.LIGHTEN_SOFT_LIGHT_PS: lighten_soft_light_ps,
    BlendModes.LIGHTEN_SOFT_LIGHT_SVG: lighten_soft_light_svg,
    BlendModes.LIGHTEN_GAMMA_LIGHT: lighten_gamma_light,
    BlendModes.LIGHTEN_GAMMA_ILLUMINATION: lighten_gamma_illumination,
    BlendModes.LIGHTEN_LIGHTER_COLOR: lighten_lighter_color,
    BlendModes.LIGHTEN_PNORM_A: lighten_pnorm_a,
    BlendModes.LIGHTEN_PNORM_B: lighten_pnorm_b,
    BlendModes.LIGHTEN_SUPER_LIGHT: lighten_super_light,
    BlendModes.LIGHTEN_TINT_IFS_ILLUSIONS: lighten_tint_ifs_illusions,
    BlendModes.LIGHTEN_FOG_LIGHTEN_IFS_ILLUSIONS: lighten_fog_lighten_ifs_illusions,
    BlendModes.LIGHTEN_EASY_DODGE: lighten_easy_dodge,
    BlendModes.LIGHTEN_LUMINOSITY_SAI: lighten_luminosity_sai,
}