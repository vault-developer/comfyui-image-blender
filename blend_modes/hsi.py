from ..blend_modes_enum import BlendModes
import torch

def hsi_increase_intensity(base_image: torch.Tensor, blend_image: torch.Tensor) -> torch.Tensor:
    def get_intensity(image: torch.Tensor) -> torch.Tensor:
        return torch.mean(image, dim=-1)

    def add_intensity(rgb: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
        rgb = rgb + light.unsqueeze(-1)
        # There is an additional logic in Krita, but I'm not sure if it's necessary
        # https://github.com/KDE/krita/blob/6324ca44615fa33957c9d96fdfb16b58d4fe6674/libs/pigment/KoColorSpaceMaths.h#L800
        return rgb.clamp(0, 1)

    assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
    assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

    blend_image_lightness = get_intensity(blend_image)
    result = add_intensity(base_image, blend_image_lightness)
    return result.clamp(0, 1)

hsi_blend_functions = {
    #BlendModes.HSI_COLOR: hsi_color,
    # BlendModes.HSI_HUE: hsi_hue,
    # BlendModes.HSI_SATURATION: hsi_saturation,
    # BlendModes.HSI_INTENSITY: hsi_intensity,
    # BlendModes.HSI_DECREASE_SATURATION: hsi_decrease_saturation,
    # BlendModes.HSI_INCREASE_SATURATION: hsi_increase_saturation,
    # BlendModes.HSI_DECREASE_INTENSITY: hsi_decrease_intensity,
    BlendModes.HSI_INCREASE_INTENSITY: hsi_increase_intensity,
}