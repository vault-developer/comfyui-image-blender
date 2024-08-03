from typing import Callable
import torch

def penumbra(base_image: torch.Tensor, blend_image: torch.Tensor, lighten_color_dodge: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    result = torch.zeros_like(base_image)
    mask_dst_one = torch.isclose(blend_image, torch.tensor(1.0))
    result[mask_dst_one] = 1.0
    mask_sum_less_than_one = (blend_image + base_image) < 1.0
    result[mask_sum_less_than_one] = lighten_color_dodge(base_image[mask_sum_less_than_one], blend_image[mask_sum_less_than_one]) / 2
    mask_remaining = ~(mask_dst_one | mask_sum_less_than_one) & ~torch.isclose(base_image, torch.tensor(0.0))
    result[mask_remaining] = 1 - torch.clamp((1 - blend_image[mask_remaining]) / base_image[mask_remaining] / 2, 0, 1)

    return result
