import torch
from typing import Callable

def replace_zeros(tensor: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    return torch.where(tensor == 0, torch.tensor(epsilon, device=tensor.device), tensor)

def float_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 256).to(torch.uint8)

def uint8_to_float(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor/256, 0.0, 1.0)

def set_saturation(image: torch.Tensor, new_saturation: torch.Tensor) -> torch.Tensor:
    result = image.clone()

    min_idx = torch.argmin(result, dim=-1, keepdim=True)
    max_idx = torch.argmax(result, dim=-1, keepdim=True)
    mid_idx = torch.where(min_idx == max_idx, min_idx, 3 - min_idx - max_idx)

    min_val = torch.gather(result, -1, min_idx)
    mid_val = torch.gather(result, -1, mid_idx)
    max_val = torch.gather(result, -1, max_idx)

    result.scatter_(-1, min_idx, torch.zeros_like(min_val))
    result.scatter_(-1, mid_idx, ((mid_val - min_val) * new_saturation.unsqueeze(-1)) / replace_zeros(max_val - min_val))
    result.scatter_(-1, max_idx, new_saturation.unsqueeze(-1))

    # manage zeros
    result = torch.where(
        (max_val - min_val) > 0,
        result,
        torch.zeros_like(result)
    )

    return result.clamp(0, 1)

def set_ilvy(image: torch.Tensor, new_ilvy: torch.Tensor, get_ilvy: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    result = add_ilvy(image, new_ilvy - get_ilvy(image), get_ilvy)
    return result.clamp(0, 1)

def add_ilvy(image: torch.Tensor, new_intensity: torch.Tensor, get_ilvy: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    image = image + new_intensity.unsqueeze(-1)

    intensity = get_ilvy(image)
    min = torch.min(image, dim=-1).values
    max = torch.max(image, dim=-1).values

    # adjust overflows
    mask_min = min < 0.0
    iln = torch.where(mask_min, 1.0 / (intensity - min + 1e-8), torch.zeros_like(intensity))
    image = torch.where(mask_min.unsqueeze(-1), intensity.unsqueeze(-1) + ((image - intensity.unsqueeze(-1)) * intensity.unsqueeze(-1)) * iln.unsqueeze(-1), image)

    mask_max = (max > 1.0) & ((max - intensity) > torch.finfo(max.dtype).eps)
    il = torch.where(mask_max, 1.0 - intensity, torch.zeros_like(intensity))
    ixl = torch.where(mask_max, 1.0 / (max - intensity + 1e-8), torch.zeros_like(intensity))
    image = torch.where(mask_max.unsqueeze(-1), intensity.unsqueeze(-1) + ((image - intensity.unsqueeze(-1)) * il.unsqueeze(-1)) * ixl.unsqueeze(-1), image)

    return image.clamp(0, 1)

def inv(image: torch.Tensor):
    return 1.0 - image

def rgb2hsv_torch1(rgb: torch.Tensor) -> torch.Tensor:
    # Convert BHWC to BCHW
    rgb = rgb.permute(0, 3, 1, 2)

    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    hsv = torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    # Convert back to BHWC
    return hsv.permute(0, 2, 3, 1)

def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]

    cmax, cmax_idx = torch.max(rgb, dim=-1, keepdim=True)
    cmin = torch.min(rgb, dim=-1, keepdim=True)[0]
    delta = cmax - cmin

    hsv_h = torch.empty_like(r)
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((g - b) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((b - r) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((r - g) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.

    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax

    return torch.cat([hsv_h, hsv_s, hsv_v], dim=-1)

def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_v = hsv[..., 0:1], hsv[..., 1:2], hsv[..., 2:3]

    _c = hsv_v * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_v - _c
    _o = torch.zeros_like(_c)

    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, -1, -1, 3)

    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=-1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=-1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=-1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=-1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=-1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=-1)[idx == 5]

    rgb += _m

    return rgb