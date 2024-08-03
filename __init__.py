import torch

from .blend_modes.index import blend_functions
from .blend_modes_enum import BlendModes

class ImageBlender:
    def __init__(self):
        self.blend_functions = blend_functions

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "blend_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (
                    [mode.value for mode in BlendModes],
                    {"default": BlendModes.MIX_NORMAL.value}
                ),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "ImageBlender"

    def blend(self, base_image: torch.Tensor, blend_image: torch.Tensor, strength: float, blend_mode: str, mask: torch.Tensor = None) -> tuple:
        assert base_image.shape == blend_image.shape, "Base and blend images must have the same shape"
        assert base_image.shape[-1] == 3, "Input images must have 3 channels (RGB)"

        blend_function = self.blend_functions.get(BlendModes(blend_mode), lambda x, y: x)
        result = blend_function(base_image, blend_image)

        if mask is not None:
            # Ensure mask has the same number of channels as the images
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1).expand(-1, -1, -1, base_image.shape[-1])

            if mask.size() != base_image.size():
                print(f"WARN: Mask size {mask.size()} is different from image size {base_image.size()}, mask is ignored")
            else:
                result = result * mask + base_image * (1 - mask)

        # Apply opacity
        result = result * strength + base_image * (1 - strength)
        # Normalize the result
        result = torch.clamp(result, 0, 1)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageBlender": ImageBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlender": "ImageBlender"
}