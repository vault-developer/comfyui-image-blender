class ImageBlender:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "blend_modes": (
                    [
                        "darken",
                        "multiply",
                        "color burn",
                        "linear burn",
                        "lighten",
                        "screen",
                        "color dodge",
                        "linear dodge",
                        "overlay",
                        "soft light",
                        "hard light",
                        "vivid light",
                        "linear light",
                        "pin light",
                        "hard mix",
                    ],
                    {
                        "default": "darken",
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "blend"

    CATEGORY = "ImageBlender"

    def blend(self, image_1, image_2, blend_modes):
        image = 1.0 - image_1
        return (image,)


NODE_CLASS_MAPPINGS = {
    "ImageBlender": ImageBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlender": "ImageBlender"
}
