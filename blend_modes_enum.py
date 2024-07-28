from enum import Enum

class BlendModes(Enum):
    NORMAL = "normal"
    DISSOLVE = "dissolve"

    DARKEN = "darken"
    MULTIPLY = "multiply"
    COLOR_BURN = "color burn"
    LINEAR_BURN = "linear burn"
    DARKER_COLOR = "darker color"

    LIGHTEN = "lighten"
    SCREEN = "screen"
    COLOR_DODGE = "color dodge"
    LINEAR_DODGE = "linear dodge"
    LIGHTEN_COLOR = "lighten color"

    OVERLAY = "overlay"
    SOFT_LIGHT = "soft light"
    HARD_LIGHT = "hard light"
    VIVID_LIGHT = "vivid light"
    LINEAR_LIGHT = "linear light"
    PIN_LIGHT = "pin light"
    HARD_MIX = "hard mix"

    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    SUBTRACT = "subtract"
    DIVIDE = "divide"