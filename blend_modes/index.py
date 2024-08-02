from .normal import normal_blend_functions
from .arithmetic import arithmetic_blend_functions
from .binary import binary_blend_functions
from .darken import darken_blend_functions
from .hsi import hsi_blend_functions
from .hsl import hsl_blend_functions
from .hsv import hsv_blend_functions
from .hsy import hsy_blend_functions
from .lighten import lighten_blend_functions

blend_functions = {}

blend_functions.update(normal_blend_functions)
blend_functions.update(arithmetic_blend_functions)
blend_functions.update(binary_blend_functions)
blend_functions.update(darken_blend_functions)
blend_functions.update(hsi_blend_functions)
blend_functions.update(hsl_blend_functions)
blend_functions.update(hsv_blend_functions)
blend_functions.update(hsy_blend_functions)
blend_functions.update(lighten_blend_functions)