from .normal import normal_blend_functions
from .arithmetic import arithmetic_blend_functions
from .binary import binary_blend_functions

blend_functions = {}

blend_functions.update(normal_blend_functions)
blend_functions.update(arithmetic_blend_functions)
blend_functions.update(binary_blend_functions)