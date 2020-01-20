"""
Write modern GPU shaders in Python!
"""

# flake8: noqa

__version__ = "0.3.2"


from ._module import ShaderModule
from .py import python2shader

# from .wasl import wasl2spirv  # note the textx dependency

from ._types import void, boolean, f16, f32, f64, i16, i32, i64
from ._types import vec2, vec3, vec4
from ._types import ivec2, ivec3, ivec4
from ._types import bvec2, bvec3, bvec4
from ._types import mat2, mat3, mat4
from ._types import Vector, Matrix, Array, Struct

from . import dev
