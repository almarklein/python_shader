"""
Write modern GPU shaders in Python!
"""

# flake8: noqa

__version__ = "0.4.2"
version_info = tuple(map(int, __version__.split(".")))

from ._coreutils import ShaderError
from ._module import ShaderModule
from .py import python2shader

# from .wasl import wasl2spirv  # note the textx dependency

from ._types import void, boolean, u8, i16, i32, i64, f16, f32, f64
from ._types import vec2, vec3, vec4
from ._types import ivec2, ivec3, ivec4
from ._types import bvec2, bvec3, bvec4
from ._types import mat2, mat3, mat4
from ._types import Vector, Matrix, Array, Struct

from ._types import shadertype_as_ctype
from ._types import RES_INPUT, RES_OUTPUT
from ._types import RES_UNIFORM, RES_BUFFER, RES_SAMPLER, RES_TEXTURE

from . import dev
