"""
Standard functions available in shaders. This module allows users to discover
the functions, read their docs, and keep flake8 happy.
"""

import math


NI = "Only works in the shader."


tex_functions = {"imageLoad", "read", "imageStore", "write", "sample"}


def read(texture, tex_coords):  # noqa: N802
    """Load a pixel from a texture. The tex_coords must be i32, ivec2
    or ivec3. Returns a vec4 color. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


def write(texture, tex_coords, color):  # noqa: N802
    """Safe a pixel value to a texture. The tex_coords must be i32, ivec2
    or ivec3. Color must be vec4. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


def sample(texture, sampler, tex_coords):  # noqa: N802
    """Sample from an image. The tex_coords must be f32, vec2 or vec3;
    the data is interpolated. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


# %% Funcions from extension instruction sets

# For the function definitions and docs, see:
# https://www.khronos.org/registry/spir-v/specs/unified1/GLSL.std.450.html
#
# For the non-hardcoded extensions, the enum nr and number of args are
# validated in our unit tests.

ext_functions = {}


def extension(nr, set_name="GLSL.std.450", result_type=""):
    def wrapper(func):
        assert not func.__defaults__
        assert not func.__kwdefaults__
        assert not func.__code__.co_kwonlyargcount
        ext_functions[func.__name__] = {
            "nr": nr,
            "set_name": set_name,
            "result_type": result_type,
            "nargs": func.__code__.co_argcount,
        }
        return func

    return wrapper


def hardcoded_extension(func):
    assert not func.__defaults__
    assert not func.__kwdefaults__
    assert not func.__code__.co_kwonlyargcount
    ext_functions[func.__name__] = None
    return func


@extension(1, result_type="same")
def round(x):
    """Round x to the nearest whole number, with x a float scalar or vector.
    Fractions of .5 may round down or up, depending on the implementation.
    """
    raise NotImplementedError()


@extension(2, result_type="same")
def round_even(x):
    """Round x to the nearest even whole number, with x a float scalar or vector."""
    raise NotImplementedError()


@extension(3, result_type="same")
def trunc(x):
    """Like floor(), but rounds towards zero (-3.2 becomes -3)."""
    raise NotImplementedError()


@hardcoded_extension  # nr is 4 or 5
def abs(x):
    """The absolute value of x. The type of x can be an int or float
    scalar or vector.
    """
    return abs(x)


@hardcoded_extension  # nr is 6 or 7
def sign(x):
    """Get the sign of x, with x a float scalar or vector. The result is
    1.0 if x > 0, -1 if x < 0, and 0.0 otherwise. If x is NaN, the result
    can be any of the former.
    """
    return math.sign(x)


@extension(8, result_type="same")
def floor(x):
    """Round x to the nearest whole number smaller than or equal to x."""
    return math.floor(x)


@extension(9, result_type="same")
def ceil(x):
    """Round x to the nearest whole number larger than or equal tox."""
    return math.ceil(x)


@extension(10, result_type="same")
def fract(x):
    """Returns x - floor(x)"""
    return math.ceil(x)


@extension(11, result_type="same")
def radians(x):
    """Converts degress to radians."""
    return x * math.pi / 180


@extension(12, result_type="same")
def degrees(x):
    """Converts degress to radians."""
    return x * 180 / math.pi


@extension(13, result_type="same")
def sin(x):
    """Get sin(x)"""
    return math.sin(x)


@extension(14, result_type="same")
def cos(x):
    """Get cos(x)"""
    return math.cos(x)


@extension(15, result_type="same")
def tan(x):
    """Get tan(x)"""
    return math.tan(x)


@extension(16, result_type="same")
def asin(x):
    """Get the arc sine of x, an angle in radians.
    The range of result values is [-π / 2, π / 2].
    """
    return math.asin(x)


@extension(17, result_type="same")
def acos(x):
    """Get the arc cosine of x, an angle in radians.
    The range of result values is [0, π].
    """
    return math.acos(x)


@extension(18, result_type="same")
def atan(x):
    """Get the arc tangent of x, an angle in radians.
    The range of result values is [-π / 2, π / 2].
    """
    return math.atan(x)


@extension(19, result_type="same")
def sinh(x):
    """Get sinh(x)"""
    return math.sinh(x)


@extension(20, result_type="same")
def cosh(x):
    """Get cosh(x)"""
    return math.cosh(x)


@extension(21, result_type="same")
def tanh(x):
    """Get tanh(x)"""
    return math.tanh(x)


@extension(22, result_type="same")
def asinh(x):
    """Get asinh(x)"""
    return math.asinh(x)


@extension(23, result_type="same")
def acosh(x):
    """Get acosh(x)"""
    return math.acosh(x)


@extension(24, result_type="same")
def atanh(x):
    """Get atanh(x)"""
    return math.atanh(x)


@extension(25, result_type="same")
def atan2(x, y):
    """Get the arc tangent of x, an angle in radians.
    The signs of x and y are used to determine what quadrant the angle is in.
    The range of result values is [-π, π]
    """
    return math.atan2(x, y)


@extension(26, result_type="same")
def pow(x, y):
    """Calculate x**y, with x and y float scalars or vectors."""
    return x ** y


@extension(27, result_type="same")
def exp(x):
    """Calculate e**x."""
    return math.e ** x


@extension(28, result_type="same")
def log(x):
    """Calculate the natural logatihm of x: log(x)"""
    return math.log(x)


@extension(29, result_type="same")
def exp2(x):
    """Calculate 2**x"""
    return 2 ** x


@extension(30, result_type="same")
def log2(x):
    """Calculate the base-2 logatihm of x: log2(x)"""
    return math.log2(x)


@extension(31, result_type="same")
def sqrt(x):
    """Calculate x**0.5, with x a float scalar or vector."""
    return x ** 0.5


# 32: InverseSqrt -> just do **2
# 33: determinant


@hardcoded_extension
def matrix_inverse(m):  # is nr 34
    """Invert the given square matrix."""
    raise NotImplementedError()


# 35: Modf -> weird op
# 36: ModfStruct


@hardcoded_extension  # is nr 37-39
def min(x, y):
    """The minimum of x and y, with both x and y floats, ints, or float/int vectors."""
    raise NotImplementedError()


@hardcoded_extension
def max(x, y):  # is nr 40-42
    """The maximum of x and y, with both x and y floats, ints, or float/int vectors."""
    raise NotImplementedError()


@hardcoded_extension  # is nr 43-45
def clamp(x, min_val, max_val):
    """Return min(max_val, max(min_val, x)), with x and y floats or float vectors."""
    return min(max_val, max(min_val, x))


# @extension(46, result_type="same")


@hardcoded_extension  # is nr 46
def mix(x, y, a):
    """Return x * (1 - a) + y * a, with x, y floats or float vectors. A can be
    a float-vector or float (also if x and y are vectors).
    """
    return x * (1 - a) + y * a


@extension(48, result_type="same")
def step(edge, x):
    """Return 0.0 if x < edge; otherwise result is 1.0, with x a float scalar or vector."""
    return 0.0 if x < edge else 1.0


@extension(49, result_type="same")
def smooth_step(edge0, edge1, x):
    """the result is 0.0 if x ≤ edge0 and 1.0 if x ≥ edge1 and performs
    smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    """
    raise NotImplementedError()


# 50: Fma
# 51: Frexp
# 52: FrexpStruct
# 53: Ldexp
# 54-65: pack and unpack operations


@extension(66, result_type="component")
def length(v):
    """Calculate the length (a.k.a. norm) of the given vector."""
    return sum(x ** 2 for x in v) ** 0.5


@extension(67, result_type="component")
def distance(p0, p1):
    """Calculate the distance between two points."""
    raise NotImplementedError()


@extension(68, result_type="same")
def cross(p0, p1):
    """The cross product of two vectors."""
    raise NotImplementedError()


@extension(69, result_type="same")
def normalize(v):
    """Get the normalized version of the given float vector."""
    raise NotImplementedError()


@extension(70, result_type="same")
def face_forward(n, i, n_ref):
    """If the dot product of n_ref and i is negative, the result is n, otherwise it is -n."""
    raise NotImplementedError()


@extension(71, result_type="same")
def reflect(i, n):
    """For the incident vector i and surface orientation n, the result is the reflection direction:
    i - 2 * dot(n, u) * n

    The vector n must already be normalized in order to achieve the desired result.
    """
    raise NotImplementedError()


# 72: refract (needs hardcoded_extension
# 73-75: LSB/MSB bit stuff
# 76-78: interpolate stuff


@extension(79, result_type="same")
def nmin(x, y):
    """The minimum of x and y, with both x and y floats, ints, or float/int vectors.
    If x or y is NaN, the other is returned. If both are NaN, the result is NaN.
    """
    raise NotImplementedError()


@extension(80, result_type="same")
def nmax(x, y):
    """The maximum of x and y, with both x and y floats, ints, or float/int vectors.
    If x or y is NaN, the other is returned. If both are NaN, the result is NaN.
    """
    raise NotImplementedError()


@extension(81, result_type="same")
def nclamp(x, min_val, max_val):
    """Return min(max_val, max(min_val, x)), with x and y floats or float vectors.
    If x or y is NaN, the other is returned. If both are NaN, the result is NaN.
    """
    raise NotImplementedError()


# %% all

__all__ = list(tex_functions) + list(ext_functions)
