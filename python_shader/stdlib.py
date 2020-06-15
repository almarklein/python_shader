"""
Standard functions available in shaders. This module allows users to discover
the functions, read their docs, and keep flake8 happy.
"""


NI = "Only works in the shader."


tex_functions = {"imageLoad", "read", "imageStore", "write", "sample"}


def read(texture, tex_coords):  # noqa: N802
    """ Load a pixel from a texture. The tex_coords must be i32, ivec2
    or ivec3. Returns a vec4 color. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


def write(texture, tex_coords, color):  # noqa: N802
    """ Safe a pixel value to a texture. The tex_coords must be i32, ivec2
    or ivec3. Color must be vec4. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


def sample(texture, sampler, tex_coords):  # noqa: N802
    """ Sample from an image. The tex_coords must be f32, vec2 or vec3;
    the data is interpolated. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


# %% Funcions from extension instruction sets

# https://www.khronos.org/registry/spir-v/specs/unified1/GLSL.std.450.html
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


@extension(26, result_type="same")
def pow(x, y):
    """ Calculate x**y, with x and y float scalars or vectors.
    """
    return x ** y


@extension(31, result_type="same")
def sqrt(x):
    """ Calculate x**0.5, with x a float scalar or vector.
    """
    return x ** 0.5


@extension(66, result_type="component")
def length(v):
    """ Calculate the length (a.k.a. norm) of the given vector.
    """
    return sum(x ** 2 for x in v) ** 0.5


# %% Extension instructions that we hard-code
# ( because of the types)


def hardcoded_extension(func):
    assert not func.__defaults__
    assert not func.__kwdefaults__
    assert not func.__code__.co_kwonlyargcount
    ext_functions[func.__name__] = None
    return func


@hardcoded_extension
def abs(x):
    """ The absolute value of x. The type of x can be an int or float
    scalar or vector.
    """
    return abs(x)
