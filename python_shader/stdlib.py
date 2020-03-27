"""
Standard functions available in shaders. This module allows users to discover
the functions, read their docs, and keep flake8 happy.
"""


# todo: rename texture -> image?


NI = "Only works in the shader."


def read(texture, tex_coords):  # noqa: N802
    """ Load a pixel from a texture. The tex_coords must be i32, ivec2
    or ivec3. Returns a vec4 color. Can also be used as a method of a
    texture object.
    """
    raise NotImplementedError(NI)


def write(texture, tex_coords, color):  # noqa: N802
    """ Load a pixel from a texture. The tex_coords must be i32, ivec2
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
