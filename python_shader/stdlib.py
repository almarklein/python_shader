"""
Standard functions available in shaders. This module allows users to discover
the functions, read their docs, and keep flake8 happy.
"""


# todo: rename this module?
# todo: follow glsl names, or use image-y terms like spirv does?
# todo: rename texture -> image?


NI = "Only works in the shader."


def imageLoad(texture, tex_coords):  # noqa: N802
    """ Load a pixel from a texture. The tex_coords must be i32, ivec2
    or ivec3. Returns a vec4 color.
    """
    raise NotImplementedError(NI)


def imageStore(texture, tex_coords, color):  # noqa: N802
    """ Load a pixel from a texture. The tex_coords must be i32, ivec2
    or ivec3. Color must be vec4.
    """
    raise NotImplementedError(NI)


def sampler2D(texture, sampler):  # noqa: N802
    """ Create a sampled image from a sampler and texture.
    """
    raise NotImplementedError(NI)


def texture(sampled_image, tex_coords):  # noqa: N802
    """ Sample from an image. The tex_coords must be f32, vec2 or vec3;
    the data is interpolated.
    """
    raise NotImplementedError(NI)


def textureLod(sampled_image, tex_coords, lod):  # noqa: N802
    """ Sample from an image using an explicit level of detail. The
    tex_coords must be f32, vec2 or vec3; the data is interpolated.
    """
    raise NotImplementedError(NI)
