"""
A few compute examples. Compute shaders do not have vertex buffer inputs, nor
outputs. Any data you need must be provided as uniform, buffer or texture.
"""

from python_shader import python2shader, i32, f32, ivec3, Array


@python2shader
def compute_shader_copy(
    index: ("input", "GlobalInvocationId", i32),
    data1: ("buffer", 0, Array(i32)),
    data2: ("buffer", 1, Array(i32)),
):
    data2[index] = data1[index]


@python2shader
def compute_shader_multiply(
    index: ("input", "GlobalInvocationId", i32),
    data1: ("buffer", 0, Array(i32)),
    data2: ("buffer", 1, Array(f32)),
    data3: ("buffer", 2, Array(f32)),
):
    data3[index] = f32(data1[index]) * data2[index]


@python2shader
def compute_shader_tex_colorwap(
    index: ("input", "GlobalInvocationId", ivec3),
    tex1: ("texture", 0, "2d rgba8ui"),
    tex2: ("texture", 1, "2d rgba8ui"),
):
    color = tex1.read(index.xy)
    color = color.bgra
    tex2.write(index.xy, color)


# (The above shader also tests the case where we have to bound textures
# of equal type. Which is a case that needs a bit of special handling.)
