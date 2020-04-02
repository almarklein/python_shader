"""
Shaders that use textures.
"""

from python_shader import python2shader, ivec3, vec2, vec4


# Take two values from a 2-element 2D texture, and store into a a scalar 2D
# texture. Note that read() and write() always operate on either vec4 or ivec4.
@python2shader
def compute_shader_tex_add(
    index: ("input", "GlobalInvocationId", ivec3),
    tex1: ("texture", 0, "2d rg16i"),
    tex2: ("texture", 1, "2d r32f"),
):
    val = tex1.read(index.xy).xy  # ivec2
    val = vec2(val)  # cast to vec2
    tex2.write(index.xy, vec4(val.x + val.y, 0.0, 0.0, 0.0))


# A simple fragment shader that applies a texture to e.g. a mesh.
@python2shader
def fragment_shader_tex(
    tex: ("texture", 0, "2d f32"),
    sampler: ("sampler", 1, ""),
    tcoord: ("input", 0, vec2),
    out_color: ("output", 0, vec4),
):
    out_color = tex.sample(sampler, tcoord)  # noqa
