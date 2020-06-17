"""
Minimal shaders to draw a triangle.
"""

from pyshader import python2shader, i32, vec2, vec3, vec4


@python2shader
def vertex_shader(
    index=("input", "VertexId", i32),
    out_pos=("output", "Position", vec4),
    out_color=("output", 0, vec3),
):
    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[index]
    out_pos = vec4(p, 0.0, 1.0)  # noqa
    out_color = vec3(p, 0.5)  # noqa


@python2shader
def fragment_shader(
    color=("input", 0, vec3), out_color=("output", 0, vec4),
):
    out_color = vec4(color, 1.0)  # noqa
