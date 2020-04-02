"""
Some shaders that could be used to render a mesh
"""

from python_shader import python2shader, vec3, vec4, mat4


@python2shader
def vertex_shader(
    vertex_pos: ("input", 0, vec3),
    transform: ("uniform", (0, 0), mat4),
    out_pos: ("output", "Position", vec4),
):
    out_pos = transform * vec4(vertex_pos, 1.0)  # noqa


# A fragment shader that applies a uniform color to the mesh
@python2shader
def fragment_shader_flat(
    color: ("uniform", (0, 1), vec3), out_color: ("output", 0, vec4),
):
    out_color = vec4(color, 1.0)  # noqa
