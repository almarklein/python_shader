"""
Compile a wasl shader to SpirV, validate it, and show the SpirV disassembly.

WASL is experimental, currently broken, and probably being deprecated :)
"""


from pyshader.wasl import wasl2shader

vertex_code = """
fn main (
    index: input i32 VertexId,  # VertexID or VertexIndex
    pos: output vec4 Position,
) {

    positions = Array(
        vec2(+0.0, -0.5),
        vec2(+0.5, +0.5),
        vec2(-0.5, +0.5),
    )

    pos = vec4(positions[index], 0.0, 1.0)
}
""".lstrip()

module = wasl2shader(vertex_code, "vert")

print(module.to_spirv())
