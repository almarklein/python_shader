"""
Compile a shader written in Python to SpirV, and show the SpirV disassembly.
"""

from python_shader import python2shader, i32, vec2, vec3, vec4


@python2shader
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


# Get the raw bytes
raw_spirv = vertex_shader.to_spirv()

# For developers: uncomment the lines below to validate and read the SpirV.
# Note that thsese requires the Vulkan SDK!

# from python_shader import dev
# dev.disassemble(vertex_shader)
# dev.validate(vertex_shader)
