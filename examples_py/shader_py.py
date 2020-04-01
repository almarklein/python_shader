"""
Compile a shader written in Python to SpirV, and show the SpirV disassembly.
"""

from python_shader import python2shader, i32, vec2, vec3, vec4, Array


@python2shader
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


@python2shader
def fragment_shader(input, output):
    input.define("color", 0, vec3)
    output.define("color", 0, vec4)

    output.color = vec4(input.color, 1.0)


@python2shader
def compute_shader(input, buffer):
    input.define("index", "GlobalInvocationId", i32)
    buffer.define("data1", 0, Array(i32))
    buffer.define("data2", 1, Array(i32))

    buffer.data2[input.index] = buffer.data1[input.index]


# Get the raw bytes
raw_vert = vertex_shader.to_spirv()
raw_frag = fragment_shader.to_spirv()
raw_comp = compute_shader.to_spirv()

# For developers: uncomment the lines below to validate and read the SpirV.
# Note that thsese requires the Vulkan SDK!

# from python_shader import dev
# print(dev.disassemble(vertex_shader))
# dev.validate(vertex_shader)
