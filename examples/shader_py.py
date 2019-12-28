"""
Compile a shader written in Python to SpirV, and show the SpirV disassembly.
"""

from spirv import python2spirv, i32, vec2, vec3, vec4


@python2spirv
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


# You can uncomment these, but they need spirv-tools to be installed
# vertex_shader.validate()
# vertex_shader.disassble()

# Similar output as disassble(), but derived from our own internal state
print(vertex_shader.gen.to_text())

# Get the raw bytes
raw_spirv = vertex_shader.to_bytes()
