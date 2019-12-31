import python_shader
from python_shader import i32, vec2, vec3, vec4


def test_shader01():
    return  # todo: broken

    def vertex_shader(input, output):
        input.define("index", "VertexId", i32)
        output.define("pos", "Position", vec4)
        output.define("color", 0, vec3)

        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

        p = positions[input.index]
        output.pos = vec4(p, 0.0, 1.0)
        output.color = vec3(p, 0.5)

    m = python_shader.python2shader(vertex_shader)
    assert m.input is vertex_shader
    assert isinstance(m.to_spirv(), bytes)
