import spirv

from pytest import raises


vertex_code = """
#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 positions[3] = vec2[3](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

}
""".lstrip()


class FakeModule:

    def __init__(self, spirv):
        self._spirv = spirv

    def to_spirv(self):
        return self._spirv


def test_run():

    bb = spirv.dev.glsl2spirv(vertex_code, "vertex")
    spirv.dev.validate(bb)
    spirv.dev.validate(FakeModule(bb))
    x1 = spirv.dev.disassemble(bb)
    x2 = spirv.dev.disassemble(FakeModule(bb))
    assert x1 == x2
    assert isinstance(x1, str)
    assert "Version" in x1
    assert "OpTypeVoid" in x1


def test_fails():

    # Shader type myst be vertex, fragment or compute
    with raises(ValueError):
        spirv.dev.glsl2spirv(vertex_code, "invalid_type")

    # Code must be str
    with raises(TypeError):
        spirv.dev.glsl2spirv(vertex_code.encode(), "vertex")

    # Code must actually glsl
    with raises(Exception):
        spirv.dev.glsl2spirv("not valid glsls", "vertex")

    # Input must be bytes or ShaderModule-ish
    with raises(Exception):
        spirv.dev.validate(523)
    with raises(Exception):
        spirv.dev.disassemble(523)

    # Not valid spirv
    with raises(Exception):
        spirv.dev.validate(b"xxxxx")

    # Cannot disassemble invalid spirv
    with raises(Exception):
        spirv.dev.disassemble(b"xxxxx")
