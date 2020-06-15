import pyshader

from pytest import raises, mark

from testutils import can_use_vulkan_sdk


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


@mark.skipif(not can_use_vulkan_sdk, reason="No Vulkan SDK")
def test_run():

    bb = pyshader.dev.glsl2spirv(vertex_code, "vertex")
    pyshader.dev.validate(bb)
    pyshader.dev.validate(FakeModule(bb))
    x1 = pyshader.dev.disassemble(bb)
    x2 = pyshader.dev.disassemble(FakeModule(bb))
    assert x1 == x2
    assert isinstance(x1, str)
    assert "Version" in x1
    assert "OpTypeVoid" in x1


@mark.skipif(not can_use_vulkan_sdk, reason="No Vulkan SDK")
def test_fails():

    # Shader type myst be vertex, fragment or compute
    with raises(ValueError):
        pyshader.dev.glsl2spirv(vertex_code, "invalid_type")

    # Code must be str
    with raises(TypeError):
        pyshader.dev.glsl2spirv(vertex_code.encode(), "vertex")

    # Code must actually glsl
    with raises(Exception):
        pyshader.dev.glsl2spirv("not valid glsls", "vertex")

    # Input must be bytes or ShaderModule-ish
    with raises(Exception):
        pyshader.dev.validate(523)
    with raises(Exception):
        pyshader.dev.disassemble(523)

    # Not valid spirv
    with raises(Exception):
        pyshader.dev.validate(b"xxxxx")

    # Cannot disassemble invalid spirv
    with raises(Exception):
        pyshader.dev.disassemble(b"xxxxx")
