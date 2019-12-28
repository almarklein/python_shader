import spirv.glsl

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

def test_api():
    assert callable(spirv.glsl.glsl2spirv)



def test_run():

    module = spirv.glsl.glsl2spirv(vertex_code, "vert")
    assert module.input == vertex_code
    module.validate()
    x = module.disassble()
    assert isinstance(x, str)
    assert "Version" in x
    assert "OpTypeVoid" in x


def test_fails():
    with raises(ValueError):  # shader type myst be vert, frag or comp
        spirv.glsl.glsl2spirv(vertex_code, "invalid_type")

    with raises(TypeError):  # code must be str
        spirv.glsl.glsl2spirv(vertex_code.encode(), "vert")

    with raises(Exception):  # code must actually glsl
        spirv.glsl.glsl2spirv("not valid glsls", "vert")

