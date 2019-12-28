import spirv
import spirv.glsl

from pytest import raises


def test_api():
    assert isinstance(spirv.__version__, str)
    assert isinstance(spirv.SpirVModule, type)

    assert callable(spirv.python2spirv)
    for name in "f32 i32 vec2 vec3 vec4 ivec2 ivec3 ivec4".split():
        assert hasattr(spirv, name)
    for name in "mat2 mat3 mat4".split():
        assert hasattr(spirv, name)


def test_spirv_module_class():

    SpirVModule = spirv.SpirVModule
    m = SpirVModule(42, b"aa", "stub")
    assert m.input == 42
    assert m.to_bytes() == b"aa"
    assert "stub" in repr(m)

    # This module is not valid at all
    with raises(Exception):
        m.validate()
    with raises(Exception):
        m.disassble()


def test_spirv_constants():
    cc = spirv._spirv_constants
    assert cc.AccessQualifier_ReadWrite
    assert cc.WordCountShift
    assert isinstance(repr(cc.Version), str)
    assert isinstance(int(cc.Version), int)
    assert str(int(cc.Version)) == str(cc.Version)
