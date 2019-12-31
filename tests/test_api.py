import python_shader


def test_api():
    assert isinstance(python_shader.__version__, str)
    assert isinstance(python_shader.ShaderModule, type)

    assert callable(python_shader.python2shader)
    for name in "f32 i32 vec2 vec3 vec4 ivec2 ivec3 ivec4".split():
        assert hasattr(python_shader, name)
    for name in "mat2 mat3 mat4".split():
        assert hasattr(python_shader, name)


def test_shader_module_class():

    # Create shader module object
    ShaderModule = python_shader.ShaderModule
    entrypoint = ("CO_ENTRYPOINT", ("main", "vertex", []))
    m = ShaderModule(42, [entrypoint], "stub")

    # Validate some stuff
    assert m.input == 42
    assert m.to_bytecode()[0] is entrypoint
    assert "stub" in repr(m)
    assert m.description in repr(m)

    # Generate spirv
    bb = m.to_spirv()
    assert isinstance(bb, bytes)


def test_spirv_constants():
    cc = python_shader._spirv_constants
    assert cc.AccessQualifier_ReadWrite
    assert cc.WordCountShift
    assert isinstance(repr(cc.Version), str)
    assert int(cc.Version) == cc.Version
    # assert str(int(cc.Version)) == str(cc.Version)  # not on Python 3.8 :)
