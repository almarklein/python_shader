import pyshader


def test_api():
    assert isinstance(pyshader.__version__, str)
    assert isinstance(pyshader.ShaderModule, type)

    assert callable(pyshader.python2shader)
    for name in "f32 i32 vec2 vec3 vec4 ivec2 ivec3 ivec4".split():
        assert hasattr(pyshader, name)
    for name in "mat2 mat3 mat4".split():
        assert hasattr(pyshader, name)


def test_shader_module_class():

    # Create shader module object
    ShaderModule = pyshader.ShaderModule
    entrypoint = ("CO_ENTRYPOINT", "main", "vertex", {})
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
    cc = pyshader._spirv_constants
    assert cc.AccessQualifier_ReadWrite
    assert cc.WordCountShift
    assert isinstance(repr(cc.Version), str)
    assert int(cc.Version) == cc.Version
    # assert str(int(cc.Version)) == str(cc.Version)  # not on Python 3.8 :)


def test_that_bytecode_generator_matches_opcode_definitions():
    cls1 = pyshader.opcodes.OpCodeDefinitions
    cls2 = pyshader._generator_bc.Bytecode2SpirVGenerator
    count = 0

    for name, func1 in cls1.__dict__.items():
        if not name.startswith("co_"):
            continue
        count += 1

        assert name in cls2.__dict__, f"{name} not implemented"
        func2 = cls2.__dict__[name]

        fcode1 = func1.__code__
        fcode2 = func2.__code__
        argnames1 = [fcode1.co_varnames[i] for i in range(fcode1.co_argcount)][1:]
        argnames2 = [fcode2.co_varnames[i] for i in range(fcode2.co_argcount)][1:]
        print(name)
        assert argnames1 == argnames2

    for name, func2 in cls2.__dict__.items():
        if not name.startswith("co_"):
            continue
        assert name in cls1.__dict__, f"{name} is not a known opcode"

    assert count > 12  # Just make sure we're not skipping all


def test_some_internal_apis_too():
    x = pyshader._generator_base.AnyId()
    assert "?" in repr(x)
    x.id = 23
    assert "23" in repr(x)
    x = pyshader._generator_base.AnyId("foo")
    assert "foo" in repr(x)
