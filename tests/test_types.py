import ctypes

from python_shader import _types as types, shadertype_as_ctype

from pytest import raises


def test_simple():

    assert types.type_from_name("f32") is types.f32
    assert types.f32.__name__ == "f32"

    for t in [types.i16, types.i32, types.i64]:
        assert isinstance(t, type) and issubclass(t, types.Numeric)
        assert types.type_from_name(t.__name__) is t

    for t in [types.f16, types.f32, types.f64]:
        assert isinstance(t, type) and issubclass(t, types.Numeric)
        assert types.type_from_name(t.__name__) is t

    for t in [types.boolean, types.void]:
        assert isinstance(t, type) and issubclass(t, types.ShaderType)
        assert types.type_from_name(t.__name__) is t


def test_vector():

    assert types.type_from_name("Vector(2,f32)") is types.vec2
    assert types.vec2.__name__ == "Vector(2,f32)"

    for t in [types.vec2, types.vec3, types.vec4]:
        assert isinstance(t, type) and issubclass(t, types.Vector)
        assert types.type_from_name(t.__name__) is t

    for t in [types.ivec2, types.ivec3, types.ivec4]:
        assert isinstance(t, type) and issubclass(t, types.Vector)
        assert types.type_from_name(t.__name__) is t

    for t in [types.bvec2, types.bvec3, types.bvec4]:
        assert isinstance(t, type) and issubclass(t, types.Vector)
        assert types.type_from_name(t.__name__) is t


def test_matrix():

    assert types.type_from_name("Matrix(2,2,f32)") is types.mat2
    assert types.mat2.__name__ == "Matrix(2,2,f32)"

    for t in [types.mat2, types.mat3, types.mat4]:
        assert isinstance(t, type) and issubclass(t, types.Matrix)
        assert types.type_from_name(t.__name__) is t

    for name in ["Matrix(2,3,f32)", "Matrix(3,4,f32)", "Matrix(4,2,f32)"]:
        assert isinstance(t, type) and issubclass(t, types.Matrix)
        t = types.type_from_name(name)
        assert t.__name__ == name


def test_array():

    for n, subt in [
        (1, "f32"),
        (12, "i16"),
        (5, "Matrix(2,4,f32)"),
        (6, "Struct(foo=f32,bar=i16)"),
    ]:
        # Array with a length
        name = f"Array({n},{subt})"
        t = types.type_from_name(name)
        assert isinstance(t, type) and issubclass(t, types.Array)
        assert t.__name__ == name
        assert t.subtype.__name__ == subt
        assert t.length == n
        # Array with undefined length
        name = f"Array({subt})"
        t = types.type_from_name(name)
        assert isinstance(t, type) and issubclass(t, types.Array)
        assert t.__name__ == name
        assert t.subtype.__name__ == subt
        assert t.length == 0


def test_struct():

    for kwargs in [
        dict(),
        dict(foo=types.f32),
        dict(foo=types.i32, bar=types.Array(12, types.vec3)),
    ]:
        fields = ",".join(f"{key}={val.__name__}" for key, val in kwargs.items())
        name = f"Struct({fields})"
        t = types.type_from_name(name)
        assert isinstance(t, type) and issubclass(t, types.Struct)
        assert t.__name__ == name
        assert t.keys == tuple(kwargs.keys())
        assert set(t.keys).difference(dir(t)) == set()
        assert t.length == len(kwargs)
        for i, key in enumerate(t.keys):
            assert getattr(t, key) == kwargs[key]

    # A struct within a struct
    T = types.Struct(
        foo=types.Struct(
            spam=types.Vector(2, types.f32), eggs=types.Struct(x=types.i16, y=types.i16)
        ),
        bar=types.Array(types.f64),
    )
    name = T.__name__
    print(name)
    assert types.type_from_name(name) is T


def test_integrity():
    for name in [
        "f32",
        "boolean",
        "i64",
        "void",
        "Vector(3,f32)",
        "Matrix(2,2,f32)",
        "Array(f32)",
    ]:
        assert name in types._subtypes


def test_that_gpu_types_cannot_be_instantiated():

    # Abstract classes cannot be instantiated
    for cls in [
        types.ShaderType,
        types.Scalar,
        types.Numeric,
        types.Float,
        types.Int,
        types.Composite,
        types.Aggregate,
    ]:
        with raises(RuntimeError) as info:
            cls()
        assert "cannot instantiate" in str(info.value).lower()
        assert "abstract" in str(info.value).lower()

    # Actually, concrete classes cannot be instantiated either
    for cls in [
        types.f32,
        types.vec2,
        types.mat3,
        types.Array(2, types.f32),
        types.Struct(foo=types.f32, bar=types.vec2),
    ]:
        with raises(RuntimeError) as info:
            cls()
        assert "cannot instantiate" in str(info.value).lower()


def test_ctypes_interop():

    # Some meta-testing
    assert ctypes.c_float * 2 == ctypes.c_float * 2
    assert ctypes.c_float * 2 != ctypes.c_float * 3

    # Pre-create struct classes
    s1 = types.Struct(foo=types.f32, bar=types.vec2)
    s2 = type(
        "xxx",
        (ctypes.Structure,),
        {"_fields_": [("foo", ctypes.c_float), ("bar", ctypes.c_float * 2)]},
    )

    for shadertype, ctype1 in [
        (types.f32, ctypes.c_float),
        (types.vec2, ctypes.c_float * 2),
        (types.vec4, ctypes.c_float * 4),
        (types.mat4, ctypes.c_float * 16),
        (types.Array(12, types.ivec2), ctypes.c_int32 * 2 * 12),
        (s1, s2),
    ]:
        ctype2 = shadertype_as_ctype(shadertype)
        assert ctypes.sizeof(ctype1) == ctypes.sizeof(ctype2)
        if not issubclass(ctype1, ctypes.Structure):
            assert ctype1 == ctype2
        else:
            # For structs we cannot compare types like that
            assert ctype1._fields_ == ctype2._fields_


if __name__ == "__main__":
    test_simple()
    test_vector()
    test_matrix()
    test_array()
    test_struct()
    test_integrity()
    test_that_gpu_types_cannot_be_instantiated()
    test_ctypes_interop()
