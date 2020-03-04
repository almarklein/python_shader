from python_shader import _types as types


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
        assert isinstance(t, type) and issubclass(t, types.GpuType)
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


if __name__ == "__main__":
    test_simple()
    test_vector()
    test_matrix()
    test_array()
    test_struct()
    test_integrity()
