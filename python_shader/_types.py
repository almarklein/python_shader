"""
Types on the GPU (from SpirV spec):

* Basic types are boolean, int, float. Latter two are numerics, all three are scalars.
* Vector is two or more values of scalars (float, int, bool). For lengt > 4 need capabilities.
* Matrix is 2, 3, or 4 float vectors (each vector is a column).
* Array is homogeneous collection of non-void-type objects.
* Structure is heterogeneous collection of non-void-type objects.
* image, sampler, ...

Here we follow the SpirV type hierarchy. We define abstract types, which
can be specialized (made concrete) by calling them. By convention,
abstract types start with a capital letter, concrete types are lowercase.

"""

import ctypes


_subtypes = {}


def _create_type(name, base, props):
    """ Create a new type, memoize on name.
    """
    if name not in _subtypes:
        assert not props.get("is_abstract", True), "can only create concrete types"
        _subtypes[name] = type(name, (base,), props)
    return _subtypes[name]


def type_from_name(name):
    """ Get a ShaderType from its name.
    """
    original_name = name
    name = name.replace(" ", "").lower()
    return _type_from_name(name, original_name)


def _type_from_name(name, original_name):
    if name in _subtypes:
        return _subtypes[name]
    elif name.startswith("vector"):
        inner, commas = _select_between_braces(name[6:], original_name)
        assert len(commas) == 1
        n, _, subtypestr = inner.partition(",")
        subtype = _type_from_name(subtypestr, original_name)
        return Vector(int(n), subtype)
    elif name.startswith("matrix"):
        inner, commas = _select_between_braces(name[6:], original_name)
        assert len(commas) == 2
        cols, rows, subtypestr = inner.split(",", 2)
        subtype = _type_from_name(subtypestr, original_name)
        return Matrix(int(cols), int(rows), subtype)
    elif name.startswith("array"):
        inner, commas = _select_between_braces(name[5:], original_name)
        assert len(commas) in (0, 1)
        if len(commas) == 0:  # subtypestr = inner
            subtype = _type_from_name(inner, original_name)
            return Array(subtype)
        else:
            n, _, subtypestr = inner.partition(",")
            subtype = _type_from_name(subtypestr, original_name)
            return Array(int(n), subtype)
    elif name.startswith("struct"):
        inner, commas = _select_between_braces(name[6:], original_name)
        commas.insert(0, -1)
        commas.append(999999)
        parts = [inner[commas[i] + 1 : commas[i + 1]] for i in range(len(commas) - 1)]
        fields = {}
        for part in [part for part in parts if part]:
            key, _, subtypestr = part.partition("=")
            fields[key.strip()] = _type_from_name(subtypestr, original_name)
        return Struct(**fields)
    else:
        raise TypeError(f"Invalid ShaderType string '{original_name}': '{name}'")


def _select_between_braces(s, original_name):
    """ Assuming s starts with an opening brace, return the part between
    braces and the position of the comma's at the root level.
    """
    assert s[0] == "("
    level = 0
    commas = []
    for i in range(len(s)):
        if s[i] == "(":
            level += 1
        elif s[i] == ")":
            level -= 1
            if level == 0:
                break
        elif level == 1 and s[i] == ",":
            commas.append(i - 1)  # 1 offset because we drop the first char
    if level != 0:
        raise TypeError(f"No end-brace in ShaderType string '{original_name}': '{s}'")
    return s[1:i], commas


def shadertype_as_ctype(shadertype):
    """ Get a ctypes type equivalent to the given ShaderType.
    """
    if isinstance(shadertype, str):
        shadertype = type_from_name(shadertype)
    if not isinstance(shadertype, type) and issubclass(shadertype, ShaderType):
        raise TypeError("Expected str or ShaderType subclass.")
    if hasattr(shadertype, "_as_ctype"):
        return shadertype._as_ctype()
    else:
        return shadertype._ctype


# %% Really abstract types


class ShaderType:
    """ The root base class of all GPU types.
    """

    is_abstract = True

    def __init__(self):
        if self.is_abstract:
            name = self.__class__.__name__
            raise RuntimeError(
                f"Cannot instantiate {name} because it is an abstract class."
            )
        else:
            name = self.__class__.__name__
            raise RuntimeError(f"Cannot instantiate ShaderType subclass {name} (yet).")

    @classmethod
    def _as_ctype(cls):
        return cls._ctype


class Scalar(ShaderType):
    """ Base class for scalar types (float, int, bool).
    """


class Numeric(Scalar):
    """ Base class for numeric scalars (float, int).
    """


class Float(Numeric):
    """ Base class for float numerics (f16, f32, f64).
    """


class Int(Numeric):
    """ Base class for int numerics (i16, i32, i64).
    """


class Composite(ShaderType):
    """ Base class for composite types (Vector, Matrix, Aggregates).
    """


class Aggregate(Composite):
    """ Base class for Array and Struct types.
    """


# %% Abstract types (but can be used to construct composite types)


class Vector(Composite):
    """ Base class for Vector types. Concrete types are templated based on
    length and subtype.
    """

    subtype = None
    length = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) != 2:
                raise TypeError(
                    "Vector specialization needs 2 args: Vector(n, subtype)"
                )
            n, subtype = args
            n = int(n)
            if not isinstance(subtype, type) and issubclass(subtype, Scalar):
                raise TypeError("Vector subtype must be a Scalar type.")
            elif subtype.is_abstract:
                raise TypeError("Vector subtype cannot be an abstract ShaderType.")
            if n < 2 or n > 4:
                raise TypeError("Vector can have 2, 3 or 4 elements.")
            props = dict(subtype=subtype, length=n, is_abstract=False)
            return _create_type(f"Vector({n},{subtype.__name__})", Vector, props)
        else:
            return super().__new__(cls, *args)

    @classmethod
    def _as_ctype(cls):
        return cls.subtype._ctype * cls.length


class Matrix(Composite):
    """ Base class for Matrix types. Concrete types are templated based on
    cols, rows and subtype. Subtype can only be Float.
    """

    subtype = None
    cols = 0
    rows = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) != 3:
                raise TypeError(
                    "Matrix specialization needs 3 args: Matrix(cols, rows, subtype)"
                )
            cols, rows, subtype = args
            cols, rows = int(cols), int(rows)
            if not isinstance(subtype, type) and issubclass(subtype, Float):
                raise TypeError("Matrix subtype must be a Float type.")
            elif subtype.is_abstract:
                raise TypeError("Matrix subtype cannot be an abstract ShaderType.")
            if cols < 2 or cols > 4:
                raise TypeError("Matrix can have 2, 3 or 4 columns.")
            if rows < 2 or rows > 4:
                raise TypeError("Matrix can have 2, 3 or 4 rows.")
            props = dict(subtype=subtype, cols=cols, rows=rows, is_abstract=False)
            return _create_type(
                f"Matrix({cols},{rows},{subtype.__name__})", Matrix, props
            )
        else:
            return super().__new__(cls, *args)

    @classmethod
    def _as_ctype(cls):
        return cls.subtype._ctype * (cls.cols * cls.rows)  # C-contiguous


class Array(Aggregate):
    """ Base class for Array types. Concrete types are templated based on
    length and subtype. Subtype can be any ShaderType except void.
    """

    subtype = None
    length = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) == 1:
                n = 0
                subtype = args[0]
            elif len(args) == 2:
                n, subtype = args
                n = int(n)
                if n < 1:
                    raise TypeError("Array must have at least 1 element.")
            else:
                raise TypeError("Array specialization needs 2 args: Array(n, subtype)")
            # Validate
            if not isinstance(subtype, type) and issubclass(subtype, ShaderType):
                raise TypeError("Array subtype must be a ShaderType.")
            elif issubclass(subtype, void):
                raise TypeError("Array subtype cannot be void.")
            elif subtype.is_abstract:
                raise TypeError("Array subtype cannot be an abstract ShaderType.")
            props = dict(subtype=subtype, length=n, is_abstract=False)
            if n == 0:  # means it's length is unknown)
                return _create_type(f"Array({subtype.__name__})", Array, props)
            else:
                return _create_type(f"Array({n},{subtype.__name__})", Array, props)
        else:
            return super().__new__(cls, *args)

    @classmethod
    def _as_ctype(cls):
        sub_ctype = cls.subtype._as_ctype()
        return sub_ctype * cls.length


class Struct(Aggregate):
    """ Base class for Struct types. Not implemented.
    """

    def __new__(cls, **kwargs):
        if cls.is_abstract:
            n = len(kwargs)
            # Validate
            for key, subtype in kwargs.items():
                if not isinstance(subtype, type) and issubclass(subtype, ShaderType):
                    raise TypeError("Struct subtype must be a ShaderType.")
                elif issubclass(subtype, void):
                    raise TypeError("Struct subtype cannot be void.")
                elif subtype.is_abstract:
                    raise TypeError("Struct subtype cannot be an abstract ShaderType.")
                if not isinstance(
                    key, str
                ):  # and key.isidentifier(): -> allow . in name?
                    raise TypeError("Struct keys must be str.")
            # Return type
            keys = tuple(kwargs.keys())
            type_names = [
                f"{key}={subtype.__name__}" for key, subtype in kwargs.items()
            ]
            props = kwargs.copy()
            props.update(dict(length=n, keys=keys, _kwargs=kwargs, is_abstract=False))
            return _create_type(f"Struct({','.join(type_names)})", Struct, props)
        else:
            return super().__new__(cls, **kwargs)

    @classmethod
    def _as_ctype(cls):
        type_fields = [(key, val._as_ctype()) for key, val in cls._kwargs.items()]
        type_name = "C_" + cls.__name__
        return type(type_name, (ctypes.Structure,), {"_fields_": type_fields})

    @classmethod
    def get_subtype(cls, key):
        if isinstance(key, int):
            return cls._kwargs[cls.keys[key]]
        else:
            return cls._kwargs[key]


# The base types that can be used to create composite types
base_types = dict(Vector=Vector, Matrix=Matrix, Array=Array, Struct=Struct)


# %% Concrete leaf types


class void(ShaderType):
    is_abstract = False
    _ctype = ctypes.c_void_p


class boolean(Scalar):
    is_abstract = False
    _ctype = ctypes.c_bool


class f16(Float):
    is_abstract = False
    # _ctype = ctypes.c_float16 ??  maybe use uin16 and map f16 onto that data?


class f32(Float):
    is_abstract = False
    _ctype = ctypes.c_float


class f64(Float):
    is_abstract = False
    _ctype = ctypes.c_double


# For now, we simply have 3 signed ints,
# and an unsigned byte for when things need to be compact.


class u8(Int):
    is_abstract = False
    _ctype = ctypes.c_uint8


class i16(Int):
    is_abstract = False
    _ctype = ctypes.c_int16


class i32(Int):
    is_abstract = False
    _ctype = ctypes.c_int32


class i64(Int):
    is_abstract = False
    _ctype = ctypes.c_int64


# Types that are at the leaf of a composite type
leaf_types = dict(
    void=void,
    boolean=boolean,
    u8=u8,
    i16=i16,
    i32=i32,
    i64=i64,
    f16=f16,
    f32=f32,
    f64=f64,
)
_subtypes.update(leaf_types)


# %% Convenient concrete types

vec2 = Vector(2, f32)
vec3 = Vector(3, f32)
vec4 = Vector(4, f32)

ivec2 = Vector(2, i32)
ivec3 = Vector(3, i32)
ivec4 = Vector(4, i32)

bvec2 = Vector(2, boolean)
bvec3 = Vector(3, boolean)
bvec4 = Vector(4, boolean)

mat2 = Matrix(2, 2, f32)
mat3 = Matrix(3, 3, f32)
mat4 = Matrix(4, 4, f32)

mat2x2 = Matrix(2, 2, f32)
mat3x2 = Matrix(3, 2, f32)
mat4x2 = Matrix(4, 2, f32)
mat2x3 = Matrix(2, 3, f32)
mat3x3 = Matrix(3, 3, f32)
mat4x3 = Matrix(4, 3, f32)
mat2x4 = Matrix(2, 4, f32)
mat3x4 = Matrix(3, 4, f32)
mat4x4 = Matrix(4, 4, f32)

convenience_types = dict(
    vec2=vec2,
    vec3=vec3,
    vec4=vec4,
    ivec2=ivec2,
    ivec3=ivec3,
    ivec4=ivec4,
    bvec2=bvec2,
    bvec3=bvec3,
    bvec4=bvec4,
    mat2=mat2,
    mat3=mat3,
    mat2x2=mat2x2,
    mat3x2=mat3x2,
    mat4x2=mat4x2,
    mat2x3=mat2x3,
    mat3x3=mat3x3,
    mat4x3=mat4x3,
    mat2x4=mat2x4,
    mat3x4=mat3x4,
    mat4x4=mat4x4,
)
_subtypes.update(convenience_types)


# %% How to expose it all

# Types that can be referenced by name.
gpu_types_map = {}
gpu_types_map.update(leaf_types)
gpu_types_map.update(base_types)  # Only the last level, e,g. not ShaderType
gpu_types_map.update(convenience_types)


# %% Shader resource enum

RES_INPUT = "input"
RES_OUTPUT = "output"
RES_UNIFORM = "uniform"
RES_BUFFER = "buffer"
RES_SAMPLER = "sampler"
RES_TEXTURE = "texture"
