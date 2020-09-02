"""
Tests for the Python to SpirV compiler chain.

These tests validate that the Python bytecode to our internal bytecode
is consistent between Python versions and platforms. This is important
because the Python bytecode is not standardised.

These tests also validate that the (internal) bytecode to SpirV compilation
is consistent, and (where possible) validates the SpirV using spirv-val.

Consistency is validated by means of hashes (of the bytecode and SpirV)
which are present at the bottom of this module. Run this module as a
script to get new hashes when needed:

    * When the compiler is changed in a way to produce different results.
    * When tests are added or changed.

"""

import ctypes

import pyshader
from pyshader import stdlib, f32, i32, vec2, vec3, vec4, ivec3, ivec4, Array

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from pytest import mark, raises
from testutils import can_use_vulkan_sdk, can_use_wgpu_lib
from testutils import validate_module, run_test_and_print_new_hashes


def test_null_shader():
    @python2shader_and_validate
    def vertex_shader():
        pass


def test_triangle_shader():
    @python2shader_and_validate
    def vertex_shader(
        index: ("input", "VertexId", i32),
        pos: ("output", "Position", vec4),
        color: ("output", 0, vec3),
    ):
        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
        p = positions[index]
        pos = vec4(p, 0.0, 1.0)  # noqa
        color = vec3(p, 0.5)  # noqa

    @python2shader_and_validate
    def fragment_shader(
        in_color: ("input", 0, vec3),
        out_color: ("output", 0, vec4),
    ):
        out_color = vec4(in_color, 1.0)  # noqa


@mark.skipif(not can_use_vulkan_sdk, reason="No Vulkan SDK")
def test_no_duplicate_constants():
    def vertex_shader():
        positions = [vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 1.0)]  # noqa

    m = pyshader.python2shader(vertex_shader)
    text = pyshader.dev.disassemble(m.to_spirv())
    # One for 1.0, one for 0.0
    assert text.count("OpConstant %float") == 2
    # One for the vector, one for the array
    assert text.count("OpConstantComposite") == 2


def test_compute_shader():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 1, Array(i32)),
    ):
        data2[index.x] = data1[index.x]


def test_cannot_assign_same_slot():
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 0, Array(i32)),
    ):
        data2[index.x] = data1[index.x]

    with raises(pyshader.ShaderError) as err:
        pyshader.python2shader(compute_shader).to_spirv()
    assert "already taken" in str(err.value)


def test_texture_2d_f32():
    # This shader can be used with float and int-norm texture formats

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, vec2),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "2d f32"),
        sampler: ("sampler", (0, 2), ""),
    ):
        outcolor = tex.sample(sampler, texcoord)  # noqa


def test_texture_1d_i32():
    # This shader can be used with non-norm integer texture formats

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, f32),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "1d i32"),
        sampler: ("sampler", (0, 2), ""),
    ):
        outcolor = vec4(tex.sample(sampler, texcoord))  # noqa


def test_texture_3d_r16i():
    # This shader explicitly specifies r16i format

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, vec3),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "3d r16i"),
        sampler: ("sampler", (0, 2), ""),
    ):
        # outcolor = vec4(tex.sample(sampler, texcoord))  # noqa
        outcolor = vec4(stdlib.sample(tex, sampler, texcoord))  # noqa


def test_texcomp_2d_rg32i():
    # compute shaders always need the format speci

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "2d rg32i"),
    ):
        color = tex.read(index.xy)
        color = ivec4(color.x + 1, color.y + 2, color.z + 3, color.a + 4)
        tex.write(index.xy, color)


def test_tuple_unpacking():
    # Python implementations deal with tuple packing/unpacking differently.
    # Python 3.8+ has rot_four, pypy3 resolves by changing the order of the
    # store ops in the bytecode itself, and seems to even ditch unused variables.
    @python2shader_and_validate_nobc
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, "Array(vec2)"),
    ):
        i = f32(index.x)
        a, b = 1.0, 2.0  # Cover Python storing this as a tuple const
        c, d = a + i, b + 1.0
        c, d = d, c
        c += 100.0
        c, d = d, c
        c += 200.0
        c, d, _ = c, d, 0.0  # 3-tuple
        c, d, _, _ = c, d, 0.0, 0.0  # 4-tuple
        c, d, _, _, _ = c, d, 0.0, 0.0, 0.0  # 5-tuple
        data2[index.x] = vec2(c, d)

    skip_if_no_wgpu()

    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers({}, out_arrays, compute_shader, n=10)
    res = list(out[1])
    assert res[0::2] == [200 + i + 1 for i in range(10)]
    assert res[1::2] == [100 + 3 for i in range(10)]


# %% test fails


def test_fail_unvalid_names():
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        color = foo  # noqa

    with raises(pyshader.ShaderError) as info:
        pyshader.python2shader(compute_shader)

    # assert "color = foo" in str(info.value).lower()


def test_fail_unvalid_stlib_name():
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        color = stdlib.foo  # noqa

    with raises(pyshader.ShaderError) as info:
        pyshader.python2shader(compute_shader)

    # assert "color = stdlib.foo" in str(info.value).lower()


def test_cannot_use_unresolved_globals():
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        color = stdlib + 1.0  # noqa

    with raises(pyshader.ShaderError) as info:
        pyshader.python2shader(compute_shader)

    # assert "color = stdlib + 1.0" in str(info.value).lower()


def test_cannot_call_non_funcs():
    def compute_shader1(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "2d rg32i"),
    ):
        a = 1.0
        a(1.0)

    def compute_shader2(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "2d rg32i"),
    ):
        a = 1.0()  # noqa

    with raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader1)
    with raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader2)


def test_cannot_use_tuples_in_other_ways():
    def compute_shader1(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        v = 3.0, 4.0  # noqa

    def compute_shader2(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        a = 3.0
        b = 4.0
        v = a, b  # noqa

    def compute_shader3(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        v = vec2(3.0, 4.0)
        a, b = v

    with raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader1)

    with raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader2)

    with raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader3)


def test_cannot_add_int_and_floats():
    def compute_shader1(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        foo = 3.0
        bar = foo + index.x  # noqa

    x = pyshader.python2shader(compute_shader1)
    with raises(pyshader.ShaderError) as info:
        x.to_spirv()
    err = info.value.args[0]
    assert "source file" in err.lower()
    assert "test_py.py" in err.lower()
    assert "bar = foo + index.x" in err.lower()


def test_errror_reports_the_correct_name1():
    # Sometimes, an object can be known by multiple names ...

    def compute_shader1(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        foo = 3.0
        bar = foo  # noqa
        spam = foo + 1  # noqa

    def compute_shader2(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        foo = 3.0
        bar = foo  # noqa
        spam = bar + 1  # noqa

    with raises(pyshader.ShaderError) as info1:
        pyshader.python2shader(compute_shader1).to_spirv()
    assert "spam = foo + 1" in str(info1.value).lower()
    assert "variables: foo, 1" in str(info1.value).lower()

    with raises(pyshader.ShaderError) as info2:
        pyshader.python2shader(compute_shader2).to_spirv()
    assert "spam = bar + 1" in str(info2.value).lower()
    assert "variables: bar, 1" in str(info2.value).lower()


def test_errror_reports_the_correct_name2():
    # ... and sometimes that name is an array (a VariableAccessId internally)

    def compute_shader1(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        foo = [1, 2, 3]
        bar = foo  # noqa
        spam = foo[0] + 1.0  # noqa

    def compute_shader2(
        index: ("input", "GlobalInvocationId", ivec3),
    ):
        foo = [1, 2, 3]
        bar = foo  # noqa
        spam = bar[0] + 1.0  # noqa

    with raises(pyshader.ShaderError) as info1:
        pyshader.python2shader(compute_shader1).to_spirv()
    assert "spam = foo[0] + 1.0" in str(info1.value).lower()
    assert "variables: foo[0], 1.0" in str(info1.value).lower()

    with raises(pyshader.ShaderError) as info2:
        pyshader.python2shader(compute_shader2).to_spirv()
    assert "spam = bar[0] + 1.0" in str(info2.value).lower()
    assert "variables: bar[0], 1.0" in str(info2.value).lower()


# %% Utils for this module


def python2shader_and_validate(func):
    m = pyshader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


def python2shader_and_validate_nobc(func):
    m = pyshader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES, check_bytecode=False)
    return m


def skip_if_no_wgpu():
    if not can_use_wgpu_lib:
        raise pytest.skip(msg="SpirV validated, but not run (cannot use wgpu)")


HASHES = {
    "test_null_shader.vertex_shader": ("bc099a07b86d70f2", "a48ffae9d0f09a5c"),
    "test_triangle_shader.vertex_shader": ("000514d8367ef0fa", "928cd41a713e3878"),
    "test_triangle_shader.fragment_shader": ("6da8c966525c9c7f", "1dc83bfcd11e229b"),
    "test_compute_shader.compute_shader": ("7cf577981390626b", "47d2f371faa48ed0"),
    "test_texture_2d_f32.fragment_shader": ("564804a234e76fe1", "2fe982d3e5542180"),
    "test_texture_1d_i32.fragment_shader": ("0c1ad1a8f909c442", "7f4ad10ae75030fa"),
    "test_texture_3d_r16i.fragment_shader": ("f1069cfd9c74fa1d", "14f0b7e61c2ea4dc"),
    "test_texcomp_2d_rg32i.compute_shader": ("7dbaa7fe613cf33d", "609468500982bfbd"),
    "test_tuple_unpacking.compute_shader": ("d48f10f99c448f65", "4158f1eb0a23c9d6"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
