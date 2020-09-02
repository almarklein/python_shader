"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""

import os
import math
import json
import random
import ctypes

import pyshader

from pyshader import f32, i32, ivec3, vec2, vec4, Array  # noqa

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_close
from testutils import validate_module, run_test_and_print_new_hashes


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# %% Builtin math


def test_add_sub1():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec2(a + 1.0, a - 1.0)

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [i + 1 for i in values1]
    assert res[1::2] == [i - 1 for i in values1]


def test_add_sub2():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec2(a + 1.0, a - 1.0) + 20.0

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [20.0 + i + 1 for i in values1]
    assert res[1::2] == [20.0 + i - 1 for i in values1]


def test_add_sub3():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        a -= -1.0
        b = vec2(a, a)
        b += 2.0
        data2[index] = b

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [i + 3 for i in values1]
    assert res[1::2] == [i + 3 for i in values1]


def test_mul_div1():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec2(a * 2.0, a / 2.0)

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [i * 2 for i in values1]
    assert res[1::2] == [i / 2 for i in values1]


def test_mul_div2():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = 2.0 * vec2(a * 2.0, a / 2.0) * 3.0

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [6 * i * 2 for i in values1]
    assert res[1::2] == [6 * i / 2 for i in values1]


def test_mul_div3():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        a /= -1.0
        b = vec2(a, a)
        b *= 2.0
        data2[index] = b

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [-i * 2 for i in values1]
    assert res[1::2] == [-i * 2 for i in values1]


def test_mul_dot():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        a = vec2(data1[index], data1[index])
        data2[index] = a @ a

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res == [i ** 2 * 2 for i in values1]


def test_integer_div():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 1, Array(i32)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = 12 // a

    skip_if_no_wgpu()

    values1 = [(i - 5) or 12 for i in range(10)]

    inp_arrays = {0: (ctypes.c_int * 10)(*values1)}
    out_arrays = {1: ctypes.c_int * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    # NOTE: the shader // truncates, not floor like Python
    res = list(out[1])
    assert res == [math.trunc(12 / i) for i in values1]


def test_mul_modulo():
    # There are two module functions, one in which the result takes the sign
    # of the divisor and one in which it takes the sign of the divident.
    # In Python these are `%` and math.fmod respectively. Here we test that
    # the SpirV code matches that (fmod and frem).
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(vec2)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec2(a.x % a.y, math.fmod(a.x, a.y))

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]
    values2 = [-2 if i % 2 else 2 for i in range(10)]
    values = sum(zip(values1, values2), ())

    inp_arrays = {0: (ctypes.c_float * 20)(*values)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::2] == [i % j for i, j in zip(values1, values2)]
    assert res[1::2] == [math.fmod(i, j) for i, j in zip(values1, values2)]


def test_math_constants():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        if index % 2 == 0:
            data2[index] = math.pi
        else:
            data2[index] = math.e

    skip_if_no_wgpu()

    out = compute_with_buffers({}, {1: ctypes.c_float * 10}, compute_shader, n=10)

    res = list(out[1])
    assert iters_close(res, [math.pi, math.e] * 5)


# %% Extension functions

# We test a subset; we test the definition of all functions in test_ext_func_definitions


def test_pow():
    # note hat a**2 is converted to a*a and a**0.5 to sqrt(a)
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec4)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec4(a ** 2, a ** 0.5, a ** 3.0, a ** 3.1)

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 40}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res[0::4] == [i ** 2 for i in values1]
    assert iters_close(res[1::4], [i ** 0.5 for i in values1])
    assert res[2::4] == [i ** 3 for i in values1]
    assert iters_close(res[3::4], [i ** 3.1 for i in values1])


def test_sqrt():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec4)),
    ):
        index = index_xyz.x
        a = data1[index]
        data2[index] = vec4(a ** 0.5, math.sqrt(a), stdlib.sqrt(a), 0.0)

    skip_if_no_wgpu()

    values1 = [i for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 40}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    ref = [i ** 0.5 for i in values1]
    assert iters_close(res[0::4], ref)
    assert iters_close(res[1::4], ref)
    assert iters_close(res[2::4], ref)


def test_length():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(vec2)),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        data2[index] = length(data1[index])

    skip_if_no_wgpu()

    values1 = [random.uniform(-2, 2) for i in range(20)]

    inp_arrays = {0: (ctypes.c_float * 20)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=10)

    res = list(out[1])
    ref = [(values1[i * 2] ** 2 + values1[i * 2 + 1] ** 2) ** 0.5 for i in range(10)]
    assert iters_close(res, ref)


def test_normalize():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
        index = index_xyz.x
        v = data1[index]
        data2[index] = normalize(vec2(v, v))

    skip_if_no_wgpu()

    values1 = [i - 5 for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=10)

    res = list(out[1])
    assert iters_close(res[:10], [-(2 ** 0.5) / 2 for i in range(10)])
    assert iters_close(res[-8:], [+(2 ** 0.5) / 2 for i in range(8)])
    assert math.isnan(res[10]) and math.isnan(res[11])  # or can this also be inf?


# %% Extension functions that need more care

# Mostly because they operate on more types than just float and vec.
# We'll want to test all "hardcoded" functions here.


def test_abs():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(i32)),
        data3: ("buffer", 2, Array(vec2)),
    ):
        index = index_xyz.x
        v1 = abs(data1[index])  # float
        v2 = abs(data2[index])  # int
        data3[index] = vec2(f32(v1), v2)

    skip_if_no_wgpu()

    values1 = [random.uniform(-2, 2) for i in range(10)]
    values2 = [random.randint(-100, 100) for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1), 1: (ctypes.c_int * 10)(*values2)}
    out_arrays = {2: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=10)

    res = list(out[2])
    assert iters_close(res[0::2], [abs(v) for v in values1])
    assert res[1::2] == [abs(v) for v in values2]


def test_min_max_clamp():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(vec4)),
        data2: ("buffer", 1, Array(vec4)),
        data3: ("buffer", 2, Array(vec4)),
    ):
        index = index_xyz.x
        v = data1[index].x
        mi = data1[index].y
        ma = data1[index].z

        data2[index] = vec4(min(v, ma), max(v, mi), clamp(v, mi, ma), 0.0)
        data3[index] = vec4(nmin(v, ma), nmax(v, mi), nclamp(v, mi, ma), 0.0)

    skip_if_no_wgpu()

    the_vals = [-4, -3, -2, -1, +0, +0, +1, +2, +3, +4]
    min_vals = [-2, -5, -5, +2, +2, -1, +3, +1, +1, -6]
    max_vals = [+2, -1, -3, +3, +3, +1, +9, +9, +2, -3]
    stubs = [0] * 10
    values = sum(zip(the_vals, min_vals, max_vals, stubs), ())

    inp_arrays = {0: (ctypes.c_float * 40)(*values)}
    out_arrays = {1: ctypes.c_float * 40, 2: ctypes.c_float * 40}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=10)

    res1 = list(out[1])
    res2 = list(out[2])
    ref_min = [min(the_vals[i], max_vals[i]) for i in range(10)]
    ref_max = [max(the_vals[i], min_vals[i]) for i in range(10)]
    ref_clamp = [min(max(min_vals[i], the_vals[i]), max_vals[i]) for i in range(10)]
    # Test normal variant
    assert res1[0::4] == ref_min
    assert res1[1::4] == ref_max
    assert res1[2::4] == ref_clamp
    # Test NaN-safe variant
    assert res2[0::4] == ref_min
    assert res2[1::4] == ref_max
    assert res2[2::4] == ref_clamp


def test_mix():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(vec4)),
        data2: ("buffer", 1, Array(vec4)),
    ):
        index = index_xyz.x
        v = data1[index]
        v1 = mix(v.x, v.y, v.z)
        v2 = mix(vec2(v.x, v.x), vec2(v.y, v.y), v.z)
        data2[index] = vec4(v1, v2.x, v2.y, 0.0)

    skip_if_no_wgpu()

    values1 = [-4, -3, -2, -1, +0, +0, +1, +2, +3, +4]
    values2 = [-2, -5, -5, +2, +2, -1, +3, +1, +1, -6]
    weights = [0.1 * i for i in range(10)]
    stubs = [0] * 10
    values = sum(zip(values1, values2, weights, stubs), ())

    inp_arrays = {0: (ctypes.c_float * 40)(*values)}
    out_arrays = {1: ctypes.c_float * 40}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=10)

    res = list(out[1])
    ref = [values1[i] * (1 - w) + values2[i] * w for i, w in enumerate(weights)]
    assert iters_close(res[0::4], ref)
    assert iters_close(res[1::4], ref)
    assert iters_close(res[2::4], ref)


# %% Extension function definitions


def test_ext_func_definitions():
    # The above tests touch a subset of all extension functions.
    # This test validates that the extension functions that we define
    # in stdlib.py have the correct enum nr and number of arguments.

    # Prepare meta data about instructions
    instructions = {}
    with open(os.path.join(THIS_DIR, "extinst.glsl.std.450.grammar.json"), "r") as f:
        meta = json.load(f)
    for x in meta["instructions"]:
        normalized_name = x["opname"].lower()
        instructions[normalized_name] = x["opcode"], len(x["operands"])

    # Check each function
    count = 0
    for name, info in pyshader.stdlib.ext_functions.items():
        if not info:
            continue  # skip the hardcoded functions
        normalized_name = name.replace("_", "")
        if normalized_name not in instructions:
            normalized_name = "f" + normalized_name
        assert normalized_name in instructions, f"Could not find meta data for {name}()"
        nr, nargs = instructions[normalized_name]
        assert (
            info["nr"] == nr
        ), f"Invalid enum nr for {name}: {info['nr']} instead of {nr}"
        assert (
            info["nargs"] == nargs
        ), f"Invalud nargs for {name}: {info['nargs']} instead of {nargs}"
        count += 1

    print(f"Validated {count} extension functions!")


# %% Utils for this module


def python2shader_and_validate(func):
    m = pyshader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


def skip_if_no_wgpu():
    if not can_use_wgpu_lib:
        raise pytest.skip(msg="SpirV validated, but not run (cannot use wgpu)")


HASHES = {
    "test_add_sub1.compute_shader": ("0527d3b9170a0d7f", "b1e717b91b81e8cd"),
    "test_add_sub2.compute_shader": ("06041924ea937b3b", "77e95233b349c31d"),
    "test_add_sub3.compute_shader": ("1a5b936748fe67f2", "a5cfa3f7fe9d3686"),
    "test_mul_div1.compute_shader": ("44839ef53adee679", "e7514d6513dc2bb0"),
    "test_mul_div2.compute_shader": ("2739b4d6acc07ce0", "f0e5fc19e10f859e"),
    "test_mul_div3.compute_shader": ("00af6e8bc63a91c9", "ddc0adcfdb3db901"),
    "test_mul_dot.compute_shader": ("7685b289189dacc8", "578228eee09a367a"),
    "test_integer_div.compute_shader": ("81f060cf8490a1d5", "26e519038f4d721f"),
    "test_mul_modulo.compute_shader": ("e6959fb01f225afa", "1adb9e2258f75084"),
    "test_math_constants.compute_shader": ("a1e6b93df962fb0c", "14bafe94b1c0a731"),
    "test_pow.compute_shader": ("24ffb3a4eb70238c", "6a6add70fa8df1f9"),
    "test_sqrt.compute_shader": ("e8bb1d3bcc8195fb", "03bf1d04c4eae6ac"),
    "test_length.compute_shader": ("9c2fde7273986a94", "1d8127d6851f3a5f"),
    "test_normalize.compute_shader": ("e8baaba5e7d45866", "bff9f990b48b5a58"),
    "test_abs.compute_shader": ("125349dd0de21476", "741cdb6a809fa2cb"),
    "test_min_max_clamp.compute_shader": ("c8120f18a3ef642b", "2251056b61e2137f"),
    "test_mix.compute_shader": ("4ce78e40035a5426", "337f4057b1278a63"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
