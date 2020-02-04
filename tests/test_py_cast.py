"""
Tests related to casting and vector/array composition.
"""


import ctypes

import python_shader
from python_shader import f32, f64, u8, i16, i32, i64  # noqa
from python_shader import bvec2, ivec2, ivec3, vec2, vec3, vec4, Array  # noqa

import wgpu.backend.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_equal
from testutils import validate_module, run_test_and_print_new_hashes


def test_cast_i32_f32():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i32))
        buffer.define("data2", 1, Array(f32))
        buffer.data2[input.index] = f32(buffer.data1[input.index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 0, 4, 100, 32767, 32768, 999999]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values1)


def test_cast_u8_f32():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(u8))
        buffer.define("data2", 1, Array(f32))
        buffer.data2[input.index] = f32(buffer.data1[input.index])

    skip_if_no_wgpu()

    values1 = [0, 1, 4, 127, 128, 255]

    inp_arrays = {0: (ctypes.c_ubyte * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values1)


def test_cast_f32_i32():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(f32))
        buffer.define("data2", 1, Array(i32))
        buffer.data2[input.index] = i32(buffer.data1[input.index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_f32_f32():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(f32))
        buffer.define("data2", 1, Array(f32))
        buffer.data2[input.index] = f32(buffer.data1[input.index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_f32_f64():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(f32))
        buffer.define("data2", 1, Array(f64))
        buffer.data2[input.index] = f64(buffer.data1[input.index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_double * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_i64_i16():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i64))
        buffer.define("data2", 1, Array(i16))
        buffer.data2[input.index] = i16(buffer.data1[input.index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 0, 4, 100, 32767, 32768, 999999]
    values2 = [-16959, -100, -4, 0, 4, 100, 32767, -32768, 16959]

    inp_arrays = {0: (ctypes.c_longlong * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_short * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values2)


def test_cast_i16_u8():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i16))
        buffer.define("data2", 1, Array(u8))
        buffer.data2[input.index] = u8(buffer.data1[input.index])

    skip_if_no_wgpu()

    values1 = [-3, -2, -1, 0, 1, 2, 3, 127, 128, 255, 256, 300]
    values2 = [253, 254, 255, 0, 1, 2, 3, 127, 128, 255, 0, 44]

    inp_arrays = {0: (ctypes.c_short * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_ubyte * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values2)


def test_cast_vec_ivec2_vec2():
    # This triggers the direct number-vector conversion
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(ivec2))
        buffer.define("data2", 1, Array(vec2))
        buffer.data2[input.index] = vec2(buffer.data1[input.index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 1, 4, 100, 32767, 32760, 0, 999999]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=5)
    assert iters_equal(out[1], values1)


def test_cast_vec_any_vec4():
    # Look how all args in a vector are converted :)
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data2", 1, Array(vec4))
        buffer.data2[input.index] = vec4(7.0, 3, ivec2(False, 2.7))

    skip_if_no_wgpu()

    values2 = [7.0, 3.0, 0.0, 2.0] * 2

    inp_arrays = {}
    out_arrays = {1: ctypes.c_float * len(values2)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=2)
    assert iters_equal(out[1], values2)


def test_cast_vec_ivec3_vec3():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(ivec3))
        buffer.define("data2", 1, Array(vec3))
        buffer.data2[input.index] = vec3(buffer.data1[input.index])

    skip_if_no_wgpu()

    # vec3's are padded to 16 bytes! I guess it's a "feature"
    # https://stackoverflow.com/questions/38172696
    values1 = [-999999, -100, -4, 1, 4, 100, 32767, 32760, 999999]
    values2 = [-999999, -100, -4, 0, 4, 100, 32767, 0, 999999]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=3)
    assert iters_equal(out[1], values2)


def test_cast_ivec2_bvec2():
    # This triggers the per-element vector conversion
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(ivec2))
        buffer.define("data2", 1, Array(ivec2))
        tmp = bvec2(buffer.data1[input.index])
        buffer.data2[input.index] = ivec2(tmp)  # ext visible storage cannot be bool

    skip_if_no_wgpu()

    values1 = [-999999, -100, 0, 1, 4, 100, 32767, 32760, 0, 999999]
    values2 = [True, True, False, True, True, True, True, True, False, True]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_int32 * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=5)
    assert iters_equal(out[1], values2)


# %% Utils for this module


def python2shader_and_validate(func):
    m = python_shader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


def skip_if_no_wgpu():
    if not can_use_wgpu_lib:
        raise pytest.skip(msg="SpirV validated, but not run (cannot use wgpu)")


HASHES = {
    "test_cast_i32_f32.compute_shader": ("d706594ebae7e631", "3914e7dbbb8738a3"),
    "test_cast_u8_f32.compute_shader": ("7dc48f8105ca9a4f", "e3d7a1a541c4dfae"),
    "test_cast_f32_i32.compute_shader": ("d4d7bdea8afce48b", "35064513c21443a2"),
    "test_cast_f32_f32.compute_shader": ("eb95481dde049870", "5c359370f12aaf05"),
    "test_cast_f32_f64.compute_shader": ("ef2254814ec474b8", "5689829983c94b14"),
    "test_cast_i64_i16.compute_shader": ("bd23f3fc85748fdb", "fcf2872482050bfb"),
    "test_cast_i16_u8.compute_shader": ("1e9f81eb17b89e9a", "81ea5397bbfd7c86"),
    "test_cast_vec_ivec2_vec2.compute_shader": ("d45b2f3931b26a71", "0eeb980b0658a970"),
    "test_cast_vec_any_vec4.compute_shader": ("8c198d2037dedd28", "7a99f4902cfad233"),
    "test_cast_vec_ivec3_vec3.compute_shader": ("ced1656635fab68b", "376fffdc9d560eb5"),
    "test_cast_ivec2_bvec2.compute_shader": ("dfa6a74dd68aee89", "3155595307fbc43a"),
}

if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
