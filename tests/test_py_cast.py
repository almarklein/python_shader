"""
Tests related to casting and vector/array composition.
"""


import ctypes

import python_shader
from python_shader import RES_INPUT, RES_BUFFER
from python_shader import InputResource, BufferResource
from python_shader import f32, f64, u8, i16, i32, i64  # noqa
from python_shader import bvec2, ivec2, ivec3, vec2, vec3, vec4, Array  # noqa

import wgpu.backend.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_equal
from testutils import validate_module, run_test_and_print_new_hashes


# todo: this first test uses a 3-tuple to specify IO, the rest uses resource object. Pick one!
def test_cast_i32_f32():
    @python2shader_and_validate
    def compute_shader(
        index: (RES_INPUT, "GlobalInvocationId", i32),
        data1: (RES_BUFFER, 0, Array(i32)),
        data2: (RES_BUFFER, 1, Array(f32)),
    ):
        data2[index] = f32(data1[index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 0, 4, 100, 32767, 32768, 999999]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values1)


def test_cast_u8_f32():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(u8)),
        data2: BufferResource(1, Array(f32)),
    ):
        data2[index] = f32(data1[index])

    skip_if_no_wgpu()

    values1 = [0, 1, 4, 127, 128, 255]

    inp_arrays = {0: (ctypes.c_ubyte * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values1)


def test_cast_f32_i32():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(f32)),
        data2: BufferResource(1, Array(i32)),
    ):
        data2[index] = i32(data1[index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_f32_f32():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(f32)),
        data2: BufferResource(1, Array(f32)),
    ):
        data2[index] = f32(data1[index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_f32_f64():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(f32)),
        data2: BufferResource(1, Array(f64)),
    ):
        data2[index] = f64(data1[index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 20)(*range(20))}
    out_arrays = {1: ctypes.c_double * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], range(20))


def test_cast_i64_i16():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(i64)),
        data2: BufferResource(1, Array(i16)),
    ):
        data2[index] = i16(data1[index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 0, 4, 100, 32767, 32768, 999999]
    values2 = [-16959, -100, -4, 0, 4, 100, 32767, -32768, 16959]

    inp_arrays = {0: (ctypes.c_longlong * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_short * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    assert iters_equal(out[1], values2)


def test_cast_i16_u8():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(i16)),
        data2: BufferResource(1, Array(u8)),
    ):
        data2[index] = u8(data1[index])

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
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(ivec2)),
        data2: BufferResource(1, Array(vec2)),
    ):
        data2[index] = vec2(data1[index])

    skip_if_no_wgpu()

    values1 = [-999999, -100, -4, 1, 4, 100, 32767, 32760, 0, 999999]

    inp_arrays = {0: (ctypes.c_int32 * len(values1))(*values1)}
    out_arrays = {1: ctypes.c_float * len(values1)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=5)
    assert iters_equal(out[1], values1)


def test_cast_vec_any_vec4():
    # Look how all args in a vector are converted :)
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data2: BufferResource(1, Array(vec4)),
    ):
        data2[index] = vec4(7.0, 3, ivec2(False, 2.7))

    skip_if_no_wgpu()

    values2 = [7.0, 3.0, 0.0, 2.0] * 2

    inp_arrays = {}
    out_arrays = {1: ctypes.c_float * len(values2)}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=2)
    assert iters_equal(out[1], values2)


def test_cast_vec_ivec3_vec3():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(ivec3)),
        data2: BufferResource(1, Array(vec3)),
    ):
        data2[index] = vec3(data1[index])

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
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(ivec2)),
        data2: BufferResource(1, Array(ivec2)),
    ):
        tmp = bvec2(data1[index])
        data2[index] = ivec2(tmp)  # ext visible storage cannot be bool

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
    "test_cast_i32_f32.compute_shader": ("650afceb4c0ce63c", "568470cadb12744f"),
    "test_cast_u8_f32.compute_shader": ("11a73d85333ae2c0", "d9c0798a59e9d3d9"),
    "test_cast_f32_i32.compute_shader": ("5f5a096fed9545cd", "ac7437359946a691"),
    "test_cast_f32_f32.compute_shader": ("079d2619043e5c76", "5753fa9dde151f3f"),
    "test_cast_f32_f64.compute_shader": ("1babbd427a5e9f2f", "7653c236a8c69ac8"),
    "test_cast_i64_i16.compute_shader": ("2e5e0da8e448810a", "f1fbaeb20d9d021e"),
    "test_cast_i16_u8.compute_shader": ("db5bc49702e17398", "19bc9aaed0e9cf09"),
    "test_cast_vec_ivec2_vec2.compute_shader": ("90aff2369c3790b7", "347b16127223b904"),
    "test_cast_vec_any_vec4.compute_shader": ("4c449d6964bdcc8d", "4d45d1589bce0b3e"),
    "test_cast_vec_ivec3_vec3.compute_shader": ("afafda12012f0054", "2833169c567e1c65"),
    "test_cast_ivec2_bvec2.compute_shader": ("88f3f231809bff20", "f1824ce347d142e0"),
}

if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
