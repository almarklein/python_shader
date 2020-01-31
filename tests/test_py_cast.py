"""
Tests related to casting and vector/array composition.
"""


import ctypes

import python_shader
from python_shader import f32, f64, i16, i32, i64, vec2, vec3, vec4, Array  # noqa

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

    inp_arrays = {0: (ctypes.c_int32 * 20)(*range(20))}
    out_arrays = {1: ctypes.c_float * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


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


def test_cast_i16_i64():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i16))
        buffer.define("data2", 1, Array(i64))
        buffer.data2[input.index] = i64(buffer.data1[input.index])

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_short * 20)(*range(20))}
    out_arrays = {1: ctypes.c_longlong * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


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
    "test_cast_i32_f32.compute_shader": ("60408a1ad7fe1e52", "f4085ece0c193fc7"),
    "test_cast_f32_i32.compute_shader": ("4e506c8a9685bb1d", "77c6b09212a45247"),
    "test_cast_f32_f32.compute_shader": ("41ec80be0c5fbd4a", "18c97b2740ef4da1"),
    "test_cast_f32_f64.compute_shader": ("02b0ba02f800c7e7", "93cbd40879309ea2"),
    "test_cast_i16_i64.compute_shader": ("864957a7eefe218f", "183f98e77568fec0"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
