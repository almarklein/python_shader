"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""


import ctypes

import python_shader

from python_shader import f32, i32, vec2, vec3, vec4, Array  # noqa

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_close
from testutils import validate_module, run_test_and_print_new_hashes


def test_add_sub1():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
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
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
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


def test_mul_div1():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
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
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec2)),
    ):
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


def test_pow():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(vec4)),
    ):
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
    "test_add_sub1.compute_shader": ("f5f5e1f5d546615f", "2edf296df860a93d"),
    "test_add_sub2.compute_shader": ("eac80cea3cae0305", "785f2c0acdbe0cd3"),
    "test_mul_div1.compute_shader": ("889f742ee3d3a695", "3b804bb4b7b52de0"),
    "test_mul_div2.compute_shader": ("bb5f1d05c0b02dab", "7e9591cb2d93d067"),
    "test_pow.compute_shader": ("c83ff35156e57f86", "4c41b41333f94ee9"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
