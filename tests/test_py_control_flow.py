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
from testutils import can_use_wgpu_lib
from testutils import validate_module, run_test_and_print_new_hashes


# todo: test_tertiary_op


def test_if1():
    # Simple
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        if index < 2:
            data2[index] = 40.0
        elif index < 4:
            data2[index] = 41.0
        elif index < 8:
            data2[index] = 42.0
        else:
            data2[index] = 43.0

    skip_if_no_wgpu()

    values1 = [i for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res == [40, 40, 41, 41, 42, 42, 42, 42, 43, 43]


def test_if2():
    # More nesting
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        if index < 2:
            if index == 0:
                data2[index] = 40.0
            else:
                data2[index] = 41.0
        elif index < 4:
            data2[index] = 42.0
            if index > 2:
                data2[index] = 43.0
        elif index < 8:
            data2[index] = 45.0
            if index <= 6:
                if index <= 5:
                    if index == 4:
                        data2[index] = 44.0
                    elif index == 5:
                        data2[index] = 45.0
                elif index == 6:
                    data2[index] = 46.0
            else:
                data2[index] = 47.0
        else:
            if index == 9:
                data2[index] = 49.0
            else:
                data2[index] = 48.0

    skip_if_no_wgpu()

    values1 = [i for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


def test_if3():
    # And and or
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        if index < 2 or index > 7 or index == 4:
            data2[index] = 40.0
        elif index > 3 and index < 6:
            data2[index] = 41.0
        else:
            data2[index] = 43.0

    skip_if_no_wgpu()

    values1 = [i for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res == [40, 40, 43, 43, 40, 41, 43, 43, 40, 40]


def test_if4():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        a = data1[index]
        if index < 2:
            a = 100.0
        elif index < 8:
            a = a + 10.0
            if index < 6:
                a = a + 1.0
            else:
                a = a + 2.0
        else:
            a = 200.0
            if index < 9:
                a = a + 1.0
        data2[index] = a

    skip_if_no_wgpu()

    values1 = [i for i in range(10)]

    inp_arrays = {0: (ctypes.c_float * 10)(*values1)}
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    res = list(out[1])
    assert res == [100, 100, 2 + 11, 3 + 11, 4 + 11, 5 + 11, 6 + 12, 7 + 12, 201, 200]


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
    "test_add_sub.compute_shader": ("a34d4efe22c15a39", "2edf296df860a93d"),
    "test_mul_div.compute_shader": ("efa94cce5c444ff1", "3b804bb4b7b52de0"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
