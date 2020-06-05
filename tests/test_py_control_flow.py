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


# %% if


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


# %% ternary


def test_ternary1():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        data2[index] = 40.0 if index == 0 else 41.0

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 41, 41, 41, 41, 41, 41, 41, 41]


def test_ternary2():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        data2[index] = (
            40.0
            if index == 0
            else ((41.0 if index == 1 else 42.0) if index < 3 else 43.0)
        )

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 42, 43, 43, 43, 43, 43, 43, 43]


def test_ternary3():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        data2[index] = (
            (10.0 * 4.0)
            if index == 0
            else ((39.0 + 2.0) if index == 1 else (50.0 - 8.0))
        )

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 42, 42, 42, 42, 42, 42, 42, 42]


def test_ternary_with_control_flow1():
    python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = False
    try:

        @python2shader_and_validate
        def compute_shader(
            index: ("input", "GlobalInvocationId", i32),
            data2: ("buffer", 1, Array(f32)),
        ):
            data2[index] = 40.0 if index == 0 else 41.0

    finally:
        python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = True

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 41, 41, 41, 41, 41, 41, 41, 41]


def test_ternary_with_control_flow2():
    python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = False
    try:

        @python2shader_and_validate
        def compute_shader(
            index: ("input", "GlobalInvocationId", i32),
            data2: ("buffer", 1, Array(f32)),
        ):
            data2[index] = (
                40.0
                if index == 0
                else ((41.0 if index == 1 else 42.0) if index < 3 else 43.0)
            )

    finally:
        python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = True

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 42, 43, 43, 43, 43, 43, 43, 43]


def test_ternary_with_control_flow3():
    python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = False
    try:

        @python2shader_and_validate
        def compute_shader(
            index: ("input", "GlobalInvocationId", i32),
            data2: ("buffer", 1, Array(f32)),
        ):
            data2[index] = (
                (10.0 * 4.0)
                if index == 0
                else ((39.0 + 2.0) if index == 1 else (50.0 - 8.0))
            )

    finally:
        python_shader.py.OPT_CONVERT_TERNARY_TO_SELECT = True

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [40, 41, 42, 42, 42, 42, 42, 42, 42, 42]


# %% or / and


# %% loops


def test_loop1():
    # Simples form

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        for i in range(index):
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_loop2():
    # With a ternary in the body

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        for i in range(index):
            val = val + (1.0 if i < 5 else 2.0)

        data2[index] = val

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]


def test_loop3():
    # With an if in the body

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        for i in range(index):
            if i < 5:
                val = val + 1.0
            else:
                val = val + 2.0
        data2[index] = val

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]


def test_loop4():
    # A loop in a loop

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        for i in range(index):
            for j in range(3):
                val = val + 10.0
                for k in range(2):
                    val = val + 2.0
            for k in range(10):
                val = val - 1.0
        data2[index] = val

    skip_if_no_wgpu()
    out_arrays = {1: ctypes.c_float * 10}
    out = compute_with_buffers({}, out_arrays, compute_shader)
    res = list(out[1])
    assert res == [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]


# %% discard


def test_discard():

    # A fragment shader for drawing red dots
    @python2shader_and_validate
    def fragment_shader(in_coord: ("input", "PointCoord", vec2),):
        r2 = ((in_coord.x - 0.5) * 2.0) ** 2 + ((in_coord.y - 0.5) * 2.0) ** 2
        if r2 > 1.0:
            return  # discard
        out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader output

    assert ("co_return",) in fragment_shader.to_bytecode()
    assert "OpKill" in fragment_shader.gen.to_text()


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
