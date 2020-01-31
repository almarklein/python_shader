"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""


import ctypes

import python_shader
from python_shader import f32, i32, vec2, vec3, vec4, Array  # noqa

import wgpu.backend.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_equal
from testutils import validate_module, run_test_and_print_new_hashes


def test_index():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data2", 1, Array(i32))
        buffer.data2[input.index] = input.index

    skip_if_no_wgpu()

    inp_arrays = {}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


def test_copy():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i32))
        buffer.define("data2", 1, Array(i32))
        buffer.data2[input.index] = buffer.data1[input.index]

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_int32 * 20)(*range(20))}
    out_arrays = {1: ctypes.c_int32 * 20}
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
    "test_index.compute_shader": ("344f1e42e6addbf2", "54535768a2a9eaea"),
    "test_copy.compute_shader": ("02822d1f23bee04d", "49a4b4a0735b8d14"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
