"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""


import ctypes

import python_shader
from python_shader import InputResource, BufferResource
from python_shader import f32, i32, vec2, vec3, vec4, Array  # noqa

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_equal
from testutils import validate_module, run_test_and_print_new_hashes


def test_index():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data2: BufferResource(1, Array(i32)),
    ):
        data2[index] = index

    skip_if_no_wgpu()

    inp_arrays = {}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


def test_copy():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(i32)),
        data2: BufferResource(1, Array(i32)),
    ):
        data2[index] = data1[index]

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
    "test_index.compute_shader": ("5b4be829de8c83e5", "e677b4cc6992630a"),
    "test_copy.compute_shader": ("7b03b3564a72be3c", "46f084870ce2681b"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
