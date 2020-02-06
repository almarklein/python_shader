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

import python_shader
from python_shader import InputResource, OutputResource, BufferResource
from python_shader import i32, vec2, vec3, vec4, Array

from pytest import mark, raises
from testutils import can_use_vulkan_sdk, validate_module, run_test_and_print_new_hashes


def test_null_shader():
    @python2shader_and_validate
    def vertex_shader():
        pass


def test_triangle_shader():
    @python2shader_and_validate
    def vertex_shader(
        index: InputResource("VertexId", i32),
        pos: OutputResource("Position", vec4),
        color: OutputResource(0, vec3),
    ):
        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
        p = positions[index]
        pos = vec4(p, 0.0, 1.0)  # noqa
        color = vec3(p, 0.5)  # noqa

    @python2shader_and_validate
    def fragment_shader(
        in_color: InputResource(0, vec3), out_color: OutputResource(0, vec4),
    ):
        out_color = vec4(in_color, 1.0)  # noqa


@mark.skipif(not can_use_vulkan_sdk, reason="No Vulkan SDK")
def test_no_duplicate_constants():
    def vertex_shader():
        positions = [vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 1.0)]  # noqa

    m = python_shader.python2shader(vertex_shader)
    text = python_shader.dev.disassemble(m.to_spirv())
    assert 2 <= text.count("OpConst") <= 3


def test_compute_shader():
    @python2shader_and_validate
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(i32)),
        data2: BufferResource(1, Array(i32)),
    ):
        data2[index] = data1[index]


def test_cannot_assign_same_slot():
    def compute_shader(
        index: InputResource("GlobalInvocationId", i32),
        data1: BufferResource(0, Array(i32)),
        data2: BufferResource(0, Array(i32)),
    ):
        data2[index] = data1[index]

    with raises(TypeError) as err:
        python_shader.python2shader(compute_shader).to_spirv()
    assert "already taken" in str(err.value)


# %% Utils for this module


def python2shader_and_validate(func):
    m = python_shader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


HASHES = {
    "test_null_shader.vertex_shader": ("bc099a07b86d70f2", "a48ffae9d0f09a5c"),
    "test_triangle_shader.vertex_shader": ("d06683f706fdcc8f", "53d4b596bc25b5a0"),
    "test_triangle_shader.fragment_shader": ("38f694941020f1cb", "6febd7dab6d72c8d"),
    "test_compute_shader.compute_shader": ("7fb8036d7f706f98", "46f084870ce2681b"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
