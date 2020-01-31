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
from python_shader import i32, vec2, vec3, vec4, Array

from pytest import mark
from testutils import can_use_vulkan_sdk, validate_module, run_test_and_print_new_hashes


def test_null_shader():
    @python2shader_and_validate
    def vertex_shader(input, output):
        pass


def test_triangle_shader():
    @python2shader_and_validate
    def vertex_shader(input, output):
        input.define("index", "VertexId", i32)
        output.define("pos", "Position", vec4)
        output.define("color", 0, vec3)

        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

        p = positions[input.index]
        output.pos = vec4(p, 0.0, 1.0)
        output.color = vec3(p, 0.5)

    @python2shader_and_validate
    def fragment_shader(input, output):
        input.define("color", 0, vec3)
        output.define("color", 0, vec4)

        output.color = vec4(input.color, 1.0)


@mark.skipif(not can_use_vulkan_sdk, reason="No Vulkan SDK")
def test_no_duplicate_constants():
    def vertex_shader():
        positions = [vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 1.0)]  # noqa

    m = python_shader.python2shader(vertex_shader)
    text = python_shader.dev.disassemble(m.to_spirv())
    assert 2 <= text.count("OpConst") <= 3


def test_compute_shader():
    @python2shader_and_validate
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i32))
        buffer.define("data2", 1, Array(i32))

        buffer.data2[input.index] = buffer.data1[input.index]


# %% Utils for this module


def python2shader_and_validate(func):
    m = python_shader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


HASHES = {
    "test_null_shader.vertex_shader": ("801b48feb24d7aea", "1cd71e87b1ba5a32"),
    "test_triangle_shader.vertex_shader": ("9cdef2b9fc3befa3", "ef7285ff8c51a8f1"),
    "test_triangle_shader.fragment_shader": ("02bcc64a3e8b05d3", "33063b6db28277db"),
    "test_compute_shader.compute_shader": ("02822d1f23bee04d", "3a3e6992643df7ea"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
