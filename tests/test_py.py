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
from python_shader import stdlib, f32, i32, vec2, vec3, vec4, ivec3, ivec4, Array

from pytest import mark, raises
from testutils import can_use_vulkan_sdk, validate_module, run_test_and_print_new_hashes


def test_null_shader():
    @python2shader_and_validate
    def vertex_shader():
        pass


def test_triangle_shader():
    @python2shader_and_validate
    def vertex_shader(
        index: ("input", "VertexId", i32),
        pos: ("output", "Position", vec4),
        color: ("output", 0, vec3),
    ):
        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
        p = positions[index]
        pos = vec4(p, 0.0, 1.0)  # noqa
        color = vec3(p, 0.5)  # noqa

    @python2shader_and_validate
    def fragment_shader(
        in_color: ("input", 0, vec3), out_color: ("output", 0, vec4),
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
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 1, Array(i32)),
    ):
        data2[index] = data1[index]


def test_cannot_assign_same_slot():
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 0, Array(i32)),
    ):
        data2[index] = data1[index]

    with raises(python_shader.ShaderError) as err:
        python_shader.python2shader(compute_shader).to_spirv()
    assert "already taken" in str(err.value)


def test_texture_2d_f32():
    # This shader can be used with float and int-norm texture formats

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, vec2),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "2d f32"),
        sampler: ("sampler", (0, 2), ""),
    ):
        outcolor = tex.sample(sampler, texcoord)  # noqa


def test_texture_1d_i32():
    # This shader can be used with non-norm integer texture formats

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, f32),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "1d i32"),
        sampler: ("sampler", (0, 2), ""),
    ):
        outcolor = vec4(tex.sample(sampler, texcoord))  # noqa


def test_texture_3d_r16i():
    # This shader explicitly specifies r16i format

    @python2shader_and_validate
    def fragment_shader(
        texcoord: ("input", 0, vec3),
        outcolor: ("output", 0, vec4),
        tex: ("texture", (0, 1), "3d r16i"),
        sampler: ("sampler", (0, 2), ""),
    ):
        # outcolor = vec4(tex.sample(sampler, texcoord))  # noqa
        outcolor = vec4(stdlib.sample(tex, sampler, texcoord))  # noqa


def test_texcomp_2d_rg32i():
    # compute shaders always need the format speci

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "2d rg32i"),
    ):
        color = tex.read(index.xy)
        color = ivec4(color.x + 1, color.y + 2, color.z + 3, color.a + 4)
        tex.write(index.xy, color)


# %% Utils for this module


def python2shader_and_validate(func):
    m = python_shader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


HASHES = {
    "test_null_shader.vertex_shader": ("bc099a07b86d70f2", "a48ffae9d0f09a5c"),
    "test_triangle_shader.vertex_shader": ("829ed988549d24fc", "53d4b596bc25b5a0"),
    "test_triangle_shader.fragment_shader": ("a617056d738350de", "6febd7dab6d72c8d"),
    "test_compute_shader.compute_shader": ("7b03b3564a72be3c", "46f084870ce2681b"),
    "test_texture_2d_f32.fragment_shader": ("91424c7a5253087f", "2fe982d3e5542180"),
    "test_texture_1d_i32.fragment_shader": ("ccb700086b9676d6", "7f4ad10ae75030fa"),
    "test_texture_3d_r16i.fragment_shader": ("4b7fd0d410a5ea46", "14f0b7e61c2ea4dc"),
    "test_texcomp_2d_rg32i.compute_shader": ("559fca30c0d12a98", "609468500982bfbd"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
