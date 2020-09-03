"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""


import ctypes

import pyshader

from pyshader import f32, i32, vec2, ivec3, vec3, vec4, Array  # noqa

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, can_use_vulkan_sdk
from testutils import validate_module, run_test_and_print_new_hashes


def generate_list_of_floats_from_shader(n, compute_shader):
    inp_arrays = {}
    out_arrays = {1: ctypes.c_float * n}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)
    return list(out[1])


# %% if


def test_if1():
    # Simple
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        if index < 2:
            data2[index] = 40.0
        elif index < 4:
            data2[index] = 41.0
        elif index < 8:
            data2[index] = 42.0
        else:
            data2[index] = 43.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 40, 41, 41, 42, 42, 42, 42, 43, 43]


def test_if2():
    # More nesting
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


def test_if3():
    # And and or
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        if index < 2 or index > 7 or index == 4:
            data2[index] = 40.0
        elif index > 3 and index < 6:
            data2[index] = 41.0
        else:
            data2[index] = 43.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 40, 43, 43, 40, 41, 43, 43, 40, 40]


def test_if4():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        a = f32(index)
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [100, 100, 2 + 11, 3 + 11, 4 + 11, 5 + 11, 6 + 12, 7 + 12, 201, 200]


def test_if5():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        x = False
        if index < 2:
            data2[index] = 40.0
        elif index < 4:
            data2[index] = 41.0
        elif index < 8:
            x = True
        else:
            data2[index] = 43.0
        if x:
            data2[index] = 42.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 40, 41, 41, 42, 42, 42, 42, 43, 43]


# %% ternary


def test_ternary1():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        data2[index] = 40.0 if index == 0 else 41.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 41, 41, 41, 41, 41, 41, 41, 41]


def test_ternary2():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        data2[index] = (
            40.0
            if index == 0
            else ((41.0 if index == 1 else 42.0) if index < 3 else 43.0)
        )

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 42, 43, 43, 43, 43, 43, 43, 43]


def test_ternary3():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        data2[index] = (
            (10.0 * 4.0)
            if index == 0
            else ((39.0 + 2.0) if index == 1 else (50.0 - 8.0))
        )

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 42, 42, 42, 42, 42, 42, 42, 42]


# %% more or / and


def test_andor1():
    # Implicit conversion to truth values is not supported

    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        if index < 5:
            val = f32(index - 3) and 99.0
        else:
            val = f32(index - 6) and 99.0
        data2[index] = val

    with pytest.raises(pyshader.ShaderError):
        pyshader.python2shader(compute_shader)


def test_andor2():
    # or a lot
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        if index == 2 or index == 3 or index == 5:
            data2[index] = 40.0
        elif index == 2 or index == 6 or index == 7:
            data2[index] = 41.0
        else:
            data2[index] = 43.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [43, 43, 40, 40, 43, 40, 41, 41, 43, 43]


def test_andor3():
    # and a lot
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        mod = index % 2
        if index < 4 and mod == 0:
            data2[index] = 2.0
        elif index > 5 and mod == 1:
            data2[index] = 3.0
        else:
            data2[index] = 1.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [2, 1, 2, 1, 1, 1, 1, 3, 1, 3]


def test_andor4():
    # mix it up
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        mod = index % 2
        if index < 4 and mod == 0 or index == 5:
            data2[index] = 2.0
        elif index > 5 and mod == 1 or index == 4:
            data2[index] = 3.0
        else:
            data2[index] = 1.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [2, 1, 2, 1, 3, 2, 1, 3, 1, 3]


def test_andor5():
    # in a ternary
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        mod = index % 2
        if index < 5:
            data2[index] = 40.0 if (index == 1 or index == 3 or index == 4) else 41.0
        else:
            data2[index] = 42.0 if (index > 6 and mod == 1) else 43.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [41, 40, 41, 40, 40, 43, 43, 42, 43, 42]


# %% loops


def test_loop0():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            pass
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_loop0b():
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            for j in range(index):
                pass
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_loop1():
    # Simplest form

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_loop2():
    # With a ternary in the body

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            val = val + (1.0 if i < 5 else 2.0)

        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]


def test_loop3():
    # With an if in the body

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            if i < 5:
                val = val + 1.0
            else:
                val = val + 2.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]


def test_loop4():
    # A loop in a loop

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]


def test_loop5a():
    # Break - this one is interesting because the stop criterion is combined with the break
    # This is a consequence of the logic to detect and simplify or-logic

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            if i == 7:
                break
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]


def test_loop5b():
    # Another break (a case fixed in #51)

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            val = val + 1.0
            if i == 7:
                break
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]


def test_loop6():
    # Test both continue and break

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(index):
            if index == 4:
                continue
            elif i == 7:
                break
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 0, 5, 6, 7, 7, 7]


def test_loop7():
    # Use start and stop

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(3, index):
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]


def test_loop8():
    # Use start and stop and step

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        for i in range(3, index, 2):
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_while1():
    # A simple while loop!

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        while val < f32(index):
            val = val + 2.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]


def test_while2a():
    # Test while with break

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        i = -1
        while i < index - 1:
            i = i + 1
            if i == 7:
                break
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]


def test_while2b():
    # Test while with break (a case fixed in #51)

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        i = -1
        while i < index - 1:
            i = i + 1
            val = val + 1.0
            if i == 7:
                break
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]


def test_while2c():
    # Test while with continue and break

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        i = -1
        while i < index - 1:
            i = i + 1
            if index == 4:
                continue
            elif i == 7:
                break
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 0, 5, 6, 7, 7, 7]


def test_while3():
    # Test while True
    # Here the if-break becomes the iter block, quite similar to a for-loop

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        i = -1
        while True:
            i = i + 1
            if i == 7 or i == index:
                break
            elif index == 4:
                continue
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 0, 5, 6, 7, 7, 7]


def test_while4():
    # Test while True again
    # Here we truely have an OpBranchConditional %true .. ..

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        i = -1
        while True:
            i = i + 1
            if i > 100:
                i = i + 1
            if i == 7 or i == index:
                break
            elif index == 4:
                continue
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 0, 5, 6, 7, 7, 7]


def test_while5():
    # A while in a while!

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        while val < f32(index):
            i = 0
            while i < 3:
                i = i + 1
                val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 3, 3, 3, 6, 6, 6, 9, 9, 9]


def test_while6():
    # A while True in a while True!

    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        val = 0.0
        while True:
            if val == 999.0:
                continue
            if val >= f32(index):
                break
            i = 0
            while True:
                i = i + 1
                if i == 999:
                    continue
                if i > 3:
                    break
                val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 3, 3, 3, 6, 6, 6, 9, 9, 9]


# %% more


def test_discard():

    # A fragment shader for drawing red dots
    @python2shader_and_validate
    def fragment_shader(
        in_coord: ("input", "PointCoord", vec2),
    ):
        r2 = ((in_coord.x - 0.5) * 2.0) ** 2 + ((in_coord.y - 0.5) * 2.0) ** 2
        if r2 > 1.0:
            return  # discard
        out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader output

    assert ("co_return",) in fragment_shader.to_bytecode()

    if can_use_vulkan_sdk:
        spirv_text = pyshader.dev.disassemble(fragment_shader.to_spirv())
        assert "OpKill" in spirv_text


def test_long_bytecode():
    # avoid regressions like issue #42
    @python2shader_and_validate
    def compute_shader(
        index_xyz=("input", "GlobalInvocationId", ivec3),
        data2=("buffer", 1, Array(f32)),
    ):
        index = index_xyz.x
        index = index_xyz.x
        if index < 2:
            a = 3 + 4
            b = a + 5
            c = a + b + 6
            d = a + b + c + 7
            e = a + b + c + d + 8 - 3  # 100
            data2[index] = f32(e - 60)
        elif index < 4:
            a = 3 + 4
            b = a + 5
            c = a + b + 6
            d = a + b + c + 7
            e = a + b + c + d + 8 - 3  # 100
            data2[index] = f32(e - 59)
        elif index < 8:
            a = 3 + 4
            b = a + 5
            c = a + b + 6
            d = a + b + c + 7
            e = a + b + c + d + 8 - 3  # 100
            data2[index] = f32(e - 58)
        else:
            a = 3 + 4
            b = a + 5
            c = a + b + 6
            d = a + b + c + 7
            e = a + b + c + d + 8 - 3  # 100
            data2[index] = f32(e - 57)
        # This loop has not effect on the output, but it does touch on
        # compiler code. In particular code related to control flow in
        # the situation where the byte addresses are larger than 255
        # so that EXTENDED_ARG instructions are used.
        for i in range(12):
            if i > index:
                break

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 40, 41, 41, 42, 42, 42, 42, 43, 43]


# %% Utils for this module


def python2shader_and_validate(func):
    m = pyshader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


def skip_if_no_wgpu():
    if not can_use_wgpu_lib:
        raise pytest.skip(msg="SpirV validated, but not run (cannot use wgpu)")


HASHES = {
    "test_if1.compute_shader": ("d6c9f62474e0776a", "6256215f8d9ccb23"),
    "test_if2.compute_shader": ("d1963f44053cdf57", "55f780c3c6c59ec3"),
    "test_if3.compute_shader": ("4c479a73bd51d6a5", "2bc27200e964ede7"),
    "test_if4.compute_shader": ("a14d264650efc4aa", "31ac27d5a38800f3"),
    "test_if5.compute_shader": ("6580a0c2925a3c32", "a735d1f259814aa9"),
    "test_ternary1.compute_shader": ("b266fa4c88d525a6", "0ab286c7d220bdc7"),
    "test_ternary2.compute_shader": ("380cd18e0bb5561b", "f138100b4be5ac2e"),
    "test_ternary3.compute_shader": ("beb18e4537da2d1e", "855af04878683ce4"),
    "test_andor2.compute_shader": ("4699d5363e68ee43", "518dd4992e18d340"),
    "test_andor3.compute_shader": ("dc9792180f64e4d6", "68aa6a5f6eb6d0ff"),
    "test_andor4.compute_shader": ("7ac7659b99d46bf2", "c46c45dfcf8fddf9"),
    "test_andor5.compute_shader": ("2b89ef239f94f2bb", "e2a3a4579cb9afe8"),
    "test_loop0.compute_shader": ("27ae7c5792fadb56", "505071c8e78b33b3"),
    "test_loop0b.compute_shader": ("c20a70d1d0085ad8", "446030f2699e9376"),
    "test_loop1.compute_shader": ("870628bb80aa8e59", "3740a8d543e6fdc0"),
    "test_loop2.compute_shader": ("5f690a0cc6b4b81c", "37058be489468bf2"),
    "test_loop3.compute_shader": ("0b48077b4ca88ed7", "65bf4899670b0a41"),
    "test_loop4.compute_shader": ("997b9f2f714b71c4", "53820e95d7bfbf50"),
    "test_loop5a.compute_shader": ("578f234b5e846c83", "074b7ea8ed2ea331"),
    "test_loop5b.compute_shader": ("90b206631325b506", "b2b364eda9fff35a"),
    "test_loop6.compute_shader": ("64572152432c0035", "73deedeec72288e4"),
    "test_loop7.compute_shader": ("3e3b04982779d035", "d9f7ec6e73ae4647"),
    "test_loop8.compute_shader": ("14e2726965e33c3f", "cf10f755ffbd5cef"),
    "test_while1.compute_shader": ("8ac71a27e3fb45de", "95f872416a208e0f"),
    "test_while2a.compute_shader": ("18923512cd5d689b", "d04a277e9a4e10f7"),
    "test_while2b.compute_shader": ("ca25131034a7dace", "cd9e8c9cb5d94d16"),
    "test_while2c.compute_shader": ("16bf0bfa0d6318bb", "06b82657f222b40f"),
    "test_while3.compute_shader": ("439d1851302d978d", "7a3fbee64999c13c"),
    "test_while4.compute_shader": ("a6945da8caec1a89", "1c19a0f2674b4447"),
    "test_while5.compute_shader": ("cc26a0617c46aad0", "8734358070b61094"),
    "test_while6.compute_shader": ("847ce5cb0d3e52d5", "3789e7ece59dfd52"),
    "test_discard.fragment_shader": ("c457fb700b255166", "dc2fc4e245787e40"),
    "test_long_bytecode.compute_shader": ("43ed16146c7f10a0", "05a51c27d830eba3"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
