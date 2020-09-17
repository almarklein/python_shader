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


def test_loop9():
    # This is a very specific shader (volumeslice from pygfx) that produces
    # wrong results at some point, which was the notch needed to implement
    # variables using VarAccessId objects. See #56.
    @python2shader_and_validate
    def compute_shader(
        index_xyz: ("input", "GlobalInvocationId", ivec3),
        data2: ("buffer", 1, Array(f32)),
    ):
        ed2pl = [[0, 4], [0, 3], [0, 5], [0, 2], [1, 5], [1, 3], [1, 4], [1, 2]]
        intersect_flag = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
        i = 1
        plane_index = ed2pl[i][0]
        vertices = [i, 0, 0, 0, 0, 0]
        i_start = i
        i_last = i
        max_iter = 6
        for iter in range(1, max_iter):
            for i in range(12):
                if i != i_last and intersect_flag[i] == 1:
                    if ed2pl[i][0] == plane_index:
                        vertices[iter] = i
                        plane_index = ed2pl[i][1]
                        i_last = i
                        break
                    elif ed2pl[i][1] == plane_index:
                        vertices[iter] = i
                        plane_index = ed2pl[i][0]
                        i_last = i
                        break
            if i_last == i_start:
                max_iter = iter
                break
        index = index_xyz.x
        data2[index] = f32(vertices[index])

    skip_if_no_wgpu()
    vertices = generate_list_of_floats_from_shader(6, compute_shader)
    print(vertices)
    assert vertices == [1, 3, 7, 5, 1, 0]


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
    "test_if1.compute_shader": ("44cc15f3c229ee9d", "24e742b065891a5f"),
    "test_if2.compute_shader": ("86d2f7c7a4c935c9", "1b8bdd178b440afc"),
    "test_if3.compute_shader": ("1c609db87eca2be8", "a1fd71cfc4368f5a"),
    "test_if4.compute_shader": ("7060b1753954d22c", "4c015cae278e707f"),
    "test_if5.compute_shader": ("6a3ea81e2cd64956", "c02b185d485d39d2"),
    "test_ternary1.compute_shader": ("156d28e5c4be6937", "7bebe09b5b2088d5"),
    "test_ternary2.compute_shader": ("d67ec1d6cd093ed4", "3ba38069a6266a05"),
    "test_ternary3.compute_shader": ("294814555a495b47", "d56fe95b099484d2"),
    "test_andor2.compute_shader": ("bb12e8e8d9b084b8", "24ced37f90452a68"),
    "test_andor3.compute_shader": ("0fd3a5e9e644355f", "72dc80fe74578c29"),
    "test_andor4.compute_shader": ("ec64940aa329c636", "6aec65f6ad6c54d2"),
    "test_andor5.compute_shader": ("e277b50c2abacd77", "286e1ee10fac74e7"),
    "test_loop0.compute_shader": ("7040fa4ca4f315d6", "9d0ff26c69754d35"),
    "test_loop0b.compute_shader": ("686a4296cbe258f0", "e2a9b4d8ae811434"),
    "test_loop1.compute_shader": ("35952fcf52dd20f0", "d8f126fe99689e42"),
    "test_loop2.compute_shader": ("ff995fa6c94115a2", "5c72a5834df3c1a4"),
    "test_loop3.compute_shader": ("805d244ecbec89a3", "ad279b0ef49f56df"),
    "test_loop4.compute_shader": ("7d5d1636c3089f12", "11a1d7b1ec854922"),
    "test_loop5a.compute_shader": ("e440f9ea91fe58b0", "7eeb59d940689c3c"),
    "test_loop5b.compute_shader": ("883e27baae98bc79", "378ba47c132c144d"),
    "test_loop6.compute_shader": ("0b3ab9bf77604e59", "34d9ada433252cbe"),
    "test_loop7.compute_shader": ("40e2d0c552374106", "cd216d288add13d3"),
    "test_loop8.compute_shader": ("1a738fac4a40cba8", "83dd5400229a0de9"),
    "test_loop9.compute_shader": ("90ecec7524972f4f", "5486ecea18245356"),
    "test_while1.compute_shader": ("a2f299b8d41c44ec", "4e515c12f8f623f3"),
    "test_while2a.compute_shader": ("da2e0f8b5f774aaa", "22a6d5dd9cb9ee86"),
    "test_while2b.compute_shader": ("e62ae12b9c5d511b", "e4c2d3f2a9e5578f"),
    "test_while2c.compute_shader": ("af3144327a1feedb", "6313ec46d193953f"),
    "test_while3.compute_shader": ("c21d6893f2bf240f", "dd2c888c2afaf011"),
    "test_while4.compute_shader": ("aff8b8bea6131cdf", "3fafc59ab6edec77"),
    "test_while5.compute_shader": ("6ee5853ff8c9085f", "a57df8d3930f2aaa"),
    "test_while6.compute_shader": ("dbf187d5ab4ff2f6", "ca7a45545785bdbb"),
    "test_discard.fragment_shader": ("bbdaa8848a180860", "9f5a7f4461e60eaf"),
    "test_long_bytecode.compute_shader": ("c0a43e86e3c7c35e", "5fb041f85d83b939"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
