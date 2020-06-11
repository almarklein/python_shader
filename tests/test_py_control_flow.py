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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 40, 41, 41, 42, 42, 42, 42, 43, 43]


def test_if2():
    # More nesting
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


def test_if3():
    # And and or
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(f32)),
        data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        data2[index] = 40.0 if index == 0 else 41.0

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [40, 41, 42, 42, 42, 42, 42, 42, 42, 42]


# %% more or / and


def test_andor1():
    # Implicit conversion to truth values is not supported

    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        if index < 5:
            val = f32(index - 3) and 99.0
        else:
            val = f32(index - 6) and 99.0
        data2[index] = val

    with pytest.raises(python_shader.ShaderError):
        python_shader.python2shader(compute_shader)


def test_andor2():
    # or a lot
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
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
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]


def test_loop5():
    # Break - this one is interesting because the stop criterion is combined with the break
    # This is a consequence of the logic to detect and simplify or-logic

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        for i in range(index):
            if i == 7:
                break
            val = val + 1.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]


def test_loop6():
    # Test both continue and break

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
        val = 0.0
        while val < f32(index):
            val = val + 2.0
        data2[index] = val

    skip_if_no_wgpu()
    res = generate_list_of_floats_from_shader(10, compute_shader)
    assert res == [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]


def test_while2():
    # Test while with continue and break

    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(f32)),
    ):
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
    "test_if1.compute_shader": ("ef6e7238394e1253", "df47945efe25f3e0"),
    "test_if2.compute_shader": ("b9c6173d367d5096", "f931b5238c6a8593"),
    "test_if3.compute_shader": ("d9965636bade920d", "5fd34c5a756c0128"),
    "test_if4.compute_shader": ("e38f5828ab33c285", "7f101ed4facfe1fd"),
    "test_if5.compute_shader": ("723487be5988c716", "0bd1a694b5cd4656"),
    "test_ternary1.compute_shader": ("71489b4284d415d0", "6055b9b66cef6a45"),
    "test_ternary2.compute_shader": ("2d240e03e769fa4a", "0fc5a0f41eee5f0d"),
    "test_ternary3.compute_shader": ("eba4597a7da5dd46", "791e0c70f0fbbf92"),
    "test_andor2.compute_shader": ("6e4e788ea8316125", "52564c851e9203a0"),
    "test_andor3.compute_shader": ("b9a6b4b5aee11a12", "e3a7271b83f5edf1"),
    "test_andor4.compute_shader": ("246983365a1efbf7", "27639564243343a1"),
    "test_andor5.compute_shader": ("a509d604b344f2ba", "dd97d5cd715efea4"),
    "test_loop0.compute_shader": ("44b6c95c699d9851", "846f19b68aa4c4e6"),
    "test_loop0b.compute_shader": ("72e8d393e8fe4385", "fae7366dfdc926dc"),
    "test_loop1.compute_shader": ("4bb929b47ff8badc", "8c9c9af924e92f14"),
    "test_loop2.compute_shader": ("c14650b95b95a90d", "76d36877e4cd6315"),
    "test_loop3.compute_shader": ("7e7eb94accafdf3c", "57b06ed205152275"),
    "test_loop4.compute_shader": ("56eb6e26210f2852", "0cff02d45e55eb77"),
    "test_loop5.compute_shader": ("9ecc1a6019df2235", "b766a0d1362fa9c2"),
    "test_loop6.compute_shader": ("d03455bd48586359", "3e1d5e4038cbedc7"),
    "test_loop7.compute_shader": ("c9891c2c45e585a3", "bd741df0657431a7"),
    "test_loop8.compute_shader": ("2658fbb85b05bf4d", "347b7c89df6fd5ac"),
    "test_while1.compute_shader": ("c7d93b33fc358ff7", "bbda0ee55cc3b891"),
    "test_while2.compute_shader": ("8801b554f198c509", "1ae62cff48b3ff3b"),
    "test_while3.compute_shader": ("7575db6c18e679ae", "34de5e3b0a44406b"),
    "test_while4.compute_shader": ("c721536fc016848f", "93b4fa7cf42b0b51"),
    "test_while5.compute_shader": ("4340589fa8649a24", "d9324dc099327352"),
    "test_while6.compute_shader": ("b4ca542b37cf60a5", "e89fc46e244b6d42"),
    "test_discard.fragment_shader": ("e7c2b2ad579b6d58", "6d3182b0b5189d45"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
