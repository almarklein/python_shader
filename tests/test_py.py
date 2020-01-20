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

import os
import hashlib
import inspect

import python_shader
from python_shader import i32, vec2, vec3, vec4, Array

from pytest import mark

from testutils import use_vulkan_sdk


def test_null_shader():
    def vertex_shader(input, output):
        pass

    m = python_shader.python2shader(vertex_shader)
    assert m.input is vertex_shader
    validate_module(vertex_shader, m)


def test_triangle_shader():
    def vertex_shader(input, output):
        input.define("index", "VertexId", i32)
        output.define("pos", "Position", vec4)
        output.define("color", 0, vec3)

        positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

        p = positions[input.index]
        output.pos = vec4(p, 0.0, 1.0)
        output.color = vec3(p, 0.5)

    m = python_shader.python2shader(vertex_shader)
    assert m.input is vertex_shader
    validate_module(vertex_shader, m)

    def fragment_shader(input, output):
        input.define("color", 0, vec3)
        output.define("color", 0, vec4)

        output.color = vec4(input.color, 1.0)

    m = python_shader.python2shader(fragment_shader)
    assert m.input is fragment_shader
    validate_module(fragment_shader, m)


@mark.skipif(not use_vulkan_sdk, reason="No Vulkan SDK")
def test_no_duplicate_constants():
    def vertex_shader():
        positions = [vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 1.0)]  # noqa

    m = python_shader.python2shader(vertex_shader)
    text = python_shader.dev.disassemble(m.to_spirv())
    assert 2 <= text.count("OpConst") <= 3


def test_compute_shader():
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("data1", 0, Array(i32))
        buffer.define("data2", 1, Array(i32))

        buffer.data2[input.index] = buffer.data1[input.index]

    m = python_shader.python2shader(compute_shader)
    assert m.input is compute_shader
    validate_module(compute_shader, m)


# %% Validation


def validate_module(func, m):

    # Get steps of code: Python, bytecode, spirv
    key = func.__qualname__.replace(".<locals>.", ".")
    text_bc = python_shader.opcodes.bc2str(m.to_bytecode())
    byte_sp = m.to_spirv()

    # Get hashes so we can compare it easier
    assert isinstance(text_bc, str)
    assert isinstance(byte_sp, bytes)
    hash_bc = hashlib.md5(text_bc.encode()).hexdigest()[:16]
    hash_sp = hashlib.md5(byte_sp).hexdigest()[:16]

    if OVERWRITE_HASHES:
        # Dev mode: print hashes so they can be copied in. MUST validate here.
        assert not os.environ.get("CI")
        python_shader.dev.validate(byte_sp)
        assert key not in HASHES  # prevent duplicates
        HASHES[key] = hash_bc, hash_sp

    else:
        # Normal mode: compare hashes with preset hashes. This allows
        # us to generate the hashes once, and then on CI we make sure
        # that any Python function results in the exact same bytecode
        # and SpirV on different platforms and Python versions.
        if HASHES[key][0] != hash_bc:
            code = inspect.getsource(func)
            assert False, f"Bytecode for {key} does not match:\n{code}\n{text_bc}"
        if HASHES[key][1] != hash_sp:
            code = inspect.getsource(func)
            assert False, f"SpirV for {key} does not match:\n{code}\n{byte_sp}"
        # If the Vulkan SKD is available, validate the module for good measure.
        # In practice there will probably be one CI build that does this.
        if use_vulkan_sdk:
            python_shader.dev.validate(byte_sp)


def run_test_and_print_new_hashes():
    global OVERWRITE_HASHES
    OVERWRITE_HASHES = True
    HASHES.clear()

    for funcname, func in globals().items():
        if funcname.startswith("test_") and callable(func):
            print(f"Running {funcname} ...")
            func()

    print("\nHASHES = {")
    for key, val in HASHES.items():
        print(f"    {key!r}: {val!r},")
    print("}")


OVERWRITE_HASHES = False


HASHES = {
    "test_null_shader.vertex_shader": ("801b48feb24d7aea", "1cd71e87b1ba5a32"),
    "test_triangle_shader.vertex_shader": ("9cdef2b9fc3befa3", "ef7285ff8c51a8f1"),
    "test_triangle_shader.fragment_shader": ("02bcc64a3e8b05d3", "33063b6db28277db"),
    "test_compute_shader.compute_shader": ("02822d1f23bee04d", "3a3e6992643df7ea"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes()
