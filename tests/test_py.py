import os
import hashlib
import inspect

import python_shader
from python_shader import i32, vec2, vec3, vec4

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
    "test_null_shader.vertex_shader": ("512ca89b1c376bde", "17b8c22d37890119"),
    "test_triangle_shader.vertex_shader": ("fe3fbcabd6ca2c19", "e1ee457967857a87"),
    "test_triangle_shader.fragment_shader": ("1b2d31e4656418c6", "69784202b4b63385"),
}


# Run this as a script to get new hashes when needed
if __name__ == "__main__":
    run_test_and_print_new_hashes()
