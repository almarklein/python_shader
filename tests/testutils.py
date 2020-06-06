import os
import sys
import inspect
import hashlib
import subprocess

import python_shader


def iters_equal(iter1, iter2):
    """ Assert that the given iterators are equal.
    """
    iter1, iter2 = list(iter1), list(iter2)
    if len(iter1) != len(iter2):
        return False
    if not all(iter1[i] == iter2[i] for i in range(len(iter1))):
        return False
    return True


def validate_module(shader_module, hashes):
    """ Validate the given shader module against the given hashes.
    """

    func = shader_module.input
    assert callable(func)

    # Get steps of code: Python, bytecode, spirv
    key = func.__qualname__.replace(".<locals>.", ".")
    bc = shader_module.to_bytecode()
    text_bc = python_shader.opcodes.bc2str(bc)
    assert bc == python_shader.opcodes.str2bc(text_bc)
    byte_sp = shader_module.to_spirv()

    # print(text_bc)
    # print(python_shader.dev.disassemble(byte_sp))

    # Get hashes so we can compare it easier
    assert isinstance(text_bc, str)
    assert isinstance(byte_sp, bytes)
    hash_bc = hashlib.md5(text_bc.encode()).hexdigest()[:16]
    hash_sp = hashlib.md5(byte_sp).hexdigest()[:16]

    overwrite_hashes = hashes.get("overwrite_hashes")
    if overwrite_hashes:
        # Dev mode: print hashes so they can be copied in. MUST validate here.
        assert not os.environ.get("CI")
        python_shader.dev.validate(byte_sp)
        assert key not in hashes  # prevent duplicates
        hashes[key] = hash_bc, hash_sp

    else:
        # Normal mode: compare hashes with preset hashes. This allows
        # us to generate the hashes once, and then on CI we make sure
        # that any Python function results in the exact same bytecode
        # and SpirV on different platforms and Python versions.
        if key not in hashes:
            assert False, f"No hash found for {key}"
        if hashes[key][0] != hash_bc:
            code = inspect.getsource(func)
            assert False, f"Bytecode for {key} does not match:\n{code}\n{text_bc}"
        if hashes[key][1] != hash_sp:
            code = inspect.getsource(func)
            assert False, f"SpirV for {key} does not match:\n{code}\n{byte_sp}"
        # If the Vulkan SKD is available, validate the module for good measure.
        # In practice there will probably be one CI build that does this.
        if can_use_vulkan_sdk:
            python_shader.dev.validate(byte_sp)


def run_test_and_print_new_hashes(ns):
    """ Run all tests in th given namespace (e.g. globals()), and print
    the new hashes. This will force validate_module() to validate the
    SpirV code.
    """

    old_hashes = ns["HASHES"]
    new_hashes = ns["HASHES"] = {"overwrite_hashes": True}

    # Run tests
    for funcname, func in ns.items():
        if funcname.startswith("test") and callable(func):
            for mark in getattr(func, "pytestmark", []):
                if mark.name == "parametrize":
                    for arg in mark.args[1]:
                        print(f"Running {funcname} {arg}...")
                        func(arg)
                    break
            else:
                print(f"Running {funcname} ...")
                func()

    new_hashes.pop("overwrite_hashes")

    # Print new hashes
    print("\nHASHES = {")
    for key, val in new_hashes.items():
        print(f"    {key!r}: {val!r},".replace("'", '"'))
    print("}\n")

    # Show a hint about what changed
    if new_hashes == old_hashes:
        print("The hashes have not changed.")
    else:
        n_more = len(set(new_hashes).difference(old_hashes))
        n_less = len(set(old_hashes).difference(new_hashes))
        keys_in_both = set(old_hashes).intersection(new_hashes)
        n_diff = sum(new_hashes[key] != old_hashes[key] for key in keys_in_both)
        n_diff1 = sum(new_hashes[key][0] != old_hashes[key][0] for key in keys_in_both)
        n_diff2 = sum(new_hashes[key][1] != old_hashes[key][1] for key in keys_in_both)
        print(
            f"Hashes changed: {n_more} added, {n_less} removed, "
            f"{n_diff1}/{n_diff2}/{n_diff} changed."
        )


def _determine_can_use_vulkan_sdk():
    # If PYTHON_SHADER_TEST_FULL is set, force using the vulkan SDK
    if os.getenv("PYTHON_SHADER_TEST_FULL", "").lower() == "true":
        return True
    try:
        subprocess.check_output(["spirv-val", "--version"])
    except Exception:
        return False
    else:
        return True


def _determine_can_use_wgpu_lib():
    code = "import wgpu.utils; wgpu.utils.get_default_device()"
    try:
        subprocess.check_output(
            [sys.executable, "-c", code,]
        )
    except Exception:
        return False
    else:
        return True


can_use_vulkan_sdk = _determine_can_use_vulkan_sdk()
can_use_wgpu_lib = _determine_can_use_wgpu_lib()
