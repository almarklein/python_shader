"""
Validate all shaders in our examples. This helps ensure that our
exampples are actually valid, but also allows us to increase test
coverage simply by writing examples.
"""

import os
import types
import importlib.util

import python_shader

import pytest
from testutils import can_use_vulkan_sdk, validate_module, run_test_and_print_new_hashes

EXAMPLES_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "examples_py"))


def get_python_shader_examples():

    shader_modules = {}  # shader descriptive name -> shader object

    # Collect shader modules
    for fname in os.listdir(EXAMPLES_DIR):
        if not fname.endswith(".py"):
            continue
        # Load module
        filename = os.path.join(EXAMPLES_DIR, fname)
        modname = fname[:-3]
        spec = importlib.util.spec_from_file_location(modname, filename)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # Collect shader module objects from the module
        for val in m.__dict__.values():
            if isinstance(val, python_shader.ShaderModule):
                fullname = modname + "." + val.input.__qualname__
                val.input.__qualname__ = fullname
                shader_modules[fullname] = val
            elif isinstance(val, types.FunctionType):
                funcname = val.__name__
                if "_shader" in funcname:
                    raise RuntimeError(f"Undecorated shader {funcname}")

    return shader_modules


shader_modules = get_python_shader_examples()


@pytest.mark.parametrize("shader_name", list(shader_modules.keys()))
def test(shader_name):
    if not can_use_vulkan_sdk:
        pytest.skip("No Vulkan SDK")
    shader = shader_modules[shader_name]
    validate_module(shader, HASHES)


HASHES = {}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
