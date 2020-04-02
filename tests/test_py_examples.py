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
from testutils import validate_module, run_test_and_print_new_hashes

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
    print("Testing shader", shader_name)
    shader = shader_modules[shader_name]
    validate_module(shader, HASHES)


HASHES = {
    "compute.compute_shader_copy": ("7b03b3564a72be3c", "46f084870ce2681b"),
    "compute.compute_shader_multiply": ("96c9e08803f51d91", "19f6fec2dca68839"),
    "textures.compute_shader_tex_add": ("d8fd12dbb01d1ef7", "e81ae2994c71064c"),
    "textures.fragment_shader_tex": ("927569ad5a038680", "e4d9b4ebfda9f897"),
    "triangle.vertex_shader": ("535c85e75318f7e9", "53d4b596bc25b5a0"),
    "triangle.fragment_shader": ("c54813968ded4543", "6febd7dab6d72c8d"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
