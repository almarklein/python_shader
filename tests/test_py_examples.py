"""
Validate all shaders in our examples. This helps ensure that our
exampples are actually valid, but also allows us to increase test
coverage simply by writing examples.
"""

import os
import types
import importlib.util

import pyshader

import pytest
from testutils import validate_module, run_test_and_print_new_hashes

EXAMPLES_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "examples_py"))


def get_pyshader_examples():

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
            if isinstance(val, pyshader.ShaderModule):
                fullname = modname + "." + val.input.__qualname__
                val.input.__qualname__ = fullname
                shader_modules[fullname] = val
            elif isinstance(val, types.FunctionType):
                funcname = val.__name__
                if "_shader" in funcname:
                    raise RuntimeError(f"Undecorated shader {funcname}")

    return shader_modules


shader_modules = get_pyshader_examples()


@pytest.mark.parametrize("shader_name", list(shader_modules.keys()))
def test(shader_name):
    print("Testing shader", shader_name)
    shader = shader_modules[shader_name]
    validate_module(shader, HASHES)


HASHES = {
    "compute.compute_shader_copy": ("0fef618daddaf07d", "c7570b16d25a33d0"),
    "compute.compute_shader_multiply": ("49e95d04924391ff", "3f3ee31245b9d16b"),
    "compute.compute_shader_tex_colorwap": ("c62f87032d09582f", "d87f2f8b15c73213"),
    "mesh.vertex_shader": ("968d9fec3eddcee7", "80db45b376a75fe3"),
    "mesh.fragment_shader_flat": ("3dfcac8707287b7e", "bca0edd57ffb8e98"),
    "textures.compute_shader_tex_add": ("db13b6f7281e0688", "4e37bc570b6462eb"),
    "textures.fragment_shader_tex": ("afb38a886eab7c1b", "28c84baac74b973e"),
    "triangle.vertex_shader": ("0557cb689b4d7c7c", "757bf0a23c44feec"),
    "triangle.fragment_shader": ("6d17c1397c52cfad", "4c6ac6942205ebfc"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
