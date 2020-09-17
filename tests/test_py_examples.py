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
    "compute.compute_shader_copy": ("6e6849aa811ccf8a", "1ac33233b60b9f13"),
    "compute.compute_shader_multiply": ("a2d0cb9798632bd1", "3229b7f2d61e79a8"),
    "compute.compute_shader_tex_colorwap": ("454cefdbf0ce1acc", "0dc6c0301d583b8e"),
    "mesh.vertex_shader": ("fdc3b4b279b3a31e", "80db45b376a75fe3"),
    "mesh.fragment_shader_flat": ("21049f547e057152", "bca0edd57ffb8e98"),
    "textures.compute_shader_tex_add": ("74c7c482a598349d", "9e271b832b0971d1"),
    "textures.fragment_shader_tex": ("7188891541d70435", "28c84baac74b973e"),
    "triangle.vertex_shader": ("738e0ac3bd22ebac", "e4209550a51f8b5a"),
    "triangle.fragment_shader": ("494975dea607787e", "4c6ac6942205ebfc"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
