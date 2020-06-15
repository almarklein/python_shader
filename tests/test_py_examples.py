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
    "compute.compute_shader_copy": ("7b03b3564a72be3c", "46f084870ce2681b"),
    "compute.compute_shader_multiply": ("3cfb0499505b4910", "8f197b15205b5a51"),
    "compute.compute_shader_tex_colorwap": ("454cefdbf0ce1acc", "46604895ec75f8a3"),
    "mesh.vertex_shader": ("fdc3b4b279b3a31e", "7af85329afdd2b25"),
    "mesh.fragment_shader_flat": ("21049f547e057152", "54f8b6c24c0822a8"),
    "textures.compute_shader_tex_add": ("8a2a1adde39897d3", "03d5f2079a960a24"),
    "textures.fragment_shader_tex": ("7188891541d70435", "adede1b40a7f1f6f"),
    "triangle.vertex_shader": ("738e0ac3bd22ebac", "53d4b596bc25b5a0"),
    "triangle.fragment_shader": ("494975dea607787e", "6febd7dab6d72c8d"),
}

if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
