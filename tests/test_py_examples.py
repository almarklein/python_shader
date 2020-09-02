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
    "compute.compute_shader_copy": ("0fef618daddaf07d", "c044d933e7470332"),
    "compute.compute_shader_multiply": ("49e95d04924391ff", "0df83cd3720f30b3"),
    "compute.compute_shader_tex_colorwap": ("c62f87032d09582f", "ccfda69295d76e89"),
    "mesh.vertex_shader": ("968d9fec3eddcee7", "5ec14e2b0f605cc8"),
    "mesh.fragment_shader_flat": ("3dfcac8707287b7e", "ebf558649fa38414"),
    "textures.compute_shader_tex_add": ("db13b6f7281e0688", "6f8ed6d74fe38e68"),
    "textures.fragment_shader_tex": ("afb38a886eab7c1b", "a3074e9bc3cfce20"),
    "triangle.vertex_shader": ("0557cb689b4d7c7c", "07bcc4f5a508dc7d"),
    "triangle.fragment_shader": ("6d17c1397c52cfad", "9c70b7411171ec9a"),
}

if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())
