import os
import tempfile
import subprocess

from ._module import SpirVModule


def glsl2spirv(glsl_code, shader_type):
    """ Compile GLSL to SPirV and return as a SpirVModule object.

    Needs Spir-V tools, which can easily be obtained by installing the
    Vulkan SDK. But for code running at end-users you probably want to
    use bytes2spirv() or file2spirv() instead.
    """

    if shader_type not in ("vert", "frag", "comp"):
        raise ValueError(
            f"Shadertype must be 'vert', 'frag' or 'comp', not {shader_type!r}."
        )
    if not isinstance(glsl_code, str):
        raise TypeError("glsl2spirv expects a string.")

    filename1 = os.path.join(tempfile.gettempdir(), f"x.{shader_type}")
    filename2 = os.path.join(tempfile.gettempdir(), "x.spv")

    with open(filename1, "wb") as f:
        f.write(glsl_code.encode())

    # Note: -O means optimize, use -O0 to disable optimization
    try:
        stdout = subprocess.check_output(
            ["glslc", "-O", "-o", filename2, filename1], stderr=subprocess.STDOUT
        )
        stdout  # noqa - not used
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    with open(filename2, "rb") as f:
        binary = f.read()

    return SpirVModule(glsl_code, binary, "compiled from GLSL")
