"""
Developer functions. These require the Vulkan SDK; don't use in end-user code!
"""

import os
import tempfile
import subprocess


def glsl2spirv(glsl_code, shader_type):
    """ Compile GLSL to SPirV and return as bytes.

    Note: needs glslc from the Vulkan SDK!
    """

    if shader_type not in ("vertex", "fragment", "compute"):
        raise ValueError(
            f"Shadertype must be 'vertex', 'fragment' or 'compute', not {shader_type!r}."
        )
    if not isinstance(glsl_code, str):
        raise TypeError("glsl2spirv expects a string.")

    ext = shader_type[:4]
    filename1 = os.path.join(tempfile.gettempdir(), f"x.{ext}")
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

    return binary


def disassemble(spirv):
    """ Disassemble the generated binary code using spirv-dis, and return as a string.

    Note: needs spirv-dis from spirv-tools or the Vulkan SDK!
    """
    if isinstance(spirv, bytes):
        data = spirv
    elif hasattr(spirv, "to_spirv"):
        data = spirv.to_spirv()
    else:
        raise TypeError("disassemble() function expects SpirV bytes or ShaderModule.")

    description = spirv.description if hasattr(spirv, "description") else "SpirV"

    filename = os.path.join(tempfile.gettempdir(), "x.spv")
    with open(filename, "wb") as f:
        f.write(data)
    try:
        stdout = subprocess.check_output(
            ["spirv-dis", filename], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        e = f"Could not disassemble {description}:\n" + err.output.decode()
        raise Exception(e)
    else:
        return stdout.decode()


def validate(spirv):
    """ Validate the generated binary code by running spirv-val. Raises an
    errror if the SpirV was found to be invalid.

    Note: needs spirv-val from spirv-tools or the Vulkan SDK!
    """
    if isinstance(spirv, bytes):
        data = spirv
    elif hasattr(spirv, "to_spirv"):
        data = spirv.to_spirv()
    else:
        raise TypeError("validate() function expects SpirV bytes or ShaderModule.")

    description = spirv.description if hasattr(spirv, "description") else "SpirV"

    filename = os.path.join(tempfile.gettempdir(), "x.spv")
    with open(filename, "wb") as f:
        f.write(data)
    try:
        stdout = subprocess.check_output(
            ["spirv-val", filename], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        out = err.output.decode()
    else:
        out = stdout.decode().strip()
    if out:
        raise Exception(f"{description} invalid:\n{out}")
    else:
        print(f"{description} seems valid!")
