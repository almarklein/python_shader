import os
import tempfile
import subprocess


def glsl2spirv(glsl, shader_type):
    assert shader_type in ("comp", "vert", "frag")
    filename1 = os.path.join(tempfile.gettempdir(), f"x.{shader_type}")
    filename2 = os.path.join(tempfile.gettempdir(), f"x.{shader_type}.spv")
    with open(filename1, "wb") as f:
        f.write(glsl.encode())

    try:
        stdout = subprocess.check_output(
            ["glslangvalidator", "-V", filename1, "-o", filename2],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    try:
        stdout = subprocess.check_output(
            ["spirv-dis", filename2], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as err:
        e = "Could not disassemle Spir-V:\n" + err.output.decode()
        raise Exception(e)
    else:
        return stdout.decode()


def print_glsl2spirv_comp(glsl):
    print(glsl2spirv(glsl, "comp"))


def print_glsl2spirv_vert(glsl):
    print(glsl2spirv(glsl, "vert"))


def print_glsl2spirv_frag(glsl):
    print(glsl2spirv(glsl, "frag"))


# %% Naked function

print_glsl2spirv_vert(
    """
#version 450

void main()
{
}
"""
)


# %% One in, one out

print_glsl2spirv_vert(
    """
#version 450

layout (location = 12) in vec3 aPos; // the position variable has attribute position 0

layout(location = 13) out vec4 vertexColor; // specify a color output to the fragment shader

void main()
{
    vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
    vertexColor = vec4(aPos, 1.0);
}
"""
)


# %% Builtin out vars

print_glsl2spirv_vert(
    """
#version 450

void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""
)


# %% Uniforms

print_glsl2spirv_vert(
    """
#version 450

//layout(location = 13) out vec4 fragColor; float foo = 3.0;
layout(location = 13) out XX { vec4 fragColor; float foo; };

//layout(binding = 0) uniform blabla { vec4 uColor; };

void main()
{
   fragColor = vec4(1.0, 1.0, 1.0, foo);//uColor;
}
"""
)


# %% Constant

print_glsl2spirv_vert(
    """
#version 450

vec3 uColor = vec3(1.0, 0.0, 0.0);

void main()
{
}
"""
)


# %% Vector composite

print_glsl2spirv_vert(
    """
#version 450

void main()
    {
    int index = 0;
    vec2 positions[3] = vec2[3]( vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5) );
    vec2 x = positions[index];
}
"""
)

# %% Compute minimal

print_glsl2spirv_comp(
    """
#version 450
//layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer PrimeIndices {
    uint[] data;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    data[index] = index;
}
"""
)
