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

layout(set = 0, binding = 0) buffer _data1 {
    int[] data1;
};
layout(set = 0, binding = 1) buffer _data2 {
    int[] data2;
};

void main() {
    data2[int(gl_GlobalInvocationID.x)] = data1[int(gl_GlobalInvocationID.x)];
}
"""
)


# %% Texturing, sampled

# https://github.com/gfx-rs/wgpu-rs/blob/master/examples/mipmap/main.rs

print_glsl2spirv_vert(
    """
#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target;
layout(set = 0, binding = 0) uniform texture2D t_Color;
layout(set = 0, binding = 1) uniform sampler s_Color;

void main() {
    // o_Target = textureLod(sampler2D(t_Color, s_Color), v_TexCoord, 0.0);
    o_Target = texture(sampler2D(t_Color, s_Color), v_TexCoord);
}
"""
)

# %% Texturing, storage


print_glsl2spirv_comp(
    """
#version 450

layout(set = 0, binding = 0, rgba8) uniform image2D tex;

void main() {
    vec4 color = imageLoad(tex, ivec2(gl_GlobalInvocationID.xy));
    color.y += 1.0;
    imageStore(tex, ivec2(gl_GlobalInvocationID.xy), color);
}
"""
)

# %% If statements

print_glsl2spirv_comp(
    """
#version 450
//layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer PrimeIndices {
    uint[] data;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index == 1) {
        data[index] = 41;
    } else if (index == 2) {
        data[index] = 42;
    } else if (index == 3) {
        data[index] = 43;
    } else {
        data[index] = index;
    }
}
"""
)

# %% for-loop

print_glsl2spirv_comp(
    """
#version 450
//layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer PrimeIndices {
    uint[] data;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint val = 0;
    for (int i=0; i<index; i++) {
        if (i == 4) { continue; }
        if (i == 7) { break; }
        val = val + 1;
    }
    data[index] = val;
}
"""
)

# %% while-loop

print_glsl2spirv_comp(
    """
#version 450
//layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer PrimeIndices {
    uint[] data;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint val = 0;
    uint i = 0;
    while (true) {
        i += 1;
        if (index == 4) { continue; }
        if (i == 7) { break; }
        val = val + 2;
    }
    data[index] = val;
}
"""
)


# %% extended instruction set


print_glsl2spirv_comp(
    """
#version 450
//layout(local_size_x = 1) in;

layout(set = 0, binding = 0) buffer PrimeIndices {
    uint[] data;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint val = 0;
    data[index] = uint(pow(val, 3.1));
}
"""
)

# %% arrays

print_glsl2spirv_comp(
    """
#version 450

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint arr[3] = uint[3](1, 2, 3);
    arr[0] = 4;
}
"""
)
