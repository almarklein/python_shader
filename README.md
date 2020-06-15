![CI status](https://github.com/pygfx/pyshader/workflows/CI/badge.svg)

# pyshader

Write modern GPU shaders in Python! Provides a Python to SpirV compiler, to
start with.


## Introduction

[SpirV](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation)
is a binary platform independent represenation for GPU shaders. This module
makes it easier to write SpirV shaders in Python.

This should be useful for anything built on top of
[wgpu-py](https://github.com/almarklein/wgpu-py).


## Current status

Very much a WIP. The `python2shader` compiler is working, but is missing
functionality for e.g. if-statements and for-loops.


## Installation

```
pip install pyshader
```


## Example usage (a simple mesh shader)

```py
from pyshader import python2shader, vec3, vec4, mat4

@python2shader
def vertex_shader(
    vertex_pos: ("input", 0, vec3),
    transform: ("uniform", (0, 0), mat4),
    out_pos: ("output", "Position", vec4),
):
    out_pos = transform * vec4(vertex_pos, 1.0)

@python2shader
def fragment_shader_flat(
    color: ("uniform", (0, 1), vec3), out_color: ("output", 0, vec4),
):
    out_color = vec4(color, 1.0)  # noqa
```


## Developers

If you want to use `pyshader.dev.validate()`,
`pyshader.dev.disassemble()`, or `pyshader.dev.glsl2spirv()`,
you need to seperately install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).


## License

This code is distributed under the 2-clause BSD license.


## API

The pyshader lib is a mini-framework allowing compilation from/to different
shader representations. It features an internal bytecode representation
(not (yet) part of the public API). At the moment, the only compilation part
is from a Python function, to bytecode, to SpirV. More paths may be added
in the future, e.g. [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html).


### The `ShaderModule` class

A `ShaderModule` is a representation of a shader. It's input is the shader
source, e.g. a Python function. It can then be converted to bytecode
and/or to SpirV.

* `input`: property that holds the input source (e.g. the Python function object).
* `to_bytecode`: method  to get the bytecode representing this shader module.
* `to_spirv`: method to get the binary representation of the SpirV module (bytes).


### The `python2shader(func)` function

Convert a Python function to a ShaderModule object. Takes the bytecode
of the given function and converts it to our internal bytecode. From there
it can be converted to binary SpirV. All in dependency-free pure Python.


### Types

GPU programming feels a bit different. This is for example expressed
by the heavy use of types representing vectors and matrices. Pyshader
has is's own type system to represent GPU specific types.

There are a handful of leaf types:

* void
* boolean  -> True or False
* f16, f32, f64  -> floating point number of various size
* u8  -> unsigned byte
* i16, i32, i64  -> signed integers of various size

Then there is the `Vector` class. One can create a vector type by
specifying the number of elements (2-4) and one of the numeric leaf
types, e.g. `Vector(2, f32)`. Similarly the `Matrix` class can be used
to create matrix types, e.g. `Matrix(4, 4, f32)`.

For convenience, there are several builtin vector and matrix types:

* Float vectors: vec2, vec3, vec4
* Integer vectors: ivec2, ivec3, ivec4
* Boolean vectors: bvec2, bvec3, bvec4
* Square matrices: mat2, mat3, mat4
* Other matrics: mat3x2, mat4x2, mat2x3, mat4x3, mat2x4, mat3x4

Further, one can specify types where each element is any of the above
types, e.g. `Array(100, vec4)`. In some cases one can also define an
array of undefined size: `Array(vec2)`.

Structs can be created using e.g. `Struct(foo=f32, bar=ivec4)`. Note
that arrays can contain structs, and structs can contain arrays.


### Python shader syntax

To write shaders in Python, you need to follow some rules. Let's start
with your function's name. It must contain one of "compute", "vertex"
or "fragment", to indicate the type of shader.


#### Function arguments

Each argument of your function must be annotated with a 3-element tuple:

```
@python2shader
def your_vertex_shader(
    argument_name: (resource_type, slot, type_info)
):
    ...
```

There are 6 possible resource types. These are specified as a string, but
we also provides an enum for convenience:

* `RES_INPUT`: For vertex shaders this means a vertex buffer. For
  fragment shaders it means the output of the vertex shader. These can
  also be builtin inputs (see info on slot below).
* `RES_OUTPUT`: For vertex shaders these will be available as inputs
  to the fragment shader, or builtin outputs. For fragment shaders
  this is e.g. the output color. Note that shaders do not have return values:
  you must assign to the output argument. Yes, this looks a bit weird.
* `RES_UNIFORM`: Small(ish) data in a uniform buffer. This will
  typically be a struct combining all uniform data.
* `RES_BUFFER`: A storage buffer that can be written to or read from.
* `RES_TEXTURE`: A texture object.
* `RES_SAMPLER`: A sampler (defines how a texture must be sampled).

For input and output resources, the `slot` is an integer, or a string specifying
the name of the builtin input/output, e.g. "VertexId" or "Position". For the other
resource types the slot is a 2-tuple specifying bind group and binding.
Integers are also allowed, implying bind group zero.

For most resource types, `type_info` is a type as specified in the
previous section. These can also be specified as a string. For textures,
`type_info` must contain the dimension ("1d", "2d", "3d" or "cube"),
and the texture format.


#### Strict typing

Pyshader uses type inference, so you don't have to worry about
specifying types except for the function's input arguments. The typing
is strict though, and there is no implicit conversion from integers
to floats; you need to explicitly cast them.


#### Vector element access

The Python shader syntax supports a nice feature from GLSL to easily access
the elements of a vector:
```py
    v = vec4(1.0, 2.0, 3.0, 4.0)
    # These are all equivalent
    v2 = vec2(v[0], v[1])
    v2 = v.xy  # xyzw
    v2 = v.rg  # rgba
    # Can also do this
    v3 = v.xzz
    scalar = v.y
```

#### Examples

Check out the Python shader examples to learn more:
https://github.com/pygfx/pyshader/tree/master/examples_py

