[![Build Status](https://dev.azure.com/almarklein/python_shader/_apis/build/status/almarklein.python_shader?branchName=master)](https://dev.azure.com/almarklein/python_shader/_build/latest?definitionId=3&branchName=master)


# python_shader

Write modern GPU shaders in Python! Provides a Python to SpirV compiler, to
start with.


## Introduction

[SpirV](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation)
is a binary platform independent represenation for GPU shaders. This module
makes it easier to write SpirV shaders in Python.


## Status

Very much a WIP. The plan is to build out the `python2shader()` compiler to
allow writing any shader in Python. Currently, it does not even implement basic
arithmetic yet.

There's also [WSL](https://gpuweb.github.io/WSL/) which we want to "take
into account" when it becomes a thing.


## Installation

```
pip install python-shader
```


## Usage

Decorate a function to turn it into a `ShaderModule` object:

```
import python_shader

@python_shader.python2shader
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)
```

You can then get the bytes representing the SpirV via `vertex_shader.to_spirv()`.
The module object can be used as-is in [wgpu-py](https://github.com/almarklein/wgpu-py).


## Developers

If you want to use `python_shader.dev.validate()`,
`python_shader.dev.disassemble()`, or `python_shader.dev.glsl2spirv()`,
you need to seperately install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).


## License

This code is distributed under the 2-clause BSD license.
