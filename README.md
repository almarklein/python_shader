[![Build Status](https://dev.azure.com/almarklein/spirv-py/_apis/build/status/almarklein.spirv-py?branchName=master)](https://dev.azure.com/almarklein/spirv-py/_build/latest?definitionId=1&branchName=master)


# SpirV-Py

Tools to work with SpirV in Python, including a Python to SpirV compiler.

## Introduction

[SpirV](https://en.wikipedia.org/wiki/Standard_Portable_Intermediate_Representation)
is a binary platform independent represenation for GPU shaders. This module
makes it easier to work with SpirV shaders from Python.


## Status

Very much a WIP. The plan is to build out the `python2spirv()` compiler to
allow writing any shader in Python.

There's also [WSL](https://gpuweb.github.io/WSL/) which we want to "take
into account" when it becomes a thing.


## Installation

```
pip install spirv
```

If you want to use `module.validation()`, `module.disassemble()`, or `glsl2spirv()`,
you need to seperately install `spirv-tools`, e.g. via the
[Vulkan SDK](https://www.lunarg.com/vulkan-sdk/).


## Usage

This library presents the `SpirVModule` class that wraps SpirV code.
As for documentation, [this link](https://github.com/almarklein/spirv-py/blob/master/spirv/_module.py)
should suffice for now.

There are a few ways to get a `SpirVModule` object:

```py

# From raw bytes
spirv.bytes2spirv()

# From a filename or file object
spirv.file2spirv()

# Compile Python code to SpirV
spirv.python2spirv(python_function)

# From glsl code (need spirv-tools)
spirv.glsl.glsl2spirv(code, shader_type])
```


## License

This code is distributed under the 2-clause BSD license.
