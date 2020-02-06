import inspect
from dis import dis as pprint_bytecode

from ._module import ShaderModule
from .opcodes import OpCodeDefinitions as op
from ._dis import dis
from . import _types
from ._types import spirv_types_map


def python2shader(func):
    """ Convert a Python function to a ShaderModule object.

    This function takes the bytecode of the given function and converts it to
    a more standardized (and shader specific) bytecode. From there it be
    converted to binary SpirV. All in dependency-free pure Python.
    """

    if not inspect.isfunction(func):
        raise TypeError("python2shader expects a Python function.")

    # Detect shader type
    possible_types = "vertex", "fragment", "compute"
    shader_types = [t for t in possible_types if t in func.__name__.lower()]
    if len(shader_types) == 1:
        shader_type = shader_types[0]
    elif len(shader_types) == 0:
        raise NameError(
            "Shader entrypoint must contain 'vertex', 'fragment' or 'compute' to specify shader type."
        )
    else:
        raise NameError("Ambiguous function name: is it a vert, frag or comp shader?")

    # Convert to bytecode
    converter = PyBytecode2Bytecode()
    converter.convert(func, shader_type)
    bytecode = converter.dump()

    return ShaderModule(func, bytecode, f"shader from {func.__name__}")


class PyBytecode2Bytecode:
    """ Convert Python bytecode to our own well-defined bytecode.
    Python bytecode depends on other variables on the code object, and differs
    between Python functions. This class converts this, so that the next step
    of code generation becomes simpler.
    """

    def convert(self, py_func, shader_type):
        self._py_func = py_func
        self._co = self._py_func.__code__

        self._opcodes = []

        self._input = {}
        self._output = {}
        self._uniform = {}
        self._buffer = {}

        # todo: odd, but name must be the same for vertex and fragment shader??
        entrypoint_name = "main"  # py_func.__name__
        self.emit(op.co_entrypoint, entrypoint_name, shader_type, {})

        KINDMAP = {
            "input": (op.co_input, self._input),
            "output": (op.co_output, self._output),
            "uniform": (op.co_uniform, self._uniform),
            "buffer": (op.co_buffer, self._buffer),
        }

        # Parse function inputs
        for i in range(py_func.__code__.co_argcount):
            # Get name and resource object
            argname = py_func.__code__.co_varnames[i]
            if argname not in py_func.__annotations__:
                raise TypeError("Shader arguments must be annotated.")
            resource = py_func.__annotations__.get(argname, None)
            if not isinstance(resource, _types.BaseShaderResource):
                raise TypeError(
                    f"Python-shader arg {argname} must be a resource object (e.g. InputResource)."
                )
            # Get opcode, dict
            try:
                opcode, resource_dict = KINDMAP[resource.kind]
            except KeyError:
                raise TypeError(
                    f"Python-shader arg {argname} has unknown resource type {type(resource)})."
                )
            # Emit and store in our dict
            # todo: change opcode args to just slot, argname, typename
            argname_alt = resource.kind + "." + argname
            self.emit(opcode, resource.slot, {argname_alt: resource.subtype.__name__})
            resource_dict[argname] = resource.subtype

        self._convert()
        self.emit(op.co_func_end)

    def emit(self, opcode, *args):
        if callable(opcode):
            fcode = opcode.__code__
            opcode = fcode.co_name  # a method of OpCodeDefinitions class
            argnames = [fcode.co_varnames[i] for i in range(fcode.co_argcount)][1:]
            if len(args) != len(argnames):
                raise RuntimeError(
                    f"Got {len(args)} args for {opcode}({', '.join(argnames)})"
                )
        self._opcodes.append((opcode, *args))

    def dump(self):
        return self._opcodes

    def _convert(self):

        # co.co_code  # bytes
        #
        # co.co_name
        # co.co_filename
        # co.co_firstlineno
        #
        # co.co_argcount
        # co.co_kwonlyargcount
        # co.co_nlocals
        # co.co_consts
        # co.co_varnames
        # co.co_names  # nonlocal names
        # co.co_cellvars
        # co.co_freevars
        #
        # co.co_stacksize  # the maximum depth the stack can reach while executing the code
        # co.co_flags  # flags if this code object has nested scopes/generators/etc.
        # co.co_lnotab  # line number table  https://svn.python.org/projects/python/branches/pep-0384/Objects/lnotab_notes.txt

        # Pointer in the bytecode stream
        self._pointer = 0

        # Bytecode is a stack machine.
        self._stack = []

        # Python variable names -> (SpirV object id, type_id)
        # self._aliases = {}

        # Parse
        while self._pointer < len(self._co.co_code):
            opcode = self._next()
            opname = dis.opname[opcode]
            method_name = "_op_" + opname.lower()
            method = getattr(self, method_name, None)
            if method is None:
                pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opname} yet (no {method_name}()).")
            else:
                method()

    def _next(self):
        res = self._co.co_code[self._pointer]
        self._pointer += 1
        return res

    def _peak_next(self):
        return self._co.co_code[self._pointer]

    # %%

    def _op_pop_top(self):
        self._next()
        self._stack.pop()
        self.emit(op.co_pop_top)

    def _op_return_value(self):
        self._next()
        result = self._stack.pop()
        assert result is None
        # for now, there is no return in our-bytecode

    def _op_load_fast(self):
        # store a variable that is used in an inner scope.
        i = self._next()
        name = self._co.co_varnames[i]
        if name in self._input:
            self.emit(op.co_load_name, "input." + name)
            self._stack.append("input." + name)
        elif name in self._output:
            self.emit(op.co_load_name, "output." + name)
            self._stack.append("output." + name)
        elif name in self._uniform:
            self.emit(op.co_load_name, "uniform." + name)
            self._stack.append("uniform." + name)
        elif name in self._buffer:
            self.emit(op.co_load_name, "buffer." + name)
            self._stack.append("buffer." + name)
        else:
            # Normal load
            self.emit(op.co_load_name, name)
            self._stack.append(name)

    def _op_store_fast(self):
        i = self._next()
        name = self._co.co_varnames[i]
        ob = self._stack.pop()  # noqa - ob not used
        # we don't prevent assigning to input here, that's the task of bc generator
        if name in self._input:
            self.emit(op.co_store_name, "input." + name)
        elif name in self._output:
            self.emit(op.co_store_name, "output." + name)
        elif name in self._uniform:
            self.emit(op.co_store_name, "uniform." + name)
        elif name in self._buffer:
            self.emit(op.co_store_name, "buffer." + name)
        else:
            # Normal store
            self.emit(op.co_store_name, name)

    def _op_load_const(self):
        i = self._next()
        ob = self._co.co_consts[i]
        if isinstance(ob, (float, int, bool)):
            self.emit(op.co_load_constant, ob)
            self._stack.append(ob)
        elif ob is None:
            self._stack.append(None)  # todo: for the final return ...
        else:
            raise NotImplementedError("Only float/int/bool constants supported.")

    def _op_load_global(self):
        i = self._next()
        name = self._co.co_names[i]
        self.emit(op.co_load_name, name)
        self._stack.append(name)

    def _op_load_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        raise NotImplementedError(f"{ob}.{name} load")

    def _op_store_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        value = self._stack.pop()  # noqa
        raise NotImplementedError(f"{ob}.{name} store")

    def _op_call_function(self):
        nargs = self._next()
        args = self._stack[-nargs:]
        args  # todo: not used?
        self._stack[-nargs:] = []
        func = self._stack.pop()
        if func in spirv_types_map and spirv_types_map[func].is_abstract:
            # A type definition
            type_str = f"{func}({','.join(args)})"
            self._stack.append(type_str)
        else:
            # Normal call
            assert isinstance(func, str)
            self.emit(op.co_call, nargs)
            self._stack.append(None)

    def _op_binary_subscr(self):
        self._next()  # because always 1 arg even if dummy
        index = self._stack.pop()
        ob = self._stack.pop()  # noqa - ob not ised
        if isinstance(index, tuple):
            self.emit(op.co_load_index, len(index))
        else:
            self.emit(op.co_load_index)
        self._stack.append(None)

    def _op_store_subscr(self):
        self._next()  # because always 1 arg even if dummy
        index = self._stack.pop()  # noqa
        ob = self._stack.pop()  # noqa
        val = self._stack.pop()  # noqa
        self.emit(op.co_store_index)

    def _op_build_tuple(self):
        # todo: but I want to be able to do ``x, y = y, x`` !
        raise SyntaxError("No tuples in SpirV-ish Python")

        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = tuple(reversed(res))

        if dis.opname[self._peak_next()] == "BINARY_SUBSCR":
            self._stack.append(res)
            # No emit, in the SpirV bytecode we pop the subscript indices off the stack.
        else:
            raise NotImplementedError("Tuples are not supported.")

    def _op_build_list(self):
        # Litaral list
        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = list(reversed(res))
        self._stack.append(res)
        self.emit(op.co_load_array, n)

    def _op_build_map(self):
        raise SyntaxError("Dict not allowed in Shader-Python")

    def _op_build_const_key_map(self):
        # The version of BUILD_MAP specialized for constant keys. Py3.6+
        raise SyntaxError("Dict not allowed in Shader-Python")

    def _op_binary_add(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binop, "add")

    def _op_binary_subtract(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binop, "sub")

    def _op_binary_multiply(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binop, "mul")

    def _op_binary_true_divide(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binop, "div")
