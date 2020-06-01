import inspect
from dis import dis as pprint_bytecode
from dis import cmp_op

from ._coreutils import ShaderError
from ._module import ShaderModule
from .opcodes import OpCodeDefinitions as op
from ._dis import dis
from . import stdlib
from ._types import gpu_types_map


def python2shader(func):
    """ Convert a Python function to a ShaderModule object.

    Takes the bytecode of the given function and converts it to our
    internal bytecode. From there it can be converted to binary SpirV.
    All in dependency-free pure Python.
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
        self._texture = {}
        self._sampler = {}

        # todo: allow user to specify name otherwise?
        entrypoint_name = "main"  # py_func.__name__
        self.emit(op.co_entrypoint, entrypoint_name, shader_type, {})

        KINDMAP = {
            "input": self._input,
            "output": self._output,
            "uniform": self._uniform,
            "buffer": self._buffer,
            "sampler": self._sampler,
            "texture": self._texture,
        }

        # Parse function inputs
        for i in range(py_func.__code__.co_argcount):
            # Get name and resource object
            argname = py_func.__code__.co_varnames[i]
            if argname not in py_func.__annotations__:
                raise TypeError("Shader arguments must be annotated.")
            resource = py_func.__annotations__.get(argname, None)
            if resource is None:
                raise TypeError(f"Python-shader arg {argname} is not decorated.")
            elif isinstance(resource, tuple) and len(resource) == 3:
                kind, slot, subtype = resource
                assert isinstance(kind, str)
                assert isinstance(slot, (int, str, tuple))
                assert isinstance(subtype, (type, str))
            else:
                raise TypeError(
                    f"Python-shader arg {argname} must be a 3-tuple, "
                    + f"not {type(resource)}."
                )
            kind = kind.lower()
            subtype = subtype.__name__ if isinstance(subtype, type) else subtype
            # Get dict to store ref in
            try:
                resource_dict = KINDMAP[kind]
            except KeyError:
                raise TypeError(
                    f"Python-shader arg {argname} has unknown resource kind '{kind}')."
                )
            # Emit and store in our dict
            self.emit(op.co_resource, kind + "." + argname, kind, slot, subtype)
            resource_dict[argname] = subtype

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

        # Keep track of labels
        self._labels = {}

        # Python variable names -> (SpirV object id, type_id)
        # self._aliases = {}

        # Parse
        while self._pointer < len(self._co.co_code):
            if self._pointer in self._labels:
                last_opcode = self._opcodes[-1][0]
                if not last_opcode.startswith("co_branch"):
                    self.emit(op.co_branch, self._labels[self._pointer])
                self.emit(op.co_label, self._labels[self._pointer])
            opcode = self._next()
            opname = dis.opname[opcode]
            method_name = "_op_" + opname.lower()
            method = getattr(self, method_name, None)
            if method is None:
                pprint_bytecode(self._co)
                raise RuntimeError(
                    f"Cannot parse py's {opname} yet (no {method_name}())."
                )
            else:
                method()

        # Python bytecode can be a bit dirty/inconsistent with control flow.
        # find jumps that immediately jump elsewhere and resolve these.
        labels_to_replace = {}
        for i in reversed(range(len(self._opcodes) - 1)):
            if (
                self._opcodes[i][0] == "co_label"
                and self._opcodes[i + 1][0] == "co_branch"
            ):
                labels_to_replace[self._opcodes[i][1]] = self._opcodes[i + 1][1]
                self._opcodes.pop(i)
                self._opcodes.pop(i)
        # Handle if there's more of these
        for key in list(labels_to_replace):
            while labels_to_replace[key] in labels_to_replace:
                labels_to_replace[key] = labels_to_replace[labels_to_replace[key]]
        # Resolve
        for i in range(len(self._opcodes)):
            if self._opcodes[i][0] == "co_branch":
                if self._opcodes[i][1] in labels_to_replace:
                    self._opcodes[i] = (
                        "co_branch",
                        labels_to_replace[self._opcodes[i][1]],
                    )
            elif self._opcodes[i][0] == "co_branch_conditional":
                if self._opcodes[i][1] in labels_to_replace:
                    self._opcodes[i] = (
                        "co_branch_conditional",
                        labels_to_replace[self._opcodes[i][1]],
                        self._opcodes[i][2],
                    )
                if self._opcodes[i][2] in labels_to_replace:
                    self._opcodes[i] = (
                        "co_branch_conditional",
                        self._opcodes[i][1],
                        labels_to_replace[self._opcodes[i][2]],
                    )

        # In `a or b` many languages don't evaluate `b` if `a` evualtes
        # to truethy. This introduces more complex control flow, with
        # multiple branches passing through the same block. SpirV does
        # not allow this. Sadly for us, the bytecode has already
        # resolved `or`'s into control flow ... so we have to detect
        # the pattern.

        def _get_block_to_resolve():
            conditional_branches = {}
            cur_block = None
            cur_block_i = 0
            for i in range(len(self._opcodes)):
                opcode, *args = self._opcodes[i]
                if opcode == "co_label":
                    cur_block = args[0]
                    cur_block_i = i
                elif opcode == "co_branch_conditional":
                    # Detect that this conditional branch is part of an earlier comparison
                    if args[0] in conditional_branches:
                        other, ii = conditional_branches[args[0]]
                        if other == cur_block:
                            return ii, cur_block_i, i
                    elif args[1] in conditional_branches:
                        other, ii = conditional_branches[args[1]]
                        if other == cur_block:
                            return ii, cur_block_i, i
                    # Register this branch (note that this may overwrite keys, which is ok)
                    conditional_branches[args[0]] = args[1], i
                    conditional_branches[args[1]] = args[0], i

        while True:
            block = _get_block_to_resolve()
            if not block:
                break
            i_ins, i_label, i_cond = block
            # Get all the labels
            labels1 = self._opcodes[i_ins][1:]  # this label and the common block
            labels2 = self._opcodes[i_cond][1:]  # the common block and the else
            # Rip out the current label
            selection = self._opcodes[i_label + 1 : i_cond]
            self._opcodes[i_label : i_cond + 1] = []
            # Determine how to combine these
            if labels1[0] == labels2[0]:  # comp1 is true or comp2 is true
                selection.append(("co_binop", "or"))
                selection.append(("co_branch_conditional", labels1[0], labels2[1]))
            elif labels1[0] == labels2[1]:  # comp1 is true or comp2 is false
                selection.append(("co_unary_op", "not"))
                selection.append(("co_binop", "or"))
                selection.append(("co_branch_conditional", labels1[0], labels2[0]))
            elif labels1[1] == labels2[0]:  # comp1 is false or comp2 is true
                selection.insert(0, ("co_unary_op", "not"))
                selection.append(("co_binop", "or"))
                selection.append(("co_branch_conditional", labels1[1], labels2[1]))
            elif labels1[1] == labels2[1]:  # comp1 is false or comp2 is false
                selection.append(("co_binop", "and"))
                selection.append(("co_unary_op", "not"))
                selection.append(("co_branch_conditional", labels1[1], labels2[0]))
            # Put it back in with the parent label
            self._opcodes[i_ins : i_ins + 1] = selection

    def _next(self):
        res = self._co.co_code[self._pointer]
        self._pointer += 1
        return res

    def _peak_next(self):
        return self._co.co_code[self._pointer]

    def _set_label(self, pointer_pos, label=None):
        label = pointer_pos if label is None else label
        if pointer_pos < self._pointer:
            raise RuntimeError(
                "Can (currently) not set labels for bytecode that has already been parsed"
            )
        self._labels[pointer_pos] = label

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
        elif name in self._sampler:
            self.emit(op.co_load_name, "sampler." + name)
            self._stack.append("sampler." + name)
        elif name in self._texture:
            self.emit(op.co_load_name, "texture." + name)
            self._stack.append("texture." + name)
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
        elif name in self._sampler:
            self.emit(op.co_store_name, "sampler." + name)
        elif name in self._texture:
            self.emit(op.co_store_name, "texture." + name)
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
            self._stack.append(None)  # Probably for the function return value
        else:
            raise ShaderError("Only float/int/bool constants supported.")

    def _op_load_global(self):
        i = self._next()
        name = self._co.co_names[i]
        if name == "stdlib":
            self._stack.append(stdlib)
        else:
            self.emit(op.co_load_name, name)
            self._stack.append(name)

    def _op_load_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()  # noqa
        if ob is stdlib:
            func_name = "stdlib." + name
            self._stack.append(func_name)
            self.emit(op.co_load_name, func_name)
        elif isinstance(ob, str) and ob.startswith("texture."):
            func_name = "texture." + name
            self._stack.append(ob)
            self._stack.append(func_name)
            self.emit(op.co_pop_top)
            self.emit(op.co_load_name, func_name)
            self.emit(op.co_load_name, ob)
        else:
            self.emit(op.co_load_attr, name)
            self._stack.append(name)

    def _op_load_method(self):
        i = self._next()
        method_name = self._co.co_names[i]
        ob = self._stack.pop()
        if ob is stdlib:
            func_name = "stdlib." + method_name
            self._stack.append(None)
            self._stack.append(func_name)
            self.emit(op.co_load_name, func_name)
        elif isinstance(ob, str) and ob.startswith("texture."):
            func_name = "texture." + method_name
            self._stack.append(ob)
            self._stack.append(func_name)
            self.emit(op.co_pop_top)
            self.emit(op.co_load_name, func_name)
            self.emit(op.co_load_name, ob)
        else:
            raise ShaderError(
                "Cannot call functions from object, except from texture and stdlib."
            )

    def _op_load_deref(self):
        self._next()
        # ext_ob_name = self._co.co_freevars[i]
        # ext_ob = self._py_func.__closure__[i]
        raise ShaderError("Shaders cannot be used as closures atm.")

    def _op_store_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        value = self._stack.pop()  # noqa
        raise ShaderError(f"{ob}.{name} store")

    def _op_call_function(self):
        nargs = self._next()
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        func = self._stack.pop()
        if func in gpu_types_map and gpu_types_map[func].is_abstract:
            # A type definition
            type_str = f"{func}({','.join(args)})"
            self._stack.append(type_str)
        elif func.startswith("texture."):
            ob = self._stack.pop()
            assert ob.startswith("texture.")  # a texture object
            self.emit(op.co_call, nargs + 1)
            self._stack.append(None)
        else:
            assert isinstance(func, str)
            self.emit(op.co_call, nargs)
            self._stack.append(None)

    def _op_call_method(self):
        nargs = self._next()
        args = self._stack[-nargs:]
        args  # not used
        self._stack[-nargs:] = []

        func = self._stack.pop()
        ob = self._stack.pop()
        assert isinstance(func, str)
        if func.startswith("texture."):
            assert ob.startswith("texture.")  # a texture object
            self.emit(op.co_call, nargs + 1)
            self._stack.append(None)
        else:  # func.startswith("stdlib.")
            assert ob is None
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
        raise ShaderError("No tuples in SpirV-ish Python yet")

        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = tuple(reversed(res))

        if dis.opname[self._peak_next()] == "BINARY_SUBSCR":
            self._stack.append(res)
            # No emit, in the SpirV bytecode we pop the subscript indices off the stack.
        else:
            raise ShaderError("Tuples are not supported.")

    def _op_build_list(self):
        # Litaral list
        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = list(reversed(res))
        self._stack.append(res)
        self.emit(op.co_load_array, n)

    def _op_build_map(self):
        raise ShaderError("Dict not allowed in Shader-Python")

    def _op_build_const_key_map(self):
        # The version of BUILD_MAP specialized for constant keys. Py3.6+
        raise ShaderError("Dict not allowed in Shader-Python")

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

    def _op_binary_power(self):
        self._next()
        exp = self._stack.pop()
        self._stack.pop()  # base
        self._stack.append(None)
        if exp == 2:  # shortcut
            self.emit(op.co_pop_top)
            self.emit(op.co_dup_top)
            self.emit(op.co_binop, "mul")
        else:
            self.emit(op.co_binop, "pow")

    def _op_compare_op(self):
        cmp = cmp_op[self._next()]
        if cmp not in ("<", "<=", "==", "!=", ">", ">="):
            raise ShaderError(f"Compare op {cmp} not supported in shaders.")
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_compare, cmp)

    def _op_jump_absolute(self):
        target = self._next()
        self._set_label(target)
        self.emit(op.co_branch, target)

    def _op_jump_forward(self):
        delta = self._next()
        target = self._pointer + delta
        self._set_label(target)
        self.emit(op.co_branch, target)

    def _op_pop_jump_if_false(self):
        target = self._next()
        condition = self._stack.pop()  # noqa
        self._set_label(self._pointer)  # Go here if condition is True
        self._set_label(target)  # Go here if condition is False
        self.emit(op.co_branch_conditional, self._pointer, target)
        # todo: spirv supports hints on what branch is the most likely

    def _op_pop_jump_if_true(self):
        target = self._next()
        condition = self._stack.pop()  # noqa
        self._set_label(self._pointer)  # Go here if condition is False
        self._set_label(target)  # Go here if condition is True
        self.emit(op.co_branch_conditional, target, self._pointer)

    # todo: these also exist, but I have yet to find out what code results in these code ops
    # _op_jump_if_true_or_pop
    # _op_jump_if_false_or_pop
