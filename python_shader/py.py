import inspect
from dis import dis as pprint_bytecode
from dis import cmp_op

from ._coreutils import ShaderError
from ._module import ShaderModule
from .opcodes import OpCodeDefinitions as op
from ._dis import dis
from . import stdlib
from ._types import gpu_types_map


OPT_CONVERT_TERNARY_TO_SELECT = True


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

        self._loop_stack = [{}]  # list of dicts

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
                slot = (
                    list(slot) if isinstance(slot, tuple) else slot
                )  # json consistency
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

        if opcode == "co_branch":
            assert not self._opcodes[-1][0].startswith("co_branch")
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
        self._labels_that_start_with_stack_item = set()
        self._labels_that_leave_item_on_stack = set()

        # Parse
        while self._pointer < len(self._co.co_code):
            if self._pointer in self._labels:
                label = self._labels[self._pointer]
                last_opcode = self._opcodes[-1][0]
                if last_opcode not in (
                    "co_branch",
                    "co_branch_conditional",
                    "co_branch_loop",
                ):
                    self.emit(op.co_branch, label)
                    self._detect_jump_is_for_ternary()
                self.emit(op.co_label, label)
                # Keep stack in shape
                # assert not self._stack
                if label in self._labels_that_start_with_stack_item:
                    self._stack.append(None)
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

        self._fix_empty_blocks()
        if OPT_CONVERT_TERNARY_TO_SELECT:
            self.fix_ternaries()
        self._fix_or_control_flow()

    def fix_ternaries(self):

        # The Python ternary operation (.. if xx else ..) results in a
        # control flow using branches/jumps, and non-empty stacks at
        # the end of certain blocks. The SpirV generator can (now)
        # handle that (using Phi merge ops). Nevertheless, it would be
        # nice to be able to do a branch-free select operation in
        # Python, and the ternary op is a great candidate for that. The
        # non-empty blocks make that we can detect the use of the
        # ternary op, so that we can inline the branches and replace
        # the co_branch_conditional with a co_select.

        def _extract_block(label):
            for i in range(0, len(self._opcodes)):
                if self._opcodes[i] == ("co_label", label):
                    break
            i1 = i
            for i in range(i1, len(self._opcodes)):
                if self._opcodes[i][0] in (
                    "co_branch",
                    "co_branch_conditional",
                    "co_return",
                ):
                    break
            i2 = i
            ops = self._opcodes[i1 : i2 + 1]
            self._opcodes[i1 : i2 + 1] = []
            return ops

        i = 0
        current_label = ""
        while i < len(self._opcodes) - 1:
            i += 1
            if self._opcodes[i][0] == "co_label":
                current_label = self._opcodes[i][1]
            elif self._opcodes[i][0] == "co_branch_conditional":
                targets = self._opcodes[i][1:]
                if (
                    targets[0] in self._labels_that_leave_item_on_stack
                    and targets[1] in self._labels_that_leave_item_on_stack
                ):
                    ops = _extract_block(targets[0]), _extract_block(targets[1])
                    merges = ops[0][-1][1], ops[1][-1][1]
                    assert merges[0] == merges[1]
                    assert merges[0] in self._labels_that_start_with_stack_item
                    self._opcodes[i:i] = ops[0][1:-1]
                    i += len(ops[0]) - 2
                    self._opcodes[i:i] = ops[1][1:-1]
                    i += len(ops[1]) - 2
                    self._opcodes[i] = ("co_select",)
                    self._opcodes.insert(i + 1, ("co_branch", merges[0]))
                    self._labels_that_leave_item_on_stack.discard(targets[0])
                    self._labels_that_leave_item_on_stack.discard(targets[1])
                    self._labels_that_leave_item_on_stack.add(current_label)
                    i = 0

        for i in reversed(range(len(self._opcodes))):
            if self._opcodes[i][0] == "co_branch":
                label = self._opcodes[i][1]
                if label in self._labels_that_start_with_stack_item:
                    if self._opcodes[i + 1] == ("co_label", label):
                        self._opcodes[i : i + 2] = []
                    else:
                        raise ShaderError("Failed to resolve ternary control flow.")

    def _fix_empty_blocks(self):

        # Sometimes Python bytecode contains an empty block (i.e. code
        # jumpt to a location, from which it jumps to another location
        # immediately). In such cases, the control flow can be
        # incosistent, with some branches jumping to that empty block,
        # and some skipping it. The code below finds such empty blocks
        # and resolve them.
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
            elif self._opcodes[i][0] in ("co_branch_conditional", "co_branch_loop"):
                op = list(self._opcodes[i])
                changed = False
                for j in range(1, len(op)):
                    if op[j] in labels_to_replace:
                        op[j] = labels_to_replace[op[j]]
                        changed = True
                if changed:
                    self._opcodes[i] = tuple(op)

    def _fix_or_control_flow(self):

        # In `a or b` many languages don't evaluate `b` if `a` evaluates
        # to truethy. This introduces more complex control flow, with
        # multiple branches passing through the same block. SpirV does
        # not allow this. Sadly for us, the bytecode has already
        # resolved `or`'s into control flow ... so we have to detect
        # the pattern. In `a and b`, `b` is not evaluated when `a`
        # evaluates to falsy. But in this case the resulting control
        # flow is fine, and we're probably unable to detect it reliably.

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
                selection.append(("co_binary_op", "or"))
                selection.append(("co_branch_conditional", labels1[0], labels2[1]))
            elif labels1[0] == labels2[1]:  # comp1 is true or comp2 is false
                selection.append(("co_unary_op", "not"))
                selection.append(("co_binary_op", "or"))
                selection.append(("co_branch_conditional", labels1[0], labels2[0]))
            elif labels1[1] == labels2[0]:  # comp1 is false or comp2 is true
                selection.insert(0, ("co_unary_op", "not"))
                selection.append(("co_binary_op", "or"))
                selection.append(("co_branch_conditional", labels1[1], labels2[1]))
            elif labels1[1] == labels2[1]:  # comp1 is false or comp2 is false
                selection.append(("co_binary_op", "and"))
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
        if pointer_pos in self._labels:
            return
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
        if self._pointer == len(self._co.co_code):
            pass
        else:
            self.emit(op.co_return)

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
        elif name == "range":
            self._stack.append(name)
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
        elif func == "range":
            if self._loop_stack[-1].get("range_specified", -1) != 0:
                raise ShaderError("Can only use range() to specify a for-loop")
            self._loop_stack[-1]["range_specified"] = 1
            if len(args) == 1:
                self.emit(op.co_load_constant, 0)
                self.emit(op.co_rot_two)
                self.emit(op.co_load_constant, 1)
            elif len(args) == 2:
                self.emit(op.co_load_constant, 1)
            elif len(args) == 3:
                step = args[2]
                if not (isinstance(step, int) and step > 0):
                    raise ShaderError("range() step must be a constant int > 0")
            else:
                raise ShaderError("range() must have 1, 2 or 3 args.")
            self._stack.append("range")
            # nothing to emit yet
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
        self.emit(op.co_binary_op, "add")

    def _op_binary_subtract(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binary_op, "sub")

    def _op_binary_multiply(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binary_op, "mul")

    def _op_binary_true_divide(self):
        self._next()
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_binary_op, "div")

    def _op_binary_power(self):
        self._next()
        exp = self._stack.pop()
        self._stack.pop()  # base
        self._stack.append(None)
        if exp == 2:  # shortcut
            self.emit(op.co_pop_top)
            self.emit(op.co_dup_top)
            self.emit(op.co_binary_op, "mul")
        else:
            self.emit(op.co_binary_op, "pow")

    def _op_compare_op(self):
        cmp = cmp_op[self._next()]
        if cmp not in ("<", "<=", "==", "!=", ">", ">="):
            raise ShaderError(f"Compare op {cmp} not supported in shaders.")
        self._stack.pop()
        self._stack.pop()
        self._stack.append(None)
        self.emit(op.co_compare, cmp)

    def _detect_jump_is_for_ternary(self):
        if self._stack:
            for i in reversed(range(len(self._opcodes))):
                if self._opcodes[i][0] == "co_label":
                    break
            assert self._opcodes[i][0] == "co_label"
            source_label = self._opcodes[i][1]
            merge_label = self._opcodes[-1][1]
            self._labels_that_leave_item_on_stack.add(source_label)
            self._labels_that_start_with_stack_item.add(merge_label)
            self._stack.pop(-1)

    def _op_jump_absolute(self):
        target = self._next()
        self._set_label(target)
        self.emit(op.co_branch, target)
        self._detect_jump_is_for_ternary()

    def _op_jump_forward(self):
        delta = self._next()
        target = self._pointer + delta
        self._set_label(target)
        if self._opcodes[-1][0].startswith("co_branch"):
            # Is this a Python bug? Below is a snippet of seen Python bytecode.
            # There are no jumps to 28. Maybe there *could* be? If so, we would
            # emit a co_label, and this IF wouldn't triger (and all is well).
            # 26 JUMP_ABSOLUTE           14
            # 28 JUMP_FORWARD            10 (to 40)
            return
        self.emit(op.co_branch, target)
        self._detect_jump_is_for_ternary()

    def _op_pop_jump_if_false(self):
        target = self._next()
        condition = self._stack.pop()  # noqa
        self._set_label(self._pointer)  # Go here if condition is True
        self._set_label(target)  # Go here if condition is False
        self.emit(op.co_branch_conditional, self._pointer, target)
        self._detect_jump_is_for_ternary()
        # todo: spirv supports hints on what branch is the most likely

    def _op_pop_jump_if_true(self):
        target = self._next()
        condition = self._stack.pop()  # noqa
        self._set_label(self._pointer)  # Go here if condition is False
        self._set_label(target)  # Go here if condition is True
        self.emit(op.co_branch_conditional, target, self._pointer)
        self._detect_jump_is_for_ternary()

    # todo: these also exist, and get triggered when a OR or AND is stored as a value
    # _op_jump_if_true_or_pop
    # _op_jump_if_false_or_pop

    def _op_setup_loop(self):
        # This is Python indicating that there is a loop, it indicates
        # where control flow goes further. We will do most of the work
        # in op_for_iter though, because we don't know enough yet.
        delta = self._next()
        loop_info = {
            "merge_label": self._pointer + delta,
            "range_specified": 0,
        }
        self._loop_stack.append(loop_info)

    def _op_break_loop(self):
        self._next()
        self.emit(op.co_branch, self._loop_stack[-1]["merge_label"])

    def _op_continue_loop(self):
        # The Python bytecode seems to just contain the right jumps, and this
        # method is never triggered. But that may differ per Python version.
        target1 = self._next()  # for-iter
        target2 = self._loop_stack[-1]["continue_label"]
        assert target1 == target2
        self.emit(op.co_branch, target2)

    def _op_get_iter(self):
        self._next()
        func = self._stack.pop()
        if func != "range":
            raise ShaderError("Can only use a loop with range()")
        self._stack.append(func)
        # Note: in op_call_function we've already made sure that there are three arg values on the stack

    def _op_for_iter(self):
        delta = self._next()
        here = self._pointer - 2
        target = self._pointer + delta
        loop_info = self._loop_stack[-1]
        assert loop_info.get("merge_label", 0) == target + 2

        # Check that range is specified and prevent further use of range()
        assert self._stack.pop() == "range"
        assert loop_info["range_specified"] == 1, "Loop iter must be a range()"
        loop_info["range_specified"] = 2

        # Consume next codepoint - the storing of the iter value
        next_op = self._next()
        assert dis.opname[next_op] == "STORE_FAST"
        iter_name = self._co.co_varnames[self._next()]
        loop_info["iter_name"] = iter_name

        # Now we have the info we need to setup the loop. We have
        # several blocks to consider. But first define more labels.

        # The header_label represents the "loop header". Any "back edge"
        # (a jump back to the start) must go here (and not to
        # iter_label). In Python, the body jumps to this (FOR_ITER)
        # instruction, so we use it for the header_label.
        loop_info["header_label"] = here - 1
        # The iter_label is the block that follows it, containing a conditional branch.
        loop_info["iter_label"] = here + 1
        # The continue label is what a continue op jumps to. Since we
        # increment the iter value in the iter_label block, we don't have
        # a separate continue block and just jump to the loop start.
        loop_info["continue_label"] = here
        # The body_label represents the body pf the loop. It is what follows after
        # the current Python bytecode instruction.
        loop_info["body_label"] = self._pointer
        # Make sure all labels exist
        self._labels[here] = here
        for key in ["body_label", "merge_label"]:
            self._set_label(loop_info[key])

        # Block 0 (the current block) - prepare iter variable
        # Note that in the range() call, we've put three variables on the stack
        self.emit(op.co_store_name, iter_name + "-step")
        self.emit(op.co_store_name, iter_name + "-stop")
        self.emit(op.co_store_name, iter_name + "-start")
        self.emit(op.co_load_name, iter_name + "-start")
        self.emit(op.co_store_name, iter_name)
        self.emit(op.co_branch, loop_info["header_label"])
        # Block 1 - the "header" of the loop
        self.emit(op.co_label, loop_info["header_label"])
        self.emit(
            op.co_branch_loop,
            loop_info["iter_label"],
            loop_info["continue_label"],
            loop_info["merge_label"],
        )
        # Block 2 - the block that decides whether to break from the loop
        self.emit(op.co_label, loop_info["iter_label"])
        self.emit(op.co_load_name, iter_name)
        self.emit(op.co_load_name, iter_name + "-stop")
        self.emit(op.co_compare, "<")
        self.emit(
            op.co_branch_conditional, loop_info["body_label"], loop_info["merge_label"],
        )
        # Block 3 - the body (can consist of more blocks
        # ... that's what gets processed next
        # Block 4 - the continue label (we emit that in _op_pop_block)

    def _op_pop_block(self):
        self._next()
        # Pop loop from the stack
        loop_info = self._loop_stack.pop(-1)
        # Check merge location
        assert loop_info.get("merge_label", -1) == self._pointer

        # Insert the continue block, which is where the iter value is incremented.
        iter_name = loop_info["iter_name"]
        self.emit(op.co_label, loop_info["continue_label"])
        self.emit(op.co_load_name, iter_name)
        self.emit(op.co_load_name, iter_name + "-step")
        self.emit(op.co_binary_op, "add")
        self.emit(op.co_store_name, iter_name)
        self.emit(op.co_branch, loop_info["header_label"])
