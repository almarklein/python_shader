import math
import inspect
from dis import dis as pprint_bytecode
from dis import cmp_op

from ._coreutils import ShaderError
from ._module import ShaderModule
from .opcodes import OpCodeDefinitions as op
from ._dis import dis
from ._types import gpu_types_map
from .stdlib import __all__ as stdlib_func_names


EXTENDED_ARG = dis.opmap["EXTENDED_ARG"]


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

    def show_bytecode(self):
        """ For debugging purposes.
        """
        pprint_bytecode(self._co)

    def convert(self, py_func, shader_type):

        # Attributes of code objects: co_code, co_name, co_filename, co_firstlineno,
        # co_argcount, co_kwonlyargcount, co_nlocals, co_consts, co_varnames,
        # co_names, co_cellvars, co_freevars, co_stacksize, co_flags, co_lnotab
        # -> co_lnotab  is line number table
        #    https://svn.python.org/projects/python/branches/pep-0384/Objects/lnotab_notes.txt
        self._py_func = py_func
        self._co = self._py_func.__code__
        self._py_bytecode = self._co.co_code

        self._opcodes = []  # The resulting "bytecode"

        self._input = {}
        self._output = {}
        self._uniform = {}
        self._buffer = {}
        self._texture = {}
        self._sampler = {}

        # Keep track of labels
        self._labels = {}

        # Protected labels wont automatically generate a co_label,
        # and cannot be resolved if block is empty
        self._protected_labels = set()

        # Bytecode is a stack machine.
        self._stack = []

        # Collect info about loop locations beforehand
        self._loops_to_handle = self._pre_detect_loops()

        # The loop_info objects are popped from the above lists and put on this stack
        self._loop_stack = [{}]  # prepend empty dict to be able to do get()

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

        defaults = list(py_func.__defaults__ or [])
        while len(defaults) < py_func.__code__.co_argcount:
            defaults.insert(0, None)

        # Parse function inputs
        for i in range(py_func.__code__.co_argcount):
            # Get name and resource object
            argname = py_func.__code__.co_varnames[i]
            resource = py_func.__annotations__.get(argname, None)
            if resource is None:
                resource = defaults[i]
            if resource is None:
                raise TypeError(
                    f"pyshader arg {argname} needs type info either as default value or annotation."
                )
            elif isinstance(resource, tuple) and len(resource) == 3:
                kind, slot, subtype = resource
                assert isinstance(kind, str)
                assert isinstance(slot, (int, str, tuple))
                assert isinstance(subtype, (type, str))
                slot = list(slot) if isinstance(slot, tuple) else slot  # json
            else:
                raise TypeError(
                    f"pyshader arg {argname} type info must be a 3-tuple, not {type(resource)}."
                )
            kind = kind.lower()
            subtype = subtype.__name__ if isinstance(subtype, type) else subtype
            # Get dict to store ref in
            try:
                resource_dict = KINDMAP[kind]
            except KeyError:
                raise TypeError(
                    f"pyshader arg {argname} has unknown resource kind '{kind}')."
                )
            # Emit and store in our dict
            self.emit(op.co_resource, kind + "." + argname, kind, slot, subtype)
            resource_dict[argname] = subtype

        self._convert()
        self.emit(op.co_func_end)

    def _stack_pop(self, allow_global=False):
        ob = self._stack.pop()
        if not allow_global:
            if isinstance(ob, str) and ob.startswith("."):
                raise ShaderError(f"Invalid use of (global) {ob[1:]}")
        return ob

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

        self._pointer = 0
        while self._pointer < len(self._py_bytecode):
            if (
                self._loops_to_handle
                and self._pointer == self._loops_to_handle[0]["start"]
            ):
                self._start_loop(self._loops_to_handle.pop(0))
            elif self._pointer == self._loop_stack[-1].get("end"):
                self._end_loop()
            elif (
                self._pointer in self._labels
                and self._pointer not in self._protected_labels
            ):
                label = self._labels[self._pointer]
                last_opcode = self._opcodes[-1][0]
                if last_opcode not in (
                    "co_branch",
                    "co_branch_conditional",
                    "co_branch_loop",
                ):
                    self.emit(op.co_branch, label)
                self.emit(op.co_label, label)
            opname, arg = self._next()
            method_name = "_op_" + opname.lower()
            method = getattr(self, method_name, None)
            if method is None:
                pprint_bytecode(self._co)
                raise RuntimeError(
                    f"Cannot parse py's {opname} yet (no {method_name}())."
                )
            else:
                method(arg)

        # Some post-processing (order is important)
        self._fix_empty_blocks()
        self._fix_or_control_flow()
        self._fix_consistent_labels()

        # Note: at some point we tried to detect ternary ops (xx if yy else zz)
        # and resolved them into op_select. This detection relied on the fact that
        # a ternary op leaves an item at the stack in both its branches.
        # However, this can also happen in: a = b + (c if d else e)
        # In this statement b is put on the stack before entering the ternary,
        # so we'd detect that as part of a ternary. Maybe this can be detected
        # too, but things get complex quickly, and I did not feel confident in
        # this approach anymore. We could consider giving at another shot later,
        # if it matters significantly for performance.

    def _pre_detect_loops(self):

        # Loops can be detected by a jump that goes backwards in the bytecode.
        # We have to examine the bytecode to find the loop structure, and this
        # consists mostly of looking at jumps, so we first detect all jumps.

        # Collect jumps in the bytecode
        jumps = {}
        jump_ops = (
            "JUMP_ABSOLUTE",
            "JUMP_FORWARD",
            "POP_JUMP_IF_FALSE",
            "POP_JUMP_IF_TRUE",
            "JUMP_IF_FALSE_OR_POP",
            "JUMP_IF_TRUE_OR_POP",
        )

        self._pointer = 0
        while self._pointer < len(self._py_bytecode):
            i = self._pointer
            opname, arg = self._next()
            if "JUMP" in opname:
                assert opname in jump_ops
                jumps[i] = (i + 2 + arg) if opname == "JUMP_FORWARD" else arg

        # Look for loop starts
        loop_starts = []
        for i, target in jumps.items():
            if target < i and target not in loop_starts:
                loop_starts.append(target)

        # Sort the starts: this is the order in which thery are encountered!
        loop_starts.sort()

        # Return list of loop_info objects
        loop_infos = []
        for i in range(len(loop_starts)):
            loop_info = self._pre_detect_loop(jumps, loop_infos, loop_starts[i])
            loop_infos.append(loop_info)
        return loop_infos

    def _pre_detect_loop(self, jumps, prev_loops, loop_start):

        # The structure of a for-loop required by SpirV / our internal bytecode:
        #
        # * block zero: the block from which the loop starts
        # * header block: we only have a co_branch_loop here
        # * iter block: ending in a co_branch_conditional that goes to the body or merge block
        # * body block: the loop body
        # * continue block: may increase iter variable, jumps to header block
        # * merge block: the loop ends here

        # We only know that loop_start is the start of the "header block"
        # (in the py bytecode). So we are going to trace all the info we need ...

        # Look for jumps to the header -> to find the end, and continue's
        jumps_to_start = []
        for i, target in jumps.items():
            if target < i and target == loop_start:
                jumps_to_start.append(i)

        # Now we know the end (but there may be two positions to jump to)
        assert len(jumps_to_start) > 0
        our_ends = [jumps_to_start[-1] + 2]
        if self._peek(our_ends[0]) == "POP_BLOCK":
            our_ends.append(our_ends[0] + 2)
        ends = our_ends.copy()
        ends += [x["start"] for x in prev_loops] + [x["end"] for x in prev_loops]

        # Take a look at that first jump. If it jumps to the merge_block,
        # we have a valid iter block.
        first_jump_is_to_end = body_target = None
        for i, target in jumps.items():
            if i > loop_start:
                if target in ends:
                    first_jump_is_to_end = True
                    body_target = i + 2
                elif self._peek(target) == "BREAK_LOOP":
                    first_jump_is_to_end = True
                    body_target = i + 2
                break

        # Check what kind of loop this is
        has_for_iter = self._peek(loop_start) == "FOR_ITER"

        # Init loop info
        loop_info = {}
        loop_info["type"] = "for" if has_for_iter else "while"
        loop_info["start"] = loop_start
        loop_info["end"] = our_ends[-1]
        loop_info["first_jump_is_to_end"] = first_jump_is_to_end

        # Define the labels that we need for the loop structure
        loop_idx = len(prev_loops) + 1
        loop_info["header_label"] = f"Lh{loop_idx}"
        loop_info["iter_label"] = f"Li{loop_idx}"
        loop_info["continue_label"] = f"Lc{loop_idx}"
        loop_info["body_label"] = f"Lb{loop_idx}"
        loop_info["merge_label"] = f"Lm{loop_idx}"

        # Define label mappings
        loop_info["labelmap"] = labelmap = {}

        # The Py bytecode jumps to loop_start become branches to continue_label.
        # Also prevent continue label from being aut-created and collapsed.
        labelmap[loop_start] = loop_info["continue_label"]
        self._protected_labels.add(loop_start)

        # Any jumps to what could mean end-targets should be branches to merge_label.
        for end in ends:
            labelmap[end] = loop_info["merge_label"]
        for end in our_ends:
            self._protected_labels.add(end)

        # If we're not generating the body_label, we want it auto-emitted!
        if first_jump_is_to_end and loop_info["type"] == "while":
            self._labels[body_target] = loop_info["body_label"]
        else:
            pass  # we create a body in _start_loop()

        # Protect the custom labels for collapsing (when empty) for good measure
        for label_id in ["iter_label", "continue_label", "merge_label", "body_label"]:
            self._protected_labels.add(loop_info[label_id])

        return loop_info

    def _replace_labels(self, labels_to_replace):

        # Handle recursion
        for key in list(labels_to_replace):
            while labels_to_replace[key] in labels_to_replace:
                labels_to_replace[key] = labels_to_replace[labels_to_replace[key]]

        # Replace the labels
        for i in range(len(self._opcodes)):
            if self._opcodes[i][0] in ("co_label", "co_branch"):
                if self._opcodes[i][1] in labels_to_replace:
                    self._opcodes[i] = (
                        self._opcodes[i][0],
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

    def _fix_empty_blocks(self):
        # Sometimes Python bytecode contains an empty block (i.e. code
        # jumpt to a location, from which it jumps to another location
        # immediately). In such cases, the control flow can be
        # incosistent, with some branches jumping to that empty block,
        # and some skipping it. The code below finds such empty blocks
        # and resolve them.

        labels_to_replace = {}

        def _set_new_label(label, new_label):
            while label in labels_to_replace:
                label = labels_to_replace[label]
            labels_to_replace[label] = new_label

        for i in reversed(range(len(self._opcodes) - 1)):
            if (
                self._opcodes[i][0] == "co_label"
                and self._opcodes[i + 1][0] == "co_branch"
                and self._opcodes[i][1] not in self._protected_labels
            ):
                _set_new_label(self._opcodes[i][1], self._opcodes[i + 1][1])
                self._opcodes.pop(i)
                self._opcodes.pop(i)

        self._replace_labels(labels_to_replace)

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

    def _fix_consistent_labels(self):
        # Rename the block labels, so that they are numbered in order
        # of appearance of the co_label. This also makes the resulting
        # bytecode consistent between Python versions/implementations.

        labels_to_replace = {}

        def _set_new_label(label, new_label):
            while label in labels_to_replace:
                label = labels_to_replace[label]
            labels_to_replace[label] = new_label

        count = 0
        for i in range(len(self._opcodes)):
            if self._opcodes[i][0] == "co_label":
                label = self._opcodes[i][1]
                if not label.startswith("L"):
                    count += 1
                    _set_new_label(label, f"L{count}")

        self._replace_labels(labels_to_replace)

    def _next(self):
        assert self._pointer % 2 == 0
        opcode = self._py_bytecode[self._pointer]
        arg = self._py_bytecode[self._pointer + 1]
        # Resolve name
        opcode = dis.opname[opcode]
        # Resolve EXTENDED_ARG
        n, i = 1, self._pointer
        while self._py_bytecode[i - 2] == EXTENDED_ARG:
            arg += self._py_bytecode[i - 1] * 256 ** n
            n += 1
            i -= 2
        self._pointer += 2
        return opcode, arg

    def _peek(self, pos=None):
        pos = self._pointer if pos is None else pos
        res = self._py_bytecode[pos]
        if pos % 2 == 0:
            # Resolve name
            res = dis.opname[res]
        else:
            # Resolve EXTENDED_ARG
            n, i = 1, pos - 1
            while self._py_bytecode[i - 2] == EXTENDED_ARG:
                res += self._py_bytecode[i - 1] * 256 ** n
                n += 1
                i -= 2
        return res

    def _get_label(self, pointer_pos):
        loop_labels = self._loop_stack[-1].get("labelmap", {})
        if pointer_pos in loop_labels:
            return loop_labels[pointer_pos]
        elif pointer_pos not in self._labels:
            # Labels are set to bytecode index at first. Later we turn
            # them into values that are consistent across Python
            # versions. The final label starts with "L", and labels
            # starting with "L" will not be renamed.
            self._labels[pointer_pos] = str(pointer_pos)
        return self._labels[pointer_pos]

    # %%

    def _op_extended_arg(self, arg):
        pass

    def _op_dup_top(self, arg):
        ob = self._stack_pop()
        self._stack.extend([ob, ob])
        self.emit(op.co_dup_top)

    def _op_pop_top(self, arg):
        self._stack_pop()
        self.emit(op.co_pop_top)

    def _op_rot_two(self, arg):
        ob1 = self._stack_pop()
        ob2 = self._stack_pop()
        self._stack.extend([ob1, ob2])
        self.emit(op.co_reverse_stack, 2)  # rotate and reverse are same for n = 2

    def _op_rot_three(self, arg):
        ob1 = self._stack_pop()
        ob2 = self._stack_pop()
        ob3 = self._stack_pop()
        self._stack.extend([ob1, ob3, ob2])
        self.emit(op.co_rotate_stack, 3)

    def _op_rot_four(self, arg):  # py 3.8+
        ob1 = self._stack_pop()
        ob2 = self._stack_pop()
        ob3 = self._stack_pop()
        ob4 = self._stack_pop()
        self._stack.extend([ob1, ob4, ob3, ob2])
        self.emit(op.co_rotate_stack, 4)

    def _op_return_value(self, arg):
        result = self._stack_pop()
        assert result is None
        if self._pointer == len(self._py_bytecode):
            pass
        else:
            self.emit(op.co_return)

    def _op_load_fast(self, i):
        # store a variable that is used in an inner scope.
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

    def _op_store_fast(self, i):
        name = self._co.co_varnames[i]
        ob = self._stack_pop()  # noqa - ob not used
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

    def _op_load_const(self, i):
        ob = self._co.co_consts[i]
        if isinstance(ob, (float, int, bool)):
            self.emit(op.co_load_constant, ob)
            self._stack.append(ob)
        elif isinstance(ob, tuple):
            if self._peek() != "UNPACK_SEQUENCE":
                raise ShaderError(
                    "Const tuples are not supported (though you can do `a, b = c, d`)"
                )
            for x in ob:
                if isinstance(x, (float, int, bool)):
                    self.emit(op.co_load_constant, x)
                    self._stack.append(x)
                else:
                    raise ShaderError("Only float/int/bool constants supported.")
            self._stack.append(("tuple", len(ob)))  # signal for UNPACK_SEQUENCE
        elif ob is None:
            self._stack.append(None)  # Probably for the function return value
        else:
            raise ShaderError("Only float/int/bool constants supported.")

    def _op_load_global(self, i):
        # Loading a global in Python can mean different things. We need
        # to check here what it is, and make sure that the loaded thing
        # gets used and results in the correct emitted bytecode. We do
        # not emit code here, but we move a special value on the stack.
        # That value is a string name prepended with a dot, to indicate
        # it being global.
        #
        # When popping a value off the stack, one must indictate whether
        # globals are allowed. This happens only in call_function,
        # call_method, and load_attr. We must make sure that such
        # globals are handled correctly, and do not "slip through",
        # otherwise the user can get really strange error messages
        # because the stack is broken.

        name = self._co.co_names[i]

        if name in gpu_types_map:
            # A type definition
            self._stack.append(".type." + name)
        elif name in stdlib_func_names:
            # An stdlib function, like texture sampling, or ext instruction
            self._stack.append(".stdlib." + name)
        elif name in ("math", "stdlib"):
            # Namespaces, need load_attr on these
            self._stack.append("." + name)
        elif name in ("range",):
            # Builtin functions that we resolve in this compiler
            self._stack.append(".py." + name)
        else:
            raise ShaderError(f"Unknown variable name {name!r}")
        # todo: loading constants from the Python globals() scope
        # todo: loading other Python shader functions

    def _op_load_attr(self, i):
        name = self._co.co_names[i]
        ob = self._stack_pop(True)  # allow global
        if not isinstance(ob, str):
            # Likely vector swizzling
            self.emit(op.co_load_attr, name)
            self._stack.append(None)
        elif ob == ".stdlib":
            if name not in stdlib_func_names:
                raise ShaderError(f"No stdlib function {name}")
            self._stack.append(".stdlib." + name)  # new global on the stack
        elif ob.startswith(".math"):
            ob = getattr(math, name, None)
            if isinstance(ob, float):
                self.emit(op.co_load_constant, ob)  # e.g. math.pi
                self._stack.append(None)
            elif name == "fmod":
                self._stack.append(".py.rem")  # new global on the stack
            elif name in stdlib_func_names:
                self._stack.append(".stdlib." + name)  # new global on the stack
            else:
                raise ShaderError(f"No math constant/function {name}")
        elif ob.startswith("texture."):
            # Calling a texture sampling function as a method on a texture
            # object. Not a global! We need to communicate to call_funcion/call_method
            # that this is such a function.
            self._stack.append(ob)
            self._stack.append("texture." + name)
        elif ob.startswith("."):
            # Catch invalid use of globals
            raise ShaderError(f"Cannot load attribute '{name}' from '{ob}'")
        else:
            self.emit(op.co_load_attr, name)
            self._stack.append(None)

    def _op_load_method(self, i):
        self._stack.append(self._stack[-1])  # for _op_load_attr
        return self._op_load_attr(i)

    def _op_load_deref(self, arg):
        # ext_ob_name = self._co.co_freevars[i]
        # ext_ob = self._py_func.__closure__[i]
        raise ShaderError("Shaders cannot be used as closures atm.")

    def _op_store_attr(self, i):
        name = self._co.co_names[i]
        ob = self._stack_pop()
        value = self._stack_pop()  # noqa
        raise ShaderError(f"{ob}.{name} store")

    def _op_call_function(self, nargs):
        args = [self._stack_pop(True) for i in range(nargs)]
        args.reverse()

        func = self._stack_pop(True)  # allow global

        if not isinstance(func, str):
            raise ShaderError(f"Cannot call object '{func}'.")
        self._call_function(func, args)

    def _op_call_method(self, nargs):
        args = [self._stack_pop() for i in range(nargs)]
        args.reverse()

        func = self._stack_pop(True)  # allow global

        ob = self._stack_pop(True)  # noqa - need to get rid of this here
        assert func.startswith("texture.") or func.startswith(".")

        assert isinstance(func, str)
        self._call_function(func, args)

    def _call_function(self, func, args):
        nargs = len(args)
        funcname = func.split(".", 2)[-1]

        # Args can be globals, but only for e.g. Vector(2, f32)
        if func.startswith(".type."):
            args, ori_args = [], args
            for arg in ori_args:
                if isinstance(arg, str) and arg.startswith(".type."):
                    args.append(arg[6:])
                else:
                    args.append(arg)
        for arg in args:
            if isinstance(arg, str) and arg.startswith("."):
                raise ShaderError(f"Cannot call {func} with arg {arg}")

        if func.startswith("texture."):
            # A texture function called as a method of a texture object
            # This is syntactic sugar. We just need to increase nargs.
            ob = self._stack_pop()
            assert ob.startswith("texture.")  # a texture object
            self.emit(op.co_call, funcname, nargs + 1)
            self._stack.append(None)
        elif func.startswith(".type."):
            # A type definition
            if "(" not in funcname and gpu_types_map[funcname].is_abstract:
                type_str = f"{funcname}({','.join(str(arg) for arg in args)})"
                self._stack.append(".type." + type_str)
            else:
                self.emit(op.co_call, funcname, nargs)
                self._stack.append(None)
        elif func == ".py.rem":
            assert nargs == 2
            self.emit(op.co_binary_op, "rem")
            self._stack.append(None)
        elif func == ".py.range":
            if not (
                self._peek(self._pointer) == "GET_ITER"
                and self._peek(self._pointer + 2) == "FOR_ITER"
            ):
                raise ShaderError("range() can only be used as a for-loop iter.")
            loop_info = self._loops_to_handle[0]
            assert loop_info["start"] == self._pointer + 2
            loop_info["range_is_set"] = True
            if nargs == 1:
                self.emit(op.co_load_constant, 0)
                self.emit(op.co_reverse_stack, 2)
                self.emit(op.co_load_constant, 1)
            elif nargs == 2:
                self.emit(op.co_load_constant, 1)
            elif nargs == 3:
                step = args[2]
                if not (isinstance(step, int) and step > 0):
                    raise ShaderError("range() step must be a constant int > 0")
            else:
                raise ShaderError("range() must have 1, 2 or 3 args.")
            self._stack.append("range")
            # nothing to emit yet
        elif func.startswith((".stdlib.", ".math.")):
            self.emit(op.co_call, funcname, nargs)
            self._stack.append(None)
        elif func.startswith("."):
            raise ShaderError(f"Unknown external function {func}.")
        else:
            raise ShaderError(f"Cannot call object {func}.")

    def _op_binary_subscr(self, arg):
        index = self._stack_pop()
        ob = self._stack_pop()  # noqa - ob not ised
        if isinstance(index, tuple):
            self.emit(op.co_load_index, len(index))
        else:
            self.emit(op.co_load_index)
        self._stack.append(None)

    def _op_store_subscr(self, arg):
        index = self._stack_pop()  # noqa
        ob = self._stack_pop()  # noqa
        val = self._stack_pop()  # noqa
        self.emit(op.co_store_index)

    def _op_build_tuple(self, n):
        if self._peek() == "UNPACK_SEQUENCE":
            # We don't actually build a tuple, but mark that the stack has the values
            self._stack.append(("tuple", n))
        else:
            raise ShaderError(
                "Tuples are not supported (though you can do `a, b = c, d`)"
            )

    def _op_unpack_sequence(self, n):
        x = self._stack_pop()
        if isinstance(x, tuple) and x and x[0] == "tuple":
            # If the number of elements matches, we are all good
            if x[1] == n:
                self.emit(op.co_reverse_stack, n)
                objects = [self._stack.pop() for i in range(n)]
                self._stack.extend(objects)
            else:
                raise ShaderError(f"Cannot unpack a {x[1]} tuple into a {n}-tuple")
        else:
            raise ShaderError(
                "Cannot unpack arbitrary sequences (though you can do `a, b = c, d`)"
            )

    def _op_build_list(self, n):
        # Litaral list
        res = [self._stack_pop() for i in range(n)]
        res = list(reversed(res))
        self._stack.append(res)
        self.emit(op.co_load_array, n)

    def _op_build_map(self, arg):
        raise ShaderError("Dict not allowed in Shader-Python")

    def _op_build_const_key_map(self, arg):
        # The version of BUILD_MAP specialized for constant keys. Py3.6+
        raise ShaderError("Dict not allowed in Shader-Python")

    def _op_unary_positive(self, arg):
        self._stack_pop()
        self._stack.append(None)
        # this is a no-op

    def _op_unary_negative(self, arg):
        self._stack_pop()
        self._stack.append(None)
        self.emit(op.co_unary_op, "neg")

    def _op_unary_not(self, arg):
        self._stack_pop()
        self._stack.append(None)
        self.emit(op.co_unary_op, "not")

    def _binary_op(self, binop):
        self._stack_pop()
        self._stack_pop()
        self._stack.append(None)
        self.emit(op.co_binary_op, binop)

    def _inplace_op(self, binop):
        val = self._stack_pop()  # noqa
        name = self._stack_pop()
        self._stack.append(None)
        assert isinstance(name, str)
        self.emit(op.co_binary_op, binop)

    def _op_inplace_add(self, arg):
        self._inplace_op("add")

    def _op_inplace_subtract(self, arg):
        self._inplace_op("sub")

    def _op_inplace_multiply(self, arg):
        self._inplace_op("mul")

    def _op_inplace_true_divide(self, arg):
        self._inplace_op("fdiv")

    def _op_inplace_floor_divide(self, arg):
        self._inplace_op("idiv")

    def _op_binary_add(self, arg):
        self._binary_op("add")

    def _op_binary_subtract(self, arg):
        self._binary_op("sub")

    def _op_binary_multiply(self, arg):
        self._binary_op("mul")

    def _op_binary_matrix_multiply(self, arg):
        self._binary_op("mmul")

    def _op_binary_true_divide(self, arg):
        # We use the fdiv opcode that only works for floats. Python
        # auto-converts ints to float when dividing. A shader does not.
        # To avoid confusion, users have to use the normal division for
        # floats, and the // division for ints.
        self._binary_op("fdiv")

    def _op_binary_floor_divide(self, arg):
        self._binary_op("idiv")

    def _op_binary_power(self, arg):
        exp = self._stack_pop()
        self._stack_pop()  # base
        self._stack.append(None)
        if exp == 2:  # shortcut
            self.emit(op.co_pop_top)
            self.emit(op.co_dup_top)
            self.emit(op.co_binary_op, "mul")
        elif exp == 0.5:
            self.emit(op.co_pop_top)
            self.emit(op.co_call, "sqrt", 1)
        else:
            self.emit(op.co_call, "pow", 2)

    def _op_binary_modulo(self, arg):
        self._binary_op("mod")

    def _op_compare_op(self, arg):
        cmp = cmp_op[arg]
        if cmp not in ("<", "<=", "==", "!=", ">", ">="):
            raise ShaderError(f"Compare op {cmp} not supported in shaders.")
        self._stack_pop()
        self._stack_pop()
        self._stack.append(None)
        self.emit(op.co_compare, cmp)

    def _op_jump_absolute(self, target):
        label = self._get_label(target)
        if label.startswith("Lm") and self._opcodes[-1][0] == "co_pop_top":
            # This is a break in Python 3.8+ - I think it pops the iterator
            self._opcodes.pop(-1)
        self.emit(op.co_branch, label)

    def _op_jump_forward(self, delta):
        target = self._pointer + delta
        if self._opcodes[-1][0].startswith("co_branch"):
            # Is this a Python bug? Below is a snippet of seen Python bytecode.
            # There are no jumps to 28. Maybe there *could* be? If so, we would
            # emit a co_label, and this IF wouldn't triger (and all is well).
            # 26 JUMP_ABSOLUTE           14
            # 28 JUMP_FORWARD            10 (to 40)
            return
        self.emit(op.co_branch, self._get_label(target))

    def _op_pop_jump_if_false(self, target):
        condition = self._stack_pop()  # noqa
        self.emit(
            op.co_branch_conditional,
            self._get_label(self._pointer),
            self._get_label(target),
        )
        # todo: spirv supports hints on what branch is the most likely

    def _op_pop_jump_if_true(self, target):
        condition = self._stack_pop()  # noqa
        self.emit(
            op.co_branch_conditional,
            self._get_label(target),
            self._get_label(self._pointer),
        )

    def _op_jump_if_true_or_pop(self, target):
        # This is xx OR yy, but only when a result is needed
        # So not inside ``if xx or yy:``, but in ``if bool(xx or yy):``

        # The xx is now on the stack. In the next instructions yy will be
        # pushed on the stack, and at target, we continue. That's where we
        # need to insert the OR.

        # self._insert_at[target] = ("co_binary_op", "or")

        # ... except that determining if an arbitrary object is true
        # or false is not trivial. We could add something like co_bool,
        # but maybe we should avoid that temptation, as it does not fit
        # a strongly typed language well ...
        raise ShaderError(
            "Implicit bool conversions not supported. Maybe use ``x if y else z``?"
        )

    def _op_jump_if_false_or_pop(self, target):
        # Same as _op_jump_if_true_or_pop, but for AND
        # self._insert_at[target] = ("co_binary_op", "and")
        raise ShaderError(
            "Implicit bool conversions not supported. Maybe use ``x if y else z``?"
        )

    def _start_loop(self, loop_info):

        # This gets called right before the first instruction of the loop
        # gets processed. We need to emit some loop-related code here.

        self._loop_stack.append(loop_info)

        if loop_info["type"] == "for":
            # Check that the range is set
            if not loop_info.get("range_is_set"):
                raise ShaderError("Shader for-loop must use range()")

            # Consume next codepoint - the storing of the iter value
            assert self._peek(self._pointer) == "FOR_ITER"
            assert self._peek(self._pointer + 2) == "STORE_FAST"
            iter_name_index = self._peek(self._pointer + 2 + 1)
            iter_name = self._co.co_varnames[iter_name_index]
            loop_info["iter_name"] = iter_name

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
                op.co_branch_conditional,
                loop_info["body_label"],
                loop_info["merge_label"],
            )
            # Block 3 - the body (can consist of more blocks)
            self.emit(op.co_label, loop_info["body_label"])
            # ... the body is what gets processed next
            # The continue_label and merge_label get emitted in _end_loop

        elif loop_info["type"] == "while":

            # Block 0 - the current block
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
            if loop_info["first_jump_is_to_end"]:
                # The self._labels[target] = loop_info["body_label"] has been applied,
                # so the body label (and the branch to it) get generated as we go.
                pass
            else:
                self.emit(op.co_load_constant, True)
                self.emit(
                    op.co_branch_conditional,
                    loop_info["body_label"],
                    loop_info["merge_label"],
                )
                self.emit(op.co_label, loop_info["body_label"])
            # The continue_label and merge_label get emitted in _end_loop

        else:
            raise RuntimeError(f"invalid loop type {loop_info['type'] }")

    def _end_loop(self):

        # This gets called right before the first instruction after the loop.
        # We need to emit some instructions to close up the loop.

        loop_info = self._loop_stack.pop(-1)

        if loop_info["type"] == "for":
            # For-loop: this is where the iter value is incremented.
            iter_name = loop_info["iter_name"]
            self.emit(op.co_label, loop_info["continue_label"])
            self.emit(op.co_load_name, iter_name)
            self.emit(op.co_load_name, iter_name + "-step")
            self.emit(op.co_binary_op, "add")
            self.emit(op.co_store_name, iter_name)
            self.emit(op.co_branch, loop_info["header_label"])
            self.emit(op.co_label, loop_info["merge_label"])
        else:
            # While-loop: just jump to the header. We add two no-op instruction
            # to avoid the branch from being collapsed by our fix_empty_blocks()
            # Note that this does not cause any SPIRV code (except
            # perhaps an unused definition of a constant 0.0)
            self.emit(op.co_label, loop_info["continue_label"])
            self.emit(op.co_branch, loop_info["header_label"])
            self.emit(op.co_label, loop_info["merge_label"])

    def _op_setup_loop(self, delta):
        # This is Python < 3.8 indicating that there is a loop coming. We don't use it.
        self._pointer + delta
        assert self._loops_to_handle[0]["end"]

    def _op_break_loop(self, arg):
        # Python < 3.8
        self.emit(op.co_branch, self._loop_stack[-1]["merge_label"])

    def _op_continue_loop(self, target):
        # This bytecode op is present in Python < 3.8, but does not seem to be
        # used in 3.6 and 3.7 either ...
        target1 = target  # for-iter
        target2 = self._loop_stack[-1]["continue_label"]
        assert target1 == target2
        self.emit(op.co_branch, target2)

    def _op_get_iter(self, arg):
        func = self._stack_pop()
        if func != "range":
            raise ShaderError("Can only use a loop with range()")
        self._stack.append(func)
        # Note: in op_call_function we've already made sure that there are three arg values on the stack

    def _op_for_iter(self, delta):
        # This is the start of a for-loop, but we don't trigger using the method,
        # because our logic needs to take while-loops into account too.
        # But we can do some checks for good measure :)

        target = self._pointer + delta
        here = self._pointer - 2

        next_op, next_val = self._next()  # STORE_FAST, iter variable name
        loop_info = self._loop_stack[-1]

        assert here == loop_info["start"]
        assert target in (loop_info["end"], loop_info["end"] - 2)
        assert next_op == "STORE_FAST"
        assert self._co.co_varnames[next_val] == loop_info["iter_name"]

    def _op_pop_block(self, arg):
        pass
