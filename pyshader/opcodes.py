""" The opcodes of our bytecode.

Bytecode describing a stack machine is a pretty nice representation to
generate SpirV code, because the code gets visited in a flow, making
it relatively easy to do type inference.

By defining our own bytecode, we can implement a single generator that
consumes it, and use the bytecode as a target for different source
languages. Also, we can target the bytecode towards SpirV, which helps
keeping the generator relatively simple.

Our bytecode consists of a list of tuples, in which the first element
is a (str) opcode, and the remaining elements its arguments. These
opcodes are to be executed in a stack machine.

The term bytecode is a bit odd, because we never really store it as
bytes. But the meaning of the term "bytecode" most closely represents
this intermediate representation of code.

"""

import json


def bc2str(opcodes):
    """ Serialize opcodes to str, one opcode + args per line (hint: it's json).
    """
    lines = [json.dumps(op)[1:-1] for op in opcodes]
    return "\n".join(lines)


def str2bc(s):
    """ Get a list of opcodes (+args) from string.
    """
    opcodes = []
    for line in s.splitlines():
        line = line.strip()
        if line:
            opcodes.append(tuple(json.loads("[" + line + "]")))
    return opcodes


class OpCodeDefinitions:
    """ Abstract class that defines the bytecode ops as methods, making
    it easy to document them (using docstring and arguments).

    Code that produces bytecode can use this as class as a kind of enum
    for the opcodes (and for documentation). Code that consumes bytecode
    can subclass this class and implement the methods.
    """

    def co_func(self, name):
        """ Define a function. WIP
        """
        raise NotImplementedError()

    def co_entrypoint(self, name, shader_type, execution_modes):
        """ Define the start of an entry point function.
        * name (str): The function name.
        * shader_type (str): 'vertex', 'fragment' or 'compute'.
        * execution_modes (dict): a dict with execution modes.
        """
        raise NotImplementedError()

    def co_func_end(self):
        """ Define the end of a function (or entry point).
        """
        raise NotImplementedError()

    def co_return(self):
        """ Return from a function. When the function is a fragment shader
        entrypoint, this means discard.
        """
        raise NotImplementedError()

    def co_call(self, funcname, nargs):
        """ Call a function. The arguments are on the stack. The
        funcname should match a shader-specific type (e.g. f32 or Array),
        a texture sample/read/write function, a function in the stdlib,
        or another defined function (once we implement that).
        """
        raise NotImplementedError()

    def co_resource(self, name, kind, slot, typename):
        """ Define a shader resource, to be available under the given name.
        Kind can be 'input', 'output', 'uniform', 'buffer', 'texture' or 'sampler'.
        Slot is typically an int defining the location/binding slot,
        but can also be a string specifying a builtin (for input and output).
        """
        raise NotImplementedError()

    def co_pop_top(self):
        """ Pop the top of the stack.
        """
        raise NotImplementedError()

    def co_dup_top(self):
        """ Duplicate the top of the stack.
        """
        raise NotImplementedError()

    def co_rotate_stack(self, n):
        """ Rotate the top n elements on the stack, moving the top item
        to the nth position.
        """
        raise NotImplementedError()

    def co_reverse_stack(self, n):
        """ Reverse the top n elements on the stack.
        """
        raise NotImplementedError()

    def co_load_name(self, name):
        """ Load a local variable onto the stack.
        """
        raise NotImplementedError()

    def co_store_name(self, name):
        """ Store the TOS under the given name, so it can be referenced later
        using co_load_name.
        """
        raise NotImplementedError()

    def co_load_index(self):
        """ Implements TOS = TOS1[TOS].
        """
        raise NotImplementedError()

    def co_store_index(self):
        """ Implements TOS1[TOS] = TOS2.
        """
        raise NotImplementedError()

    def co_load_attr(self, name):
        """ Implements TOS = TOS.name. Mostly intended for vector swizzling
        (e.g. pos.xyzw and color.rgba)
        """
        raise NotImplementedError()

    def co_load_constant(self, value):
        """ Load a constant value onto the stack.
        The value can be a float, int, bool. Tuple for vec?
        """
        raise NotImplementedError()

    def co_load_array(self, nargs):
        """ Build an array composed of the nargs last elements on the stack,
        and push that on the stack.
        """
        raise NotImplementedError()

    def co_binary_op(self, op):
        """ Implements TOS = TOS1 ?? TOS, where ?? is the given operation,
        which can be: add, sub, mul, div, fdiv, idiv, and, or, ...
        """
        raise NotImplementedError()

    def co_unary_op(self, op):
        """ A unary operation: neg, not.
        """
        raise NotImplementedError()

    def co_compare(self, cmp):
        """ Comparison operation. cmp can be "<", "<=", "==", "!=", ">", ">=".
        """
        raise NotImplementedError()

    def co_label(self, label):
        """ Start a block with the given label.
        The label must be a unique int or string.
        """
        raise NotImplementedError()

    def co_branch(self, label):
        """ Branch to the target label.
        """
        raise NotImplementedError()

    def co_branch_conditional(self, true_label, false_label):
        """ Branch to true_label if TOS is True, else branch to false_label.

        The control flow of the bytecode must be such that for each
        pair of branches there is a unique label where they merge and
        where no other branches pass through. The exception is that a
        branch-pair, and their sub-branches (and theirs, etc.) can merge
        at the same label (the compiler can resolve this).
        """
        raise NotImplementedError()

    def co_branch_loop(self, iter_label, continue_label, merge_label):
        """ Indicate the beginning of a loop (in a new block).
        """
        raise NotImplementedError()

    def co_select(self):
        """ Select between two values based on a bool.
        If TOS3, select TOS2, otherwise select TOS1. Push
        the selected object on the stack.
        """
        raise NotImplementedError()
