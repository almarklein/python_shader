""" The opcodes of our bytecode.
"""

# todo: the name is misleading. It's a stack machine representation, but it is never represented in bytes.

import json


def bc2str(opcodes):
    lines = [json.dumps(op)[1:-1] for op in opcodes]
    return "\n".join(lines)


def str2bc(s):
    opcodes = []
    for line in s.splitlines():
        line = line.strip()
        if line:
            opcodes.append(tuple(json.loads("[" + line + "]")))
    return opcodes


class ByteCodeDefinitions:
    """ Abstract class that defines the bytecode ops as methods, making
    it easy to document them (using docstring and arguments.

    Code that produces bytecode can use this as class as a kind of enum
    for the opcodes (and for documentation). Code that consumes bytecode
    can subclass this class and implement the methods.
    """

    # %% High level stuff

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

    def co_call(self, nargs):
        """ Call a function. WIP
        """
        raise NotImplementedError()

    # %% IO

    def co_input(self, location, name_type_items):
        """ Define shader input.
        """
        raise NotImplementedError()

    def co_output(self, location, name_type_items):
        """ Define sader output.
        """
        raise NotImplementedError()

    def co_uniform(self, location, name_type_items):
        """ Define shader uniform.
        """
        raise NotImplementedError()

    def co_buffer(self, location, name_type_items):
        """ Define storage buffer.
        """
        raise NotImplementedError()

    # %% Basics

    def co_pop_top(self):
        """ Pop the top of the stack.
        """
        raise NotImplementedError()

    # todo: local is not the right name, since we use it for globals and io too
    def co_load_local(self, varname):
        """ Load a local variable onto the stack.
        """
        raise NotImplementedError()

    def co_store_local(self, varname):
        """ Store the TOS under the given name, so it can be referenced later
        using co_load_local.
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

    # %% Math and more

    def co_binop(self, op):
        """ Implements TOS = TOS1 ?? TOS, where ?? is the given operation,
        which can be: add, sub, mul, div, ...
        """
        raise NotImplementedError()
        # todo: use a generic binop or one op for each op?

    def co_add(self):
        """ Implements TOS = TOS1 + TOS.
        """
        raise NotImplementedError()

    def co_sub(self):
        """ Implements TOS = TOS1 - TOS.
        """
        raise NotImplementedError()

    def co_mul(self):
        """ Implements TOS = TOS1 * TOS.
        """
        raise NotImplementedError()

    def co_div(self):
        """ Implements TOS = TOS1 / TOS. Float types only.
        """
        raise NotImplementedError()
