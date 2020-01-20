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


CO_FUNC = "CO_FUNC"
CO_ENTRYPOINT = "CO_ENTRYPOINT"
CO_FUNC_END = "CO_FUNC_END"
CO_INPUT = "CO_INPUT"
CO_OUTPUT = "CO_OUTPUT"
CO_SET_OUTPUT = "CO_SET_OUTPUT"  # todo :.....
CO_UNIFORM = "CO_UNIFORM"
CO_BUFFER = "CO_BUFFER"
CO_ASSIGN = "CO_ASSIGN"
CO_LOAD_CONSTANT = "CO_LOAD_CONSTANT"
CO_LOAD = "CO_LOAD"
CO_BINARY_OP = "CO_BINARY_OP"
CO_STORE = "CO_STORE"
CO_CALL = "CO_CALL"
CO_INDEX = "CO_INDEX"  # INDEX_GET
CO_INDEX_SET = "CO_INDEX_SET"
CO_POP_TOP = "CO_POP_TOP"
CO_BUILD_ARRAY = "CO_BUILD_ARRAY"
