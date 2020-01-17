"""
Implements the base class for generating SpirV code.
"""

import io
import struct

from . import _spirv_constants as cc
from . import _types

# todo: add debug info, most notably line numbers, but can also do source code and object names!
# todo: SpirV supports specialization: setting constants at runtime, can be useful?


# Storage class members, used in OpTypePointer, OpTypeForwardPointer, Opvariable, OpGenericCastToPtrExplicit
STORAGE_CLASSES = dict(
    input=cc.StorageClass_Input,
    output=cc.StorageClass_Output,
    uniform=cc.StorageClass_Uniform,
    buffer=cc.StorageClass_StorageBuffer,
    image=cc.StorageClass_Image,
    # uniform_constant=cc.StorageClass_UniformConstant,
    # private=cc.StorageClass_Private,  # a constant within this module
    # function=cc.StorageClass_Function,  # scoped to the current function invokation
)


def str_to_words(s):
    # In SpirV, words are 32bit. Op counting is per word, not per immediate or per byte.
    b = s.encode()
    padding = 4 - (len(b) % 4)  # 4, 3, 2 or 1 -> always at least 1 for 0-termination
    b += padding * b"\x00"
    assert len(b) % 4 == 0 and b[-1] == 0, b
    words = []
    for i in range(0, len(b), 4):
        words.append(b[i : i + 4])
        # words.append(struct.unpack("<I", b[i : i + 4])[0])
    return words


class AnyId:
    """ Anything that has an id in a SpirV module
    """

    def __init__(self, name=""):
        self.name = name
        self.id = None

    def __repr__(self):
        clsname = self.__class__.__name__
        id = self.id or "?"

        if self.name:
            return f"<{clsname} %{self.name} ({id})>"
        else:
            return f"<{clsname} %{id}>"

    @property
    def display_name(self):
        if self.name:
            return f"%{self.name}"
        else:
            return f"%{self.id or '?'}"

    def resolve(self, gen):
        if self.id is None:
            self.id = len(gen._ids)
            gen._ids[self.id] = self
        return self


class TypeId(AnyId):
    """ A type in a SpirV module.
    """

    def __init__(self, type, name=""):
        super().__init__(name=name)
        self.type = type


class ValueId(AnyId):
    """ Anything that represents a concrete value in a SpirV module.
    """

    def __init__(self, type, name=""):
        super().__init__(name=name)
        self.type = type


class VariableAccessId(ValueId):
    """ A chain of access into a SpirV variable. The type arg is the type
    for the eventual value.
    """

    def __init__(self, variable, storage_class, type, *indices, name=""):
        super().__init__(type, name=name)
        self.variable = variable  # ValueId representing the SpirV Variable
        self.storage_class = storage_class
        self.indices = indices  # ValueId's

    def index(self, index):
        """ Index into the variable chain, so we go one level deeper.
        """
        assert isinstance(index, ValueId)
        assert issubclass(index.type, _types.Int)
        assert issubclass(self.type, _types.Array)  # todo: also support structs
        indices = list(self.indices) + [index]
        return VariableAccessId(
            self.variable, self.storage_class, self.type.subtype, *indices
        )

    def resolve_chain(self, gen):
        """ Generate OpAccessChain instruction and return pointer id object for result.
        """
        if len(self.indices) == 0:
            return self.variable
        else:
            result_type_id = gen.obtain_type_id(self.type)
            pointer_id = gen.obtain_id("pointer")
            gen.gen_instruction(
                "types",
                cc.OpTypePointer,
                pointer_id,
                self.storage_class,
                result_type_id,
            )
            result_id = gen.obtain_id("access-chain-result")
            gen.gen_func_instruction(
                cc.OpAccessChain, pointer_id, result_id, self.variable, *self.indices
            )
        return result_id

    def resolve_load(self, gen):
        """ Generate OpAccessChain instruction followed by OpLoad and return result id.
        """
        temp_id = self.resolve_chain(gen)
        id, type_id = gen.obtain_value(self.type)
        gen.gen_func_instruction(cc.OpLoad, type_id, id, temp_id)
        return id

    def resolve_store(self, gen, val):
        """ Generate OpAccessChain instruction followed by OpStore.
        """
        temp_id = self.resolve_chain(gen)
        gen.gen_func_instruction(cc.OpStore, temp_id, val)
        return val

    # Default resolve is a load
    resolve = resolve_load


class BaseSpirVGenerator:
    """ Base class that can be used by compiler implementations in the
    last compile step to generate the SpirV code. It has an internal
    representation of SpirV module and provides an API to generate
    instructions.
    """

    def convert(self, input):
        """ Generate the Spir-V code. After this, dump() can be used to
        produce the binary blob that represents the Spir-V module.
        """

        # Start clean
        self._init()

        # Do the thing!
        self._convert(input)

        # Wrap up
        self._post_convert()

    def _convert(self, input):
        """ Subclasses should implement this.
        """
        raise NotImplementedError()

    def _init(self):

        self._ids = {0: None}  # maps id -> info. For objects, info is a type in _types
        self._type_name_to_id = {}
        self.scope_stack = []  # stack of dicts: name -> id, type, type_id
        # todo: can we do without a stack, pass everything into funcs?

        # Section 2.4 of the Spir-V spec specifies the Logical Layout of a Module
        self._sections = {
            "capabilities": [],  # 1. All OpCapability instructions.
            "extensions": [],  # 2. Optional OpExtension instructions.
            "extension_imports": [],  # 3. Optional OpExtInstImport instructions.
            "memory_model": [],  # 4. The single required OpMemoryModel instruction.
            "entry_points": [],  # 5. All entry point declarations, using OpEntryPoint.
            "execution_modes": [],  # 6. All execution-mode declarations, using OpExecutionMode or OpExecutionModeId.
            "debug": [],  # 7. The debug instructions, which must be grouped in a specific following order.
            "annotations": [],  # 8. All annotation instructions, e.g. OpDecorate.
            "types": [],  # 9. All type declarations (OpTypeXXX instructions),
            # all constant instructions, and all global
            # variable declarations (all OpVariable instructions whose
            # Storage Class is notFunction). This is the preferred
            # location for OpUndef instructions, though they can also
            # appear in function bodies. All operands in all these
            # instructions must be declared before being used. Otherwise,
            # they can be in any order. This section is the ﬁrst section
            # to allow use of OpLine debug information.
            "function_defs": [],  # 10. All function declarations. A function
            # declaration is as follows.
            # a. Function declaration, using OpFunction.
            # b. Function parameter declarations, using OpFunctionParameter.
            # c. Function end, using OpFunctionEnd.
            "functions": [],  # 11. All function deﬁnitions (functions with a body).
            # A function deﬁnition is as follows:
            # a. Function deﬁnition, using OpFunction.
            # b. Function parameter declarations, using OpFunctionParameter.
            # c. Block, Block ...
            # d. Function end, using OpFunctionEnd.
        }

    def _post_convert(self):
        """ After most of the generation has been done, we set the required capabilities
        and massage the order of instructions a bit.
        """

        # Define memory model (1 instruction)
        self.gen_instruction(
            "memory_model",
            cc.OpMemoryModel,
            cc.AddressingModel_Logical,
            cc.MemoryModel_Simple,
        )

        # Define capabilities. Therea area lot more, and we probably should detect
        # todo: detect capabilities from SpirV stuff being used
        # the cases when we need to define them, and/or let the user define them somehow.
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Matrix)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Shader)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Geometry)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float16)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float64)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_ImageBasic)

        # Move OpVariable to the start of a function
        # Variables are used to refer to either internal variables, or IO, and load/store
        # is used to move variables from/to the stack.
        func_instructions = self._sections["functions"]
        insert_point = -1
        for i in range(len(func_instructions)):
            if func_instructions[i][0] == cc.OpFunction:
                insert_point = -1
            elif insert_point < 0:
                if func_instructions[i][0] == cc.OpLabel:
                    insert_point = i + 1
            elif func_instructions[i][0] == cc.OpVariable:
                func_instructions.insert(insert_point, func_instructions.pop(i))
                insert_point += 1

        # Get ids of global variables
        global_OpVariable_s = []
        for instr in self._sections["types"]:
            if instr[0] == cc.OpVariable:
                # todo: can remove the if below when we move to 1.4 or above
                if instr[-1] in (cc.StorageClass_Input, cc.StorageClass_Output):
                    global_OpVariable_s.append(instr[2])
        # We assume one function, so all are used in our single function
        self._sections["entry_points"][0] = self._sections["entry_points"][0] + tuple(
            global_OpVariable_s
        )

    # %% Utility for compiler

    def to_text(self):
        """ Generate a textual (dis-assembly-like) representation.
        """

        lines = []
        edge = 22

        def disp(pre, pro):
            pre = pre or ""
            line = str(pre.rjust(edge)) + str(pro)
            lines.append(line)

        disp("header ".ljust(edge, "-"), "")
        disp("MagicNumber: ", hex(cc.MagicNumber))
        disp("Version: ", hex(cc.Version))
        disp("VendorId: ", hex(0))
        disp("Bounds: ", len(self._ids))
        disp("Reserved: ", hex(0))

        seen_ids = set()
        for section_name, instructions in self._sections.items():
            # disp(section_name.upper(), "-" * 20)
            disp((section_name + " ").ljust(edge, "-"), "")
            for instruction in instructions:
                instruction_str = repr(instruction[0])
                ret = None
                for i in instruction[1:]:
                    if isinstance(i, AnyId):
                        i_str = i.display_name
                        if instruction[0] == cc.OpDecorate:
                            pass
                        elif i.id not in seen_ids:
                            seen_ids.add(i.id)
                            ret = i.display_name + " = "
                            i_str = f"({i.id})"
                        instruction_str += " " + i_str
                    else:
                        instruction_str += " " + repr(i)
                disp(ret, instruction_str)

        return "\n".join(lines)

    def dump(self):
        """ Generated a bytes object representing the Spir-V module.
        """

        f = io.BytesIO()

        def write_word(w):
            if isinstance(w, bytes):
                assert len(w) == 4
                f.write(w)
            else:
                f.write(struct.pack("<I", w))

        # We pin to version 1.3, since higher versions seem not well supported by drivers
        # version = cc.Version
        # version = struct.unpack("<I", struct.pack("<bbbb", 0, 5, 1, 0))
        version = 66304

        # Write header
        write_word(cc.MagicNumber)  # Magic number
        write_word(version)  # SpirV version
        write_word(0)  # Vendor id - can be zero, let's use zero until we are registered
        write_word(len(self._ids))  # Bound (of ids)
        write_word(0)  # Reserved

        # Write instructions
        for instructions in self._sections.values():
            for opcode, *instr_words in instructions:
                words = []
                for word in instr_words:
                    if isinstance(word, AnyId):
                        words.append(word.id)
                    elif isinstance(word, str):
                        words.extend(str_to_words(word))
                    else:
                        words.append(word)
                write_word(((len(words) + 1) << 16) | opcode)
                for word in words:
                    write_word(word)

        return f.getvalue()

    # %% Utils for subclasses

    def gen_instruction(self, section_name, opcode, *words):
        words = [
            word.resolve(self) if isinstance(word, AnyId) else word for word in words
        ]
        self._sections[section_name].append((opcode, *words))

    def gen_func_instruction(self, opcode, *words):
        words = [
            word.resolve(self) if isinstance(word, AnyId) else word for word in words
        ]
        self._sections["functions"].append((opcode, *words))

    def obtain_id(self, name=""):
        """ Get a new raw id for anything that's not a value or type.
        """
        return AnyId(name=name)

    def obtain_value(self, the_type, name=""):
        """ Create id for a new value. Returns (value_id, type_id).
        """
        assert isinstance(
            the_type, type
        ), f"obtain_value requires a type, not {the_type}"
        assert issubclass(the_type, _types.SpirVType), f"not a spirv type: {the_type}"
        type_id = self.obtain_type_id(the_type)
        value_id = ValueId(the_type)
        return value_id, type_id
        # todo: return only value, and support value.type_id?

    def obtain_variable(self, type, name, storage_class):
        """ Create a variable in the current scope. Generates a variable
        definition instruction. Return the new id.
        """
        pass
        # todo: XXXXX

    def obtain_type_id(self, the_type):
        """ Get the id for the given type_name. Generates a type
        definition instruction as needed.
        """
        assert isinstance(
            the_type, type
        ), f"obtain_type_id requires a type, not {the_type}"
        assert issubclass(the_type, _types.SpirVType), f"not a spirv type: {the_type}"
        assert not the_type.is_abstract, f"not a concrete spirv type: {the_type}"

        # Already know this type?
        if the_type.__name__ in self._type_name_to_id:
            return self._type_name_to_id[the_type.__name__]

        if issubclass(the_type, _types.void):
            type_id = TypeId(the_type)
            self.gen_instruction("types", cc.OpTypeVoid, type_id)
        elif issubclass(the_type, _types.boolean):
            type_id = TypeId(the_type)
            self.gen_instruction("types", cc.OpTypeBool, type_id)
        elif issubclass(the_type, _types.Int):
            type_id = TypeId(the_type)
            bits = 32
            if issubclass(the_type, _types.i16):
                # todo: need OpCapability
                bits = 16
            elif issubclass(the_type, _types.i64):
                bits = 64
            self.gen_instruction(
                "types", cc.OpTypeInt, type_id, bits, 0
            )  # no signedness semantics
        elif issubclass(the_type, _types.Float):
            type_id = TypeId(the_type)
            bits = 32
            if issubclass(the_type, _types.f16):
                # todo: need OpCapability
                bits = 16
            elif issubclass(the_type, _types.f64):
                bits = 64
            self.gen_instruction("types", cc.OpTypeFloat, type_id, bits)
        elif issubclass(the_type, _types.Vector):
            sub_type_id = self.obtain_type_id(the_type.subtype)
            type_id = TypeId(the_type)
            self.gen_instruction(
                "types", cc.OpTypeVector, type_id, sub_type_id, the_type.length
            )
        elif issubclass(the_type, _types.Matrix):
            raise NotImplementedError()
            # OpTypeMatrix
        elif issubclass(the_type, _types.Array):
            count = the_type.length
            sub_type_id = self.obtain_type_id(the_type.subtype)
            if not count:
                # An array for which the length is not known at compile time
                # Use OpArrayLength to get the array length at compile time
                type_id = TypeId(the_type)
                self.gen_instruction(
                    "types", cc.OpTypeRuntimeArray, type_id, sub_type_id
                )
            else:
                # Handle count
                count_value_id, count_type_id = self.obtain_value(
                    _types.i32, "array_count"
                )
                self.gen_instruction(
                    "types", cc.OpConstant, count_type_id, count_value_id, count
                )
                # Handle toplevel array type
                type_id = TypeId(the_type)
                self.gen_instruction(
                    "types", cc.OpTypeArray, type_id, sub_type_id, count_value_id
                )
        elif issubclass(the_type, _types.Struct):
            type_id = TypeId(the_type)
            subtype_ids = [
                self.obtain_type_id(subtype) for subtype in the_type.subtypes
            ]
            self.gen_instruction("types", cc.OpTypeStruct, type_id, *subtype_ids)
        else:
            raise NotImplementedError(the_type)

        # todo: also OpTypeImage and OpTypeSampledImage
        self._type_name_to_id[the_type.__name__] = type_id
        return type_id
