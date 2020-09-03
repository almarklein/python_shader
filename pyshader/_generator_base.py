"""
Implements the base class for generating SpirV code.

Notes on exceptions for code that parses/converts/compiles shading code:

* Raise ShaderError when the user-provided code cannot be compiled.
* Raise RuntimeError when the parser/compiler is in an unexpected state.
* Raise NotImplementedError only to indicate that it's an abstract method.
* Always provide an error message.
"""

import io
import struct

from ._coreutils import ShaderError
from . import _spirv_constants as cc
from . import _types


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
    """Anything that has an id in a SpirV module"""

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
    """A type in a SpirV module."""

    def __init__(self, type, name=""):
        super().__init__(name=name)
        self.type = type


class ValueId(AnyId):
    """Anything that represents a concrete value in a SpirV module."""

    def __init__(self, type, name=""):
        super().__init__(name=name)
        self.type = type

    def clone(self, name=""):
        assert self.id is not None
        x = ValueId(self.type, name)
        x.id = self.id
        return x


class VariableAccessId(ValueId):
    """A chain of access into a SpirV variable. The type arg is the type
    for the eventual value. This is a subclass of ValueId because it can usually
    be used in place of one. However, it's only a wrapper class, and it's id
    is always None.
    """

    def __init__(self, variable, storage_class, type, *indices, name=""):
        super().__init__(type, name=name)
        self.variable = variable  # ValueId representing the SpirV Variable
        self.storage_class = storage_class
        self.indices = indices  # ValueId's
        # self.id -> not used

    def clone(self, name=""):
        assert self.variable.id is not None
        x = VariableAccessId(
            self.variable, self.storage_class, self.type, *self.indices, name=name
        )
        return x

    def index(self, index, field=None):
        """Index into the variable chain, so we go one level deeper."""
        assert isinstance(index, ValueId)
        assert issubclass(index.type, _types.Int)
        indices = list(self.indices) + [index]
        if issubclass(self.type, _types.Struct):
            assert isinstance(field, int)
            name = f"{self.name}.{self.type.keys[field]}" if self.name else ""
            return VariableAccessId(
                self.variable,
                self.storage_class,
                self.type.get_subtype(field),
                *indices,
                name=name,
            )
        elif issubclass(self.type, _types.Array):
            assert field is None
            name = f"{self.name}[{index.name or '..'}]" if self.name else ""
            return VariableAccessId(
                self.variable,
                self.storage_class,
                self.type.subtype,
                *indices,
                name=name,
            )
        elif issubclass(self.type, _types.Vector):
            assert field is None
            name = f"{self.name}[{index.name or '..'}]" if self.name else ""
            return VariableAccessId(
                self.variable,
                self.storage_class,
                self.type.subtype,
                *indices,
                name=name,
            )
        else:
            raise ShaderError(f"VariableAccessId cannot index into {self.type}")

    def resolve_chain(self, gen):
        """Generate OpAccessChain instruction and return pointer id object for result."""
        if len(self.indices) == 0:
            return self.variable
        else:
            result_type_id = gen.obtain_type_id(self.type)
            pointer_id = gen.obtain_id()
            gen.gen_instruction(
                "types",
                cc.OpTypePointer,
                pointer_id,
                self.storage_class,
                result_type_id,
            )
            result_id = gen.obtain_id()
            gen.gen_func_instruction(
                cc.OpAccessChain, pointer_id, result_id, self.variable, *self.indices
            )
        return result_id

    def resolve_load(self, gen):
        """Generate OpAccessChain instruction followed by OpLoad and return result id."""
        temp_id = self.resolve_chain(gen)
        id, type_id = gen.obtain_value(self.type, self.name)
        gen.gen_func_instruction(cc.OpLoad, type_id, id, temp_id)
        return id

    def resolve_store(self, gen, val):
        """Generate OpAccessChain instruction followed by OpStore."""
        temp_id = self.resolve_chain(gen)
        gen.gen_func_instruction(cc.OpStore, temp_id, val)
        return val

    # Default resolve is a load
    resolve = resolve_load


class WordPlaceholder:
    """Object that holds an integer value (or 4 bytes), which value
    can be changed as more of the code is parsed. This e.g. allows
    specifying types with knowledge that we encounter later in the
    program.
    """

    def __init__(self, initial_value):
        assert isinstance(initial_value, (int, bytes))
        self.value = initial_value

    def __repr__(self):
        return f"~{self.value}"


class BaseSpirVGenerator:
    """Base class that can be used by compiler implementations in the
    last compile step to generate the SpirV code. It has an internal
    representation of SpirV module and provides an API to generate
    instructions. This class it not aware of our bytecode representation.
    """

    def convert(self, input):
        """Generate the Spir-V code. After this, dump() can be used to
        produce the binary blob that represents the Spir-V module.
        """

        # Start clean
        self._init()

        # Do the thing!
        self._convert(input)

        # Wrap up
        self._post_convert()

    def _convert(self, input):
        """Subclasses should implement this."""
        raise NotImplementedError()  # noqa

    def _init(self):

        self._ids = {0: None}  # maps id -> info. For objects, info is a type in _types
        self._constants = {}
        self._type_hash_to_id = {}
        self._capabilities = set()
        self._execution_modes = {}
        self._extentded_instruction_sets = {}

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
        """After most of the generation has been done, we set the required capabilities
        and massage the order of instructions a bit.
        """

        # Define memory model (1 instruction)
        self.gen_instruction(
            "memory_model",
            cc.OpMemoryModel,
            cc.AddressingModel_Logical,
            cc.MemoryModel_Simple,
        )

        # Write execution modes
        for mode_name, mode_args in self._execution_modes.items():
            self.gen_instruction(
                "execution_modes",
                cc.OpExecutionMode,
                self._entry_point_id,
                getattr(cc, "ExecutionMode_" + mode_name),
                *mode_args,
            )

        # Remove duplicate types. This is required because some types are not
        # "complete" until the shader has been fully parsed. In particular the
        # OpTypeImage.
        type_instructions = self._sections["types"]
        seen_type_defs = {}
        to_remove = []
        for i in range(len(type_instructions)):
            words = type_instructions[i]
            if words[0] not in (cc.OpTypeImage,):
                continue
            # Resolve WordPlaceholder's
            words = tuple(
                w.value if isinstance(w, WordPlaceholder) else w for w in words
            )
            type_instructions[i] = words
            # Get hash of the parts except the TypeId itself, see if we already have it.
            # If so, replace id with the id of the other, and remove from list.
            h = hash(words[:1] + words[2:])
            if h in seen_type_defs:
                words[1].id = seen_type_defs[h][1].id
                to_remove.append(i)
            else:
                seen_type_defs[h] = words
        for i in reversed(to_remove):
            type_instructions.pop(i)

        # Define capabilities. We "collect" these as certain features are used.
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Shader)
        for capability_op in sorted(self._capabilities):
            self.gen_instruction("capabilities", cc.OpCapability, capability_op)

        # Move OpVariable to the start of a function
        # Variables are used to refer to either internal variables, or IO, and load/store
        # is used to move variables from/to the stack.
        func_instructions = self._sections["functions"]
        insert_point = -1
        for i in range(len(func_instructions)):
            if func_instructions[i][0] == cc.OpFunction:
                insert_point = i + 2
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
        """Generate a textual (dis-assembly-like) representation."""

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
                        if instruction[0] in (
                            cc.OpDecorate,
                            cc.OpLoopMerge,
                            cc.OpSelectionMerge,
                            cc.OpBranch,
                            cc.OpBranchConditional,
                        ):
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
        """Generated a bytes object representing the Spir-V module."""

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
                    elif isinstance(word, WordPlaceholder):
                        words.append(word.value)
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
        # Resolve all args for this instruction
        words_resolved = []
        for word in words:
            if isinstance(word, AnyId):
                word = word.resolve(self)
            words_resolved.append(word)
        # Store
        self._sections[section_name].append((opcode, *words_resolved))

    def gen_func_instruction(self, opcode, *words):
        self.gen_instruction("functions", opcode, *words)

    def obtain_id(self, name=""):
        """Get a new raw id for anything that's not a value or type."""
        return AnyId(name=name)

    def obtain_value(self, the_type, name=""):
        """Create id for a new value. Returns (value_id, type_id)."""
        type_id = self.obtain_type_id(the_type)
        value_id = ValueId(the_type, name)
        return value_id, type_id
        # todo: return only value, and support value.type_id?

    def obtain_constant(self, value, the_type=None):
        """Get the id object for the constant of given value.
        Existing constants are re-used.
        """
        # First derive SpirV type from value
        if isinstance(value, float):
            the_type = _types.f32 if the_type is None else the_type
            assert the_type.__name__ != "f16", "Cannot yet create f16 constants."
            M = {"f32": "<f", "f64": "<d"}
            struct_type = M[the_type.__name__]
            bb = struct.pack(struct_type, value)
        elif isinstance(value, bool):  # test before int because issubclass(bool, int)
            the_type = _types.boolean
        elif isinstance(value, int):
            the_type = _types.i32 if the_type is None else the_type
            M = {"u8": "<B", "i16": "<h", "i32": "<i", "i64": "<q"}
            struct_type = M[the_type.__name__]
            bb = struct.pack(struct_type, value)
        else:
            raise RuntimeError(f"Cannot get a constant for {value}")
        # Make sure that we have it
        key = the_type.__name__, value
        if key not in self._constants:
            name = repr(value).lower()
            id, type_id = self.obtain_value(the_type, name)
            if the_type is _types.boolean:
                opcode = cc.OpConstantTrue if value else cc.OpConstantFalse
                self.gen_instruction("types", opcode, type_id, id)
            else:
                self.gen_instruction("types", cc.OpConstant, type_id, id, bb)
            self.gen_instruction("debug", cc.OpName, id.id, name)
            self._constants[key] = id
        # Return cached
        return self._constants[key]

    def obtain_variable(self, the_type, storage_class, name=""):
        """Create a variable in the current scope. Generates an OpVariable
        definition instruction and returns a VariableAccessId to access it.
        """
        # Create id and type_id
        var_id, var_type_id = self.obtain_value(the_type, name)
        #  Create pointer for variable
        var_pointer_id = self.obtain_id()
        self.gen_instruction(
            "types", cc.OpTypePointer, var_pointer_id, storage_class, var_type_id
        )
        # Generate the variable instruction
        where = "types"
        if storage_class in (cc.StorageClass_Function, cc.StorageClass_Private):
            where = "functions"
        self.gen_instruction(
            where, cc.OpVariable, var_pointer_id, var_id, storage_class
        )
        # Mark the name of this variable
        if name:
            self.gen_instruction("debug", cc.OpName, var_id.id, name)
        # Return object that can be used to access the variable with multi-level indexing
        return VariableAccessId(var_id, storage_class, the_type, name=name)

    def obtain_type_id(self, the_type):
        """Get the id for the given type. Generates a type
        definition instruction as needed.
        """
        if isinstance(the_type, tuple):
            if not (
                the_type
                and isinstance(the_type[0], cc.Enum)
                and the_type[0].name.startswith("OpType")
            ):
                raise RuntimeError(
                    "ShaderType can be tuple only if it specifies the OpTypeXYZ"
                )
            type_hash = hash(the_type)
        else:
            if not (
                isinstance(the_type, type) and issubclass(the_type, _types.ShaderType)
            ):
                raise RuntimeError(f"not a ShaderType subclass: {the_type}")
            assert not the_type.is_abstract, f"not a concrete spirv type: {the_type}"
            type_hash = the_type.__name__

        # Already know this type?
        if type_hash in self._type_hash_to_id:
            return self._type_hash_to_id[type_hash]

        if isinstance(the_type, tuple):
            type_id = TypeId(the_type)  # all info is now on TypeId instance
            self.gen_instruction("types", the_type[0], type_id, *the_type[1:])
        elif issubclass(the_type, _types.void):
            type_id = TypeId(the_type)
            self.gen_instruction("types", cc.OpTypeVoid, type_id)
        elif issubclass(the_type, _types.boolean):
            type_id = TypeId(the_type)
            self.gen_instruction("types", cc.OpTypeBool, type_id)
        elif issubclass(the_type, _types.Int):
            type_id = TypeId(the_type)
            if issubclass(the_type, _types.u8):
                self._capabilities.add(cc.Capability_Int8)
                self.gen_instruction("types", cc.OpTypeInt, type_id, 8, 0)
            elif issubclass(the_type, _types.i16):
                self._capabilities.add(cc.Capability_Int16)
                self.gen_instruction("types", cc.OpTypeInt, type_id, 16, 1)
            elif issubclass(the_type, _types.i32):
                self.gen_instruction("types", cc.OpTypeInt, type_id, 32, 1)
            elif issubclass(the_type, _types.i64):
                self._capabilities.add(cc.Capability_Int64)
                self.gen_instruction("types", cc.OpTypeInt, type_id, 64, 1)
            else:
                raise RuntimeError(f"Unknown integer type: {the_type}")
        elif issubclass(the_type, _types.Float):
            type_id = TypeId(the_type)
            if issubclass(the_type, _types.f16):
                self._capabilities.add(cc.Capability_Float16)
                self.gen_instruction("types", cc.OpTypeFloat, type_id, 16)
            elif issubclass(the_type, _types.f32):
                self.gen_instruction("types", cc.OpTypeFloat, type_id, 32)
            elif issubclass(the_type, _types.f64):
                self._capabilities.add(cc.Capability_Float64)
                self.gen_instruction("types", cc.OpTypeFloat, type_id, 64)
            else:
                raise RuntimeError(f"Unknown float type: {the_type}")
        elif issubclass(the_type, _types.Vector):
            sub_type_id = self.obtain_type_id(the_type.subtype)
            type_id = TypeId(the_type)
            self.gen_instruction(
                "types", cc.OpTypeVector, type_id, sub_type_id, the_type.length
            )
        elif issubclass(the_type, _types.Matrix):
            sub_vector_type = _types.Vector(the_type.rows, the_type.subtype)
            column_type_id = self.obtain_type_id(sub_vector_type)
            type_id = TypeId(the_type)
            self.gen_instruction(
                "types", cc.OpTypeMatrix, type_id, column_type_id, the_type.cols
            )
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
                count_value_id = self.obtain_constant(count)
                # Handle toplevel array type
                type_id = TypeId(the_type)
                self.gen_instruction(
                    "types", cc.OpTypeArray, type_id, sub_type_id, count_value_id
                )
        elif issubclass(the_type, _types.Struct):
            type_id = TypeId(the_type)
            subtype_ids = [
                self.obtain_type_id(the_type.get_subtype(key)) for key in the_type.keys
            ]
            self.gen_instruction("types", cc.OpTypeStruct, type_id, *subtype_ids)
        else:
            raise RuntimeError(f"Unknown GPU type {the_type}")

        self._type_hash_to_id[type_hash] = type_id
        return type_id

    def obtain_extended_instruction_set(self, set_name):
        """Obtain the extended instruction set object by the instruction set name.
        The used instruction sets are defined near the top of the SpirV file. The
        resulting id is used in OpExtInst instructions.
        """
        if set_name not in self._extentded_instruction_sets:
            id = self.obtain_id(set_name)
            self.gen_instruction("extension_imports", cc.OpExtInstImport, id, set_name)
            self._extentded_instruction_sets[set_name] = id
        return self._extentded_instruction_sets[set_name]
