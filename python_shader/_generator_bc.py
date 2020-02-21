"""
Implements generating SpirV code from our bytecode.
"""

from ._generator_base import BaseSpirVGenerator, ValueId, VariableAccessId
from . import _spirv_constants as cc
from . import _types

from .opcodes import OpCodeDefinitions

# todo: build in some checks
# - expect no func or entrypoint inside a func definition
# - expect other opcodes only inside a func definition
# - expect input/output/uniform at the very start (or inside an entrypoint?)


class Bytecode2SpirVGenerator(OpCodeDefinitions, BaseSpirVGenerator):
    """ A generator that operates on our own well-defined bytecode.

    In essence, this class implements BaseSpirVGenerator by implementing
    the opcode methods of OpCodeDefinitions.
    """

    def _convert(self, bytecode):

        self._stack = []

        # External variables per storage class
        self._input = {}
        self._output = {}
        self._uniform = {}
        self._buffer = {}
        self._slotmap = {}  # (namespaceidentifier, slot) -> name
        self._image = {}  # differentiate between texture and sampler?

        # Resulting values may be given a name so we can pick them up
        self._aliases = {}

        # Parse
        for opcode, *args in bytecode:
            method = getattr(self, opcode.lower(), None)
            if method is None:
                # pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opcode} yet.")
            else:
                method(*args)

    def co_func(self, name):
        raise NotImplementedError()

    def co_entrypoint(self, name, shader_type, execution_modes):
        # Special function definition that acts as an entrypoint

        # Get execution_model flag
        modelmap = {
            "compute": cc.ExecutionModel_GLCompute,  # see also ExecutionModel_Kernel
            "vertex": cc.ExecutionModel_Vertex,
            "fragment": cc.ExecutionModel_Fragment,
            "geometry": cc.ExecutionModel_Geometry,
        }
        execution_model_flag = modelmap.get(shader_type.lower(), None)
        if execution_model_flag is None:
            raise ValueError(f"Unknown execution model: {shader_type}")

        # Define entry points
        # Note that we must add the ids of all used OpVariables that this entrypoint uses.
        entry_point_id = self.obtain_id(name)
        self.gen_instruction(
            "entry_points", cc.OpEntryPoint, execution_model_flag, entry_point_id, name
        )

        # Define execution modes for each entry point
        assert isinstance(execution_modes, dict)
        modes = execution_modes.copy()
        if execution_model_flag == cc.ExecutionModel_Fragment:
            if "OriginLowerLeft" not in modes and "OriginUpperLeft" not in modes:
                modes["OriginLowerLeft"] = []
        if execution_model_flag == cc.ExecutionModel_GLCompute:
            if "LocalSize" not in modes:
                modes["LocalSize"] = [1, 1, 1]
        for mode_name, mode_args in modes.items():
            self.gen_instruction(
                "execution_modes",
                cc.OpExecutionMode,
                entry_point_id,
                getattr(cc, "ExecutionMode_" + mode_name),
                *mode_args,
            )

        # Declare funcion
        return_type_id = self.obtain_type_id(_types.void)
        func_type_id = self.obtain_id("func_declaration")
        self.gen_instruction(
            "types", cc.OpTypeFunction, func_type_id, return_type_id
        )  # 0 args

        # Start function definition
        func_id = entry_point_id
        func_control = 0  # can specify whether it should inline, etc.
        self.gen_func_instruction(
            cc.OpFunction, return_type_id, func_id, func_control, func_type_id
        )
        self.gen_func_instruction(cc.OpLabel, self.obtain_id("label"))

    def co_func_end(self):
        # End function or entrypoint
        self.gen_func_instruction(cc.OpReturn)
        self.gen_func_instruction(cc.OpFunctionEnd)

    def co_call(self, nargs):

        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        func = self._stack.pop()

        if isinstance(func, type):
            assert not func.is_abstract
            if issubclass(func, _types.Vector):
                result = self._vector_packing(func, args)
            elif issubclass(func, _types.Array):
                result = self._array_packing(args)
            elif issubclass(func, _types.Scalar):
                if len(args) != 1:
                    raise TypeError("Scalar convert needs exactly one argument.")
                result = self._convert_scalar(func, args[0])
            self._stack.append(result)
        else:
            raise NotImplementedError()

    # %% IO

    def co_resource(self, name, kind, slot, typename):

        bindgroup = 0
        if isinstance(slot, tuple):
            bindgroup, slot = slot

        # Triage over input kind
        if kind == "input":
            storage_class, iodict = cc.StorageClass_Input, self._input
            # This is also called "shaderLocation" or "slot" in wgpu
            location_or_binding = cc.Decoration_Location
        elif kind == "output":
            storage_class, iodict = cc.StorageClass_Output, self._output
            location_or_binding = cc.Decoration_Location
        elif kind == "uniform":  # slot == binding
            storage_class, iodict = cc.StorageClass_Uniform, self._uniform
            location_or_binding = cc.Decoration_Binding
        elif kind == "buffer":  # slot == binding
            # note: this should be cc.StorageClass_StorageBuffer in SpirV 1.4+
            storage_class, iodict = cc.StorageClass_Uniform, self._buffer
            location_or_binding = cc.Decoration_Binding
        else:
            raise RuntimeError(f"Invalid IO kind {kind}")

        # Check if slot is taken.
        if kind in ("input", "output"):
            # Locations must be unique per kind.
            namespace_id = kind
        else:
            # Bindings must be unique within a bind group.
            namespace_id = "bindgroup-" + str(bindgroup)
        slotmap_key = (namespace_id, slot)
        if slotmap_key in self._slotmap:
            other_name = self._slotmap[slotmap_key]
            raise TypeError(
                f"The {namespace_id} {slot} for {name} already taken by {other_name}."
            )
        else:
            self._slotmap[slotmap_key] = name

        # Get the root variable
        if kind in ("input", "output"):
            var_name = "var-" + name
            var_type = _types.type_from_name(typename)
            subtypes = None
        else:
            # Block - Consider the variable to be a struct
            var_name = "var-" + name
            var_type = _types.type_from_name(typename)
            # Block needs to be a struct
            if issubclass(var_type, _types.Struct):
                subtypes = None
            else:
                subtypes = {name: var_type}
                var_type = _types.Struct(**subtypes)
                var_name = "var-" + var_type.__name__

        # Create VariableAccessId object
        var_access = self.obtain_variable(var_type, storage_class, var_name)
        var_id = var_access.variable

        # Dectorate block for uniforms and buffers
        if kind == "uniform":
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_Block
            )
        elif kind == "buffer":
            # todo: according to docs, in SpirV 1.4+, BufferBlock is deprecated
            # and one should use Block with StorageBuffer. But this crashes.
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_BufferBlock
            )

        # Define slot of variable
        if kind in ("buffer", "image", "uniform"):
            assert isinstance(slot, int)
            # Default to descriptor set zero
            self.gen_instruction(
                "annotations",
                cc.OpDecorate,
                var_id,
                cc.Decoration_DescriptorSet,
                bindgroup,
            )
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_Binding, slot
            )
        elif isinstance(slot, int):
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, location_or_binding, slot
            )
        elif isinstance(slot, str):
            # Builtin input or output
            try:
                slot = cc.builtins[slot]
            except KeyError:
                raise NameError(f"Not a known builtin io variable: {slot}")
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_BuiltIn, slot
            )

        # Store internal info to derefererence the variables
        if subtypes is None:
            if name in iodict:
                raise NameError(f"{kind} {name} already exists")
            iodict[name] = var_access
        else:
            for i, subname in enumerate(subtypes):
                index_id = self.obtain_constant(i)
                if subname in iodict:
                    raise NameError(f"{kind} {subname} already exists")
                iodict[subname] = var_access.index(index_id, i)

    # %% Basics

    def co_pop_top(self):
        self._stack.pop()

    def co_load_name(self, name):
        # store a variable that is used in an inner scope.
        if name in self._aliases:
            ob = self._aliases[name]
        elif name in self._input:
            ob = self._input[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._output:
            ob = self._output[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._uniform:
            ob = self._uniform[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._buffer:
            ob = self._buffer[name]
            assert isinstance(ob, VariableAccessId)
        elif name in _types.spirv_types_map:  # todo: use type_from_name instead?
            ob = _types.spirv_types_map[name]
        else:
            raise NameError(f"Using invalid variable: {name}")
        self._stack.append(ob)

    def co_store_name(self, name):
        ob = self._stack.pop()
        if name in self._output:
            ac = self._output[name]
            ac.resolve_store(self, ob)
        elif name in self._buffer:
            ac = self._buffer[name]
            ac.resolve_store(self, ob)
        elif name in self._input:
            raise SyntaxError("Cannot store to input")
        elif name in self._uniform:
            raise SyntaxError("Cannot store to uniform")

        self._aliases[name] = ob

    def co_load_index(self):
        index = self._stack.pop()
        container = self._stack.pop()

        # Get type of object and index
        element_type = container.type.subtype
        # assert index.type is int

        if isinstance(container, VariableAccessId):
            result_id = container.index(index)

        elif issubclass(container.type, _types.Array):

            # todo: maybe ... the variable for a constant should be created only once ... instead of every time it gets indexed
            # Put the array into a variable
            var_access = self.obtain_variable(container.type, cc.StorageClass_Function)
            container_variable = var_access.variable
            var_access.resolve_store(self, container.id)

            # Prepare result id and type
            result_id, result_type_id = self.obtain_value(element_type)

            # Create pointer into the array
            pointer1 = self.obtain_id("pointer")
            pointer2 = self.obtain_id("pointer")
            self.gen_instruction(
                "types",
                cc.OpTypePointer,
                pointer1,
                cc.StorageClass_Function,
                result_type_id,
            )
            self.gen_func_instruction(
                cc.OpInBoundsAccessChain, pointer1, pointer2, container_variable, index
            )

            # Load the element from the array
            self.gen_func_instruction(cc.OpLoad, result_type_id, result_id, pointer2)
        else:
            raise NotImplementedError()

        self._stack.append(result_id)

        # OpVectorExtractDynamic: Extract a single, dynamically selected, component of a vector.
        # OpVectorInsertDynamic: Make a copy of a vector, with a single, variably selected, component modified.
        # OpVectorShuffle: Select arbitrary components from two vectors to make a new vector.
        # OpCompositeInsert: Make a copy of a composite object, while modifying one part of it. (updating an element)

    def co_store_index(self):
        index = self._stack.pop()
        ob = self._stack.pop()
        val = self._stack.pop()  # noqa

        if isinstance(ob, VariableAccessId):
            # Create new variable access for this last indexing op
            ac = ob.index(index)
            assert val.type is ac.type
            # Then resolve the chain to a store op
            ac.resolve_store(self, val)
        else:
            raise NotImplementedError()

    def co_load_attr(self, name):
        ob = self._stack.pop()

        if isinstance(ob, VariableAccessId) and issubclass(ob.type, _types.Struct):
            # Struct attribute access
            if name not in ob.type.keys:
                raise TypeError(f"Attribute {name} invalid for {ob.type.__name__}.")
            # Create new variable access for this attr op
            index = ob.type.keys.index(name)
            ac = ob.index(self.obtain_constant(index), index)
            self._stack.append(ac)
        elif issubclass(ob.type, _types.Vector):
            indices = []
            for c in name:
                if c in "xr":
                    indices.append(0)
                elif c in "yg":
                    indices.append(1)
                elif c in "zb":
                    indices.append(2)
                elif c in "wa":
                    indices.append(3)
                else:
                    raise AttributeError(name)
            # elements = []
            # for i in indices:
            #     elements.append(ob.index(self.obtain_constant(i)))
            result_type = _types.Vector(len(indices), ob.type.subtype)
            result_id, type_id = self.obtain_value(result_type)
            self.gen_func_instruction(
                cc.OpVectorShuffle, type_id, result_id, ob, ob, *indices
            )
            self._stack.append(result_id)
        else:
            # todo: not implemented for non VariableAccessId
            raise NotImplementedError()

    def co_load_constant(self, value):
        id = self.obtain_constant(value)
        self._stack.append(id)
        # Also see OpConstantNull OpConstantSampler OpConstantComposite

    def co_load_array(self, nargs):
        # Literal array
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        result = self._array_packing(args)
        self._stack.append(result)

    # %% Math and more

    def co_binop(self, op):

        val2 = self._stack.pop()
        val1 = self._stack.pop()

        # The ids that will be in the instruction, can be reset
        id1, id2 = val1, val2

        # Predefine some types
        scalar_or_vector = _types.Scalar, _types.Vector
        FOPS = dict(add=cc.OpFAdd, sub=cc.OpFSub, mul=cc.OpFMul, div=cc.OpFDiv)
        IOPS = dict(add=cc.OpIAdd, sub=cc.OpISub, mul=cc.OpIMul)

        # Get reference types
        type1 = val1.type
        reftype1 = type1
        if issubclass(type1, (_types.Vector, _types.Matrix)):
            reftype1 = type1.subtype
        type2 = val2.type
        reftype2 = type2
        if issubclass(type2, (_types.Vector, _types.Matrix)):
            reftype2 = type2.subtype

        tn1 = type1.__name__
        tn2 = type2.__name__

        if reftype1 is not reftype2:
            # Let's start by excluding cases where the subtypes differ.
            raise TypeError(
                f"Cannot {op} two values with different (sub)types: {tn1} and {tn2}"
            )

        elif type1 is type2 and issubclass(type1, scalar_or_vector):
            # Types are equal and scalar or vector. Covers a lot of cases.
            result_id, type_id = self.obtain_value(type1)
            if issubclass(reftype1, _types.Float):
                opcode = FOPS[op]
            elif issubclass(reftype1, _types.Int):
                opcode = IOPS[op]
            else:
                raise TypeError("Cannot {op} values of type {tn1}.")

        elif issubclass(type1, _types.Scalar) and issubclass(type2, _types.Vector):
            # Convenience - add/mul vectors with scalars
            if not issubclass(reftype1, _types.Float):
                raise TypeError(f"Scalar {op} Vector only supported for float subtype.")
            result_id, type_id = self.obtain_value(type2)  # result is vector
            if op == "mul":
                opcode = cc.OpVectorTimesScalar
                id1, id2 = val2, val1  # swap to put vector first
            else:
                opcode = FOPS[op]
                val3 = self._vector_packing(type2, [val1] * type2.length)
                id1, id2 = val1, val3

        elif issubclass(type1, _types.Vector) and issubclass(type2, _types.Scalar):
            # Convenience - add/mul vectors with scalars, opposite order
            if not issubclass(reftype1, _types.Float):
                raise TypeError(f"Vector {op} Scalar only supported for float subtype.")
            result_id, type_id = self.obtain_value(type1)  # result is vector
            if op == "mul":
                opcode = cc.OpVectorTimesScalar
                id1, id2 = val1, val2
            else:
                opcode = FOPS[op]
                val3 = self._vector_packing(type1, [val2] * type1.length)
                id1, id2 = val1, val3

        elif op != "mul":
            # The remaining cases are all limited to multiplication
            raise TypeError(f"Cannot {op} {tn1} and {tn2}, multiply only.")

        elif not issubclass(reftype1, _types.Float):
            # The remaining cases are all limited to float types
            raise TypeError(f"Cannot {op} {tn1} and {tn2}, float only.")

        # With that out of the way, the remaining cases are quite short to write.

        elif issubclass(type1, _types.Matrix) and issubclass(type2, _types.Matrix):
            # Multiply two matrices
            if type1.cols != type2.rows:
                raise TypeError(f"Cannot {op} two matrices with incompatible shapes.")
            type3 = _types.Matrix(type2.cols, type1.rows, type1.subtype)
            result_id, type_id = self.obtain_value(type3)
            opcode = cc.OpMatrixTimesMatrix

        elif issubclass(type1, _types.Matrix) and issubclass(type2, _types.Scalar):
            # Matrix times vector
            result_id, type_id = self.obtain_value(type1)  # Result is a matrix
            opcode = cc.OpMatrixTimesScalar
            id1, id2 = val1, val2

        elif issubclass(type1, _types.Matrix) and issubclass(type2, _types.Scalar):
            # Matrix times vector, opposite order
            result_id, type_id = self.obtain_value(type2)  # Result is a matrix
            opcode = cc.OpMatrixTimesScalar
            id1, id2 = val2, val1  # reverse

        elif issubclass(type1, _types.Matrix) and issubclass(type2, _types.Vector):
            # Matrix times Vector
            if type2.length != type1.cols:
                raise TypeError(f"Incompatible shape for {tn1} x {tn2}")
            type3 = _types.Vector(type1.rows, type1.subtype)
            result_id, type_id = self.obtain_value(type3)
            opcode = cc.OpMatrixTimesVector

        elif issubclass(type1, _types.Vector) and issubclass(type2, _types.Matrix):
            # Vector times Matrix
            if type1.length != type2.rows:
                raise TypeError(f"Incompatible shape for {tn1} x {tn2}")
            type3 = _types.Vector(type2.cols, type2.subtype)
            result_id, type_id = self.obtain_value(type3)
            opcode = cc.OpVectorTimesMatrix

        else:
            raise TypeError(f"Cannot {op} values of {tn1} and {tn2}.")

        self.gen_func_instruction(opcode, type_id, result_id, id1, id2)
        self._stack.append(result_id)

    # %% Helper methods

    def _convert_scalar(self, out_type, arg):
        return self._convert_scalar_or_vector(out_type, out_type, arg, arg.type)

    def _convert_numeric_vector(self, out_type, arg):
        if not (
            issubclass(arg.type, _types.Vector) and arg.type.length == out_type.length
        ):
            raise TypeError("Vector conversion needs vectors of equal length.")
        return self._convert_scalar_or_vector(
            out_type, out_type.subtype, arg, arg.type.subtype
        )

    def _convert_scalar_or_vector(self, out_type, out_el_type, arg, arg_el_type):

        # This function only works for vectors for numeric types (no bools)
        if out_type is not out_el_type:
            assert issubclass(out_el_type, _types.Numeric) and issubclass(
                arg_el_type, _types.Numeric
            )

        # Is a conversion actually needed?
        if arg.type is out_type:
            return arg

        # Otherwise we need a new value
        result_id, type_id = self.obtain_value(out_type)

        argtname = arg_el_type.__name__
        outtname = out_el_type.__name__

        if issubclass(out_el_type, _types.Float):
            if issubclass(arg_el_type, _types.Float):
                self.gen_func_instruction(cc.OpFConvert, type_id, result_id, arg)
            elif issubclass(arg_el_type, _types.Int):
                op = cc.OpConvertSToF if argtname.startswith("u") else cc.OpConvertUToF
                self.gen_func_instruction(op, type_id, result_id, arg)
            elif issubclass(arg_el_type, _types.boolean):
                zero = self.obtain_constant(0.0, out_el_type)
                one = self.obtain_constant(1.0, out_el_type)
                self.gen_func_instruction(
                    cc.OpSelect, type_id, result_id, arg, one, zero
                )
            else:
                raise TypeError(f"Cannot convert to float: {arg.type}")

        elif issubclass(out_el_type, _types.Int):
            if issubclass(arg_el_type, _types.Float):
                op = cc.OpConvertFToU if outtname.startswith("u") else cc.OpConvertFToS
                self.gen_func_instruction(cc.OpConvertFToS, type_id, result_id, arg)
            elif issubclass(arg_el_type, _types.Int):
                op = cc.OpUConvert if outtname.startswith("u") else cc.OpSConvert
                self.gen_func_instruction(cc.OpSConvert, type_id, result_id, arg)
            elif issubclass(arg_el_type, _types.boolean):
                zero = self.obtain_constant(0, out_type)
                one = self.obtain_constant(1, out_type)
                self.gen_func_instruction(
                    cc.OpSelect, type_id, result_id, arg, one, zero
                )
            else:
                raise TypeError(f"Cannot convert to int: {arg.type}")

        elif issubclass(out_el_type, _types.boolean):
            if issubclass(arg_el_type, _types.Float):
                zero = self.obtain_constant(0.0, arg_el_type)
                self.gen_func_instruction(
                    cc.OpFOrdNotEqual, type_id, result_id, arg, zero
                )
            elif issubclass(arg_el_type, _types.Int):
                zero = self.obtain_constant(0, arg_el_type)
                self.gen_func_instruction(cc.OpINotEqual, type_id, result_id, arg, zero)
            elif issubclass(arg_el_type, _types.boolean):
                return arg  # actually covered above
            else:
                raise TypeError(f"Cannot convert to bool: {arg.type}")
        else:
            raise TypeError(f"Cannot convert to {out_type}")

        return result_id

    def _vector_packing(self, vector_type, args):

        # Vector conversion of numeric types is easier
        if (
            len(args) == 1
            and issubclass(vector_type.subtype, _types.Numeric)
            and issubclass(args[0].type.subtype, _types.Numeric)
        ):
            return self._convert_numeric_vector(vector_type, args[0])

        n, t = vector_type.length, vector_type.subtype  # noqa
        composite_ids = []

        # Deconstruct
        for arg in args:
            if not isinstance(arg, ValueId):
                raise RuntimeError("Expected a SpirV object")
            if issubclass(arg.type, _types.Scalar):
                comp_id = arg
                if arg.type is not t:
                    comp_id = self._convert_scalar(t, arg)
                composite_ids.append(comp_id)
            elif issubclass(arg.type, _types.Vector):
                # todo: a contiguous subset of the scalars consumed can be represented by a vector operand instead!
                # -> I think this means we can simply do composite_ids.append(arg)
                for i in range(arg.type.length):
                    comp_id, comp_type_id = self.obtain_value(arg.type.subtype)
                    self.gen_func_instruction(
                        cc.OpCompositeExtract, comp_type_id, comp_id, arg, i
                    )
                    if arg.type.subtype is not t:
                        comp_id = self._convert_scalar(t, comp_id)
                    composite_ids.append(comp_id)
            else:
                raise TypeError(f"Invalid type to compose vector: {arg.type}")

        # Check the length
        if len(composite_ids) != n:
            raise TypeError(
                f"{vector_type} did not expect {len(composite_ids)} elements"
            )

        assert (
            len(composite_ids) >= 2
        ), "When constructing a vector, there must be at least two Constituent operands."

        # Construct
        result_id, vector_type_id = self.obtain_value(vector_type)
        self.gen_func_instruction(
            cc.OpCompositeConstruct, vector_type_id, result_id, *composite_ids
        )
        # todo: or OpConstantComposite
        return result_id

    def _array_packing(self, args):
        n = len(args)
        if n == 0:
            raise IndexError("No support for zero-sized arrays.")

        # Check that all args have the same type
        element_type = args[0].type
        composite_ids = args
        for arg in args:
            assert arg.type is element_type, "array type mismatch"

        # Create array class
        array_type = _types.Array(n, element_type)

        result_id, type_id = self.obtain_value(array_type)
        self.gen_func_instruction(
            cc.OpCompositeConstruct, type_id, result_id, *composite_ids
        )
        # todo: or OpConstantComposite

        return result_id
