"""
Implements generating SpirV code from our bytecode.
"""

import ctypes

from ._generator_base import (
    BaseSpirVGenerator,
    ValueId,
    VariableAccessId,
    WordPlaceholder,
)
from ._coreutils import ShaderError
from . import _spirv_constants as cc
from . import _types
from .stdlib import tex_functions, ext_functions
from .opcodes import OpCodeDefinitions

# todo: build in some checks
# - expect no func or entrypoint inside a func definition
# - expect other opcodes only inside a func definition
# - expect input/output/uniform at the very start (or inside an entrypoint?)


image_formats_that_need_no_ext = {
    cc.ImageFormat_Rgba32f,
    cc.ImageFormat_Rgba16f,
    cc.ImageFormat_Rgba32i,
    cc.ImageFormat_Rgba32ui,
    cc.ImageFormat_Rgba16i,
    cc.ImageFormat_Rgba16ui,
    cc.ImageFormat_Rgba8,
    cc.ImageFormat_Rgba8i,
    cc.ImageFormat_Rgba8ui,
    cc.ImageFormat_Rgba8Snorm,
    cc.ImageFormat_R32f,
    cc.ImageFormat_R32i,
    cc.ImageFormat_R32ui,
}


class Bytecode2SpirVGenerator(OpCodeDefinitions, BaseSpirVGenerator):
    """ A generator that operates on our own well-defined bytecode.

    In essence, this class implements BaseSpirVGenerator by implementing
    the opcode methods of OpCodeDefinitions.
    """

    def show_bytecode(self):
        """ For debugging purposes.
        """
        for x in self._bytecode:
            print(x)

    def _convert(self, bytecode):

        self._bytecode = bytecode
        self._execution_model_flag = None

        self._stack = []
        self._stack_for_phi = {}  # label -> []

        # Track loops. The bottom of the stack is an empty dict, for convenience
        self._loop_stack = [{}]

        # External variables per storage class
        self._input = {}
        self._output = {}
        self._uniform = {}
        self._buffer = {}
        self._sampler = {}
        self._texture = {}
        self._slotmap = {}  # (namespaceidentifier, slot) -> name

        self._decorated_array_types = set()

        # We keep track of sampler for each combination of texture and sampler
        self._texture_samplers = {}

        # Keep track what id a name was saved by. Some need to be stored in variables.
        self._name_ids = {}  # name -> ValueId
        self._name_variables = {}  # name -> VariableAccessId

        # Labels for control flow
        self._labels = {}
        self._root_branch = {"depth": 0, "label": "", "children": ()}
        self._current_branch = self._root_branch

        # Pre-parse the bytecode, to detect what variables names need
        # to be stored in a func-variable, and what can be used via the
        # (faster and easier) SSA form.
        cur_block_label = ""
        saved_in_blocks = {}  # name -> set of labels
        self._need_name_var_save = {}  # label -> set of names
        self._need_name_var_load = {}  # label -> set of names
        for opcode, *args in bytecode:
            if opcode == "co_label":
                cur_block_label = args[0]
            elif opcode == "co_store_name":
                name = args[0]
                saved_in_blocks.setdefault(name, set()).add(cur_block_label)
            elif opcode == "co_load_name":
                name = args[0]
                blocks_where_name_is_saved = saved_in_blocks.get(name, ())
                if cur_block_label in blocks_where_name_is_saved:
                    pass  # no need to use a variable
                else:
                    s = self._need_name_var_load.setdefault(cur_block_label, set())
                    s.add(name)
        # Also mark the blocks where we need to save the variable
        for names in self._need_name_var_load.values():
            for name in names:
                blocks_where_name_is_saved = saved_in_blocks.get(name, ())
                for block_label in blocks_where_name_is_saved:
                    s = self._need_name_var_save.setdefault(block_label, set())
                    s.add(name)

        # Parse
        for opcode, *args in bytecode:
            method = getattr(self, opcode.lower(), None)
            if method is None:
                # pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opcode} yet.")
            else:
                method(*args)

    def _get_label_id(self, label_value):
        if label_value not in self._labels:
            label_id = self.obtain_id(f"label-{label_value}")
            label_id.resolve(self)
            self._labels[label_value] = label_id
        return self._labels[label_value]

    def co_func(self, name):
        raise ShaderError("No sub-functions yet")

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
        self._execution_model_flag = execution_model_flag
        if execution_model_flag is None:
            raise ShaderError(f"Unknown execution model: {shader_type}")

        # Define entry points
        # Note that we must add the ids of all used OpVariables that this entrypoint uses.
        self._entry_point_id = entry_point_id = self.obtain_id(name)
        self.gen_instruction(
            "entry_points", cc.OpEntryPoint, execution_model_flag, entry_point_id, name
        )

        # Define execution modes for each entry point
        assert isinstance(execution_modes, dict)
        self._execution_modes.update(execution_modes)
        modes = self._execution_modes
        if execution_model_flag == cc.ExecutionModel_Fragment:
            if "OriginLowerLeft" not in modes and "OriginUpperLeft" not in modes:
                modes["OriginLowerLeft"] = []
        if execution_model_flag == cc.ExecutionModel_GLCompute:
            if "LocalSize" not in modes:
                modes["LocalSize"] = [1, 1, 1]

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
        if self._current_branch is not self._root_branch:
            raise RuntimeError("Function ends with unresolved sub-branches!")
        # End function or entrypoint
        self.gen_func_instruction(cc.OpReturn)
        self.gen_func_instruction(cc.OpFunctionEnd)

    def co_return(self):
        # A discard is only allowed in a fragment shader. Later we might
        # add helper functions, in which case co_return has another
        # function.
        if self._execution_model_flag == cc.ExecutionModel_Fragment:
            self.gen_func_instruction(cc.OpKill)
        else:
            raise ShaderError("Unexpected return/discard")

    def co_call(self, funcname, nargs):

        assert len(self._stack) >= nargs
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []

        assert isinstance(funcname, str)

        if funcname in _types.gpu_types_map:
            # A common type, below we also check for more complex type expressions
            ty = _types.gpu_types_map[funcname]
            self._stack.append(self._typecast(ty, args))
        elif funcname in tex_functions:
            self._texture_call(funcname, args)
        elif funcname in ext_functions:
            self._ext_instruction_call(funcname, args)
        else:
            # Well, it could be a more special type ... try to convert!
            try:
                ty = _types.type_from_name(funcname)
            except Exception:
                ty = None
            if ty is not None:
                self._stack.append(self._typecast(ty, args))
            else:
                raise ShaderError(f"Using invalid function call: {funcname}")

    def _typecast(self, ty, args):
        assert not ty.is_abstract
        if issubclass(ty, _types.Vector):
            result = self._vector_packing(ty, args)
        elif issubclass(ty, _types.Array):
            result = self._array_packing(args)
        elif issubclass(ty, _types.Scalar):
            if len(args) != 1:
                raise ShaderError("Scalar convert needs exactly one argument.")
            result = self._convert_scalar(ty, args[0])
        return result

    def _texture_call(self, funcname, args):
        if funcname in ("imageLoad", "read"):
            tex, coord = args
            self._capabilities.add(cc.Capability_StorageImageReadWithoutFormat)
            tex.depth.value, tex.sampled.value = 0, 2
            if coord.type not in (_types.i32, _types.ivec2, _types.ivec3):
                raise ShaderError("Expected texture coords to be i32, ivec2 or ivec3.")
            vec_sample_type = _types.Vector(4, tex.sample_type)
            result_id, type_id = self.obtain_value(vec_sample_type)
            self.gen_func_instruction(
                cc.OpImageRead, type_id, result_id, tex, coord,
            )
            self._stack.append(result_id)
        elif funcname in ("imageStore", "write"):
            tex, coord, color = args
            self._capabilities.add(cc.Capability_StorageImageWriteWithoutFormat)
            tex.depth.value, tex.sampled.value = 0, 2
            if coord.type not in (_types.i32, _types.ivec2, _types.ivec3):
                raise ShaderError("Expected texture coords to be i32, ivec2 or ivec3.")
            if tex.sample_type is _types.i32 and color.type is not _types.ivec4:
                raise ShaderError(
                    f"Expected texture value to be ivec4, not {color.type}"
                )
            elif tex.sample_type is _types.f32 and color.type is not _types.vec4:
                raise ShaderError(
                    f"Expected texture value to be vec4, not {color.type}"
                )
            self.gen_func_instruction(cc.OpImageWrite, tex, coord, color)
            self._stack.append(None)  # this call returns None, gets popped
        elif funcname == "sample":  # -> from a texture
            tex, sam, coord = args
            tex.depth.value, tex.sampled.value = 0, 1
            sample_type = tex.sample_type
            result_id, type_id = self.obtain_value(_types.Vector(4, sample_type))
            self.gen_func_instruction(
                cc.OpImageSampleExplicitLod,  # or cc.OpImageSampleImplicitLod,
                type_id,
                result_id,
                self.get_texture_sampler(tex, sam),
                coord,
                cc.ImageOperandsMask_MaskNone | cc.ImageOperandsMask_Lod,
                self.obtain_constant(0.0),
            )
            self._stack.append(result_id)
        else:
            raise RuntimeError(f"Unknown texture func {funcname}")

    def _ext_instruction_call(self, funcname, args):
        # An extension instruction call. If there is an info dict for
        # this function name, all args must be float or float-vector,
        # and the result is either the same, or the same as the
        # component type. All ext instructions that do not fall into
        # this category are handled seperately here. These are what we
        # call the "hardcoded" functions in stdlib.py.

        # https://www.khronos.org/registry/spir-v/specs/unified1/GLSL.std.450.html
        set_name = "GLSL.std.450"  # The most common

        info = ext_functions.get(funcname, None)

        if info:
            # One of the many float/vec-float functions that we can handle automatically
            set_name, nr, nargs = info["set_name"], info["nr"], info["nargs"]
            # Check
            ty = args[0].type
            if issubclass(ty, _types.Float):
                pass  # ok
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                pass  # ok
            else:
                raise ShaderError(f"Arg 0 of {funcname} must be float or float-vector.")
            for i in range(1, len(args)):
                if args[i].type is not ty:
                    raise ShaderError(
                        f"Arg {i} of {funcname} must be float or float-vector."
                    )
            # Get result object
            result_type = info["result_type"]
            if result_type == "same":
                result_type = args[0].type
            elif result_type == "component":
                result_type = args[0].type.subtype
            else:
                result_type = s_types.type_from_name(result_type)
        elif funcname == "abs":
            nargs = 1
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                nr = 4
            elif issubclass(ty, _types.Int):
                nr = 5
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                nr = 4
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Int):
                nr = 5
            else:
                raise ShaderError("abs() expects (vector of) int or float.")
        elif funcname == "sign":
            nargs = 1
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                nr = 6
            elif issubclass(ty, _types.Int):
                nr = 6
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                nr = 6
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Int):
                nr = 7
            else:
                raise ShaderError("sign() expects (vector of) int or float.")
        elif funcname == "matrix_inverse":
            nargs = 1
            result_type = ty = args[0].type
            if issubclass(ty, _types.Matrix) and ty.rows == ty.cols:
                nr = 34
            else:
                raise ShaderError("matrix_inverse() expects square matrix.")
        elif funcname == "min":
            nargs = 2
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                nr = 37
            elif issubclass(ty, _types.Int):
                nr = 39
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                nr = 37
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Int):
                nr = 39
            else:
                raise ShaderError("min() expects (vector of) int or float.")
        elif funcname == "max":
            nargs = 2
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                nr = 40
            elif issubclass(ty, _types.Int):
                nr = 42
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                nr = 40
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Int):
                nr = 42
            else:
                raise ShaderError("max() expects (vector of) int or float.")
        elif funcname == "clamp":
            nargs = 3
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                nr = 43
            elif issubclass(ty, _types.Int):
                nr = 45
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                nr = 43
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Int):
                nr = 45
            else:
                raise ShaderError("clamp() expects (vector of) int or float.")
        elif funcname == "mix":
            nargs = 3
            nr = 46
            result_type = ty = args[0].type
            if issubclass(ty, _types.Float):
                pass
            elif issubclass(ty, _types.Vector) and issubclass(ty.subtype, _types.Float):
                if issubclass(args[2].type, _types.Float):
                    # Support mix(x, y, a) with X and Y vectors, and A a float
                    result_id, vector_type_id = self.obtain_value(ty)
                    self.gen_func_instruction(
                        cc.OpCompositeConstruct,
                        vector_type_id,
                        result_id,
                        *[args[2]] * ty.length,
                    )
                    args[2] = result_id
            else:
                raise ShaderError("mix() expects (vector of) float.")
        else:
            raise RuntimeError(f"Unknown extension instruction {funcname}")

        # Check
        if nargs != len(args):
            raise ShaderError(
                f"Ext function {funcname} expects {info['nargs']} args, got {nargs}."
            )

        # Generate instruction
        result_id, type_id = self.obtain_value(result_type)
        instr_set = self.obtain_extended_instruction_set(set_name)
        self.gen_func_instruction(
            cc.OpExtInst, type_id, result_id, instr_set, nr, *args
        )
        self._stack.append(result_id)

    # %% IO

    def co_resource(self, name, kind, slot, typename):

        bindgroup = 0
        if isinstance(slot, (tuple, list)):
            bindgroup, slot = slot

        # --> https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)

        # todo: should we check if the incoming code has the proper amount of skipping
        # in input and output slots?

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
        elif kind == "sampler":
            storage_class, iodict = cc.StorageClass_UniformConstant, self._sampler
            location_or_binding = cc.Decoration_Binding
        elif kind == "texture":
            storage_class, iodict = cc.StorageClass_UniformConstant, self._texture
            location_or_binding = cc.Decoration_Binding
        else:
            raise ShaderError(f"Invalid IO kind {kind}")

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
            raise ShaderError(
                f"The {namespace_id} {slot} for {name} already taken by {other_name}."
            )
        else:
            self._slotmap[slotmap_key] = name

        # Get the root variable
        if kind in ("input", "output"):
            var_name = "var-" + name
            var_type = _types.type_from_name(typename)
            subtypes = None
        elif kind in ("uniform", "buffer"):
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
        elif kind == "sampler":
            var_name = "var-" + name
            var_type = (cc.OpTypeSampler,)
            subtypes = None
        elif kind == "texture":
            var_name = "var-" + name
            # Get a list of type info parts
            type_info = typename.lower().replace(",", " ").split()
            # Get dimension of the texture, and whether it is arrayed
            arrayed = 0
            if "1d" in type_info or "1d-array" in type_info:
                dim = cc.Dim_Dim1D
                arrayed = 1 if "1d-array" in type_info else 0
                self._capabilities.add(cc.Capability_Image1D)
                # self._capabilities.add(cc.Capability_Sampled1D)
            elif "2d" in type_info or "2d-array" in type_info:
                dim = cc.Dim_Dim2D
                arrayed = 1 if "2d-array" in type_info else 0
            elif "3d" in type_info or "3d-array" in type_info:
                dim = cc.Dim_Dim3D
                arrayed = 1 if "3d-array" in type_info else 0
            elif "cube" in type_info or "cube-array" in type_info:
                dim = cc.Dim_Cube
                arrayed = 1 if "cube-array" in type_info else 0
            else:
                raise ShaderError("Texture type info does not specify dimensionality.")
            # Get format
            fmt = cc.ImageFormat_Unknown
            sample_type = None  # can be set through format or specified explicitly
            for part in type_info:
                if part.startswith("r"):
                    part = part.replace("uint", "ui").replace("sint", "i")
                    part = part.replace("int", "i").replace("float", "f")
                    try:
                        fmt = getattr(cc, "ImageFormat_R" + part[1:])
                    except AttributeError:
                        continue
                    if part.endswith(("f", "norm")):
                        sample_type = _types.f32
                    else:
                        sample_type = _types.i32
                    break
            if fmt and fmt not in image_formats_that_need_no_ext:
                self._capabilities.add(cc.Capability_StorageImageExtendedFormats)
            # Get sample type (type of each of the 4 components when sampling)
            if "i32" in type_info:
                sample_type = _types.i32
            elif "f32" in type_info:
                sample_type = _types.f32
            if sample_type is None:  # note that it can have been set from fmt
                raise ShaderError(
                    "Texture type info does not specify format nor sample type."
                )
            # Get whether the texture is sampled - 0: unknown, 1: sampled, 2: storage
            sampled = WordPlaceholder(0)  # -> to be set later
            # Get depth, ms - 0: no, 1: yes, 2: unknown
            depth = WordPlaceholder(2)
            # Get multisampling
            ms = 1 if "ms" in type_info else 0
            # We now have all the info!
            stype = self.obtain_type_id(sample_type)
            var_type = (cc.OpTypeImage, stype, dim, depth, arrayed, ms, sampled, fmt)
            subtypes = None
        else:
            assert False  # unreachable

        # Create VariableAccessId object
        type_id = self.obtain_type_id(var_type)
        var_access = self.obtain_variable(var_type, storage_class, var_name)
        var_id = var_access.variable

        # On textures, store some more info that we need when sampling
        if kind == "texture":
            var_access.sample_type = sample_type
            var_access.sampled = sampled  # a word placeholder
            var_access.depth = depth  # a word placeholder

        # Dectorate block for uniforms and buffers
        if kind == "uniform":
            assert issubclass(var_type, _types.Struct)
            offset = 0
            for i, key in enumerate(var_type.keys):
                subtype = var_type.get_subtype(key)
                offset += self._annotate_uniform_subtype(type_id, subtype, i, offset)
            self.gen_instruction(
                "annotations", cc.OpDecorate, type_id, cc.Decoration_Block
            )
        elif kind == "buffer":
            # todo: according to docs, in SpirV 1.4+, BufferBlock is deprecated
            # and one should use Block with StorageBuffer. But this crashes.
            # Generate an ArrayStride on the storage buffer array
            if issubclass(var_type, _types.Struct) and len(var_type.keys) == 1:
                array_type = var_type.get_subtype(0)
                if issubclass(array_type, _types.Array):
                    stride = ctypes.sizeof(array_type.subtype._as_ctype())
                    if stride > 0 and array_type not in self._decorated_array_types:
                        self._decorated_array_types.add(array_type)
                        self.gen_instruction(
                            "annotations",
                            cc.OpDecorate,
                            self.obtain_type_id(array_type),
                            cc.Decoration_ArrayStride,
                            stride,
                        )
                    self.gen_instruction(
                        "annotations",
                        cc.OpMemberDecorate,
                        type_id,
                        0,
                        cc.Decoration_Offset,
                        0,
                    )
            self.gen_instruction(
                "annotations", cc.OpDecorate, type_id, cc.Decoration_BufferBlock
            )

        # Define slot of variable
        if kind in ("buffer", "texture", "uniform", "sampler"):
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
            if slot == "FragDepth":
                self._execution_modes["DepthReplacing"] = []
            try:
                slot = cc.builtins[slot]
            except KeyError:
                raise ShaderError(f"Not a known builtin io variable: {slot}")
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_BuiltIn, slot
            )

        # Store internal info to derefererence the variables
        if subtypes is None:
            if name in iodict:
                raise ShaderError(f"{kind} {name} already exists")
            iodict[name] = var_access
        else:
            for i, subname in enumerate(subtypes):
                index_id = self.obtain_constant(i)
                if subname in iodict:
                    raise ShaderError(f"{kind} {subname} already exists")
                iodict[subname] = var_access.index(index_id, i)

    def _annotate_uniform_subtype(self, type_id, subtype, i, offset):
        """ Annotates the given uniform struct subtype and return its size in bytes.
        """
        a = "annotations"
        if issubclass(subtype, _types.Matrix):
            # Stride for col or row depending on what is major
            stride = subtype.rows * ctypes.sizeof(subtype.subtype._ctype)
            self.gen_instruction(
                "annotations", cc.OpMemberDecorate, type_id, i, cc.Decoration_ColMajor,
            )
            self.gen_instruction(
                a, cc.OpMemberDecorate, type_id, i, cc.Decoration_MatrixStride, stride,
            )
            self.gen_instruction(
                a, cc.OpMemberDecorate, type_id, i, cc.Decoration_Offset, offset,
            )
        else:
            self.gen_instruction(
                a, cc.OpMemberDecorate, type_id, i, cc.Decoration_Offset, offset,
            )
        return ctypes.sizeof(subtype._as_ctype())

    def get_texture_sampler(self, texture, sampler):
        """ texture and sampler are bot VariableAccessId.
        """
        key = (id(texture), id(sampler))
        if key not in self._texture_samplers:
            tex_type_id = self.obtain_type_id(texture.type)
            restype = (cc.OpTypeSampledImage, tex_type_id)
            result_id, type_id = self.obtain_value(restype)
            self.gen_func_instruction(
                cc.OpSampledImage, type_id, result_id, texture, sampler
            )
            self._texture_samplers[key] = result_id
        return self._texture_samplers[key]

    # %% Basics

    def co_pop_top(self):
        self._stack.pop()

    def co_dup_top(self):
        ob = self._stack.pop()
        self._stack.append(ob)
        self._stack.append(ob)

    def co_rotate_stack(self, n):
        obs = [self._stack.pop() for i in range(n)]
        obs.append(obs.pop(0))
        obs.reverse()
        self._stack.extend(obs)

    def co_reverse_stack(self, n):
        obs = [self._stack.pop() for i in range(n)]
        self._stack.extend(obs)

    def co_load_name(self, name):
        # store a variable that is used in an inner scope.
        if name in self._name_ids:
            # First check if we need to load the name from a variable. If we do,
            # load it. Thereafter we won't need to load it again (in this block).
            current_label = self._current_branch["label"]
            names_that_need_load = self._need_name_var_load.get(current_label, set())
            if name in names_that_need_load:
                ob = self._name_variables[name].resolve_load(self)
                self._name_ids[name] = ob
                names_that_need_load.discard(name)
            ob = self._name_ids[name]
        elif name in self._input:
            ob = self._input[name]
            assert isinstance(ob, VariableAccessId)
            # We could make it quicker for next time, but we loose the deeper access
            # ob = ob.resolve_load(self)
            # self._name_ids[name] = ob
        elif name in self._output:
            ob = self._output[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._uniform:
            ob = self._uniform[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._buffer:
            ob = self._buffer[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._sampler:
            ob = self._sampler[name]
            assert isinstance(ob, VariableAccessId)
        elif name in self._texture:
            ob = self._texture[name]
            assert isinstance(ob, VariableAccessId)
        else:
            raise ShaderError(f"Using invalid variable: {name}")
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
            raise ShaderError("Cannot store to input")
        elif name in self._uniform:
            raise ShaderError("Cannot store to uniform")

        # Store where the result can now be fetched by name (within this block)
        self._name_ids[name] = ob

    def co_load_index(self):
        index = self._stack.pop()
        container = self._stack.pop()

        # Get type of object and index
        element_type = container.type.subtype
        # assert index.type is int

        if isinstance(container, VariableAccessId):
            result_id = container.index(index)

        elif issubclass(container.type, _types.Array):

            # todo: maybe ... the variable should be created only once ...
            # ... instead of every time it gets indexed
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
            raise ShaderError("Can only index from Arrays")

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
            if val.type is not ac.type:
                raise ShaderError(
                    f"Cannot set-index a {val.type.__name__} object in a {ac.type.__name__} container."
                )
            # Then resolve the chain to a store op
            ac.resolve_store(self, val)
        else:
            raise ShaderError(f"Cannot set-index on {ob}")

    def co_load_attr(self, name):
        ob = self._stack.pop()

        if not isinstance(getattr(ob, "type"), type):
            raise ShaderError("Invalid attribute access")
        elif isinstance(ob, VariableAccessId) and issubclass(ob.type, _types.Struct):
            # Struct attribute access
            if name not in ob.type.keys:
                raise ShaderError(f"Attribute {name} invalid for {ob.type.__name__}.")
            # Create new variable access for this attr op
            index = ob.type.keys.index(name)
            ac = ob.index(self.obtain_constant(index), index)
            self._stack.append(ac)
        elif issubclass(ob.type, _types.Vector):
            indices = []
            # Wr support xyzw, rgba, stpq
            for c in name:
                if c in "xrs":
                    indices.append(0)
                elif c in "ygt":
                    indices.append(1)
                elif c in "zbp":
                    indices.append(2)
                elif c in "waq":
                    indices.append(3)
                else:
                    raise ShaderError(f"Invalid vector attribute {name}")
            if len(indices) == 1:
                if isinstance(ob, VariableAccessId):
                    index_id = self.obtain_constant(indices[0])
                    result_id = ob.index(index_id)
                else:
                    result_id, type_id = self.obtain_value(ob.type.subtype)
                    self.gen_func_instruction(
                        cc.OpCompositeExtract, type_id, result_id, ob, indices[0]
                    )
            else:
                result_type = _types.Vector(len(indices), ob.type.subtype)
                result_id, type_id = self.obtain_value(result_type)
                self.gen_func_instruction(
                    cc.OpVectorShuffle, type_id, result_id, ob, ob, *indices
                )
            self._stack.append(result_id)
        else:
            # todo: not implemented for non VariableAccessId
            raise ShaderError(f"Unsupported attribute access {name}")

    def co_load_constant(self, value):
        id = self.obtain_constant(value)
        self._stack.append(id)
        # Also see OpConstantNull OpConstantSampler OpConstantComposite

    def co_load_array(self, nargs):
        # Literal array
        assert len(self._stack) >= nargs
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        result = self._array_packing(args)
        self._stack.append(result)

    # %% Math and more

    def co_unary_op(self, op):

        val1 = self._stack.pop()

        # Get reference types
        type1 = val1.type
        reftype1 = type1
        if issubclass(type1, (_types.Vector, _types.Matrix)):
            reftype1 = type1.subtype
        tn1 = type1.__name__

        # Determine opcode and check types
        if op == "neg":
            if issubclass(reftype1, _types.Float):
                opcode = cc.OpFNegate
            elif issubclass(reftype1, _types.Int):
                opcode = cc.OpSNegate
            else:
                raise ShaderError(f"Cannot {op.upper()} values of type {tn1}.")
        elif op == "not":
            if issubclass(reftype1, _types.boolean):
                opcode = cc.OpLogicalNot
            else:
                raise ShaderError(f"Cannot {op.upper()} values of type {tn1}.")

        # Emit code
        result_id, type_id = self.obtain_value(type1)
        self.gen_func_instruction(opcode, type_id, result_id, val1)
        self._stack.append(result_id)

    def co_binary_op(self, op):

        val2 = self._stack.pop()
        val1 = self._stack.pop()

        # The ids that will be in the instruction, can be reset
        id1, id2 = val1, val2

        # Predefine some types
        # We specify three flavors of div: One that works for both int and float,
        # one that works for float only, and one that works for int only.
        scalar_or_vector = _types.Scalar, _types.Vector
        FOPS = dict(
            add=cc.OpFAdd,
            sub=cc.OpFSub,
            mul=cc.OpFMul,
            fdiv=cc.OpFDiv,
            div=cc.OpFDiv,
            mod=cc.OpFMod,
            rem=cc.OpFRem,
        )
        IOPS = dict(
            add=cc.OpIAdd,
            sub=cc.OpISub,
            mul=cc.OpIMul,
            idiv=cc.OpSDiv,
            div=cc.OpSDiv,
            mod=cc.OpSMod,
            rem=cc.OpSRem,
        )
        LOPS = {"and": cc.OpLogicalAnd, "or": cc.OpLogicalOr}

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
            raise ShaderError(
                f"Cannot {op.upper()} two values with different (sub)types: {tn1} and {tn2}"
            )

        elif type1 is type2 and issubclass(type1, scalar_or_vector):
            # Types are equal and scalar or vector. Covers a lot of cases.
            result_id, type_id = self.obtain_value(type1)
            if (
                issubclass(type1, _types.Vector)
                and issubclass(reftype1, _types.Float)
                and op == "mmul"
            ):
                opcode = cc.OpDot  # special case
                result_id, type_id = self.obtain_value(type1.subtype)
            elif issubclass(reftype1, _types.Float):
                opcode = FOPS[op]
            elif issubclass(reftype1, _types.Int):
                opcode = IOPS[op]
            elif issubclass(reftype1, _types.boolean):
                opcode = LOPS[op]
            else:
                raise ShaderError(f"Cannot {op.upper()} values of type {tn1}.")

        elif issubclass(type1, _types.Scalar) and issubclass(type2, _types.Vector):
            # Convenience - add/mul vectors with scalars
            if not issubclass(reftype1, _types.Float):
                raise ShaderError(
                    f"Scalar {op.upper()} Vector only supported for float subtype."
                )
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
                raise ShaderError(
                    f"Vector {op.upper()} Scalar only supported for float subtype."
                )
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
            raise ShaderError(f"Cannot {op.upper()} {tn1} and {tn2}, multiply only.")

        elif not issubclass(reftype1, _types.Float):
            # The remaining cases are all limited to float types
            raise ShaderError(f"Cannot {op.upper()} {tn1} and {tn2}, float only.")

        # With that out of the way, the remaining cases are quite short to write.

        elif issubclass(type1, _types.Matrix) and issubclass(type2, _types.Matrix):
            # Multiply two matrices
            if type1.cols != type2.rows:
                raise ShaderError(
                    f"Cannot {op.upper()} two matrices with incompatible shapes."
                )
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
                raise ShaderError(f"Incompatible shape for {tn1} x {tn2}")
            type3 = _types.Vector(type1.rows, type1.subtype)
            result_id, type_id = self.obtain_value(type3)
            opcode = cc.OpMatrixTimesVector

        elif issubclass(type1, _types.Vector) and issubclass(type2, _types.Matrix):
            # Vector times Matrix
            if type1.length != type2.rows:
                raise ShaderError(f"Incompatible shape for {tn1} x {tn2}")
            type3 = _types.Vector(type2.cols, type2.subtype)
            result_id, type_id = self.obtain_value(type3)
            opcode = cc.OpVectorTimesMatrix

        else:
            raise ShaderError(f"Cannot {op.upper()} values of {tn1} and {tn2}.")

        self.gen_func_instruction(opcode, type_id, result_id, id1, id2)
        self._stack.append(result_id)

    def co_compare(self, cmp):
        val2 = self._stack.pop()
        val1 = self._stack.pop()

        # Get reference type
        if val1.type is not val2.type:
            raise ShaderError("Cannot compare values that do not have the same type.")
        if issubclass(val1.type, _types.Vector):
            reftype = val1.type.subtype
            result_type = _types.Vector(val1.type.length, _types.boolean)
        else:
            reftype = val1.type
            result_type = _types.boolean

        # Get what kind of comparison to do
        opname_suffix_map = {
            "<": "LessThan",
            "<=": "LessThanEqual",
            "==": "Equal",
            "!=": "NotEqual",
            ">": "GreaterThan",
            ">=": "GreaterThanEqual",
        }
        opname_suffix = opname_suffix_map[cmp]

        # Get the actual opcode
        if issubclass(reftype, _types.Float):
            opcode = getattr(cc, "OpFOrd" + opname_suffix)
        elif issubclass(reftype, _types.Int):
            prefix = "OpS" if "Than" in opname_suffix else "OpI"
            opcode = getattr(cc, prefix + opname_suffix)
        else:
            raise ShaderError(f"Cannot compare values of {val1.type.__name__}.")

        # Generate instruction
        result_id, type_id = self.obtain_value(result_type)
        self.gen_func_instruction(opcode, type_id, result_id, val1, val2)
        self._stack.append(result_id)

    # %% Control flow

    def _before_moving_out_of_a_block(self):
        label = self._current_branch["label"]
        self._store_variables_for_other_blocks(label)
        self._store_stack_for_phi_op(label)

    def _store_variables_for_other_blocks(self, label):
        # Before we wrap up the current block ...
        # We may need to store variable to use in other blocks
        # Sort the names to obtain consistent bytecode.
        for name in sorted(self._need_name_var_save.get(label, ())):
            ob = self._name_ids[name]  # Get the last value
            if name not in self._name_variables:
                self._name_variables[name] = self.obtain_variable(
                    ob.type, cc.StorageClass_Function, name
                )
            var_access = self._name_variables[name]
            if ob.type is not var_access.type:
                raise ShaderError(
                    f"Name {name} used for different types {ob.type} and {var_access.type}"
                )
            var_access.resolve_store(self, ob)

    def _store_stack_for_phi_op(self, label):
        # Sometimes, when we jump out of a block, the stack is not
        # empty. When this happens the stack value is to be picked up
        # by a OpPhi, e.g. with ternary operations (.. if .. else ..).
        # We move the remaining item off the stack and push it on a
        # special dict for the phi op.
        if self._stack:
            self._stack_for_phi.setdefault(label, []).extend(self._stack)
            self._stack = []

    def _collect_stack_from_previous_block(self, prev_label):
        # Take over the remaining stack from the previous block (if this is not a merge block)
        if prev_label in self._stack_for_phi:
            self._stack.extend(self._stack_for_phi[prev_label])

    def _collect_stack_from_previous_blocks(self, *prev_labels):
        # Insert a phi op that selects a stack value based on the branch we came from
        assert len(prev_labels) >= 2
        if any(label in self._stack_for_phi for label in prev_labels):
            assert all(label in self._stack_for_phi for label in prev_labels)
            # All but the last item on the sub-stacks must be equal
            substack = self._stack_for_phi[prev_labels[0]][:-1]
            for label in prev_labels:
                substack2 = self._stack_for_phi[label][:-1]
                assert len(substack) == len(substack2)
                assert all(ob is ob2 for ob, ob2 in zip(substack, substack2))
            # We can simply copy this substack
            self._stack.extend(substack)
            # But for the last item we need a Phi op ...
            top_obs = [self._stack_for_phi[x][-1] for x in prev_labels]
            types = [ob.type for ob in top_obs]
            type = top_obs[0].type
            for ob in top_obs:
                assert (
                    ob.type is type
                ), f"Phi stack has objects with different types: {types}"
            result_id, type_id = self.obtain_value(type)
            phi_op = [cc.OpPhi, type_id, result_id]
            for label, ob in zip(prev_labels, top_obs):
                phi_op.extend([ob, self._get_label_id(label)])
            self.gen_func_instruction(*phi_op)
            self._stack.append(result_id)

    def co_label(self, label):
        # We enter a new block. This is where we resolve branch pairs that both
        # jumped to this block. If there are multiple such pairs, we need to
        # introduce extra blocks, because at each label we can merge at most
        # one branch pair (matching an OpSelectionMerge).

        # Details: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_structuredcontrolflow_a_structured_control_flow
        # * The merge block declared by a header block cannot be a merge
        #   block declared by any other header block.
        # * Each header block must strictly dominate its merge block,
        #   unless the merge block is unreachable in the CFG.
        # * A certain block A dominates a block B, where A and B are in the
        #   same function, if every path from the function’s entry point to
        #   block B includes block A. A strictly dominates B only if A
        #   dominates B and A and B are different blocks.
        # * In other words: every pair of diverging branches must merge at a
        #   unique block (i.e. label), and all control flow going through that
        #   block must also pass through the block where the branch-pair
        #   diverged (i.e.the header block).

        new_label = label

        # Get what loop we're in. If this label closes/merges the loop, pop it.
        loop_info = self._loop_stack[-1]
        if loop_info.get("merge_label") == new_label:
            self._loop_stack.pop(-1)

        # Create a mapping to obtain parents of items in the CFG.
        # We don't use a 'parent' attribute on the items, because that makes
        # these dicts really hard to read during debugging.
        parents = {}

        # First we collect the leaf branches that have jumped to this block
        def _collect_leaf_branches(branch):
            if branch["children"]:
                c1, c2 = branch["children"]
                parents[id(c1)] = parents[id(c2)] = branch
                return _collect_leaf_branches(c1) + _collect_leaf_branches(c2)
            elif branch["label"] == new_label:
                return [branch]
            else:
                return []

        leaf_branches = _collect_leaf_branches(self._root_branch)

        # Then we merge branches, one by one, deeper ones first
        def _select_branch_to_merge():
            for branch in leaf_branches:
                if branch["depth"] > 0:
                    parent = parents[id(branch)]
                    siblings = parent["children"]
                    if siblings[0]["label"] == siblings[1]["label"]:
                        # Merging what started at co_branch_conditional
                        return parent
                    elif parent is loop_info.get("branch", None):
                        # This is the main loop branch. Merge if we reached the merge label.
                        # Note that one child branch is pointing to the continue label, not to here.
                        # Don't AND this sub-if, the above elif excludes cases for the next elif.
                        if loop_info.get("merge_label") == new_label:
                            parent["children"] = tuple(
                                c for c in siblings if c in leaf_branches
                            )
                            return parent
                    elif branch["label"] == new_label:
                        # This could be a break
                        sibling_branch = siblings[int(branch is siblings[0])]
                        if sibling_branch["label"] == loop_info.get("merge_label"):
                            parent["children"] = tuple(
                                c for c in siblings if c in leaf_branches
                            )
                            return parent

        branches2merge = []
        while True:
            leaf_branches.sort(key=lambda b: -b["depth"])
            new_leaf = _select_branch_to_merge()
            if not new_leaf:
                break
            new_leaf["label"] = new_leaf["children"][0]["label"]
            # new_leaf["children"] = () -> do further down, we need that info
            for child in new_leaf["children"]:
                if child in leaf_branches:
                    leaf_branches.remove(child)
            leaf_branches.append(new_leaf)
            branches2merge.append(new_leaf)

        # If all is well, we're now left with a single branch. Make it the current.
        if len(leaf_branches) != 1:
            raise ShaderError(
                f"New block ({new_label}) should start with 1 unmerged branches, got {[b['label'] for b in leaf_branches]}"
            )
        self._current_branch = leaf_branches[0]

        # Need an extra hop?
        exta_hop = 0
        if new_label == self._loop_stack[-1].get("continue_label"):
            exta_hop = 1

        # Get hop labels for all the merges.
        hop_labels = [new_label]
        while len(hop_labels) < len(branches2merge) + exta_hop:
            hop_labels.insert(-1, f"{new_label}-hop-{len(hop_labels)}")

        # Emit the code for the extra blocks, and update previous jump ids.
        hop_label = None
        for i, branch in enumerate(branches2merge):
            # Mark the start of a new block
            hop_label = hop_labels[i]
            label_id = self._get_label_id(hop_label)
            self.gen_func_instruction(cc.OpLabel, label_id)
            # Update placeholders
            for child in branch["children"]:
                child["branch_label_placeholder"].value = label_id.id
            branch["merge_label_placeholder"].value = label_id.id
            # We may need to insert a Phi op
            if len(branch["children"]) == 2:
                self._collect_stack_from_previous_blocks(
                    *[b["prev_label"] for b in branch["children"]]
                )
            # Do we need to hop to the next block?
            if hop_label is not new_label:
                branch_label_placeholder = WordPlaceholder(
                    self._get_label_id(hop_labels[i + 1]).id
                )
                branch["branch_label_placeholder"] = branch_label_placeholder
                branch["prev_label"] = hop_label
                self.gen_func_instruction(cc.OpBranch, branch_label_placeholder)
            if i < len(branches2merge) - 1:
                self._store_stack_for_phi_op(hop_label)

        # If we did not create the main label yet, create it now
        if hop_label != new_label:
            self.gen_func_instruction(cc.OpLabel, self._get_label_id(new_label))

        # Collect stack from previous block
        if not branches2merge:
            self._collect_stack_from_previous_block(self._current_branch["prev_label"])

        # Clean up
        for branch in branches2merge:
            branch["children"] = ()

    def co_branch(self, label):
        # Before we leave this block ...
        self._before_moving_out_of_a_block()
        # Initialize branch label. We use a placeholder because we may introduce
        # additional labels for merges, so the id may need to change.
        branch_label = self._get_label_id(label)
        branch_label_placeholder = WordPlaceholder(branch_label.id)
        # Update the label for the currently running branch
        self._current_branch["prev_label"] = self._current_branch["label"]
        self._current_branch["label"] = label
        # Also update the label placeholder for the last jump
        self._current_branch["branch_label_placeholder"] = branch_label_placeholder
        # Mark the end of the block (will be set again at co_label)
        self._current_branch = None
        # Generate instruction, but not if the last instruction already marked
        # the end of a block.
        if self._sections["functions"][-1][0] not in (cc.OpKill,):
            self.gen_func_instruction(cc.OpBranch, branch_label_placeholder)

    def co_branch_conditional(self, true_label, false_label):
        condition = self._stack.pop()
        current_label = self._current_branch["label"]
        # Before we leave this block ...
        self._before_moving_out_of_a_block()
        # Setup tracing for the two new branches. SpirV wants to know
        # beforehand where the two branches meet, so we will need to
        # update the OpSelectionMerge instruction when we find the merge
        # point in co_label.
        branch1_label = WordPlaceholder(self._get_label_id(true_label).id)
        branch2_label = WordPlaceholder(self._get_label_id(false_label).id)
        new_branch1 = {
            "depth": self._current_branch["depth"] + 1,
            "children": (),
            "label": true_label,
            "prev_label": current_label,
            "branch_label_placeholder": branch1_label,
        }
        new_branch2 = {
            "depth": self._current_branch["depth"] + 1,
            "children": (),
            "label": false_label,
            "prev_label": current_label,
            "branch_label_placeholder": branch2_label,
        }
        self._current_branch["children"] = new_branch1, new_branch2
        # Introduce OpSelectionMerge, unless we've already emitted OpLoopMerge
        if self._loop_stack[-1].get("branch") is not self._current_branch:
            merge_label = WordPlaceholder(0)
            self._current_branch["merge_label_placeholder"] = merge_label
            self.gen_func_instruction(cc.OpSelectionMerge, merge_label, 0)
        # Generate the branch instruction
        self.gen_func_instruction(
            cc.OpBranchConditional, condition, branch1_label, branch2_label,
        )

    def co_branch_loop(self, iter_label, continue_label, merge_label):
        # Before we leave this block ...
        self._before_moving_out_of_a_block()
        # Get id's
        iter_id = self._get_label_id(iter_label)
        continue_id = self._get_label_id(continue_label)
        merge_id = self._get_label_id(merge_label)
        # Generate loop merge instruction
        # note: here we can specify request for unroll, min/max iters (1.4+) etc.
        merge_placeholder = WordPlaceholder(merge_id.id)
        self._current_branch["merge_label_placeholder"] = merge_placeholder
        self.gen_func_instruction(cc.OpLoopMerge, merge_placeholder, continue_id, 0)
        # Mark the current branch as a loop, ending at merge_label
        loop_info = {
            "merge_label": merge_label,
            "iter_label": iter_label,
            "continue_label": continue_label,
            "branch": self._current_branch,
        }
        self._loop_stack.append(loop_info)

        # The rest is similar to co_branch()
        branch_label_placeholder = WordPlaceholder(iter_id.id)
        # Update the label for the currently running branch
        self._current_branch["prev_label"] = self._current_branch["label"]
        self._current_branch["label"] = iter_label
        # Also update the label placeholder for the last jump
        self._current_branch["branch_label_placeholder"] = branch_label_placeholder
        # Mark the end of the block (will be set again at co_label)
        self._current_branch = None
        self.gen_func_instruction(cc.OpBranch, iter_id)

    def co_select(self):
        val2 = self._stack.pop()
        val1 = self._stack.pop()
        condition = self._stack.pop()
        if val1.type is not val2.type:
            raise ShaderError(
                "Incompatible types in op_selec: {val1.type.__name__} and {val2.type.__name__}"
            )
        result_id, type_id = self.obtain_value(val1.type)
        self.gen_func_instruction(
            cc.OpSelect, type_id, result_id, condition, val1, val2
        )
        self._stack.append(result_id)

    # %% Helper methods

    def _convert_scalar(self, out_type, arg):
        return self._convert_scalar_or_vector(out_type, out_type, arg, arg.type)

    def _convert_numeric_vector(self, out_type, arg):
        if not (
            issubclass(arg.type, _types.Vector) and arg.type.length == out_type.length
        ):
            raise ShaderError("Vector conversion needs vectors of equal length.")
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
                op = cc.OpConvertUToF if argtname.startswith("u") else cc.OpConvertSToF
                self.gen_func_instruction(op, type_id, result_id, arg)
            elif issubclass(arg_el_type, _types.boolean):
                zero = self.obtain_constant(0.0, out_el_type)
                one = self.obtain_constant(1.0, out_el_type)
                self.gen_func_instruction(
                    cc.OpSelect, type_id, result_id, arg, one, zero
                )
            else:
                raise ShaderError(f"Cannot convert to float: {arg.type}")

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
                raise ShaderError(f"Cannot convert to int: {arg.type}")

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
                raise ShaderError(f"Cannot convert to bool: {arg.type}")
        else:
            raise ShaderError(f"Cannot convert to {out_type}")

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
                raise ShaderError(f"Invalid type to compose vector: {arg.type}")

        # Check the length
        if len(composite_ids) != n:
            raise ShaderError(
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
            raise ShaderError("No support for zero-sized arrays.")

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
