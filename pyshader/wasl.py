"""
WASL is an experimental custom textual shading language.
It's currently in a broken state. It's probably better
to support WSL than roll our own shading language. This was fun though :)
"""

from textx import metamodel_from_str

from ._module import ShaderModule
from . import _generator_bc as bc


grammar = """
Program: Procedure;
Comment: /#.*$/;
Procedure: 'fn' name=ID '(' params*=IOParameter[',']  ','? ')' '{' body=Body '}';
IOParameter: name=ID ':' mode=ID type=ID location=Location;
Location: INT | ID;
Parameter: name=ID ':' type=ID;
Body: expressions+=Statement;
Statement: Assignment | Expression;
Expression: CallExpr | Sum;
CallExpr: name=ID '(' args+=Expression[','] ','? ')';
Assignment: lhs=ID '=' rhs=Expression;
Sum: lhs=Term rhs*=SumRHS;
SumRHS: op=AddOp value=Term;
Term: lhs=Factor rhs*=TermRHS;
TermRHS: op=MulOp value=Factor;
Factor: IdentifierIndexed | Identifier | Number;
MulOp: '*' | '/';
AddOp: '+' | '-';
Number: value=FLOAT;
Identifier: name=ID;
IdentifierIndexed: name=ID '[' index=Expression ']';
""".lstrip()


meta_model = metamodel_from_str(grammar, classes=[])


def wasl2shader(code, shader_type=None):
    """Compile WASL code to a ShaderModule object.

    WASL is our own defined domain specific language (DSL) to write shaders.
    It is highly experimental. The code is parsed using textx, the resulting
    AST is converted to bytecode, from which binary SpirV can be generated.
    """
    if not isinstance(code, str):
        raise TypeError("wasl2shader expects a string.")

    ast = meta_model.model_from_str(code)

    converter = Wasl2Bytecode()
    converter.convert(ast)
    bytecode = converter.dump()

    return ShaderModule(code, bytecode, "shader from WASL")


class Wasl2Bytecode:
    """Compile WASL AST to bytecode."""

    def convert(self, ast):
        self._opcodes = []
        self.visit(ast)

    def dump(self):
        return self._opcodes

    def emit(self, opcode, arg):
        self._opcodes.append((opcode, arg))

    def visit(self, node):

        method_name = "visit_" + node.__class__.__name__.lower()
        getattr(self, method_name)(node)

    def visit_procedure(self, node):
        for param in node.params:
            if param.mode == "input":
                self.emit(bc.CO_INPUT, (param.name, param.location, param.type))
            elif param.mode == "output":
                self.emit(bc.CO_OUTPUT, (param.name, param.location, param.type))
            elif param.mode == "uniform":
                raise NotImplementedError()
            else:
                raise TypeError(
                    f"Funcion argument {param.name} must be input, output or uniform, not {param.mode}."
                )

        for node in node.body.expressions:
            self.visit(node)

    def visit_assignment(self, node):
        self.visit(node.rhs)
        self.emit(bc.CO_STORE, node.lhs)

    def visit_sum(self, node):
        self.visit(node.lhs)
        for term in node.rhs:
            self.visit(term)
            1 / 0

    def visit_term(self, node):
        self.visit(node.lhs)
        for term_rhs in node.rhs:
            self.visit(term_rhs.value)
            self.emit(bc.CO_BINARY_OP, term_rhs.op)

    def visit_identifier(self, node):
        self.emit(bc.CO_LOAD, node.name)

    def visit_identifierindexed(self, node):
        self.emit(bc.CO_LOAD, node.name)
        self.visit(node.index)
        self.emit(bc.CO_INDEX, None)

    def visit_number(self, node):
        self.emit(bc.CO_LOAD_CONSTANT, node.value)

    def visit_callexpr(self, node):
        self.emit(bc.CO_LOAD, node.name)
        for arg in node.args:
            self.visit(arg)
        self.emit(bc.CO_CALL, len(node.args))
