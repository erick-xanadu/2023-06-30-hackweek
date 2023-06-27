import sys
from antlr4 import *
from openqasm_reference_parser import qasm3Lexer
from openqasm_reference_parser import qasm3Parser, qasm3ParserListener
import catalyst
from mlir_quantum.dialects.func import FuncOp, ReturnOp
from mlir_quantum.dialects.arith import ConstantOp as ArithConstantOp, AddFOp, DivFOp
from mlir_quantum.dialects.math import CosOp, SinOp
from mlir_quantum.dialects.complex import ExpOp, CreateOp, SubOp, MulOp, AddOp
from mlir_quantum.dialects.quantum import AllocOp, QubitUnitaryOp, ExtractOp, PrintStateOp, DeviceOp, DeallocOp, InitializeOp, FinalizeOp
from mlir_quantum.dialects.tensor import FromElementsOp

import mlir_quantum
from mlir_quantum.ir import Context, Module, InsertionPoint, Location, Block

def insert_qreg(ctx):
    qreg_type = mlir_quantum.ir.OpaqueType.get("quantum", "reg", ctx)
    i64 = mlir_quantum.ir.IntegerType.get_signless(64, ctx)
    size_attr = mlir_quantum.ir.IntegerAttr.get(i64, 10)
    return AllocOp(qreg_type, nqubits_attr=size_attr).results

def insert_main(ctx):
    func = FuncOp("main", ([], []))
    entry_block = Block.create_at_start(func.body)
    return func

def insert_device(ctx):
    InitializeOp()
    backend_attr = mlir_quantum.ir.StringAttr.get("backend")
    val_attr = mlir_quantum.ir.StringAttr.get("lightning.qubit")
    DeviceOp(specs=mlir_quantum.ir.ArrayAttr.get([backend_attr, val_attr]))

 
def main(argv):
    input_stream = FileStream(argv[1])
    lexer = qasm3Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = qasm3Parser(stream)
    tree = parser.program()
    walker = ParseTreeWalker()
    with Context() as ctx, Location.file("f.mlir", line=0, col=0, context=ctx):
        module = Module.create()

        ctx.allow_unregistered_dialects = True
        with InsertionPoint(module.body):
            main = insert_main(ctx)
            with InsertionPoint(main.body.blocks[0]):
                insert_device(ctx)
                qreg = insert_qreg(ctx)
                listener = SimpleListener(ctx, module, qreg)
                walker.walk(listener, tree)
                PrintStateOp()
                DeallocOp(qreg)
                FinalizeOp()
                ReturnOp([])
        print(module)

class SimpleListener(qasm3ParserListener.qasm3ParserListener):
    def __init__(self, mlir_context, mlir_module, qreg):
        self._mlir_context = mlir_context
        self._mlir_module = mlir_module
        self._qreg = qreg
        self.idQubitMap = dict()
        self.qubitsUsed = 0
        super().__init__()

    def exitLiteralExpression(self, ctx):
        assert len(ctx.children) == 1
        child = ctx.children[0]
        type = child.symbol.type
        typeName = qasm3Lexer.symbolicNames[type]
        assert "FloatLiteral" == typeName
        f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
        constant = ArithConstantOp(f64, float(child.getText()))
        child.variable = constant

    def exitQuantumDeclarationStatement(self, ctx):
        qubitName = ctx.children[1].getText()
        if qubitName in self.idQubitMap:
            raise ValueError("You are re-declaring the same symbol")
        qubit_type = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
        i64 = mlir_quantum.ir.IntegerType.get_signless(64, self._mlir_context)
        index = ArithConstantOp(i64, self.qubitsUsed).results
        extractOp = ExtractOp(qubit_type, self._qreg, idx=index)
        self.idQubitMap[qubitName] = extractOp.results
        self.qubitsUsed += 1




    def exitGateCallStatement(self, ctx):
        gateName = ctx.children[0].getText()
        assert "U" == gateName
        arglist = ctx.children[2]

        theta = arglist.children[0].children[0].variable
        phi = arglist.children[2].children[0].variable
        _lambda = arglist.children[4].children[0].variable

        qubitName = ctx.children[4].getText()
        if qubitName not in self.idQubitMap:
            raise ValueError("Attempting to use non-declared qubit.")

        qubit = self.idQubitMap[qubitName]

        f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
        complex128 = mlir_quantum.ir.ComplexType.get(f64)
        zero = ArithConstantOp(f64, 0.0)
        one = ArithConstantOp(f64, 1.0)
        mone = ArithConstantOp(f64, -1.0)
        two = ArithConstantOp(f64, 2.0)
        half = ArithConstantOp(f64, 0.5)
        zero_izero = CreateOp(complex128, zero, zero)
        one_izero = CreateOp(complex128, one, zero)
        half_izero = CreateOp(complex128, half, zero)

        # We need to perform arithmetic 
        # Use the definition of U3 found in pennylane.U3.html
        # matrix[0,0] = cos(theta/2)
        divOp = DivFOp(theta, two)
        cosOp = CreateOp(complex128, CosOp(divOp), zero)
        m00 = cosOp

        # matrix[0,1] = -exp(i * lambda) * sin(theta / 2)
        zero_imone = CreateOp(complex128, zero, mone)                        # -i
        sinOp = CreateOp(complex128, SinOp(divOp), zero)
        zero_ilambda = CreateOp(complex128, zero, _lambda)
        expOp = ExpOp(zero_ilambda)
        mulOp = MulOp(zero_imone, expOp)
        m01 = MulOp(expOp, sinOp)
    
        # matrix[1,0] = exp (i * phi) * sin (theta / 2)
        zero_iphi = CreateOp(complex128, zero, phi)
        expOp = ExpOp(zero_iphi)
        mulOp = MulOp(expOp, sinOp)
        m10 = mulOp

        # matrix[1,1] = exp (i * (phi + lambda)) * cos (theta / 2)
        addOp = AddOp(zero_iphi, zero_ilambda)
        expOp = ExpOp(addOp)
        mulOp = MulOp(expOp, cosOp)
        m11 = mulOp

        tensor_complex128_= mlir_quantum.ir.RankedTensorType.get([2, 2], complex128)
        matrix = FromElementsOp.build_generic([tensor_complex128_], [m00.results[0], m01.results[0], m10.results[0], m11.results[0]])

        # QubitUnitaryOp()
        QubitUnitaryOp([qubit[0].type], matrix.results, [qubit])
        
 
if __name__ == '__main__':
    main(sys.argv)
