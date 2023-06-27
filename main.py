import sys
from antlr4 import *
from openqasm_reference_parser import qasm3Lexer
from openqasm_reference_parser import qasm3Parser, qasm3ParserListener
import catalyst
from mlir_quantum.dialects.func import FuncOp
from mlir_quantum.dialects.arith import ConstantOp as ArithConstantOp, AddFOp
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
        half = ArithConstantOp(f64, 0.5)
        zero_izero = CreateOp(complex128, zero, zero)
        one_izero = CreateOp(complex128, one, zero)
        half_izero = CreateOp(complex128, half, zero)

        # We need to perform arithmetic 
        # matrix[0,0] = 1 + e^{i*theta}
        zero_itheta = CreateOp(complex128, zero, theta)                       # itheta
        e_to_zero_itheta = ExpOp(zero_itheta)                                 # e^{itheta}
        one_izero_plus_e_to_zero_itheta = AddOp(one_izero, e_to_zero_itheta)  # 1 + e^{itheta}
        m00 = MulOp(one_izero_plus_e_to_zero_itheta, half_izero)

        # matrix[0,1] = -ie^{i*lambda} * (1 - e^{i*theta})
        one_izero_sub_e_to_zero_itheta = SubOp(one_izero, e_to_zero_itheta)  # 1 - e^{itheta}
        lambda_izero = CreateOp(complex128, _lambda, zero)                   # ilambda
        e_to_lambda_izero = ExpOp(lambda_izero)                              # e^{ilambda}
        zero_imone = CreateOp(complex128, zero, mone)                        # -i
        tmp = MulOp(zero_imone, e_to_lambda_izero)                           # -i * e^{ilambda}
        tmp2 = MulOp(tmp, one_izero_sub_e_to_zero_itheta)                    # -i * e^{ilambda} * (1 - e^{itheta})
        m01 = MulOp(tmp2, half_izero)
    
        # matrix[1,0] = i*e^{i*phi} * (1 - e^{i*theta})
        phi_izero = CreateOp(complex128, phi, zero)                        # iphi
        e_to_phi_izero = ExpOp(phi_izero)                                  # e^{iphi}
        zero_ione = CreateOp(complex128, zero, one)                        # i
        tmp = MulOp(zero_ione, e_to_phi_izero)                             # i * e^{iphi}
        tmp2 = MulOp(tmp, one_izero_sub_e_to_zero_itheta)                  # i * e^{iphi} * (1 - e^{itheta})
        m10 = MulOp(tmp2, half_izero)

        # matrix[1,1] = 
        lambda_izero_plus_phi_izero = AddOp(lambda_izero, phi_izero)      # ilambda + iphi
        exp = ExpOp(lambda_izero_plus_phi_izero)                           # e^{ilambda + iphi}
        tmp3 = MulOp(exp, one_izero_plus_e_to_zero_itheta)
        m11 = MulOp(tmp3, half_izero)


        tensor_complex128_= mlir_quantum.ir.RankedTensorType.get([2, 2], complex128)
        matrix = FromElementsOp.build_generic([tensor_complex128_], [m00.results[0], m01.results[0], m10.results[0], m11.results[0]])

        # QubitUnitaryOp()
        QubitUnitaryOp([qubit[0].type], matrix.results, [qubit])
        
 
if __name__ == '__main__':
    main(sys.argv)
