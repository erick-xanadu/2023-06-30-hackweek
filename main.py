import sys
import subprocess
from antlr4 import *
from openqasm_reference_parser import qasm3Lexer
from openqasm_reference_parser import qasm3Parser, qasm3ParserListener
import catalyst
from mlir_quantum.dialects.func import FuncOp, ReturnOp, CallOp
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
    entry_block = func.add_entry_block()
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
            listener = SimpleListener(ctx, module, qreg, main)
            walker.walk(listener, tree)
            with InsertionPoint(main.body.blocks[0]):
                PrintStateOp()
                DeallocOp(qreg)
                FinalizeOp()
                ReturnOp([])

    with open("hadamard.mlir", "w") as output_file:
        print(module, file=output_file)
    subprocess.run(["./script.sh"])

class Frame:

    def __init__(self, ip, qubits):
        # ip is short for InsertionPoint
        self._ip = ip
        self._qubits = qubits

    def insertQubit(self, oq3Symbol, mlirSymbol):
        self._qubits[oq3Symbol] = mlirSymbol


class SimpleListener(qasm3ParserListener.qasm3ParserListener):
    def __init__(self, mlir_context, mlir_module, qreg, main):
        self._main = main
        self._mlir_context = mlir_context
        self._mlir_module = mlir_module
        self._qreg = qreg
        self.idQubitMap = dict()
        self.qubitsUsed = 0
        self.userDefinedGates = dict()
        self.currentInsertionPoint = [InsertionPoint(main.body.blocks[0])]
        super().__init__()


    def exitLiteralExpression(self, ctx):
        with self.currentInsertionPoint[-1]:
            assert len(ctx.children) == 1
            child = ctx.children[0]
            type = child.symbol.type
            typeName = qasm3Lexer.symbolicNames[type]
            assert "FloatLiteral" == typeName, typeName
            f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
            constant = ArithConstantOp(f64, float(child.getText()))
            child.variable = constant

    def exitQuantumDeclarationStatement(self, ctx):
        with self.currentInsertionPoint[-1]:
            qubitName = ctx.children[1].getText()
            if qubitName in self.idQubitMap:
                raise ValueError("You are re-declaring the same symbol")
            qubit_type = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
            i64 = mlir_quantum.ir.IntegerType.get_signless(64, self._mlir_context)
            index = ArithConstantOp(i64, self.qubitsUsed).results
            extractOp = ExtractOp(qubit_type, self._qreg, idx=index)
            self.idQubitMap[qubitName] = extractOp.results[0]
            self.qubitsUsed += 1

    def enterGateStatement(self, ctx):
        gateName = ctx.Identifier()
        if gateName in self.userDefinedGates:
            raise ValueError("Re-declaring gate.")

        # We are going to need to create a new function in the module
        self.currentInsertionPoint.append(InsertionPoint(self._mlir_module.body))
        with self.currentInsertionPoint[-1]:
            funcName = gateName.getText()
            # classical params 
            cparams = ctx.params.children if ctx.params else []
            # qubit params
            qparams = ctx.qubits.children

            # And a gate never returns anything!
            retval = []

            params = []
            f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
            for cparam in cparams:
                params.append(f64)

            qubit = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
            for qparam in qparams:
                params.append(qubit)

            func = FuncOp(funcName, (params, retval))
            entry_block = func.add_entry_block()

            # Qubits need to be declared
            for param, qparam in zip(entry_block.arguments, qparams):
                if param.type != qubit:
                    continue
                self.idQubitMap[qparam.getText()] = param


            self.currentInsertionPoint.append(InsertionPoint(func.body.blocks[0]))
        self.userDefinedGates[funcName] = func

    def exitGateStatement(self, ctx):
        self.currentInsertionPoint.pop() # function-scope
        self.currentInsertionPoint.pop() # mlir-module-scope

    def enterScope(self, ctx):
        assert type(ctx.parentCtx) == qasm3Parser.GateStatementContext

    def exitScope(self, ctx):
        with self.currentInsertionPoint[-1]:
            ReturnOp([])

    def createGPhase(self, ctx):
        with self.currentInsertionPoint[-1]:
            gamma = float(ctx.expressionList().expression()[0].getText())
            how_many_qubits_in_this_scope

    def createU3(self, ctx):
        with self.currentInsertionPoint[-1]:
            gateName = ctx.children[0].getText()
            isU3 = "U" == gateName
            assert isU3, "Not U3"

            theta = ctx.expressionList().children[0].children[0].variable
            phi = ctx.expressionList().children[2].children[0].variable
            _lambda = ctx.expressionList().children[4].children[0].variable

            qubitName = ctx.children[4].getText()
            if qubitName not in self.idQubitMap:
                raise ValueError(f"Attempting to use non-declared qubit. {qubitName}")

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
            zero_imone = CreateOp(complex128, zero, mone)
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
            QubitUnitaryOp([qubit.type], matrix.results, [qubit])


    def exitGateCallStatement(self, ctx):
        gateName = ctx.Identifier().getText() if ctx.Identifier() else ctx.GPHASE().getText()

        isU3 = "U" == gateName
        isGphase = "gphase" == gateName
        if isU3:
            self.createU3(ctx)
        elif isGphase:
            self.createGPhase(ctx)
        else:
            with self.currentInsertionPoint[-1]:
                assert self.userDefinedGates[gateName], f"{gateName} not yet defined!"
                funcName = mlir_quantum.ir.FlatSymbolRefAttr.get(gateName)
                qubits = ctx.gateOperandList().getText()
                params = []
                for qubit in qubits:
                    params.append(self.idQubitMap[qubit])
                CallOp([], funcName, params)


        
 
if __name__ == '__main__':
    main(sys.argv)
