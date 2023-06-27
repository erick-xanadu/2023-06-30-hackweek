import sys
import subprocess
from antlr4 import *
from openqasm_reference_parser import qasm3Lexer
from openqasm_reference_parser import qasm3Parser, qasm3ParserListener
import catalyst
from mlir_quantum.dialects.func import FuncOp, ReturnOp, CallOp
from mlir_quantum.dialects.arith import ConstantOp as ArithConstantOp, AddFOp, DivFOp, SubFOp, CmpIOp
from mlir_quantum.dialects.math import CosOp, SinOp
from mlir_quantum.dialects.complex import ExpOp, CreateOp, SubOp, MulOp, AddOp
from mlir_quantum.dialects.quantum import AllocOp, QubitUnitaryOp, ExtractOp, PrintStateOp, DeviceOp, DeallocOp, InitializeOp, FinalizeOp
from mlir_quantum.dialects.tensor import FromElementsOp, GenerateOp, YieldOp, SplatOp
from mlir_quantum.dialects.scf import IfOp, YieldOp as SCFYieldOp

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
        listener = SimpleListener(ctx, module)
        walker.walk(listener, tree)

    with open("hadamard.mlir", "w") as output_file:
        print(module, file=output_file)
    subprocess.run(["./script.sh"])

class SymbolTable:

    def __init__(self):
        self.table = dict()

    def _setitemunsafe(self, symbol, ref):
        self.table[symbol] = ref

    def __setitem__(self, symbol, ref):
        if symbol in self.table:
            raise ValueError("Redefinition of a symbol")
        self._setitemunsafe(symbol, ref)

    def __getitem__(self, symbol):
        return self.table.get(symbol)

    def __contains__(self, symbol):
        return symbol in self.table

class Frame:

    def __init__(self, ip, parent=None):
        self._symbols = SymbolTable()
        self._ip = ip
        self._parent = parent
        self._params = [] 
        self._qubits = []
        self._locals = None

    def __enter__(self):
        self._ip.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self._ip.__exit__(type, value, traceback)

    def __getitem__(self, symbol):
        local = self._symbols[symbol]
        if local:
            return local
        if not self._parent:
            raise ValueError(f"Symbol {symbol} has not been defined!")
        return self._parent[symbol]

    def __setitem__(self, symbol, ref):
        self._symbols[symbol] = ref

    def __contains__(self, symbol):
        local = symbol in self._symbols
        if local:
            return local
        if not self._parent:
            return False
        return symbol in self._parent

    def addParam(self, param):
        self._params.append(param)

    @property
    def params(self):
        return self._params

    def addQubit(self, qubit):
        self._qubits.append(qubit)

    @property
    def qubits(self):
        return self._qubits

class FrameStack:

    def __init__(self):
        self.stack = []

    def push(self, ip):
        self.stack.append(Frame(ip, parent=self.top()))

    def pop(self):
        return self.stack.pop()

    def top(self):
        if not self.stack:
            return None
        return self.stack[-1]

    @property
    def main(self):
        return self.stack[1]

    @property
    def module(self):
        return self.stack[0]


class SimpleListener(qasm3ParserListener.qasm3ParserListener):
    def __init__(self, mlir_context, mlir_module):
        self.stack = FrameStack()
        self.sv = None
        self._mlir_context = mlir_context
        self._mlir_module = mlir_module
        self.qubitsUsed = 0
        self.userDefinedGates = dict()
        super().__init__()

    def enterProgram(self, ctx):
        ip = InsertionPoint(self._mlir_module.body)
        self.stack.push(ip)
        with self.stack.top():
            main = insert_main(self._mlir_context)
        ip = InsertionPoint(main.body.blocks[0])
        mainFunction = self.stack.push(ip)
        self.stack.top()["main"] = mainFunction
        with self.stack.top():
            insert_device(self._mlir_context)
            self.sv = insert_qreg(self._mlir_context)

    def exitProgram(self, ctx):
        with self.stack.top():
            PrintStateOp()
            DeallocOp(self.sv)
            FinalizeOp()
            ReturnOp([])

        self.stack.pop() # main function
        self.stack.pop() # module

    def floatLiteralToMLIR(self, ctx):
        with self.stack.top():
            floatLiteral = ctx.FloatLiteral().getText()
            f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
            constant = ArithConstantOp(f64, float(floatLiteral))
            ctx.mlir = constant.results[0]

    def exitLiteralExpression(self, ctx):
        if ctx.FloatLiteral():
            self.floatLiteralToMLIR(ctx)
        else:
            raise NotImplementedError("Not yet implemented!")

    def unaryNegationToMLIR(self, ctx):
        with self.stack.top():
            value = ctx.expression().mlir
            f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
            zero = ArithConstantOp(f64, 0.0)
            subOp = SubFOp(zero, value)
            ctx.mlir = subOp.results[0]

    def exitUnaryExpression(self, ctx):
        if not ctx.MINUS():
            raise NotImplementedError("Not yet implemented!")
        self.unaryNegationToMLIR(ctx)

    def exitQuantumDeclarationStatement(self, ctx):
        with self.stack.top() as frame:
            qubitName = ctx.Identifier().getText()
            if qubitName in frame:
                raise ValueError("You are re-declaring the same symbol")
            qubit_type = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
            i64 = mlir_quantum.ir.IntegerType.get_signless(64, self._mlir_context)
            index = ArithConstantOp(i64, self.qubitsUsed).results
            extractOp = ExtractOp(qubit_type, self.sv, idx=index)
            frame[qubitName] = extractOp.results[0]
            self.qubitsUsed += 1

    def enterGateStatement(self, ctx):
        gateName = ctx.Identifier().getText()
        cparams = ctx.params.children if ctx.params else []
        qparams = ctx.qubits.children
        retval = []
        params = []

        f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
        for cparam in cparams:
            params.append(f64)

        qubit = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
        for qparam in qparams:
            params.append(qubit)

        oq3params = []
        for param in cparams + qparams:
            oq3params.append(param.getText())

        with self.stack.module as _globals:
            if gateName in _globals:
                raise ValueError(f"Re-declaring gate {gateName}.")
            func = FuncOp(gateName, (params, retval))
            entry_block = func.add_entry_block()
            _globals[gateName] = func

        ip = InsertionPoint(func.body.blocks[0])
        self.stack.push(ip)
        with self.stack.top() as frame:
            for mlirParam, oq3Param in zip(entry_block.arguments, oq3params):
                frame[oq3Param] = mlirParam
                if mlirParam.type != qubit:
                    frame.addParam(mlirParam)
                else:
                    frame.addQubit(mlirParam)

    def exitGateStatement(self, ctx):
        with self.stack.top():
            ReturnOp([]) # No gate returns ever
        self.stack.pop() # function-scope

    def createGPhase(self, ctx):
        # We are going to have the assumption here
        # that gphase is only called within gate statements.
        # This means that gphase only applies to the arguments
        # Which are quantum bits.
        gamma = ctx.expressionList().expression()[0].mlir
        with self.stack.top() as frame:
            # We need to know how many qubits are in the scope!
            N = len(frame.qubits)
            qubit_t = frame.qubits[0].type
            # We need to create an Identity matrix of size 2^Nx2^N
            f64 = mlir_quantum.ir.F64Type.get(self._mlir_context)
            zero = ArithConstantOp(f64, 0.0)
            one = ArithConstantOp(f64, 1.0)
            complex128 = mlir_quantum.ir.ComplexType.get(f64)
            shape = [2**N, 2**N]
            tensor_complex128_= mlir_quantum.ir.RankedTensorType.get(shape, complex128)
            generateOp = GenerateOp(tensor_complex128_, [])
            index = mlir_quantum.ir.IndexType.get()
            Block.create_at_start(generateOp.body, [index, index])
            with InsertionPoint(generateOp.body.blocks[0]):
                arg0, arg1 = generateOp.body.blocks[0].arguments
                zero_igamma = CreateOp(complex128, zero, gamma)
                expOp = ExpOp(zero_igamma)
                i64 = mlir_quantum.ir.IntegerType.get_signless(64)
                eq = mlir_quantum.ir.IntegerAttr.get(i64, 0)
                pred = CmpIOp(eq, arg0, arg1)
                ifOp = IfOp(pred.results[0], [complex128], hasElse=True)
                with InsertionPoint(ifOp.then_block):
                    one_izero = CreateOp(complex128, one, zero)
                    mulOp = MulOp(one_izero, expOp)
                    SCFYieldOp(mulOp)
                with InsertionPoint(ifOp.else_block):
                    zero_izero = CreateOp(complex128, zero, zero)
                    SCFYieldOp(zero_izero)
                YieldOp(ifOp.results[0])
            # Now we need to multiply the 2^n * 2^n tensor by e^i*gamma
            matrix = generateOp.results[0]
            QubitUnitaryOp([qubit_t] * N, matrix, frame.qubits)

                



    def createU3(self, ctx):
        with self.stack.top() as frame:
            gateName = ctx.children[0].getText()
            isU3 = "U" == gateName
            assert isU3, "Not U3"

            theta = ctx.expressionList().children[0].mlir
            phi = ctx.expressionList().children[2].mlir
            _lambda = ctx.expressionList().children[4].mlir

            qubitName = ctx.children[4].getText()
            if qubitName not in frame:
                raise ValueError(f"Attempting to use non-declared qubit. {qubitName}")

            qubit = frame[qubitName]

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
            with self.stack.top() as frame:
                if not (gateName in frame):
                    raise ValueError(f"User defined gate {gateName} has not been defined.")
                funcName = mlir_quantum.ir.FlatSymbolRefAttr.get(gateName)
                qubits = ctx.gateOperandList().getText()
                params = []
                for qubit in qubits:
                    params.append(frame[qubit])
                CallOp([], funcName, params)


        
 
if __name__ == '__main__':
    main(sys.argv)
