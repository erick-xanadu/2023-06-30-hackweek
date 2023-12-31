import sys

from antlr4 import *
from openqasm_reference_parser import qasm3Lexer
from openqasm_reference_parser import qasm3Parser, qasm3ParserListener
import catalyst
from mlir_quantum import ir
from mlir_quantum import runtime
from mlir_quantum.dialects import arith
from mlir_quantum.dialects import bufferization
from mlir_quantum.dialects import builtin
import subprocess
from mlir_quantum.dialects.func import FuncOp, ReturnOp, CallOp
from mlir_quantum.dialects.arith import ConstantOp as ArithConstantOp, AddFOp, DivFOp, SubFOp, CmpIOp
from mlir_quantum.dialects.math import CosOp, SinOp
from mlir_quantum.dialects.complex import ExpOp, CreateOp, SubOp, MulOp, AddOp
from mlir_quantum.dialects.quantum import AllocOp, QubitUnitaryOp, ExtractOp, PrintStateOp, DeviceOp, DeallocOp, InitializeOp, FinalizeOp
from mlir_quantum.dialects.tensor import FromElementsOp, GenerateOp, YieldOp, SplatOp, EmptyOp
from mlir_quantum.dialects.scf import IfOp, YieldOp as SCFYieldOp
from mlir_quantum.dialects.linalg import MatmulOp, YieldOp as LinalgYieldOp

import mlir_quantum
from mlir_quantum.ir import Context, Module, InsertionPoint, Location, Block

def insert_qreg(ctx):
    qreg_type = mlir_quantum.ir.OpaqueType.get("quantum", "reg", ctx)
    i64 = mlir_quantum.ir.IntegerType.get_signless(64, ctx)
    size_attr = mlir_quantum.ir.IntegerAttr.get(i64, 1)
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
    input_stream = FileStream(argv[1], encoding='utf-8')
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
        self._currentUnitary = None

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

    def push_frame(self, frame):
        self.stack.append(frame)

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

    @property
    def f64(self):
        return mlir_quantum.ir.F64Type.get(self._mlir_context)

    @property
    def i64(self):
        return mlir_quantum.ir.IntegerType.get_signless(64, self._mlir_context)

    @property
    def complex128(self):
        return mlir_quantum.ir.ComplexType.get(self.f64)

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
        self.create_builtin_U3(ctx)

    def exitProgram(self, ctx):
        with self.stack.top():
            PrintStateOp()
            FinalizeOp()
            ReturnOp([])

        self.stack.pop() # main function
        self.stack.pop() # module


    def exitScalarType(self, ctx):
        if ctx.FLOAT():
            ctx.mlir_type = self.f64
        else:
            raise NotImplementedError(f"Unsupported classical type {ctx.getText()}")

    def exitClassicalDeclarationStatement(self, ctx):
        identifier = ctx.Identifier().getText()
        typeCtx = ctx.scalarType() if ctx.scalarType() else ctx.arrayType()
        assert typeCtx, "This is expected from the grammar."
        mlirType = typeCtx.mlir_type
        declExprCtx = ctx.declarationExpression()
        with self.stack.top() as frame:
            if declExprCtx:
                value = declExprCtx.mlir
            else:
                value = ArithConstantOp(mlirType, 0.0)
            frame[identifier] = value

    def enterConstDeclarationStatement(self, ctx):
        raise NotImplementedError("Not implemented yet!")

    def floatLiteralToMLIR(self, ctx):
        with self.stack.top():
            floatLiteral = ctx.FloatLiteral().getText()
            f64 = self.f64
            constant = ArithConstantOp(f64, float(floatLiteral))
            ctx.mlir = constant.results[0]

    def decimalIntegerLiteralToMLIR(self, ctx):
        with self.stack.top():
            lit = ctx.DecimalIntegerLiteral().getText()
            i64 = self.i64
            constant = ArithConstantOp(i64, int(lit))
            ctx.mlir = constant.results[0]

    def identifierToMLIR(self, ctx):
        with self.stack.top() as frame:
            ctx.mlir = frame[ctx.Identifier().getText()]

    def exitMultiplicativeExpression(self, ctx):
        if not ctx.SLASH():
            raise NotImplementedError("Not yet implemented!")
        numerator = ctx.expression()[0]
        denominator = ctx.expression()[1]
        with self.stack.top():
            divOp = DivFOp(numerator.mlir, denominator.mlir)
            ctx.mlir = divOp.results[0]

    def exitLiteralExpression(self, ctx):
        if ctx.FloatLiteral():
            self.floatLiteralToMLIR(ctx)
        elif ctx.Identifier():
            self.identifierToMLIR(ctx)
        elif ctx.DecimalIntegerLiteral():
            self.decimalIntegerLiteralToMLIR(ctx)
        else:
            raise NotImplementedError("Not yet implemented!")

    def exitDeclarationExpression(self, ctx):
        if ctx.expression():
            ctx.mlir = ctx.expression().mlir
        else:
            raise NotImplementedError("Not yet implemented!")

    def unaryNegationToMLIR(self, ctx):
        with self.stack.top():
            value = ctx.expression().mlir
            f64 = self.f64
            zero = ArithConstantOp(f64, 0.0)
            subOp = SubFOp(zero, value)
            ctx.mlir = subOp.results[0]

    def exitUnaryExpression(self, ctx):
        if not ctx.MINUS():
            raise NotImplementedError("Not yet implemented!")
        self.unaryNegationToMLIR(ctx)

    def exitDesignator(self, ctx):
        exprCtx = ctx.expression()
        ctx.mlir = exprCtx.mlir

    def exitQuantumDeclarationStatement(self, ctx):
        qubitName = ctx.Identifier().getText()
        with self.stack.top() as frame:
            if qubitName in frame:
                raise ValueError(f"You are re-declaring the same symbol {qubitName}")
            qreg = insert_qreg(self._mlir_context)
            qubit_type = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
            i64 = self.i64
            index = ArithConstantOp(i64, 0).results[0]
            extractOp = ExtractOp(qubit_type, qreg, idx=index)
            frame[qubitName] = extractOp.results[0]
            self.qubitsUsed += 1

    def enterGateStatement(self, ctx):
        gateName = ctx.Identifier().getText()
        paramsCtx = ctx.params
        qubitsCtx = ctx.qubits
        oq3params = []
        retval = []
        params = []

        f64 = self.f64
        if paramsCtx:
            for cparam in paramsCtx.Identifier():
                oq3params.append(cparam.getText())
                params.append(f64)

        qubit = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
        for qparam in qubitsCtx.Identifier():
            oq3params.append(qparam.getText())
            params.append(qubit)

        retval = [self.getIdentityNType(len(qubitsCtx.Identifier()))]

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

                frame._currentUnitary = self.IdentityN(len(frame.qubits))

    def getIdentityNType(self, N):
            shape = [2**N, 2**N]
            return mlir_quantum.ir.RankedTensorType.get(shape, self.complex128)

    def IdentityN(self, N):
        with self.stack.top() as frame:
            i64 = self.i64
            f64 = self.f64
            complex128 = mlir_quantum.ir.ComplexType.get(f64)

            zero = ArithConstantOp(f64, 0.0)
            one = ArithConstantOp(f64, 1.0)
            zero_izero = CreateOp(complex128, zero, zero)
            one_izero = CreateOp(complex128, one, zero)

            shape = [2**N, 2**N]
            tensor_complex128_= mlir_quantum.ir.RankedTensorType.get(shape, complex128)

            generateOp = GenerateOp(tensor_complex128_, [])
            index = mlir_quantum.ir.IndexType.get()
            Block.create_at_start(generateOp.body, [index, index])

            with InsertionPoint(generateOp.body.blocks[0]):
                arg0, arg1 = generateOp.body.blocks[0].arguments
                eq = mlir_quantum.ir.IntegerAttr.get(i64, 0)
                pred = CmpIOp(eq, arg0, arg1)
                ifOp = IfOp(pred.results[0], [complex128], hasElse=True)
                with InsertionPoint(ifOp.then_block):
                    SCFYieldOp(one_izero)
                with InsertionPoint(ifOp.else_block):
                    SCFYieldOp(zero_izero)
                YieldOp(ifOp.results[0])

            matrix = generateOp.results[0]
            return matrix

    def create_builtin_U3(self, ctx):
        name = "__builtin_U"
        f64 = self.f64
        qubit = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
        paramTypes = [f64, f64, f64, qubit]
        shape = [2, 2]
        complex128 = mlir_quantum.ir.ComplexType.get(f64)
        tensor_complex128_= mlir_quantum.ir.RankedTensorType.get(shape, complex128)
        retval = tensor_complex128_

        with self.stack.module as _globals:
            if name in _globals:
                raise ValueError(f"Re-declaring gate {gateName}.")
            func = FuncOp(name, (paramTypes, [retval]))
            entry_block = func.add_entry_block()
            _globals[name] = func

        ip = InsertionPoint(func.body.blocks[0])
        self.stack.push(ip)
        with self.stack.top() as frame:
            theta, phi, _lambda, qubit = entry_block.arguments

            f64 = self.f64
            complex128 = mlir_quantum.ir.ComplexType.get(f64)
            zero = ArithConstantOp(f64, 0.0)
            one = ArithConstantOp(f64, 1.0)
            mone = ArithConstantOp(f64, -1.0)
            two = ArithConstantOp(f64, 2.0)
            half = ArithConstantOp(f64, 0.5)
            zero_izero = CreateOp(complex128, zero, zero)
            one_izero = CreateOp(complex128, one, zero)
            zero_ione = CreateOp(complex128, zero, one)
            half_izero = CreateOp(complex128, half, zero)
            zero_imone = CreateOp(complex128, zero, mone)

            itheta = CreateOp(complex128, zero, theta)
            e_itheta = ExpOp(itheta)
            addOp = AddOp(one_izero, e_itheta)
            tmp1 = MulOp(half_izero, addOp)
            m00 = tmp1

            subOp = SubOp(one_izero, e_itheta)
            ilambda = CreateOp(complex128, zero, _lambda)
            e_ilambda = ExpOp(ilambda)
            tmp = MulOp(zero_imone, e_ilambda)
            tmp1 = MulOp(half_izero, tmp)
            tmp2 = MulOp(tmp1, subOp)
            m01 = tmp2
        
            iphi = CreateOp(complex128, zero, phi)
            e_iphi = ExpOp(iphi)
            ie_iphi = MulOp(zero_ione, e_iphi)
            tmp = MulOp(ie_iphi, subOp)
            tmp1 = MulOp(tmp, half_izero)
            m10 = tmp1


            iphi_plus_lambda = AddOp(ilambda, iphi)
            e_iphi_plus_lambda = ExpOp(iphi_plus_lambda)
            tmp = MulOp(e_iphi_plus_lambda, half_izero)
            tmp1 = MulOp(tmp, addOp)
            m11 = tmp1

            tensor_complex128_= mlir_quantum.ir.RankedTensorType.get([2, 2], complex128)
            matrix = FromElementsOp.build_generic([tensor_complex128_], [m00.results[0], m01.results[0], m10.results[0], m11.results[0]])

            ReturnOp(matrix.results)
        self.stack.pop()


    def exitGateStatement(self, ctx):
        with self.stack.top() as frame:
            import pdb
            pdb.set_trace()
            ReturnOp([frame._currentUnitary])
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
            f64 = self.f64
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



    def exitGateCallStatement(self, ctx):
        gateName = ctx.Identifier().getText() if ctx.Identifier() else ctx.GPHASE().getText()

        isGphase = "gphase" == gateName
        if isGphase:
            self.createGPhase(ctx)
        else:
            with self.stack.top() as frame:
                if not (gateName in frame):
                    raise ValueError(f"User defined gate {gateName} has not been defined.")
                gateModifierCtx = ctx.gateModifier()
                if gateModifierCtx:
                    gateModifierCtx.reverse()

                is_ctrl = False if not gateModifierCtx else gateModifierCtx[0].CTRL()
                is_pow = False if not gateModifierCtx else gateModifierCtx[0].POW()

                power = []
                control = []

                funcName = mlir_quantum.ir.FlatSymbolRefAttr.get(gateName)
                params = []
                exprListCtx = ctx.expressionList()
                if exprListCtx:
                    for e in exprListCtx.expression():
                        params.append(e.mlir)
                gateOperandCtx = ctx.gateOperandList()
                qubits = []
                for idx, qubit in enumerate(gateOperandCtx.gateOperand()):
                    if idx == 0 and is_ctrl:
                        control.append(frame[qubit.getText()])
                        continue
                    params.append(frame[qubit.getText()])
                    qubits.append(frame[qubit.getText()])
                mlir_function = frame[str(funcName)[1:]]
                results = mlir_function.type.results
                callOp = CallOp(results, funcName, params)
                if frame._currentUnitary:
                    emptyOp = EmptyOp([2, 2], self.complex128)
                    op = MatmulOp([callOp.results[0], frame._currentUnitary], outputs=[emptyOp.results[0]], results=[emptyOp.results[0].type])
                    entry_block = Block.create_at_start(op.regions[0], [self.complex128, self.complex128, self.complex128])
                    
                    with InsertionPoint(entry_block):
                        a, b, c = entry_block.arguments
                        d = MulOp(a, b)
                        e = AddOp(c, d)
                        LinalgYieldOp(e)

                    frame._currentUnitary = op.results[0]
                else: # We are in the global scope and can actually apply QubitUnitaryOp
                    qubit_type = mlir_quantum.ir.OpaqueType.get("quantum", "bit")
                    QubitUnitaryOp([qubit.type for qubit in qubits], callOp.results, qubits)


        
 
if __name__ == '__main__':
    main(sys.argv)
