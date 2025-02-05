

from typing import List, Dict, Optional, Union


class IRType:
    """
    Represents a type in the IR, such as int, float, bool, string, any,
    or advanced concurrency types like thread, lock, channel.
    You can add generics, function types, etc. as needed.
    """
    def __init__(self, name: str):
        self.name = name  # e.g. "int","float","bool","string","any","thread"

    def __repr__(self):
        return self.name


class IRModule:
    """
    The entire program's IR, containing global variables, function definitions, etc.
    """
    def __init__(self):
        self.functions: List[IRFunction] = []
        self.globals: Dict[str, IRGlobalVar] = {}

    def add_function(self, func: 'IRFunction'):
        self.functions.append(func)

    def add_global(self, gvar: 'IRGlobalVar'):
        self.globals[gvar.name] = gvar

    def __repr__(self):
        lines = ["<IRModule>"]
        for gv in self.globals.values():
            lines.append(repr(gv))
        for f in self.functions:
            lines.append(repr(f))
        return "\n".join(lines)


class IRGlobalVar:
    """
    A global variable or reference in the IR.
    """
    def __init__(self, name: str, ty: IRType, init_value=None):
        self.name = name
        self.type = ty
        self.init_value = init_value

    def __repr__(self):
        return f"<Global {self.name}:{self.type} = {self.init_value}>"


class IRFunction:
    """
    A function in the IR, potentially in SSA form.
    param_types: list of IRType for parameters
    return_type: IRType
    blocks: list of IRBlock
    ssa: bool indicates whether we are in SSA form
    """
    def __init__(self, name: str,
                 param_types: List[IRType],
                 return_type: IRType,
                 ssa: bool = True):
        self.name = name
        self.param_types = param_types
        self.return_type = return_type
        self.blocks: List[IRBlock] = []
        self.temp_counter = 0
        self.ssa = ssa

    def create_temp(self, ty: IRType) -> 'IRTemp':
        """
        Generate a typed SSA temp register for this function.
        """
        self.temp_counter += 1
        reg_name = f"t{self.temp_counter}"
        return IRTemp(reg_name, ty)

    def add_block(self, block: 'IRBlock'):
        self.blocks.append(block)

    def __repr__(self):
        lines = [f"<Function {self.name} params={self.param_types} -> {self.return_type} (SSA={self.ssa})>"]
        for b in self.blocks:
            lines.append(repr(b))
        return "\n".join(lines)


class IRBlock:
    """
    A basic block: label + list of IRInstructions.
    Predecessors/successors info can be stored for advanced flow analysis.
    """
    def __init__(self, label: str):
        self.label = label
        self.instructions: List[IRInstr] = []

    def add_instr(self, instr: 'IRInstr'):
        self.instructions.append(instr)

    def __repr__(self):
        lines = [f"<Block {self.label}>"]
        for i in self.instructions:
            lines.append("  " + repr(i))
        return "\n".join(lines)


class IRValue:
    """
    Base class for IR operands (constants, temps, etc.).
    """
    pass


class IRTemp(IRValue):
    """
    A typed SSA register, e.g. %t3 : int
    """
    def __init__(self, name: str, ty: IRType):
        self.name = name
        self.ty = ty

    def __repr__(self):
        return f"%{self.name}:{self.ty}"


class IRConst(IRValue):
    """
    A constant value (int, float, string, bool).
    """
    def __init__(self, value, ty: IRType):
        self.value = value
        self.ty = ty

    def __repr__(self):
        # If string, wrap in quotes
        if self.ty.name == "string":
            return f"\"{self.value}\""
        return f"{self.value}:{self.ty}"


class IRGlobalRef(IRValue):
    """
    A reference to a global variable or function name.
    """
    def __init__(self, name: str, ty: IRType):
        self.name = name
        self.ty = ty

    def __repr__(self):
        return f"@{self.name}:{self.ty}"


class IRInstr:
    """
    Base class for IR instructions.
    Subclasses will be used for arithmetic, memory, concurrency, etc.
    """
    pass


class PhiInstr(IRInstr):
    """
    SSA phi-node: dest = phi( (val1, pred_block1), (val2, pred_block2), ... )
    """
    def __init__(self, dest: IRTemp, incomings: List[tuple]):
        """
        incomings: list of (IRValue, block_label)
        """
        self.dest = dest
        self.incomings = incomings

    def __repr__(self):
        parts = []
        for val, blk in self.incomings:
            parts.append(f"({val}, {blk})")
        joined = ", ".join(parts)
        return f"{self.dest} = PHI {joined}"


class MoveInstr(IRInstr):
    """
    move dest, src
    """
    def __init__(self, dest: IRTemp, src: IRValue):
        self.dest = dest
        self.src = src

    def __repr__(self):
        return f"MOVE {self.dest} <- {self.src}"


class BinOpInstr(IRInstr):
    """
    dest = left op right
    """
    def __init__(self, dest: IRTemp, left: IRValue, right: IRValue, op: str):
        self.dest = dest
        self.left = left
        self.right = right
        self.op = op

    def __repr__(self):
        return f"{self.dest} = {self.left} {self.op} {self.right}"


class UnOpInstr(IRInstr):
    """
    dest = op src
    """
    def __init__(self, dest: IRTemp, op: str, src: IRValue):
        self.dest = dest
        self.op = op
        self.src = src

    def __repr__(self):
        return f"{self.dest} = {self.op} {self.src}"


class LoadInstr(IRInstr):
    """
    dest = load address
    For advanced memory usage
    """
    def __init__(self, dest: IRTemp, address: IRValue):
        self.dest = dest
        self.address = address

    def __repr__(self):
        return f"{self.dest} = LOAD {self.address}"


class StoreInstr(IRInstr):
    """
    store src -> address
    """
    def __init__(self, address: IRValue, src: IRValue):
        self.address = address
        self.src = src

    def __repr__(self):
        return f"STORE {self.src} -> {self.address}"


class AtomicLoadInstr(IRInstr):
    """
    concurrency aware load
    """
    def __init__(self, dest: IRTemp, address: IRValue):
        self.dest = dest
        self.address = address

    def __repr__(self):
        return f"{self.dest} = ATOMIC_LOAD {self.address}"


class AtomicStoreInstr(IRInstr):
    """
    concurrency aware store
    """
    def __init__(self, address: IRValue, src: IRValue):
        self.address = address
        self.src = src

    def __repr__(self):
        return f"ATOMIC_STORE {self.src} -> {self.address}"


class AcquireLockInstr(IRInstr):
    """
    acquire a lock variable
    """
    def __init__(self, lockVal: IRValue):
        self.lockVal = lockVal

    def __repr__(self):
        return f"ACQUIRE_LOCK {self.lockVal}"


class ReleaseLockInstr(IRInstr):
    """
    release a lock variable
    """
    def __init__(self, lockVal: IRValue):
        self.lockVal = lockVal

    def __repr__(self):
        return f"RELEASE_LOCK {self.lockVal}"


class JumpInstr(IRInstr):
    def __init__(self, label: str):
        self.label = label

    def __repr__(self):
        return f"JUMP {self.label}"


class CJumpInstr(IRInstr):
    """
    cond: IRValue (bool)
    true_label, false_label: str
    """
    def __init__(self, cond: IRValue, true_label: str, false_label: str):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label

    def __repr__(self):
        return f"CJUMP {self.cond} ? {self.true_label} : {self.false_label}"


class ReturnInstr(IRInstr):
    """
    Return with optional value
    """
    def __init__(self, value: Optional[IRValue]):
        self.value = value

    def __repr__(self):
        if self.value is None:
            return "RETURN"
        else:
            return f"RETURN {self.value}"


class CallInstr(IRInstr):
    """
    dest = call func(args)
    if dest is None => no return
    """
    def __init__(self,
                 dest: Optional[IRTemp],
                 func: IRValue,
                 args: List[IRValue]):
        self.dest = dest
        self.func = func
        self.args = args

    def __repr__(self):
        arg_str = ", ".join(repr(a) for a in self.args)
        if self.dest:
            return f"{self.dest} = CALL {self.func}({arg_str})"
        else:
            return f"CALL {self.func}({arg_str})"


class RequestInstr(IRInstr):
    """
    Represents an HTTP request in the IR: e.g. GET, POST, HEADERS, BODY
    """
    def __init__(self,
                 dest: Optional[IRTemp],
                 method: str,
                 url: IRValue,
                 headers: Optional[IRValue],
                 body: Optional[IRValue]):
        self.dest = dest
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body

    def __repr__(self):
        base = f"REQUEST {self.method} {self.url}"
        if self.headers:
            base += f" HEADERS={self.headers}"
        if self.body:
            base += f" BODY={self.body}"
        if self.dest:
            return f"{self.dest} = {base}"
        else:
            return base


class SpawnInstr(IRInstr):
    """
    concurrency spawn => new thread or task
    spawnVal: IRValue referencing a function or closure
    If we store handle => dest
    """
    def __init__(self, dest: Optional[IRTemp], spawnVal: IRValue):
        self.dest = dest
        self.spawnVal = spawnVal

    def __repr__(self):
        if self.dest:
            return f"{self.dest} = SPAWN {self.spawnVal}"
        else:
            return f"SPAWN {self.spawnVal}"


class ThreadForkInstr(IRInstr):
    """
    Lower-level concurrency: fork a new thread entry
    """
    def __init__(self, dest: Optional[IRTemp], func: IRValue, args: List[IRValue]):
        self.dest = dest
        self.func = func
        self.args = args

    def __repr__(self):
        arg_str = ", ".join(repr(a) for a in self.args)
        if self.dest:
            return f"{self.dest} = THREAD_FORK {self.func}({arg_str})"
        else:
            return f"THREAD_FORK {self.func}({arg_str})"


class ThreadJoinInstr(IRInstr):
    """
    join a previously forked thread
    """
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"THREAD_JOIN {self.threadVal}"


class KillInstr(IRInstr):
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"KILL {self.threadVal}"


class DetachInstr(IRInstr):
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"DETACH {self.threadVal}"


class SleepInstr(IRInstr):
    def __init__(self, durationVal: IRValue):
        self.durationVal = durationVal

    def __repr__(self):
        return f"SLEEP {self.durationVal}"


class PrintInstr(IRInstr):
    def __init__(self, val: IRValue):
        self.val = val

    def __repr__(self):
        return f"PRINT {self.val}"
