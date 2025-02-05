"""
Intermediate Representation (IR) for eLang compiler/codegen.
We represent each function as a list of IRBlock's (basic blocks).
Within blocks, IRInstr objects define operations. We keep concurrency
and request instructions at this IR level for now.
"""

from typing import List, Optional, Union

class IRModule:
    """
    Top-level container for the entire program's IR.
    """
    def __init__(self):
        self.functions: List[IRFunction] = []
        self.globals = {}  # e.g. global variables, classes, etc.

    def add_function(self, func):
        self.functions.append(func)

    def __repr__(self):
        lines = ["<IRModule>"]
        for f in self.functions:
            lines.append(repr(f))
        return "\n".join(lines)


class IRFunction:
    """
    Represents a single function in IR form.
    param_types, return_type = strings from our type system ("int", "float", etc.)
    blocks = list of IRBlock
    """
    def __init__(self, name: str, param_types: List[str], return_type: str = "void"):
        self.name = name
        self.param_types = param_types
        self.return_type = return_type
        self.blocks: List[IRBlock] = []
        self.temp_counter = 0

    def create_temp(self) -> 'IRTemp':
        """
        Generate a new temporary register name for this function.
        """
        self.temp_counter += 1
        return IRTemp(f"t{self.temp_counter}")

    def add_block(self, block: 'IRBlock'):
        self.blocks.append(block)

    def __repr__(self):
        lines = [f"<Function {self.name}({','.join(self.param_types)}) -> {self.return_type}>"]
        for b in self.blocks:
            lines.append(repr(b))
        return "\n".join(lines)


class IRBlock:
    """
    A basic block containing a list of instructions, with a label.
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
    Base class for anything that can be used as an operand in instructions:
    temporaries, constants, global references, etc.
    """
    pass


class IRTemp(IRValue):
    """
    A temporary register within a function.
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"%{self.name}"


class IRConst(IRValue):
    """
    A numeric or string constant operand.
    """
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        if isinstance(self.value, str):
            return f"\"{self.value}\""
        return str(self.value)


class IRGlobal(IRValue):
    """
    A reference to a global variable or function name in the IR.
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"@{self.name}"


class IRInstr:
    """
    Base class for all instructions in our IR.
    Each subclass handles a different operation (binop, call, spawn, etc.).
    """
    pass


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
    binop dest, left, right, op
    Example: dest = left + right
    """
    def __init__(self, dest: IRTemp, left: IRValue, right: IRValue, op: str):
        self.dest = dest
        self.left = left
        self.right = right
        self.op = op  # e.g. '+', '-', '*', '/', '%', '<<', '>>'

    def __repr__(self):
        return f"{self.op.upper()} {self.dest} <- {self.left}, {self.right}"


class UnOpInstr(IRInstr):
    """
    unary dest, src, op
    Example: dest = -src
    """
    def __init__(self, dest: IRTemp, src: IRValue, op: str):
        self.dest = dest
        self.src = src
        self.op = op  # e.g. 'NEG', 'NOT'

    def __repr__(self):
        return f"{self.op.upper()} {self.dest} <- {self.src}"


class CallInstr(IRInstr):
    """
    call dest, func, args[]
    If 'dest' is None => no return value (call is used as statement).
    """
    def __init__(self, dest: Optional[IRTemp], func: IRValue, args: List[IRValue]):
        self.dest = dest
        self.func = func
        self.args = args

    def __repr__(self):
        arg_str = ", ".join(repr(a) for a in self.args)
        if self.dest is not None:
            return f"CALL {self.dest} <- {self.func}({arg_str})"
        else:
            return f"CALL {self.func}({arg_str})"


class RequestInstr(IRInstr):
    """
    Represents an HTTP request, e.g. GET/POST with optional headers/body.
    method: str = 'GET','POST', etc.
    url: IRValue
    headers: IRValue or None
    body: IRValue or None
    dest: optional IRTemp if we store the response
    """
    def __init__(self, dest: Optional[IRTemp], method: str, url: IRValue,
                 headers: Optional[IRValue], body: Optional[IRValue]):
        self.dest = dest
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body

    def __repr__(self):
        s = f"REQUEST {self.method} {self.url}"
        if self.headers:
            s += f" HEADERS={self.headers}"
        if self.body:
            s += f" BODY={self.body}"
        if self.dest:
            s = f"{self.dest} <- " + s
        return s


class JumpInstr(IRInstr):
    """
    Unconditional jump to a block label.
    """
    def __init__(self, label: str):
        self.label = label

    def __repr__(self):
        return f"JUMP {self.label}"


class CJumpInstr(IRInstr):
    """
    Conditional jump. The condition is in 'cond' (bool typed).
    If cond is true => jump to true_label, else => jump to false_label
    """
    def __init__(self, cond: IRValue, true_label: str, false_label: str):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label

    def __repr__(self):
        return f"CJUMP {self.cond} ? {self.true_label} : {self.false_label}"


class ReturnInstr(IRInstr):
    """
    Return from the function. May or may not have a value.
    """
    def __init__(self, value: Optional[IRValue]):
        self.value = value

    def __repr__(self):
        if self.value is None:
            return "RETURN"
        else:
            return f"RETURN {self.value}"


class SpawnInstr(IRInstr):
    """
    concurrency spawn => new thread or task
    spawnVal: IRValue (function pointer or some representation)
    If we need a handle, we store it in 'dest'
    """
    def __init__(self, dest: Optional[IRTemp], spawnVal: IRValue):
        self.dest = dest
        self.spawnVal = spawnVal

    def __repr__(self):
        if self.dest:
            return f"SPAWN {self.dest} <- {self.spawnVal}"
        else:
            return f"SPAWN {self.spawnVal}"


class ThreadInstr(IRInstr):
    """
    Declares/creates a new thread with some label or block
    If there's a function-like body, we might store it or reference a block
    """
    def __init__(self, thread_name: str):
        self.thread_name = thread_name

    def __repr__(self):
        return f"THREAD {self.thread_name}"


class LockInstr(IRInstr):
    """
    Acquire a lock on some variable or object
    """
    def __init__(self, lockVal: IRValue):
        self.lockVal = lockVal

    def __repr__(self):
        return f"LOCK {self.lockVal}"


class UnlockInstr(IRInstr):
    """
    Release a lock
    """
    def __init__(self, lockVal: IRValue):
        self.lockVal = lockVal

    def __repr__(self):
        return f"UNLOCK {self.lockVal}"


class SleepInstr(IRInstr):
    """
    Sleep for the duration in 'durationVal'
    """
    def __init__(self, durationVal: IRValue):
        self.durationVal = durationVal

    def __repr__(self):
        return f"SLEEP {self.durationVal}"


class KillInstr(IRInstr):
    """
    Kill a thread
    """
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"KILL {self.threadVal}"


class DetachInstr(IRInstr):
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"DETACH {self.threadVal}"


class JoinInstr(IRInstr):
    def __init__(self, threadVal: IRValue):
        self.threadVal = threadVal

    def __repr__(self):
        return f"JOIN {self.threadVal}"


class PrintInstr(IRInstr):
    """
    Print a value
    """
    def __init__(self, val: IRValue):
        self.val = val

    def __repr__(self):
        return f"PRINT {self.val}"
