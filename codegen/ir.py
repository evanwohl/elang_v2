from typing import List, Dict, Optional, Union

class IRType:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name

class IRModule:
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
    def __init__(self, name: str, ty: 'IRType', init_value=None):
        self.name = name
        self.type = ty
        self.init_value = init_value
    def __repr__(self):
        return f"<Global {self.name}:{self.type} = {self.init_value}>"

class IRFunction:
    def __init__(self, name: str, param_types: List['IRType'], return_type: 'IRType', ssa: bool=True):
        self.name = name
        self.param_types = param_types
        self.return_type = return_type
        self.blocks: List[IRBlock] = []
        self.temp_counter = 0
        self.ssa = ssa
    def create_temp(self, ty: 'IRType') -> 'IRTemp':
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
    def __init__(self, label: str):
        self.label = label
        self.instructions: List['IRInstr'] = []
    def add_instr(self, instr: 'IRInstr'):
        self.instructions.append(instr)
    def __repr__(self):
        lines = [f"<Block {self.label}>"]
        for i in self.instructions:
            lines.append("  " + repr(i))
        return "\n".join(lines)

class IRValue:
    pass

class IRTemp(IRValue):
    def __init__(self, name: str, ty: IRType):
        self.name = name
        self.ty = ty
    def __repr__(self):
        return f"%{self.name}:{self.ty}"

class IRConst(IRValue):
    def __init__(self, value, ty: IRType):
        self.value = value
        self.ty = ty
    def __repr__(self):
        if self.ty.name == "string":
            return f"\"{self.value}\""
        return f"{self.value}:{self.ty}"

class IRGlobalRef(IRValue):
    def __init__(self, name: str, ty: IRType):
        self.name = name
        self.ty = ty
    def __repr__(self):
        return f"@{self.name}:{self.ty}"

class IRInstr:
    pass

class PhiInstr(IRInstr):
    def __init__(self, dest: IRTemp, incomings: List[tuple]):
        self.dest = dest
        self.incomings = incomings
    def __repr__(self):
        parts = []
        for val, blk in self.incomings:
            parts.append(f"({val}, {blk})")
        j = ", ".join(parts)
        return f"{self.dest} = PHI {j}"

class MoveInstr(IRInstr):
    def __init__(self, dest: IRTemp, src: IRValue):
        self.dest = dest
        self.src = src
    def __repr__(self):
        return f"MOVE {self.dest} <- {self.src}"

class BinOpInstr(IRInstr):
    def __init__(self, dest: IRTemp, left: IRValue, right: IRValue, op: str):
        self.dest = dest
        self.left = left
        self.right = right
        self.op = op
    def __repr__(self):
        return f"{self.dest} = {self.left} {self.op} {self.right}"

class UnOpInstr(IRInstr):
    def __init__(self, dest: IRTemp, op: str, src: IRValue):
        self.dest = dest
        self.op = op
        self.src = src
    def __repr__(self):
        return f"{self.dest} = {self.op} {self.src}"

class LoadInstr(IRInstr):
    def __init__(self, dest: IRTemp, address: IRValue):
        self.dest = dest
        self.address = address
    def __repr__(self):
        return f"{self.dest} = LOAD {self.address}"

class StoreInstr(IRInstr):
    def __init__(self, address: IRValue, src: IRValue):
        self.address = address
        self.src = src
    def __repr__(self):
        return f"STORE {self.src} -> {self.address}"

class AtomicLoadInstr(IRInstr):
    def __init__(self, dest: IRTemp, address: IRValue):
        self.dest = dest
        self.address = address
    def __repr__(self):
        return f"{self.dest} = ATOMIC_LOAD {self.address}"

class AtomicStoreInstr(IRInstr):
    def __init__(self, address: IRValue, src: IRValue):
        self.address = address
        self.src = src
    def __repr__(self):
        return f"ATOMIC_STORE {self.src} -> {self.address}"

class AcquireLockInstr(IRInstr):
    def __init__(self, lockVal: IRValue):
        self.lockVal = lockVal
    def __repr__(self):
        return f"ACQUIRE_LOCK {self.lockVal}"

class ReleaseLockInstr(IRInstr):
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
    def __init__(self, cond: IRValue, true_label: str, false_label: str):
        self.cond = cond
        self.true_label = true_label
        self.false_label = false_label
    def __repr__(self):
        return f"CJUMP {self.cond} ? {self.true_label} : {self.false_label}"

class ReturnInstr(IRInstr):
    def __init__(self, value: Optional[IRValue]):
        self.value = value
    def __repr__(self):
        if self.value is None:
            return "RETURN"
        else:
            return f"RETURN {self.value}"

class CallInstr(IRInstr):
    def __init__(self, dest: Optional[IRTemp], func: IRValue, args: List[IRValue]):
        self.dest = dest
        self.func = func
        self.args = args
    def __repr__(self):
        a = ", ".join(repr(x) for x in self.args)
        if self.dest:
            return f"{self.dest} = CALL {self.func}({a})"
        else:
            return f"CALL {self.func}({a})"

class RequestInstr(IRInstr):
    def __init__(self, dest: Optional[IRTemp], method: str, url: IRValue, headers: Optional[IRValue], body: Optional[IRValue]):
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
        return base

class SpawnInstr(IRInstr):
    def __init__(self, dest: Optional[IRTemp], spawnVal: IRValue):
        self.dest = dest
        self.spawnVal = spawnVal
    def __repr__(self):
        if self.dest:
            return f"{self.dest} = SPAWN {self.spawnVal}"
        return f"SPAWN {self.spawnVal}"

class ThreadForkInstr(IRInstr):
    def __init__(self, dest: Optional[IRTemp], func: IRValue, args: List[IRValue]):
        self.dest = dest
        self.func = func
        self.args = args
    def __repr__(self):
        a = ", ".join(repr(x) for x in self.args)
        if self.dest:
            return f"{self.dest} = THREAD_FORK {self.func}({a})"
        return f"THREAD_FORK {self.func}({a})"

class ThreadJoinInstr(IRInstr):
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

class ChannelSendInstr(IRInstr):
    def __init__(self, channel: IRValue, val: IRValue):
        self.channel = channel
        self.val = val
    def __repr__(self):
        return f"CHANNEL_SEND {self.channel}, {self.val}"

class ChannelRecvInstr(IRInstr):
    def __init__(self, dest: IRTemp, channel: IRValue):
        self.dest = dest
        self.channel = channel
    def __repr__(self):
        return f"{self.dest} = CHANNEL_RECV {self.channel}"

class WaitAllInstr(IRInstr):
    def __init__(self, tasks: List[IRValue]):
        self.tasks = tasks
    def __repr__(self):
        s = ", ".join(repr(x) for x in self.tasks)
        return f"WAIT_ALL [{s}]"
class CreateDictInstr(IRInstr):
    def __init__(self, dest):
        self.dest = dest
    def __repr__(self):
        return f"{self.dest} = CREATE_DICT"

class DictSetInstr(IRInstr):
    def __init__(self, dict_temp, key_temp, val_temp):
        self.dict_temp = dict_temp
        self.key_temp = key_temp
        self.val_temp = val_temp
    def __repr__(self):
        return f"DICT_SET {self.dict_temp}, {self.key_temp}, {self.val_temp}"

class CreateArrayInstr(IRInstr):
    def __init__(self, dest):
        self.dest = dest
    def __repr__(self):
        return f"{self.dest} = CREATE_ARRAY"

class ArrayPushInstr(IRInstr):
    def __init__(self, arr_temp, val_temp):
        self.arr_temp = arr_temp
        self.val_temp = val_temp
    def __repr__(self):
        return f"ARRAY_PUSH {self.arr_temp}, {self.val_temp}"
