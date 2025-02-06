"""
ir_utils.py

An expanded suite of utilities for eLang's advanced IR, focusing on concurrency
and request transformations. Adds advanced IRBuilder methods, concurrency passes,
and request-optimizing passes for a "requests-focused, hyper-fast" language.
"""

from typing import List, Dict, Set, Optional
# We'll assume you have all these from your advanced ir.py
from codegen.ir import (
    IRModule, IRFunction, IRBlock, IRType, IRTemp, IRConst, IRGlobalRef,
    IRInstr, PhiInstr, MoveInstr, BinOpInstr, UnOpInstr, LoadInstr,
    StoreInstr, AtomicLoadInstr, AtomicStoreInstr, AcquireLockInstr,
    ReleaseLockInstr, JumpInstr, CJumpInstr, ReturnInstr, CallInstr,
    RequestInstr, SpawnInstr, ThreadForkInstr, ThreadJoinInstr, KillInstr,
    DetachInstr, SleepInstr, PrintInstr, ChannelSendInstr, ChannelRecvInstr,
    WaitAllInstr
)

# -----------------------------------------------------------------------
# IR BUILDER
# -----------------------------------------------------------------------

class IRBuilder:
    """
    A helper class to build IR instructions more easily, now expanded
    with concurrency+request-oriented methods.
    """

    def __init__(self, function: IRFunction):
        self.function = function
        self.current_block: Optional[IRBlock] = None

    def set_block(self, block: IRBlock):
        self.current_block = block

    def emit(self, instr: IRInstr):
        if not self.current_block:
            raise RuntimeError("No current block set in IRBuilder")
        self.current_block.add_instr(instr)
        return instr

    def create_temp(self, ty: IRType) -> IRTemp:
        return self.function.create_temp(ty)

    # ---------- Basic building methods (like before) ----------
    def binop(self, op: str, left, right, ty: IRType) -> IRTemp:
        left_val = self.ensure_value(left, ty)
        right_val = self.ensure_value(right, ty)
        dest = self.create_temp(ty)
        instr = BinOpInstr(dest, left_val, right_val, op)
        self.emit(instr)
        return dest

    def unop(self, op: str, src, ty: IRType) -> IRTemp:
        src_val = self.ensure_value(src, ty)
        dest = self.create_temp(ty)
        instr = UnOpInstr(dest, op, src_val)
        self.emit(instr)
        return dest

    def move(self, dest: IRTemp, src, ty: IRType):
        src_val = self.ensure_value(src, ty)
        instr = MoveInstr(dest, src_val)
        self.emit(instr)
        return dest

    def call(self, func, arg_list, ret_ty: IRType):
        args = []
        for a in arg_list:
            args.append(self.ensure_value(a, IRType("any")))  # or typed if you prefer
        if ret_ty.name == "void":
            instr = CallInstr(None, func, args)
            self.emit(instr)
            return None
        else:
            dest = self.create_temp(ret_ty)
            instr = CallInstr(dest, func, args)
            self.emit(instr)
            return dest

    def request(self, method: str, url, headers=None, body=None, ret_ty=IRType("any")):
        url_val = self.ensure_value(url, IRType("string"))
        hdr_val = self.ensure_value(headers, IRType("any")) if headers else None
        body_val = self.ensure_value(body, IRType("any")) if body else None
        if ret_ty.name == "void":
            instr = RequestInstr(None, method, url_val, hdr_val, body_val)
            self.emit(instr)
            return None
        else:
            dest = self.create_temp(ret_ty)
            instr = RequestInstr(dest, method, url_val, hdr_val, body_val)
            self.emit(instr)
            return dest

    def ensure_value(self, val, ty: IRType):
        if isinstance(val, (IRTemp, IRConst, IRGlobalRef)):
            return val
        # if it's a Python int/float/string => wrap as IRConst
        return IRConst(val, ty)

    # ---------- concurrency-building methods ----------
    def spawn(self, spawnVal, ret_ty=IRType("thread")):
        """
        spawnVal could be a function reference or something
        If ret_ty is "void", no handle returned
        """
        if ret_ty.name == "void":
            instr = SpawnInstr(None, spawnVal)
            self.emit(instr)
            return None
        else:
            dest = self.create_temp(ret_ty)
            instr = SpawnInstr(dest, spawnVal)
            self.emit(instr)
            return dest

    def channel_send(self, channel, val):
        instr = ChannelSendInstr(channel, val)
        self.emit(instr)

    def channel_recv(self, channel, val_ty=IRType("any")):
        dest = self.create_temp(val_ty)
        instr = ChannelRecvInstr(dest, channel)
        self.emit(instr)
        return dest

    def wait_all(self, tasks):
        instr = WaitAllInstr(tasks)
        self.emit(instr)

    def request_async(self, method: str, url, headers=None, body=None, ret_ty=IRType("any")):
        # Example "async" transformation not strictly required, just a placeholder
        url_val = self.ensure_value(url, IRType("string"))
        hdr_val = self.ensure_value(headers, IRType("any")) if headers else None
        body_val = self.ensure_value(body, IRType("any")) if body else None

        handle_temp = self.create_temp(ret_ty)
        instr = RequestInstr(handle_temp, method, url_val, hdr_val, body_val)
        self.emit(instr)
        return handle_temp

# -----------------------------------------------------------------------
# CFG & DOMINATOR LOGIC
# -----------------------------------------------------------------------

def build_cfg(fn: IRFunction):
    preds = {}
    succs = {}
    for b in fn.blocks:
        preds[b] = []
        succs[b] = []

    for i, b in enumerate(fn.blocks):
        if not b.instructions:
            # fallthrough to next block
            if i+1 < len(fn.blocks):
                nxt = fn.blocks[i+1]
                succs[b].append(nxt)
                preds[nxt].append(b)
            continue

        last = b.instructions[-1]
        if isinstance(last, JumpInstr):
            tgt = find_block_by_label(fn, last.label)
            if tgt:
                succs[b].append(tgt)
                preds[tgt].append(b)
        elif isinstance(last, CJumpInstr):
            tb = find_block_by_label(fn, last.true_label)
            fb = find_block_by_label(fn, last.false_label)
            if tb:
                succs[b].append(tb)
                preds[tb].append(b)
            if fb:
                succs[b].append(fb)
                preds[fb].append(b)
        elif isinstance(last, ReturnInstr):
            pass
        else:
            # fallthrough
            if i+1 < len(fn.blocks):
                nxt = fn.blocks[i+1]
                succs[b].append(nxt)
                preds[nxt].append(b)

    return preds, succs

def find_block_by_label(fn: IRFunction, label: str) -> Optional[IRBlock]:
    for blk in fn.blocks:
        if blk.label == label:
            return blk
    return None

def build_dominators(fn: IRFunction):
    if not fn.blocks:
        return {}
    entry = fn.blocks[0]
    preds, succs = build_cfg(fn)

    dom = {b: set(fn.blocks) for b in fn.blocks}
    dom[entry] = {entry}

    changed = True
    while changed:
        changed = False
        for b in fn.blocks:
            if b == entry:
                continue
            new_dom = set(fn.blocks)
            for p in preds[b]:
                new_dom = new_dom.intersection(dom[p])
            new_dom.add(b)
            if new_dom != dom[b]:
                dom[b] = new_dom
                changed = True
    return dom

# -----------------------------------------------------------------------
# PASS MANAGER
# -----------------------------------------------------------------------

class PassManager:
    def __init__(self):
        self.passes = []

    def add_pass(self, p):
        self.passes.append(p)

    def run(self, module: IRModule):
        for p in self.passes:
            p.run_on_module(module)

# -----------------------------------------------------------------------
# EXAMPLE PASSES
# -----------------------------------------------------------------------

class ConstFoldingPass:
    """
    Folds constant arithmetic (like 2+3 => 5) ONLY if both operands are numeric constants.
    Otherwise, we do nothing.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_instructions = []
            for instr in block.instructions:
                if isinstance(instr, BinOpInstr):
                    folded = self.try_fold_binop(instr)
                    if folded is not None:
                        # Replace the BinOpInstr with a MoveInstr of the folded constant
                        new_instructions.append(MoveInstr(instr.dest, folded))
                    else:
                        new_instructions.append(instr)
                else:
                    new_instructions.append(instr)
            block.instructions = new_instructions

    def try_fold_binop(self, instr: BinOpInstr) -> Optional[IRConst]:
        left = instr.left
        right = instr.right

        # Only fold if both sides are IRConst and numeric
        if not (isinstance(left, IRConst) and isinstance(right, IRConst)):
            return None

        # Check if both are Python numeric (int or float) and IR type is "int"/"float"
        if (left.ty.name in ("int","float") and right.ty.name in ("int","float")
            and isinstance(left.value, (int,float)) and isinstance(right.value, (int,float))):
            # We can attempt a fold for +, -, *, etc.
            if instr.op == '+':
                return IRConst(left.value + right.value, left.ty)
            elif instr.op == '-':
                return IRConst(left.value - right.value, left.ty)
            elif instr.op == '*':
                return IRConst(left.value * right.value, left.ty)
            elif instr.op == '/':
                # check for divide by zero, etc. or do integer vs float logic
                if right.value == 0:
                    return None  # skip folding if dividing by zero
                return IRConst(left.value / right.value, left.ty)
            # you can expand for more operators (%, etc.)
        return None


class DeadCodeEliminationPass:
    """
    If an instruction writes to a temp that is never used later, remove it.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        used = self.compute_used_temps(fn)
        for block in fn.blocks:
            new_insts = []
            for i in block.instructions:
                dest = self.get_dest_temp(i)
                # If there's a dest but that dest is never used => skip
                if dest is not None and dest.name not in used:
                    continue
                new_insts.append(i)
            block.instructions = new_insts

    def compute_used_temps(self, fn: IRFunction) -> Set[str]:
        used = set()
        for b in fn.blocks:
            for i in b.instructions:
                for op in self.get_operands(i):
                    if isinstance(op, IRTemp):
                        used.add(op.name)
        return used

    def get_dest_temp(self, i: IRInstr) -> Optional[IRTemp]:
        if hasattr(i, 'dest') and i.dest is not None:
            return i.dest
        return None

    def get_operands(self, i: IRInstr) -> List[IRTemp]:
        ops = []
        if isinstance(i, BinOpInstr):
            if isinstance(i.left, IRTemp): ops.append(i.left)
            if isinstance(i.right, IRTemp): ops.append(i.right)
        elif isinstance(i, UnOpInstr):
            if isinstance(i.src, IRTemp): ops.append(i.src)
        elif isinstance(i, MoveInstr):
            if isinstance(i.src, IRTemp): ops.append(i.src)
        elif isinstance(i, CallInstr):
            if isinstance(i.func, IRTemp):
                ops.append(i.func)
            for a in i.args:
                if isinstance(a, IRTemp):
                    ops.append(a)
        elif isinstance(i, RequestInstr):
            if isinstance(i.url, IRTemp): ops.append(i.url)
            if i.headers and isinstance(i.headers, IRTemp): ops.append(i.headers)
            if i.body and isinstance(i.body, IRTemp): ops.append(i.body)
        elif isinstance(i, CJumpInstr):
            if isinstance(i.cond, IRTemp):
                ops.append(i.cond)
        elif isinstance(i, PrintInstr):
            if isinstance(i.val, IRTemp):
                ops.append(i.val)
        elif isinstance(i, SpawnInstr):
            if isinstance(i.spawnVal, IRTemp):
                ops.append(i.spawnVal)
        # etc. for other concurrency instructions as needed
        return ops


class RequestBatchingPass:
    """
    Example advanced pass: merges consecutive RequestInstr's with the same domain or pattern.
    Naive demonstration only.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_insts = []
            pending_requests = []
            for instr in block.instructions:
                if isinstance(instr, RequestInstr):
                    # naive approach: if same domain as previous, combine
                    if pending_requests and self.same_domain(pending_requests[-1], instr):
                        # skip or unify
                        continue
                    else:
                        pending_requests.append(instr)
                        new_insts.append(instr)
                else:
                    new_insts.append(instr)
            block.instructions = new_insts

    def same_domain(self, r1: RequestInstr, r2: RequestInstr) -> bool:
        if isinstance(r1.url, IRConst) and isinstance(r2.url, IRConst):
            url1 = r1.url.value
            url2 = r2.url.value
            # naive check for "example.com"
            return "example.com" in str(url1) and "example.com" in str(url2)
        return False

class AsyncRequestPass:
    """
    Transforms blocking RequestInstr into a spawn-based approach.
    Purely a naive placeholder.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_insts = []
            for instr in block.instructions:
                if isinstance(instr, RequestInstr):
                    # Replace with spawn
                    if instr.dest:
                        handle = instr.dest
                        spawnVal = IRConst(f"closure_for_{instr.method}", IRType("function"))
                        sp_instr = SpawnInstr(handle, spawnVal)
                        new_insts.append(sp_instr)
                    else:
                        spawnVal = IRConst(f"closure_for_{instr.method}", IRType("function"))
                        sp_instr = SpawnInstr(None, spawnVal)
                        new_insts.append(sp_instr)
                else:
                    new_insts.append(instr)
            block.instructions = new_insts

class SchedulerPass:
    """
    Very naive reordering pass: concurrency instructions first, normal instructions after.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            concurrency_insts = []
            normal_insts = []
            for instr in block.instructions:
                if isinstance(instr, (SpawnInstr, RequestInstr, ThreadForkInstr, ChannelSendInstr, ChannelRecvInstr)):
                    concurrency_insts.append(instr)
                else:
                    normal_insts.append(instr)
            # concurrency instructions first
            block.instructions = concurrency_insts + normal_insts

# -----------------------------------------------------------------------
# A bigger pass pipeline
# -----------------------------------------------------------------------

def create_big_pass_manager() -> PassManager:
    pm = PassManager()
    pm.add_pass(ConstFoldingPass())
    pm.add_pass(RequestBatchingPass())
    pm.add_pass(AsyncRequestPass())
    pm.add_pass(SchedulerPass())
    pm.add_pass(DeadCodeEliminationPass())
    return pm

def print_ir_module(module: IRModule, title: str = ""):
    print("====================================")
    print(f" IR MODULE DUMP: {title}")
    print("====================================")
    print(module)
    print("====================================\n")
