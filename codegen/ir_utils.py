"""
ir_utils.py

An expanded suite of utilities for eLang's advanced IR, focusing on concurrency
and request transformations. Adds advanced IRBuilder methods, concurrency passes,
and request-optimizing passes for a "requests-focused, hyper-fast" language.
"""

from typing import List, Dict, Set, Optional
# We'll assume you have all these from your advanced ir.py
from ir import (
    IRModule, IRFunction, IRBlock, IRType, IRTemp, IRConst, IRGlobalRef,
    IRInstr, PhiInstr, MoveInstr, BinOpInstr, UnOpInstr, LoadInstr,
    StoreInstr, AtomicLoadInstr, AtomicStoreInstr, AcquireLockInstr,
    ReleaseLockInstr, JumpInstr, CJumpInstr, ReturnInstr, CallInstr,
    RequestInstr, SpawnInstr, ThreadForkInstr, ThreadJoinInstr, KillInstr,
    DetachInstr, SleepInstr, PrintInstr, ChannelSendInstr,  # existing concurrency
    # The new concurrency instructions we just added
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
            args.append(self.ensure_value(a, IRType("any")))  # or typed
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
        return IRConst(val, ty)

    # ---------- New concurrency-building methods ----------
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
        """
        channel => IRValue of type channel
        val => IRValue
        """
        instr = ChannelSendInstr(channel, val)
        self.emit(instr)

    def channel_recv(self, channel, val_ty=IRType("any")):
        """
        Receives a value from 'channel'. Returns an IRTemp with that type
        """
        dest = self.create_temp(val_ty)
        instr = ChannelRecvInstr(dest, channel)
        self.emit(instr)
        return dest

    def wait_all(self, tasks: List[IRTemp]):
        """
        WaitAll for multiple concurrency tasks
        """
        instr = WaitAllInstr(tasks)
        self.emit(instr)

    def request_async(self, method: str, url, headers=None, body=None, ret_ty=IRType("any")):
        """
        Build a RequestInstr in a separate concurrency way, e.g. spawn a thread or something.
        Just an example of how to unify concurrency+requests.
        """
        # e.g. might do spawn => in that spawn, do a request
        # Or build direct concurrency. We'll do a naive approach:
        # 1) create a function reference for the request
        # 2) spawn it
        # For demonstration, let's just directly return a "spawned request handle"
        # Real logic would be more sophisticated
        # We'll produce a "spawn" that calls request.
        # This is purely a demonstration, may not align with your actual concurrency model
        url_val = self.ensure_value(url, IRType("string"))
        hdr_val = self.ensure_value(headers, IRType("any")) if headers else None
        body_val = self.ensure_value(body, IRType("any")) if body else None

        # Could store the request info in IRConst or IRGlobalRef and spawn
        # Then wait on it later
        # We'll just pretend we create a "request handle"
        handle_temp = self.create_temp(ret_ty)
        # This is contrived. A real approach might create a closure function.
        instr = RequestInstr(handle_temp, method, url_val, hdr_val, body_val)
        self.emit(instr)
        # you'd then spawn it or store it for concurrency
        return handle_temp


# -----------------------------------------------------------------------
# CFG, Dominators, etc. (same as before, but you can expand concurrency)
# -----------------------------------------------------------------------

def build_cfg(fn: IRFunction):
    preds = {}
    succs = {}
    for b in fn.blocks:
        preds[b] = []
        succs[b] = []

    for i, b in enumerate(fn.blocks):
        if not b.instructions:
            # fallthrough to next?
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
# PASS MANAGER (same as before)
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
# EXAMPLE PASSES: Add More Request/Concurrency Focus
# -----------------------------------------------------------------------

class ConstFoldingPass:
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
                        new_instructions.append(MoveInstr(instr.dest, folded))
                    else:
                        new_instructions.append(instr)
                else:
                    new_instructions.append(instr)
            block.instructions = new_instructions

    def try_fold_binop(self, instr: BinOpInstr) -> Optional[IRConst]:
        left = instr.left
        right = instr.right
        if isinstance(left, IRConst) and isinstance(right, IRConst):
            if instr.op == '+':
                return IRConst(left.value + right.value, left.ty)
            # etc. expand as needed
        return None

class DeadCodeEliminationPass:
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        used = self.compute_used_temps(fn)
        for block in fn.blocks:
            new_insts = []
            for i in block.instructions:
                dest = self.get_dest_temp(i)
                if dest is not None and dest.name not in used:
                    continue  # remove
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

    def get_dest_temp(self, i: IRInstr):
        # same logic as before
        if hasattr(i, 'dest') and i.dest is not None:
            return i.dest
        return None

    def get_operands(self, i: IRInstr) -> List[IRTemp]:
        # gather IRTemp usage from i
        # expand concurrency instructions as well
        ops = []
        if isinstance(i, BinOpInstr):
            if isinstance(i.left, IRTemp): ops.append(i.left)
            if isinstance(i.right, IRTemp): ops.append(i.right)
        elif isinstance(i, MoveInstr):
            if isinstance(i.src, IRTemp): ops.append(i.src)
        elif isinstance(i, RequestInstr):
            if isinstance(i.url, IRTemp): ops.append(i.url)
            if i.headers and isinstance(i.headers, IRTemp): ops.append(i.headers)
            if i.body and isinstance(i.body, IRTemp): ops.append(i.body)
        # etc. for concurrency instructions
        return ops


class RequestBatchingPass:
    """
    Example advanced pass: merges consecutive RequestInstr's with the same
    domain, or transforms them into a single batched request. The specifics
    are language-dependent. We'll do a trivial demonstration.
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
                        # combine into one, for demonstration we just skip the new request
                        # or we could unify them
                        continue
                    else:
                        pending_requests.append(instr)
                        new_insts.append(instr)
                else:
                    # flush pending? We'll just move on
                    new_insts.append(instr)
            block.instructions = new_insts

    def same_domain(self, r1: RequestInstr, r2: RequestInstr) -> bool:
        # super naive check: if both are IRConst string for the same domain?
        if isinstance(r1.url, IRConst) and isinstance(r2.url, IRConst):
            # parse domain from string? We'll just check substring
            url1 = r1.url.value
            url2 = r2.url.value
            # e.g. if they share "example.com"
            return "example.com" in url1 and "example.com" in url2
        return False


class AsyncRequestPass:
    """
    Transforms blocking RequestInstr into a spawn-based async approach,
    for potential concurrency. This is a naive placeholder. Real logic
    might require closures or advanced concurrency.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_insts = []
            for instr in block.instructions:
                if isinstance(instr, RequestInstr):
                    # Replace with spawn of a function that does the request
                    # We'll pretend we have a global function @do_request
                    # So: handle = THREAD_FORK(@do_request, [method, url, headers, body])
                    # or spawn, etc. We'll do a naive spawn
                    # If there's a dest, we need a concurrency handle. We'll do a naive approach
                    if instr.dest:
                        # we produce a "spawnVal" which is some closure. We'll just do IRConst for example
                        handle = instr.dest
                        spawnVal = IRConst(f"closure_for_{instr.method}", IRType("function"))
                        sp_instr = SpawnInstr(handle, spawnVal)
                        new_insts.append(sp_instr)
                    else:
                        # no handle => just do spawn
                        spawnVal = IRConst(f"closure_for_{instr.method}", IRType("function"))
                        sp_instr = SpawnInstr(None, spawnVal)
                        new_insts.append(sp_instr)
                else:
                    new_insts.append(instr)
            block.instructions = new_insts


class SchedulerPass:
    """
    A naive pass that tries to reorder concurrency instructions for better parallelization:
    e.g., move SpawnInstr or RequestInstr earlier if no dependencies,
    push WaitAllInstr later, etc.
    This is extremely naive demonstration.
    """
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            # We'll just bubble up concurrency instructions if possible
            # Real sched pass would be much more advanced (dependency graph, topological sort, etc.)
            concurrency_insts = []
            normal_insts = []
            for instr in block.instructions:
                if isinstance(instr, (SpawnInstr, RequestInstr, ThreadForkInstr, ChannelSendInstr)):
                    concurrency_insts.append(instr)
                else:
                    normal_insts.append(instr)
            # We'll reorder: concurrency first, then normal, just as a naive approach
            block.instructions = concurrency_insts + normal_insts


# -----------------------------------------------------------------------
# DEMO: A bigger pass pipeline
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
