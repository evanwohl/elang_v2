# lower_concurrency.py

from codegen.ir import (
    IRModule, IRFunction, IRBlock, IRTemp, IRType, IRConst,
    SpawnInstr, AcquireLockInstr, ReleaseLockInstr, ThreadJoinInstr,
    KillInstr, DetachInstr, SleepInstr, CallInstr
)

class ConcurrencyLoweringPass:
    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_insts = []
            for instr in block.instructions:
                if isinstance(instr, SpawnInstr):
                    self.lower_spawn(new_insts, instr, fn)
                elif isinstance(instr, AcquireLockInstr):
                    self.lower_acquire_lock(new_insts, instr, fn)
                elif isinstance(instr, ReleaseLockInstr):
                    self.lower_release_lock(new_insts, instr, fn)
                elif isinstance(instr, ThreadJoinInstr):
                    self.lower_join(new_insts, instr, fn)
                elif isinstance(instr, KillInstr):
                    self.lower_kill(new_insts, instr, fn)
                elif isinstance(instr, DetachInstr):
                    self.lower_detach(new_insts, instr, fn)
                elif isinstance(instr, SleepInstr):
                    self.lower_sleep(new_insts, instr, fn)
                else:
                    new_insts.append(instr)
            block.instructions = new_insts

    def lower_spawn(self, new_insts, s, fn: IRFunction):
        if s.dest:
            tid = s.dest
        else:
            tid = fn.create_temp(IRType("thread"))
        spawn_lib = IRConst("pthread_create", IRType("function"))
        ret_temp = fn.create_temp(IRType("int"))
        new_insts.append(
            CallInstr(
                ret_temp,
                spawn_lib,
                [tid, IRConst(0, IRType("any")), s.spawnVal, IRConst(0, IRType("any"))]
            )
        )

    def lower_acquire_lock(self, new_insts, instr: AcquireLockInstr, fn: IRFunction):
        lock_func = IRConst("pthread_mutex_lock", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, lock_func, [instr.lockVar]))

    def lower_release_lock(self, new_insts, instr: ReleaseLockInstr, fn: IRFunction):
        unlock_func = IRConst("pthread_mutex_unlock", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, unlock_func, [instr.lockVar]))

    def lower_join(self, new_insts, instr: ThreadJoinInstr, fn: IRFunction):
        join_func = IRConst("pthread_join", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, join_func, [instr.threadVal, IRConst(0, IRType("any"))]))

    def lower_kill(self, new_insts, instr: KillInstr, fn: IRFunction):
        kill_func = IRConst("pthread_cancel", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, kill_func, [instr.threadVal]))

    def lower_detach(self, new_insts, instr: DetachInstr, fn: IRFunction):
        detach_func = IRConst("pthread_detach", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, detach_func, [instr.threadVal]))

    def lower_sleep(self, new_insts, instr: SleepInstr, fn: IRFunction):
        # e.g. map SleepInstr to nanosleep or a runtime function
        sleep_func = IRConst("sleep", IRType("function"))
        ret = fn.create_temp(IRType("int"))
        new_insts.append(CallInstr(ret, sleep_func, [instr.duration_expr]))
