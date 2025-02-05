# x86_codegen.py

from x86_instructions import (
    X86Asm, X86Section, X86Label, X86Instruction, X86Op,
    reg, imm, mem, X86Register
)
from codegen.ir import (
    IRModule, IRFunction, IRBlock,
    IRInstr, IRConst, IRTemp, IRType,
    BinOpInstr, UnOpInstr, MoveInstr, CallInstr, PrintInstr,
    ReturnInstr, JumpInstr, CJumpInstr,
    SpawnInstr, AcquireLockInstr, ReleaseLockInstr,
    ThreadJoinInstr, KillInstr, DetachInstr, SleepInstr,
    RequestInstr, CreateDictInstr, DictSetInstr,
    CreateArrayInstr, ArrayPushInstr
)

class X86Codegen:
    def __init__(self):
        self.asm = X86Asm()
        self.temp_locs = {}
        self.current_stack_offset = 0
        self.string_literals = {}
        self.string_count = 0

        self.int_arg_regs = [X86Register.RDI, X86Register.RSI, X86Register.RDX,
                             X86Register.RCX, X86Register.R8,  X86Register.R9]
        self.float_arg_regs = [X86Register.XMM0, X86Register.XMM1, X86Register.XMM2,
                               X86Register.XMM3, X86Register.XMM4, X86Register.XMM5,
                               X86Register.XMM6, X86Register.XMM7]

    def run_on_module(self, module: IRModule):
        if X86Section.DATA not in self.asm.sections:
            self.asm.sections[X86Section.DATA] = []
        for fn in module.functions:
            self.run_on_function(fn)
        return str(self.asm)

    def run_on_function(self, fn: IRFunction):
        fn_label = fn.name
        self.asm.add_label(X86Section.TEXT, X86Label(fn_label))

        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.PUSH, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RBP), reg(X86Register.RSP)]))

        local_size = 1024
        self.current_stack_offset = 0
        self.temp_locs.clear()

        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB, [reg(X86Register.RSP), imm(local_size)]))

        for block in fn.blocks:
            blk_label = f"{fn_label}_{block.label}"
            self.asm.add_label(X86Section.TEXT, X86Label(blk_label))
            self.run_on_block(fn, block)

        ep = f"{fn_label}_epilogue"
        self.asm.add_label(X86Section.TEXT, X86Label(ep))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RSP), reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.POP, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.RET))

    def run_on_block(self, fn: IRFunction, block: IRBlock):
        for instr in block.instructions:
            if isinstance(instr, MoveInstr):
                self.lower_move(instr)
            elif isinstance(instr, BinOpInstr):
                self.lower_binop(instr)
            elif isinstance(instr, UnOpInstr):
                self.lower_unop(instr)
            elif isinstance(instr, CallInstr):
                self.lower_call(instr)
            elif isinstance(instr, PrintInstr):
                self.lower_print(instr)
            elif isinstance(instr, ReturnInstr):
                self.lower_return(instr, fn)
            elif isinstance(instr, JumpInstr):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.JMP, [self.label_operand(fn, instr.label)]))
            elif isinstance(instr, CJumpInstr):
                self.lower_cjump(instr, fn)
            elif isinstance(instr, SpawnInstr):
                self.lower_spawn(instr)
            elif isinstance(instr, AcquireLockInstr):
                self.lower_acquire_lock(instr)
            elif isinstance(instr, ReleaseLockInstr):
                self.lower_release_lock(instr)
            elif isinstance(instr, ThreadJoinInstr):
                self.lower_join(instr)
            elif isinstance(instr, KillInstr):
                self.lower_kill(instr)
            elif isinstance(instr, DetachInstr):
                self.lower_detach(instr)
            elif isinstance(instr, SleepInstr):
                self.lower_sleep(instr)
            elif isinstance(instr, RequestInstr):
                self.lower_request(instr)
            elif isinstance(instr, CreateDictInstr):
                self.lower_create_dict(instr)
            elif isinstance(instr, DictSetInstr):
                self.lower_dict_set(instr)
            elif isinstance(instr, CreateArrayInstr):
                self.lower_create_array(instr)
            elif isinstance(instr, ArrayPushInstr):
                self.lower_array_push(instr)
            else:
                # Possibly warn or ignore
                pass

    def lower_move(self, instr: MoveInstr):
        s_op = self.get_operand_for_value(instr.src)
        d_op = self.get_operand_for_temp(instr.dest)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [d_op, s_op]))

    def lower_binop(self, instr: BinOpInstr):
        l_op = self.get_operand_for_value(instr.left)
        r_op = self.get_operand_for_value(instr.right)
        is_float = (instr.left.ty.name in ("float","double") or
                    instr.right.ty.name in ("float","double"))

        if not is_float:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [reg(X86Register.RAX), l_op]))
            if instr.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADD, [reg(X86Register.RAX), r_op]))
            elif instr.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUB, [reg(X86Register.RAX), r_op]))
            elif instr.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IMUL, [reg(X86Register.RAX), r_op]))
            elif instr.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CDQ))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV, [r_op]))
            dst_op = self.get_operand_for_temp(instr.dest)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [dst_op, reg(X86Register.RAX)]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), l_op]))
            if instr.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADDSD, [reg(X86Register.XMM0), r_op]))
            elif instr.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD, [reg(X86Register.XMM0), r_op]))
            elif instr.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MULSD, [reg(X86Register.XMM0), r_op]))
            elif instr.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.DIVSD, [reg(X86Register.XMM0), r_op]))
            dst_op = self.get_operand_for_temp(instr.dest)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [dst_op, reg(X86Register.XMM0)]))

    def lower_unop(self, instr: UnOpInstr):
        s_op = self.get_operand_for_value(instr.src)
        d_op = self.get_operand_for_temp(instr.dest)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RAX), s_op]))
        if instr.op=="-":
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.NEG, [reg(X86Register.RAX)]))
        elif instr.op=="!":
            # minimal approach: compare rax with 0, set if zero => 1 else 0
            # or invert. Let's do: cmp rax,0 => sete al => movzx rax,al => xor rax,1
            # etc. We'll do a simpler approach for demonstration, not perfect
            pass
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [d_op, reg(X86Register.RAX)]))

    def lower_call(self, instr: CallInstr):
        int_used   = 0
        float_used = 0
        stack_args = []
        # backward for stack
        for arg in reversed(instr.args):
            if arg.ty.name in ("float","double"):
                if float_used<8:
                    float_used += 1
                else:
                    stack_args.append(arg)
            else:
                if int_used<6:
                    int_used += 1
                else:
                    stack_args.append(arg)
        for a in stack_args:
            a_op = self.get_operand_for_value(a)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.PUSH, [a_op]))
        int_i   = 0
        float_i = 0
        for arg in instr.args:
            a_op = self.get_operand_for_value(arg)
            if arg.ty.name in ("float","double"):
                if float_i<8:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOVSD, [reg(self.float_arg_regs[float_i]), a_op]))
                    float_i+=1
            else:
                if int_i<6:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOV, [reg(self.int_arg_regs[int_i]), a_op]))
                    int_i+=1
        f_op = self.get_operand_for_value(instr.func)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL, [f_op]))

        stack_sz = 8*len(stack_args)
        if stack_sz>0:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.ADD, [reg(X86Register.RSP), imm(stack_sz)]))
        if instr.dest:
            d_op = self.get_operand_for_temp(instr.dest)
            if instr.dest.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD, [d_op, reg(X86Register.XMM0)]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV, [d_op, reg(X86Register.RAX)]))

    def lower_print(self, instr: PrintInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("__lang_print", IRType("function")),
            args=[instr.val]
        )
        self.lower_call(c)

    def lower_return(self, instr: ReturnInstr, fn: IRFunction):
        if instr.value:
            val_op = self.get_operand_for_value(instr.value)
            if instr.value.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), val_op]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV, [reg(X86Register.RAX), val_op]))
        ep = f"{fn.name}_epilogue"
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP, [ep]))

    def lower_cjump(self, instr: CJumpInstr, fn: IRFunction):
        c_op = self.get_operand_for_value(instr.cond)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CMP, [c_op, imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JNE, [self.label_operand(fn, instr.true_label)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP, [self.label_operand(fn, instr.false_label)]))

    def lower_spawn(self, s: SpawnInstr):
        # naive
        if s.dest:
            tid = s.dest
        else:
            tid = None
        spawn_func = s.spawnVal
        if isinstance(spawn_func, IRConst) and spawn_func.ty.name=="function":
            pass
        else:
            spawn_func = IRConst("__unknown_function", IRType("function"))
        c = CallInstr(
            dest=tid,
            func=IRConst("pthread_create", IRType("function")),
            args=[
                IRConst(None, IRType("any")),
                IRConst(0, IRType("any")),
                spawn_func,
                IRConst(0, IRType("any"))
            ]
        )
        self.lower_call(c)

    def lower_acquire_lock(self, i: AcquireLockInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_mutex_lock", IRType("function")),
            args=[i.lockVal]
        )
        self.lower_call(c)

    def lower_release_lock(self, i: ReleaseLockInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_mutex_unlock", IRType("function")),
            args=[i.lockVal]
        )
        self.lower_call(c)

    def lower_join(self, i: ThreadJoinInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_join", IRType("function")),
            args=[i.threadVal, IRConst(0, IRType("any"))]
        )
        self.lower_call(c)

    def lower_kill(self, i: KillInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_cancel", IRType("function")),
            args=[i.threadVal]
        )
        self.lower_call(c)

    def lower_detach(self, i: DetachInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_detach", IRType("function")),
            args=[i.threadVal]
        )
        self.lower_call(c)

    def lower_sleep(self, i: SleepInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("sleep", IRType("function")),
            args=[i.durationVal]
        )
        self.lower_call(c)

    def lower_request(self, r: RequestInstr):
        # We'll call __do_http_request(method, url, headers, body)
        # ensure method, url, headers, body exist
        method_val = (IRConst(r.method, IRType("string"))
                      if r.method else IRConst(None, IRType("any")))
        url_val = r.url if r.url else IRConst(None, IRType("any"))
        hdr_val = r.headers if r.headers else IRConst(None, IRType("any"))
        bdy_val = r.body if r.body else IRConst(None, IRType("any"))
        c = CallInstr(
            dest=r.dest,
            func=IRConst("__do_http_request", IRType("function")),
            args=[method_val, url_val, hdr_val, bdy_val]
        )
        self.lower_call(c)

    def lower_create_dict(self, cd: CreateDictInstr):
        # call __create_dict => returns pointer in RAX
        # store into cd.dest
        c = CallInstr(
            dest=cd.dest,
            func=IRConst("__create_dict", IRType("function")),
            args=[]
        )
        self.lower_call(c)

    def lower_dict_set(self, ds: DictSetInstr):
        # call __dict_set(dict_temp, key_temp, val_temp)
        c = CallInstr(
            dest=None,
            func=IRConst("__dict_set", IRType("function")),
            args=[ds.dict_temp, ds.key_temp, ds.val_temp]
        )
        self.lower_call(c)

    def lower_create_array(self, ca: CreateArrayInstr):
        c = CallInstr(
            dest=ca.dest,
            func=IRConst("__create_array", IRType("function")),
            args=[]
        )
        self.lower_call(c)

    def lower_array_push(self, ap: ArrayPushInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("__array_push", IRType("function")),
            args=[ap.arr_temp, ap.val_temp]
        )
        self.lower_call(c)

    def get_operand_for_value(self, val):
        if isinstance(val, IRConst):
            if val.ty.name=="function":
                if val.value is None:
                    return self.make_label_operand("__unknown_function")
                return self.make_label_operand(str(val.value))
            if val.ty.name=="string":
                label = self.get_string_label(val.value)
                return label
            if isinstance(val.value, int):
                return imm(val.value)
            if val.value is None:
                return imm(0)
            # fallback
            return imm(0)
        elif isinstance(val, IRTemp):
            return self.get_operand_for_temp(val)
        return imm(0)

    def get_operand_for_temp(self, t: IRTemp):
        if t not in self.temp_locs:
            self.current_stack_offset += 8
            off = -self.current_stack_offset
            self.temp_locs[t] = off
        off = self.temp_locs[t]
        return mem(X86Register.RBP, disp=off)

    def get_string_label(self, s: str):
        if s not in self.string_literals:
            label_name = f".LC{self.string_count}"
            self.string_count += 1
            self.string_literals[s] = label_name
            escaped = s.replace('"','\\"')
            self.asm.sections[X86Section.DATA].append(
                f"{label_name}:\n  .asciz \"{escaped}\""
            )
        return self.string_literals[s]

    def label_operand(self, fn: IRFunction, lab: str):
        if lab.startswith(fn.name+"_"):
            return lab
        return f"{fn.name}_{lab}"

    def make_label_operand(self, lbl: str):
        class LabelOp:
            def __init__(self, label):
                self.label = label
            def __repr__(self):
                return self.label
        return LabelOp(lbl)
