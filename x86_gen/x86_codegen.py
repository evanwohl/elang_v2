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

        # We'll assume a calling convention with up to 6 integer/pointer regs,
        # and up to 8 float regs. If more, we push them on stack in reverse order.
        self.int_arg_regs = [
            X86Register.RDI, X86Register.RSI, X86Register.RDX,
            X86Register.RCX, X86Register.R8, X86Register.R9
        ]
        self.float_arg_regs = [
            X86Register.XMM0, X86Register.XMM1, X86Register.XMM2, X86Register.XMM3,
            X86Register.XMM4, X86Register.XMM5, X86Register.XMM6, X86Register.XMM7
        ]

    def run_on_module(self, module: IRModule):
        if X86Section.DATA not in self.asm.sections:
            self.asm.sections[X86Section.DATA] = []
        # Codegen each function
        for fn in module.functions:
            self.run_on_function(fn)
        return str(self.asm)

    def run_on_function(self, fn: IRFunction):
        fn_label = fn.name
        self.asm.add_label(X86Section.TEXT, X86Label(fn_label))

        # prologue
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.PUSH, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RBP), reg(X86Register.RSP)]))

        local_size = 1024
        self.current_stack_offset = 0
        self.temp_locs.clear()

        # reserve local space
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB, [reg(X86Register.RSP), imm(local_size)]))

        # output basic blocks
        for block in fn.blocks:
            block_lbl = f"{fn_label}_{block.label}"
            self.asm.add_label(X86Section.TEXT, X86Label(block_lbl))
            self.run_on_block(fn, block)

        # function epilogue
        ep_label = f"{fn_label}_epilogue"
        self.asm.add_label(X86Section.TEXT, X86Label(ep_label))

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
                # Possibly log an unhandled instruction
                pass

    def lower_move(self, instr: MoveInstr):
        src_op = self.get_operand_for_value(instr.src)
        dst_op = self.get_operand_for_temp(instr.dest)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [dst_op, src_op]))

    def lower_binop(self, instr: BinOpInstr):
        left_op = self.get_operand_for_value(instr.left)
        right_op = self.get_operand_for_value(instr.right)
        is_float = (instr.left.ty.name in ("float","double") or
                    instr.right.ty.name in ("float","double"))
        dst_op = self.get_operand_for_temp(instr.dest)

        if not is_float:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [reg(X86Register.RAX), left_op]))
            if instr.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADD, [reg(X86Register.RAX), right_op]))
            elif instr.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUB, [reg(X86Register.RAX), right_op]))
            elif instr.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IMUL, [reg(X86Register.RAX), right_op]))
            elif instr.op=="/":
                # sign extend rax into rdx
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CDQ, []))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV, [right_op]))
            elif instr.op=="%":
                # example: do the IDIV => remainder is in rdx
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CDQ, []))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV, [right_op]))
                # store remainder from RDX
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV, [dst_op, reg(X86Register.RDX)]))
                return
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [dst_op, reg(X86Register.RAX)]))
        else:
            # float path
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), left_op]))
            if instr.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADDSD, [reg(X86Register.XMM0), right_op]))
            elif instr.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD, [reg(X86Register.XMM0), right_op]))
            elif instr.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MULSD, [reg(X86Register.XMM0), right_op]))
            elif instr.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.DIVSD, [reg(X86Register.XMM0), right_op]))
            # store to dest
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [dst_op, reg(X86Register.XMM0)]))

    def lower_unop(self, instr: UnOpInstr):
        s_op = self.get_operand_for_value(instr.src)
        d_op = self.get_operand_for_temp(instr.dest)
        # if float => use XMM. if int => RAX.
        is_float = (instr.src.ty.name in ("float","double"))
        if not is_float:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [reg(X86Register.RAX), s_op]))
            if instr.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.NEG, [reg(X86Register.RAX)]))
            elif instr.op=="+":
                # do nothing
                pass
            elif instr.op=="!":
                # naive approach => compare with 0 => set==0 => ...
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CMP, [reg(X86Register.RAX), imm(0)]))
                # set E = (rax==0)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SETE, ["al"]))
                # zero-extend al => rax
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVZX, [reg(X86Register.RAX), "al"]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV, [d_op, reg(X86Register.RAX)]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), s_op]))
            if instr.op=="-":
                # flip sign => XORPS with sign mask, or simpler: movsd XMM1, -0.0 then SUB
                # we'll do an easier approach: sub from zero
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.XORPS, [reg(X86Register.XMM1), reg(X86Register.XMM1)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD, [reg(X86Register.XMM1), reg(X86Register.XMM0)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), reg(X86Register.XMM1)]))
            elif instr.op=="+":
                pass
            elif instr.op=="!":
                # not well-defined for float => we won't handle
                pass
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD, [d_op, reg(X86Register.XMM0)]))

    def lower_call(self, instr: CallInstr):
        # 1) gather args into stack or registers
        int_used=0
        float_used=0
        stack_args=[]
        # we will store them in reverse order for stack
        for arg in reversed(instr.args):
            if arg.ty.name in ("float","double"):
                if float_used<8:
                    float_used+=1
                else:
                    stack_args.append(arg)
            else:
                if int_used<6:
                    int_used+=1
                else:
                    stack_args.append(arg)
        for a in stack_args:
            a_op = self.get_operand_for_value(a)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.PUSH, [a_op]))

        # now pass regs
        used_i = 0
        used_f = 0
        for arg in instr.args:
            a_op = self.get_operand_for_value(arg)
            if arg.ty.name in ("float","double"):
                if used_f<8:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOVSD, [reg(self.float_arg_regs[used_f]), a_op]))
                    used_f+=1
            else:
                if used_i<6:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOV, [reg(self.int_arg_regs[used_i]), a_op]))
                    used_i+=1

        # 2) call
        f_op = self.get_operand_for_value(instr.func)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL, [f_op]))

        # 3) cleanup stack
        if stack_args:
            cleanup_size= 8*len(stack_args)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.ADD, [reg(X86Register.RSP), imm(cleanup_size)]))

        # 4) store ret in dest if any
        if instr.dest:
            d_op = self.get_operand_for_temp(instr.dest)
            if instr.dest.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD, [d_op, reg(X86Register.XMM0)]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV, [d_op, reg(X86Register.RAX)]))

    def lower_print(self, instr: PrintInstr):
        # pass the value in %rdi
        # call __lang_print
        # The value to print is presumably a pointer or an int
        val_op = self.get_operand_for_value(instr.val)
        # move into rdi
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RDI), val_op]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL, ["__lang_print"]))

    def lower_return(self, instr: ReturnInstr, fn: IRFunction):
        if instr.value:
            v_op = self.get_operand_for_value(instr.value)
            if instr.value.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD, [reg(X86Register.XMM0), v_op]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV, [reg(X86Register.RAX), v_op]))
        ep = f"{fn.name}_epilogue"
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP, [ep]))

    def lower_cjump(self, instr: CJumpInstr, fn: IRFunction):
        cond_op = self.get_operand_for_value(instr.cond)
        # compare cond_op,0 => jne => true => jmp => false
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CMP, [cond_op, imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JNE, [self.label_operand(fn, instr.true_label)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP, [self.label_operand(fn, instr.false_label)]))

    def lower_spawn(self, s: SpawnInstr):
        # naive: call pthread_create(..., s.spawnVal, ...)
        # ignoring the real details
        # if s.dest => store thread handle
        c = CallInstr(
            dest=s.dest,
            func=IRConst("pthread_create", IRType("function")),
            args=[IRConst(None, IRType("any")),
                  IRConst(0, IRType("any")),
                  s.spawnVal,
                  IRConst(0, IRType("any"))]
        )
        self.lower_call(c)

    def lower_acquire_lock(self, a: AcquireLockInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_mutex_lock", IRType("function")),
            args=[a.lockVal]
        )
        self.lower_call(c)

    def lower_release_lock(self, r: ReleaseLockInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_mutex_unlock", IRType("function")),
            args=[r.lockVal]
        )
        self.lower_call(c)

    def lower_join(self, j: ThreadJoinInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_join", IRType("function")),
            args=[j.threadVal, IRConst(0, IRType("any"))]
        )
        self.lower_call(c)

    def lower_kill(self, k: KillInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_cancel", IRType("function")),
            args=[k.threadVal]
        )
        self.lower_call(c)

    def lower_detach(self, d: DetachInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("pthread_detach", IRType("function")),
            args=[d.threadVal]
        )
        self.lower_call(c)

    def lower_sleep(self, s: SleepInstr):
        c = CallInstr(
            dest=None,
            func=IRConst("sleep", IRType("function")),
            args=[s.durationVal]
        )
        self.lower_call(c)

    def lower_request(self, r: RequestInstr):
        # We do call __do_http_request(method, url, headers, body)
        method_val = IRConst(r.method, IRType("string")) if r.method else IRConst(None, IRType("any"))
        url_val = r.url if r.url else IRConst(None, IRType("any"))
        hdr_val = r.headers if r.headers else IRConst(None, IRType("any"))
        body_val= r.body if r.body else IRConst(None, IRType("any"))
        c = CallInstr(
            dest=r.dest,
            func=IRConst("__do_http_request", IRType("function")),
            args=[method_val, url_val, hdr_val, body_val]
        )
        self.lower_call(c)

    def lower_create_dict(self, cd: CreateDictInstr):
        c = CallInstr(
            dest=cd.dest,
            func=IRConst("__create_dict", IRType("function")),
            args=[]
        )
        self.lower_call(c)

    def lower_dict_set(self, ds: DictSetInstr):
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

    # ---------- utility
    def get_operand_for_value(self, val):
        if isinstance(val, IRConst):
            if val.ty.name=="function":
                if val.value is None:
                    return self.make_label_operand("__unknown_function")
                return self.make_label_operand(str(val.value))
            if val.ty.name=="string":
                lbl = self.get_string_label(val.value)
                return self.make_label_operand(lbl)
            if isinstance(val.value, int):
                return imm(val.value)
            if val.value is None:
                return imm(0)
            # fallback => imm(0)
            return imm(0)
        elif isinstance(val, IRTemp):
            return self.get_operand_for_temp(val)
        # fallback
        return imm(0)

    def get_operand_for_temp(self, t: IRTemp):
        if t not in self.temp_locs:
            self.current_stack_offset += 8
            off = -self.current_stack_offset
            self.temp_locs[t] = off
        off = self.temp_locs[t]
        return mem(X86Register.RBP, disp=off)

    def label_operand(self, fn: IRFunction, label: str):
        # if label is already prefixed => return
        if label.startswith(fn.name+"_"):
            return label
        return f"{fn.name}_{label}"

    def make_label_operand(self, lbl: str):
        class LabelOp:
            def __init__(self, label):
                self.label = label
            def __repr__(self):
                return self.label
        return LabelOp(lbl)

    def get_string_label(self, s: str):
        if s not in self.string_literals:
            lbl = f".LC{self.string_count}"
            self.string_count += 1
            # naive escaping
            escaped = s.replace('"','\\"').replace("\n","\\n")
            self.string_literals[s] = lbl
            self.asm.sections[X86Section.DATA].append(
                f"{lbl}:\n  .asciz \"{escaped}\""
            )
        return self.string_literals[s]
