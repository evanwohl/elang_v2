from typing import Dict
from x86_instructions import (
    X86Asm, X86Section, X86Label, X86Instruction, X86Op,
    reg, imm, mem, X86Register
)
from codegen.ir import (
    IRModule, IRFunction, IRBlock, IRGlobalVar, IRTemp, IRConst, IRGlobalRef, IRType,
    IRInstr, MoveInstr, BinOpInstr, UnOpInstr, CallInstr, PrintInstr,
    ReturnInstr, JumpInstr, CJumpInstr, SpawnInstr, AcquireLockInstr, ReleaseLockInstr,
    ThreadJoinInstr, KillInstr, DetachInstr, SleepInstr, RequestInstr,
    CreateDictInstr, DictSetInstr, CreateArrayInstr, ArrayPushInstr,
    ThreadForkInstr, ChannelSendInstr, ChannelRecvInstr, WaitAllInstr
)

class X86Codegen:
    def __init__(self):
        self.asm = X86Asm()
        self.temp_locs: Dict[IRTemp,int] = {}
        self.current_stack_offset = 0
        self.string_literals: Dict[str,str] = {}
        self.string_count = 0
        self.global_vars: Dict[str,str] = {}
        self.int_arg_regs = [
            X86Register.RDI, X86Register.RSI, X86Register.RDX,
            X86Register.RCX, X86Register.R8,  X86Register.R9
        ]
        self.float_arg_regs = [
            X86Register.XMM0, X86Register.XMM1, X86Register.XMM2, X86Register.XMM3,
            X86Register.XMM4, X86Register.XMM5, X86Register.XMM6, X86Register.XMM7
        ]

    def run_on_module(self, module: IRModule) -> str:
        if X86Section.DATA not in self.asm.sections:
            self.asm.sections[X86Section.DATA] = []
        if X86Section.BSS not in self.asm.sections:
            self.asm.sections[X86Section.BSS] = []
        if X86Section.TEXT not in self.asm.sections:
            self.asm.sections[X86Section.TEXT] = []
        for g in module.globals.values():
            self.emit_global_var(g)
        for fn in module.functions:
            self.run_on_function(fn)
        return str(self.asm)

    def emit_global_var(self, g: IRGlobalVar):
        lbl = g.name
        self.global_vars[g.name] = lbl
        if g.init_value is not None:
            v = g.init_value
            if isinstance(v,int):
                self.asm.sections[X86Section.DATA].append(f"{lbl}:\n  .quad {v}")
            elif isinstance(v,str):
                e = v.replace('"','\\"').replace("\n","\\n")
                self.asm.sections[X86Section.DATA].append(f'{lbl}:\n  .asciz "{e}"')
            else:
                self.asm.sections[X86Section.DATA].append(f"{lbl}:\n  .quad 0")
        else:
            self.asm.sections[X86Section.BSS].append(f"{lbl}:\n  .zero 8")

    def run_on_function(self, fn: IRFunction):
        fn_lbl = fn.name
        self.asm.add_label(X86Section.TEXT, X86Label(fn_lbl))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.PUSH, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RBP), reg(X86Register.RSP)]))
        self.temp_locs.clear()
        self.current_stack_offset = 0
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB, [reg(X86Register.RSP), imm(8*64)]))
        self.lower_function_params(fn)
        for blk in fn.blocks:
            b_lbl = f"{fn_lbl}_{blk.label}"
            self.asm.add_label(X86Section.TEXT, X86Label(b_lbl))
            self.run_on_block(fn, blk)
        ep = f"{fn_lbl}_epilogue"
        self.asm.add_label(X86Section.TEXT, X86Label(ep))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RSP), reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.POP, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.RET))

    def lower_function_params(self, fn: IRFunction):
        ui = 0
        uf = 0
        for p in fn.param_types:
            tmp = fn.create_temp(p)
            loc = self.reserve_slot(tmp)
            if p.name in ("float","double"):
                if uf<8:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOVSD, [
                            mem(X86Register.RBP,disp=loc), reg(self.float_arg_regs[uf])
                        ]))
                    uf+=1
            else:
                if ui<6:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOV, [
                            mem(X86Register.RBP,disp=loc), reg(self.int_arg_regs[ui])
                        ]))
                    ui+=1

    def run_on_block(self, fn: IRFunction, blk: IRBlock):
        for i in blk.instructions:
            if isinstance(i, MoveInstr):
                self.lower_move(i)
            elif isinstance(i, BinOpInstr):
                self.lower_binop(i)
            elif isinstance(i, UnOpInstr):
                self.lower_unop(i)
            elif isinstance(i, CallInstr):
                self.lower_call(i)
            elif isinstance(i, PrintInstr):
                self.lower_print(i)
            elif isinstance(i, ReturnInstr):
                self.lower_return(i, fn)
            elif isinstance(i, JumpInstr):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.JMP,[self.label_operand(fn,i.label)]))
            elif isinstance(i, CJumpInstr):
                self.lower_cjump(i,fn)
            elif isinstance(i, SpawnInstr):
                self.lower_spawn(i)
            elif isinstance(i, AcquireLockInstr):
                self.lower_acquire_lock(i)
            elif isinstance(i, ReleaseLockInstr):
                self.lower_release_lock(i)
            elif isinstance(i, ThreadJoinInstr):
                self.lower_join(i)
            elif isinstance(i, KillInstr):
                self.lower_kill(i)
            elif isinstance(i, DetachInstr):
                self.lower_detach(i)
            elif isinstance(i, SleepInstr):
                self.lower_sleep(i)
            elif isinstance(i, RequestInstr):
                self.lower_request(i)
            elif isinstance(i, CreateArrayInstr):
                self.lower_create_array_inline(i)
            elif isinstance(i, ArrayPushInstr):
                self.lower_array_push_inline(i)
            elif isinstance(i, CreateDictInstr):
                self.lower_create_dict_inline(i)
            elif isinstance(i, DictSetInstr):
                self.lower_dict_set_inline(i)
            elif isinstance(i, ThreadForkInstr):
                self.lower_thread_fork(i)
            elif isinstance(i, ChannelSendInstr):
                self.lower_channel_send(i)
            elif isinstance(i, ChannelRecvInstr):
                self.lower_channel_recv(i)
            elif isinstance(i, WaitAllInstr):
                self.lower_wait_all(i)

    def lower_move(self, i: MoveInstr):
        s = self.get_value_op(i.src)
        d = self.get_temp_op(i.dest)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[d,s]))

    def lower_binop(self, i: BinOpInstr):
        l = self.get_value_op(i.left)
        r = self.get_value_op(i.right)
        dst = self.get_temp_op(i.dest)
        is_f = i.left.ty.name in ("float","double") or i.right.ty.name in ("float","double")
        if not is_f:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RAX),l]))
            if i.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADD,[reg(X86Register.RAX),r]))
            elif i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUB,[reg(X86Register.RAX),r]))
            elif i.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IMUL,[reg(X86Register.RAX),r]))
            elif i.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CDQ,[]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV,[r]))
            elif i.op=="%":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CDQ,[]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV,[r]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[dst,reg(X86Register.RDX)]))
                return
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[dst,reg(X86Register.RAX)]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),l]))
            if i.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADDSD,[reg(X86Register.XMM0),r]))
            elif i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD,[reg(X86Register.XMM0),r]))
            elif i.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MULSD,[reg(X86Register.XMM0),r]))
            elif i.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.DIVSD,[reg(X86Register.XMM0),r]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[dst,reg(X86Register.XMM0)]))

    def lower_unop(self, i: UnOpInstr):
        s = self.get_value_op(i.src)
        d = self.get_temp_op(i.dest)
        is_f = i.src.ty.name in ("float","double")
        if not is_f:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RAX),s]))
            if i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.NEG,[reg(X86Register.RAX)]))
            elif i.op=="!":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CMP,[reg(X86Register.RAX),imm(0)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SETE,["al"]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVZX,[reg(X86Register.RAX),"al"]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),s]))
            if i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.XORPS,[reg(X86Register.XMM1),reg(X86Register.XMM1)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD,[reg(X86Register.XMM1),reg(X86Register.XMM0)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),reg(X86Register.XMM1)]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[d,reg(X86Register.XMM0)]))

    def lower_call(self, i: CallInstr):
        fn_label = self.get_func_label(i.func)
        int_used=0
        float_used=0
        stack_args=[]
        arg_locs=[]
        for a in i.args:
            if a.ty.name in ("float","double"):
                if float_used<8:
                    arg_locs.append(("floatreg",float_used))
                    float_used+=1
                else:
                    arg_locs.append(("stack",None))
            else:
                if int_used<6:
                    arg_locs.append(("intreg",int_used))
                    int_used+=1
                else:
                    arg_locs.append(("stack",None))
        for (arg,loc) in reversed(list(zip(i.args,arg_locs))):
            k,idx=loc
            if k=="stack":
                aop=self.get_value_op(arg)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.PUSH,[aop]))
        for arg, loc in zip(i.args,arg_locs):
            k,idx=loc
            aop=self.get_value_op(arg)
            if k=="intreg":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[reg(self.int_arg_regs[idx]),aop]))
            elif k=="floatreg":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[reg(self.float_arg_regs[idx]),aop]))
        st_count=sum(1 for x in arg_locs if x[0]=="stack")
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,[fn_label]))
        if st_count>0:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(st_count*8)]))
        if i.dest:
            d = self.get_temp_op(i.dest)
            if i.dest.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[d,reg(X86Register.XMM0)]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))

    def lower_print(self, i: PrintInstr):
        v = self.get_value_op(i.val)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDI),v]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["__lang_print"]))

    def lower_return(self, i: ReturnInstr, fn: IRFunction):
        if i.value:
            op = self.get_value_op(i.value)
            if i.value.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),op]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[reg(X86Register.RAX),op]))
        ep = f"{fn.name}_epilogue"
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP,[ep]))

    def lower_cjump(self, i: CJumpInstr, fn: IRFunction):
        co = self.get_value_op(i.cond)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CMP,[co,imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JNE,[self.label_operand(fn,i.true_label)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP,[self.label_operand(fn,i.false_label)]))

    def lower_spawn(self, s: SpawnInstr):
        c = CallInstr(
            dest=s.dest, func=IRConst("pthread_create", IRType("function")),
            args=[
                IRConst(None,IRType("any")),
                IRConst(0,IRType("any")),
                s.spawnVal,
                IRConst(0,IRType("any"))
            ]
        )
        self.lower_call(c)

    def lower_acquire_lock(self, a: AcquireLockInstr):
        c = CallInstr(None, IRConst("pthread_mutex_lock",IRType("function")), [a.lockVal])
        self.lower_call(c)

    def lower_release_lock(self, r: ReleaseLockInstr):
        c = CallInstr(None, IRConst("pthread_mutex_unlock",IRType("function")), [r.lockVal])
        self.lower_call(c)

    def lower_join(self, j: ThreadJoinInstr):
        c = CallInstr(None, IRConst("pthread_join",IRType("function")),
            [j.threadVal,IRConst(0,IRType("any"))])
        self.lower_call(c)

    def lower_kill(self, k: KillInstr):
        c = CallInstr(None, IRConst("pthread_cancel",IRType("function")),
            [k.threadVal])
        self.lower_call(c)

    def lower_detach(self, d: DetachInstr):
        c=CallInstr(None, IRConst("pthread_detach",IRType("function")),[d.threadVal])
        self.lower_call(c)

    def lower_sleep(self, s: SleepInstr):
        c=CallInstr(None, IRConst("sleep",IRType("function")),[s.durationVal])
        self.lower_call(c)

    def lower_request(self, r: RequestInstr):
        m = IRConst(r.method,IRType("string")) if r.method else IRConst(None,IRType("any"))
        u = r.url if r.url else IRConst(None,IRType("any"))
        h = r.headers if r.headers else IRConst(None,IRType("any"))
        b = r.body if r.body else IRConst(None,IRType("any"))
        c = CallInstr(r.dest,IRConst("__do_http_request",IRType("function")),[m,u,h,b])
        self.lower_call(c)

    def lower_thread_fork(self, tf: ThreadForkInstr):
        c=CallInstr(tf.dest,IRConst("pthread_create",IRType("function")),
            [IRConst(None,IRType("any")),IRConst(0,IRType("any")),tf.func,IRConst(0,IRType("any"))])
        self.lower_call(c)

    def lower_channel_send(self, cs: ChannelSendInstr):
        c=CallInstr(None,IRConst("__channel_send",IRType("function")),
            [cs.channel,cs.val])
        self.lower_call(c)

    def lower_channel_recv(self, cr: ChannelRecvInstr):
        c=CallInstr(cr.dest,IRConst("__channel_recv",IRType("function")),
            [cr.channel])
        self.lower_call(c)

    def lower_wait_all(self, w: WaitAllInstr):
        arr=[]
        for x in w.tasks:
            arr.append(x)
        c=CallInstr(None,IRConst("__wait_all_n",IRType("function")),
            [IRConst(len(arr),IRType("int"))]+arr)
        self.lower_call(c)

    # Inlined array/dict

    def lower_create_array_inline(self, i: CreateArrayInstr):
        d = self.get_temp_op(i.dest)
        total = 2*8 + 16*8
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDI),imm(total)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["malloc"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=0),imm(16)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),imm(0)]))

    def lower_array_push_inline(self, i: ArrayPushInstr):
        arr = self.get_value_op(i.arr_temp)
        val = self.get_value_op(i.val_temp)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RAX),arr]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RBX),mem(X86Register.RAX,disp=8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[
                mem(X86Register.RAX,index=X86Register.RBX,scale=8,disp=16),
                val
            ]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RBX),imm(1)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),reg(X86Register.RBX)]))

    def lower_create_dict_inline(self, i: CreateDictInstr):
        d=self.get_temp_op(i.dest)
        total = 144
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDI),imm(total)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["malloc"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=0),imm(8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),imm(0)]))

    def lower_dict_set_inline(self, i: DictSetInstr):
        d = self.get_value_op(i.dict_temp)
        k = self.get_value_op(i.key_temp)
        v = self.get_value_op(i.val_temp)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RAX),d]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RBX),mem(X86Register.RAX,disp=8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[
                mem(X86Register.RAX,index=X86Register.RBX,scale=16,disp=16),
                k
            ]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[
                mem(X86Register.RAX,index=X86Register.RBX,scale=16,disp=24),
                v
            ]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RBX),imm(1)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),reg(X86Register.RBX)]))

    # Utils

    def reserve_slot(self,t: IRTemp)->int:
        self.current_stack_offset+=8
        off=-self.current_stack_offset
        self.temp_locs[t]=off
        return off

    def get_temp_op(self,t: IRTemp):
        if t not in self.temp_locs:
            self.reserve_slot(t)
        off=self.temp_locs[t]
        return mem(X86Register.RBP,disp=off)

    def get_value_op(self,val):
        if isinstance(val,IRConst):
            if val.ty.name=="function":
                if isinstance(val.value,str):
                    return val.value
                return "__unknown_function"
            if val.ty.name=="string":
                s=val.value
                return self.get_string_label(s)
            if isinstance(val.value,int):
                return imm(val.value)
            if val.value is None:
                return imm(0)
            return imm(0)
        elif isinstance(val,IRTemp):
            return self.get_temp_op(val)
        elif isinstance(val,IRGlobalRef):
            gname=val.name
            if gname in self.global_vars:
                return self.global_vars[gname]
            return gname
        return imm(0)

    def get_string_label(self,s:str)->str:
        if s not in self.string_literals:
            lbl = f".LC{self.string_count}"
            self.string_count+=1
            e = s.replace('"','\\"').replace("\n","\\n")
            self.string_literals[s]=lbl
            self.asm.sections[X86Section.DATA].append(f'{lbl}:\n  .asciz "{e}"')
        return self.string_literals[s]

    def label_operand(self,fn:IRFunction,label:str)->str:
        if label.startswith(fn.name+"_"):
            return label
        return f"{fn.name}_{label}"

    def get_func_label(self,f) -> str:
        if isinstance(f,IRConst) and isinstance(f.value,str):
            return f.value
        elif isinstance(f,IRGlobalRef):
            return f.name
        return "__unknown_function"
