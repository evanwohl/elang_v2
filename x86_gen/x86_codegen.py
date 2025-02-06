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
        self.stack_offset=0
        self.strings: Dict[str,str] = {}
        self.strcount=0
        self.global_vars: Dict[str,str] = {}
        self.int_arg_regs=[X86Register.RCX, X86Register.RDX, X86Register.R8,  X86Register.R9]
        self.flt_arg_regs=[X86Register.XMM0,X86Register.XMM1,X86Register.XMM2,X86Register.XMM3]

    def run_on_module(self,m:IRModule)->str:
        if X86Section.DATA not in self.asm.sections:
            self.asm.sections[X86Section.DATA] = []
        if X86Section.BSS not in self.asm.sections:
            self.asm.sections[X86Section.BSS] = []
        if X86Section.TEXT not in self.asm.sections:
            self.asm.sections[X86Section.TEXT] = []
        self.emit_externs()
        self.emit_runtime()
        for g in m.globals.values():
            self.lower_globalvar(g)
        for fn in m.functions:
            self.lower_function(fn)
        return str(self.asm)

    def emit_externs(self):
        self.asm.sections[X86Section.TEXT].append("extern GetStdHandle")
        self.asm.sections[X86Section.TEXT].append("extern WriteFile")
        self.asm.sections[X86Section.TEXT].append("extern GetProcessHeap")
        self.asm.sections[X86Section.TEXT].append("extern HeapAlloc")
        self.asm.sections[X86Section.TEXT].append("extern HeapReAlloc")
        self.asm.sections[X86Section.TEXT].append("extern HeapFree")
        self.asm.sections[X86Section.TEXT].append("extern CreateThread")
        self.asm.sections[X86Section.TEXT].append("extern WaitForSingleObject")
        self.asm.sections[X86Section.TEXT].append("extern CloseHandle")
        self.asm.sections[X86Section.TEXT].append("extern Sleep")
        self.asm.sections[X86Section.TEXT].append("extern _httpRequest")

    def emit_runtime(self):
        self.asm.sections[X86Section.DATA].append("_stdoutHandle:\n  .quad 0")
        # Minimal SEH approach: define a label to install an SEH handler (not fully workable)
        self.asm.sections[X86Section.TEXT].append("_initRuntime:")
        self.asm.sections[X86Section.TEXT].append("""
  push rbp
  mov rbp, rsp
  ; For demonstration: get handle for stdout
  mov rcx, -11
  call GetStdHandle
  mov [_stdoutHandle], rax
  ; Possibly install SEH with RtlAddFunctionTable, etc. omitted
  leave
  ret
""")
        # We'll define a helper for thread creation, so we handle the StartRoutine
        self.asm.sections[X86Section.TEXT].append("_threadThunk:")
        self.asm.sections[X86Section.TEXT].append("""
; 64-bit Windows start routine => RCX has the pointer to our function
; We do "call rcx" to run user code
global _threadThunk
_threadThunk:
  push rbp
  mov rbp, rsp
  ; windows home space 32 bytes
  sub rsp, 32
  call rcx
  ; once done, do "return"
  add rsp, 32
  leave
  ret
""")
        # Minimal naive dictionary hashing approach
        self.asm.sections[X86Section.TEXT].append("_dictHashPut:")
        self.asm.sections[X86Section.TEXT].append("""
; RCX = dict pointer
; RDX = key pointer
; R8  = val
; We'll do a naive open addressing in 64 slots
push rbp
mov rbp, rsp
; skipping real hashing, just do pointer mod 64
; you'd do a better approach
xor rax, rax
mov rax, rdx
and rax, 63  ; just a toy
mov rbx, rcx
add rbx, 16  ; skip header
.loopD:
cmp qword [rbx+rax*8], 0
je .store
; collision => inc rax
inc rax
and rax, 63
jmp .loopD
.store:
; store pointer to key / store value in next slot?
mov [rbx+rax*8], r8
pop rbp
ret
""")
        # Minimal approach to real HTTP => call _httpRequest
        self.asm.sections[X86Section.TEXT].append("_doHttp:")
        self.asm.sections[X86Section.TEXT].append("""
push rbp
mov rbp,rsp
sub rsp,32
; RCX=method, RDX=url, R8=headers, R9=body
; we just forward to _httpRequest
mov rax, 0
call _httpRequest
add rsp,32
leave
ret
""")

    def lower_globalvar(self, gv:IRGlobalVar):
        self.global_vars[gv.name] = gv.name
        if gv.init_value is not None:
            pass

    def lower_function(self, fn:IRFunction):
        self.asm.add_label(X86Section.TEXT, fn.name)
        # standard prologue
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.PUSH,[reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RBP), reg(X86Register.RSP)]))
        self.temp_locs.clear()
        self.stack_offset=0
        # keep 32 bytes alignment for Windows calls
        # plus we store local temps
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB,[reg(X86Register.RSP), imm(8*32)]))
        self.lower_params(fn)
        for b in fn.blocks:
            bname = f"{fn.name}_{b.label}"
            self.asm.add_label(X86Section.TEXT, bname)
            self.lower_block(fn,b)
        ep = f"{fn.name}_epilogue"
        self.asm.add_label(X86Section.TEXT, ep)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV, [reg(X86Register.RSP), reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.POP, [reg(X86Register.RBP)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.RET))

    def lower_params(self, fn:IRFunction):
        iregs=0
        fregs=0
        for p in fn.param_types:
            t=fn.create_temp(p)
            off=self.reserve_stack(t)
            if p.name in ("float","double"):
                if fregs<4:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOVSD,[
                            mem(X86Register.RBP, disp=off),
                            reg(self.flt_arg_regs[fregs])
                        ]))
                    fregs+=1
            else:
                if iregs<4:
                    self.asm.add_instruction(X86Section.TEXT,
                        X86Instruction(X86Op.MOV,[
                            mem(X86Register.RBP, disp=off),
                            reg(self.int_arg_regs[iregs])
                        ]))
                    iregs+=1

    def lower_block(self, fn:IRFunction, blk:IRBlock):
        for ins in blk.instructions:
            if isinstance(ins, MoveInstr):
                s=self.op_val(ins.src)
                d=self.op_tmp(ins.dest)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[d,s]))
            elif isinstance(ins, BinOpInstr):
                self.lower_binop(ins)
            elif isinstance(ins, UnOpInstr):
                self.lower_unop(ins)
            elif isinstance(ins, CallInstr):
                self.lower_call(ins)
            elif isinstance(ins, PrintInstr):
                self.lower_print(ins)
            elif isinstance(ins, ReturnInstr):
                self.lower_return(ins,fn)
            elif isinstance(ins, JumpInstr):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.JMP,[self.blocklbl(fn,ins.label)]))
            elif isinstance(ins, CJumpInstr):
                co=self.op_val(ins.cond)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CMP,[co,imm(0)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.JNE,[self.blocklbl(fn,ins.true_label)]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.JMP,[self.blocklbl(fn,ins.false_label)]))
            elif isinstance(ins,SpawnInstr):
                self.lower_spawn(ins)
            elif isinstance(ins, SleepInstr):
                dur=self.op_val(ins.durationVal)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[reg(X86Register.RCX), dur]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CALL, ["Sleep"]))
            elif isinstance(ins, RequestInstr):
                self.lower_request(ins)
            elif isinstance(ins, CreateArrayInstr):
                self.lower_create_array(ins)
            elif isinstance(ins, ArrayPushInstr):
                self.lower_array_push(ins)
            elif isinstance(ins, CreateDictInstr):
                self.lower_create_dict(ins)
            elif isinstance(ins, DictSetInstr):
                self.lower_dict_set(ins)
            else:
                pass

    def lower_binop(self, i: BinOpInstr):
        L=self.op_val(i.left)
        R=self.op_val(i.right)
        D=self.op_tmp(i.dest)
        f= i.left.ty.name in ("float","double") or i.right.ty.name in ("float","double")
        if not f:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RAX), L]))
            if i.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADD,[reg(X86Register.RAX), R]))
            elif i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUB,[reg(X86Register.RAX), R]))
            elif i.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IMUL,[reg(X86Register.RAX),R]))
            elif i.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CQO,[]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV,[R]))
            elif i.op=="%":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.CQO,[]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.IDIV,[R]))
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[D, reg(X86Register.RDX)]))
                return
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[D, reg(X86Register.RAX)]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),L]))
            if i.op=="+":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.ADDSD,[reg(X86Register.XMM0),R]))
            elif i.op=="-":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.SUBSD,[reg(X86Register.XMM0),R]))
            elif i.op=="*":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MULSD,[reg(X86Register.XMM0),R]))
            elif i.op=="/":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.DIVSD,[reg(X86Register.XMM0),R]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[D, reg(X86Register.XMM0)]))

    def lower_unop(self, i:UnOpInstr):
        s=self.op_val(i.src)
        d=self.op_tmp(i.dest)
        f=i.src.ty.name in ("float","double")
        if not f:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RAX), s]))
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
                X86Instruction(X86Op.MOV,[d, reg(X86Register.RAX)]))
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
                X86Instruction(X86Op.MOVSD,[d, reg(X86Register.XMM0)]))

    def lower_call(self, i:CallInstr):
        fnlbl=self.func_lbl(i.func)
        argslots=[]
        use_i=0
        use_f=0
        stackNeeded=0
        for a in i.args:
            if a.ty.name in ("float","double"):
                if use_f<4:
                    argslots.append(("flt", use_f))
                    use_f+=1
                else:
                    argslots.append(("stk", None))
                    stackNeeded+=1
            else:
                if use_i<4:
                    argslots.append(("int", use_i))
                    use_i+=1
                else:
                    argslots.append(("stk", None))
                    stackNeeded+=1
        # Win64 requires 32 bytes "shadow space" even if no stack args
        # plus stack args beyond that
        # We'll do naive approach: push stack-based in reverse
        for arg,sl in reversed(list(zip(i.args,argslots))):
            k,ix=sl
            if k=="stk":
                op=self.op_val(arg)
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.PUSH,[op]))
        # now place registers
        for arg,sl in zip(i.args,argslots):
            k,ix=sl
            op=self.op_val(arg)
            if k=="int":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[reg(self.int_arg_regs[ix]),op]))
            elif k=="flt":
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[reg(self.flt_arg_regs[ix]),op]))
        # allocate the 32 byte home space
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB,[reg(X86Register.RSP),imm(32)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,[fnlbl]))
        # restore
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(32)]))
        if stackNeeded>0:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(stackNeeded*8)]))
        if i.dest:
            d=self.op_tmp(i.dest)
            if i.dest.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[d, reg(X86Register.XMM0)]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[d, reg(X86Register.RAX)]))

    def lower_print(self,i:PrintInstr):
        v=self.op_val(i.val)
        t=i.val.ty.name
        if t in ("int","bool"):
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RCX),v]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.CALL,["_printInt"]))
        elif t=="string":
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[reg(X86Register.RDI),v]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.CALL,["_printStr"]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.CALL,["_printNL"]))
        else:
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),v]))
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.CALL,["_printFloat"]))

    def lower_return(self,i:ReturnInstr,fn:IRFunction):
        if i.value:
            val=self.op_val(i.value)
            if i.value.ty.name in ("float","double"):
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOVSD,[reg(X86Register.XMM0),val]))
            else:
                self.asm.add_instruction(X86Section.TEXT,
                    X86Instruction(X86Op.MOV,[reg(X86Register.RAX),val]))
        ep=f"{fn.name}_epilogue"
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP,[ep]))

    def lower_create_array(self, i:CreateArrayInstr):
        d=self.op_tmp(i.dest)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX), imm(16+8*8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["_heapAlloc"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=0),imm(8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),imm(0)]))

    def lower_array_push(self,i:ArrayPushInstr):
        arr=self.op_val(i.arr_temp)
        val=self.op_val(i.val_temp)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RAX),arr]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RBX),mem(X86Register.RAX,disp=8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX),mem(X86Register.RAX,disp=0)]))
        labOk=f".arrOK_{self.strcount}"
        labRs=f".arrRs_{self.strcount}"
        self.strcount+=1
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CMP,[reg(X86Register.RBX), reg(X86Register.RCX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JL,[labOk]))
        self.asm.add_label(X86Section.TEXT, labRs)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RCX),reg(X86Register.RCX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=0),reg(X86Register.RCX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDX),reg(X86Register.RCX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.IMUL,[reg(X86Register.RDX),imm(8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RDX),imm(16)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX),reg(X86Register.RAX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.PUSH,[reg(X86Register.RDX)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["_heapRealloc"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RBX),mem(X86Register.RAX,disp=8)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.JMP,[labOk]))
        self.asm.add_label(X86Section.TEXT, labOk)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[
                mem(X86Register.RAX,index=X86Register.RBX,scale=8,disp=16),
                val
            ]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RBX),imm(1)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=8),reg(X86Register.RBX)]))

    def lower_create_dict(self, i:CreateDictInstr):
        d=self.op_tmp(i.dest)
        # 64 slot dictionary
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX),imm(8+8*64)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["_heapAlloc"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))
        # store capacity at disp=0 => 64
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[mem(X86Register.RAX,disp=0),imm(64)]))
        # init slot region
        # skipping an actual init loop => assume _heapAlloc returns zeroed memory if we used HeapAllocZero?
        pass

    def lower_dict_set(self, i:DictSetInstr):
        # call _dictHashPut
        # RCX=dict, RDX=key, R8=val
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX), self.op_val(i.dict_temp)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDX), self.op_val(i.key_temp)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.R8),  self.op_val(i.val_temp)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL, ["_dictHashPut"]))

    def lower_request(self, i:RequestInstr):
        # call _doHttp => method, url, headers, body => rcx, rdx, r8, r9
        m=self.op_val(IRConst(i.method,"string")) if i.method else imm(0)
        u=i.url if i.url else IRConst(None,"any")
        h=i.headers if i.headers else IRConst(None,"any")
        b=i.body if i.body else IRConst(None,"any")
        rc=self.op_val(u)
        rh=self.op_val(h)
        rb=self.op_val(b)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX), m]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDX), rc]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.R8), rh]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.R9), rb]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB,[reg(X86Register.RSP),imm(32)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL, ["_doHttp"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(32)]))
        if i.dest:
            # pretend _doHttp returns in RAX => store
            d=self.op_tmp(i.dest)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))

    def lower_spawn(self, i:SpawnInstr):
        # Windows style => pass function pointer in RCX to create a thread that calls _threadThunk
        # create param struct if multiple args => skipping
        func=self.op_val(i.spawnVal)
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX),imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDX),imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.R8),imm(0)]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.R9),imm(0)]))
        # push shadow space
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.SUB,[reg(X86Register.RSP),imm(32)]))
        # pass a function pointer => we store in RCX? Actually for CreateThread  is weird
        # We want "start address" => so we put _threadThunk in RCX + param
        # We'll pass _threadThunk, param => skipping for brevity
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RCX), imm("_threadThunk")]))
        # RDX= param => we store user function in param
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.MOV,[reg(X86Register.RDX), func]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.CALL,["CreateThread"]))
        self.asm.add_instruction(X86Section.TEXT,
            X86Instruction(X86Op.ADD,[reg(X86Register.RSP),imm(32)]))
        if i.dest:
            d=self.op_tmp(i.dest)
            self.asm.add_instruction(X86Section.TEXT,
                X86Instruction(X86Op.MOV,[d,reg(X86Register.RAX)]))

    def reserve_stack(self, t:IRTemp)->int:
        self.stack_offset+=8
        off=-self.stack_offset
        self.temp_locs[t]=off
        return off

    def op_tmp(self,t:IRTemp):
        if t not in self.temp_locs:
            self.reserve_stack(t)
        off=self.temp_locs[t]
        return mem(X86Register.RBP, disp=off)

    def op_val(self,v):
        if isinstance(v,IRConst):
            if isinstance(v.value,int):
                return imm(v.value)
            if v.value is None:
                return imm(0)
            if v.ty=="string" or v.ty.name=="string":
                return self.strlit(v.value)
            return imm(0)
        elif isinstance(v,IRTemp):
            return self.op_tmp(v)
        elif isinstance(v,IRGlobalRef):
            nm=v.name
            if nm in self.global_vars: return self.global_vars[nm]
            return nm
        return imm(0)

    def blocklbl(self,fn:IRFunction, l:str)->str:
        if l.startswith(fn.name+"_"): return l
        return fn.name+"_"+l

    def func_lbl(self,fv)->str:
        if isinstance(fv,IRConst) and isinstance(fv.value,str):
            return fv.value
        elif isinstance(fv,IRGlobalRef):
            return fv.name
        return "__unknown"

    def strlit(self, s:str)->str:
        if s not in self.strings:
            lab=f".LC{self.strcount}"
            self.strcount+=1
            e=s.replace('"','\\"').replace("\n","\\n")
            self.strings[s]=lab
            self.asm.sections[X86Section.DATA].append(f'{lab}:\n  .asciz "{e}"')
        return self.strings[s]
