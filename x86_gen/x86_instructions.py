# x86_instructions.py

class X86OpObj:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name

class X86Op:
    MOV   = X86OpObj("MOV")
    MOVSD = X86OpObj("MOVSD")
    ADD   = X86OpObj("ADD")
    SUB   = X86OpObj("SUB")
    IMUL  = X86OpObj("IMUL")
    IDIV  = X86OpObj("IDIV")
    CQO   = X86OpObj("CQO")
    AND   = X86OpObj("AND")
    OR    = X86OpObj("OR")
    XOR   = X86OpObj("XOR")
    XORPS = X86OpObj("XORPS")
    MULSD = X86OpObj("MULSD")
    DIVSD = X86OpObj("DIVSD")
    ADDSD = X86OpObj("ADDSD")
    SUBSD = X86OpObj("SUBSD")
    PUSH  = X86OpObj("PUSH")
    POP   = X86OpObj("POP")
    RET   = X86OpObj("RET")
    CALL  = X86OpObj("CALL")
    JMP   = X86OpObj("JMP")
    JE    = X86OpObj("JE")
    JNE   = X86OpObj("JNE")
    JL    = X86OpObj("JL")
    JLE   = X86OpObj("JLE")
    JG    = X86OpObj("JG")
    JGE   = X86OpObj("JGE")
    SETE  = X86OpObj("SETE")
    MOVZX = X86OpObj("MOVZX")
    CMP   = X86OpObj("CMP")
    TEST  = X86OpObj("TEST")
    CDQ   = X86OpObj("CDQ")  # if needed for 32-bit
    NEG   = X86OpObj("NEG")

class X86Register:
    RAX = "rax"
    RBX = "rbx"
    RCX = "rcx"
    RDX = "rdx"
    RBP = "rbp"
    RSP = "rsp"
    RSI = "rsi"
    RDI = "rdi"
    R8  = "r8"
    R9  = "r9"
    XMM0= "xmm0"
    XMM1= "xmm1"
    XMM2= "xmm2"
    XMM3= "xmm3"
    XMM4= "xmm4"
    XMM5= "xmm5"
    XMM6= "xmm6"
    XMM7= "xmm7"

def reg(rname: str):
    return X86Operand(kind="reg", value=rname)

def imm(n: int):
    return X86Operand(kind="imm", value=n)

def mem(base: str, disp: int=0, index=None, scale=1):
    return X86Operand(kind="mem", value=(base, disp, index, scale))

class X86Operand:
    def __init__(self, kind: str, value):
        self.kind=kind
        self.value=value

    def __repr__(self):
        if self.kind=="reg":
            return self.value
        if self.kind=="imm":
            return f"${self.value}"
        if self.kind=="mem":
            base,disp,idx,scl=self.value
            out=""
            out+="["
            out+=base
            if idx:
                out+=f"+{idx}*{scl}"
            if disp!=0:
                if disp>0: out+=f"+{disp}"
                else: out+=f"{disp}"
            out+="]"
            return out
        return f"<?{self.kind}:{self.value}?>"

class X86Label:
    def __init__(self, name:str):
        self.name=name
    def __repr__(self):
        return f"{self.name}:"

class X86Section:
    TEXT="text"
    DATA="data"
    BSS="bss"

class X86Instruction:
    def __init__(self,op,operands=None):
        self.op=op
        self.operands=operands or []

    def __repr__(self):
        op_str = self.op.name.lower()  # safe: self.op is X86OpObj
        ops_strs=", ".join(repr(o) for o in self.operands)
        return f"{op_str} {ops_strs}".strip()

class X86Asm:
    def __init__(self):
        self.sections={
            X86Section.TEXT:[],
            X86Section.DATA:[],
            X86Section.BSS:[]
        }

    def add_label(self, section, lbl: X86Label):
        self.sections[section].append(repr(lbl))

    def add_instruction(self, section, instr: X86Instruction):
        self.sections[section].append(instr)

    def __str__(self):
        out=[]
        if self.sections[X86Section.DATA]:
            out.append(".data")
            for x in self.sections[X86Section.DATA]:
                out.append(f"  {x}")
        if self.sections[X86Section.BSS]:
            out.append(".bss")
            for x in self.sections[X86Section.BSS]:
                out.append(f"  {x}")
        if self.sections[X86Section.TEXT]:
            out.append(".text")
            for x in self.sections[X86Section.TEXT]:
                if isinstance(x,X86Instruction):
                    out.append(f"  {x}")
                else:
                    out.append(f"{x}")
        return "\n".join(out)
