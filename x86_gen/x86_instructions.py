from enum import Enum, auto

class X86Register(Enum):
    # General-purpose 64-bit integer registers
    RAX = auto()
    RBX = auto()
    RCX = auto()
    RDX = auto()
    RSI = auto()
    RDI = auto()
    RBP = auto()
    RSP = auto()
    R8  = auto()
    R9  = auto()
    R10 = auto()
    R11 = auto()
    R12 = auto()
    R13 = auto()
    R14 = auto()
    R15 = auto()

    # SSE/AVX/AVX-512 registers (for floating-point/vector ops)
    XMM0  = auto()
    XMM1  = auto()
    XMM2  = auto()
    XMM3  = auto()
    XMM4  = auto()
    XMM5  = auto()
    XMM6  = auto()
    XMM7  = auto()
    XMM8  = auto()
    XMM9  = auto()
    XMM10 = auto()
    XMM11 = auto()
    XMM12 = auto()
    XMM13 = auto()
    XMM14 = auto()
    XMM15 = auto()
    # If you want to go beyond XMM15 up to XMM31 for AVX-512, add them here

    YMM0  = auto()
    YMM1  = auto()
    YMM2  = auto()
    YMM3  = auto()
    YMM4  = auto()
    YMM5  = auto()
    YMM6  = auto()
    YMM7  = auto()
    YMM8  = auto()
    YMM9  = auto()
    YMM10 = auto()
    YMM11 = auto()
    YMM12 = auto()
    YMM13 = auto()
    YMM14 = auto()
    YMM15 = auto()
    # Similarly, if needed, add YMM16..YMM31

    ZMM0  = auto()
    ZMM1  = auto()
    ZMM2  = auto()
    ZMM3  = auto()
    ZMM4  = auto()
    ZMM5  = auto()
    ZMM6  = auto()
    ZMM7  = auto()
    ZMM8  = auto()
    ZMM9  = auto()
    ZMM10 = auto()
    ZMM11 = auto()
    ZMM12 = auto()
    ZMM13 = auto()
    ZMM14 = auto()
    ZMM15 = auto()
    # AVX-512 can extend up to ZMM31, add if needed

class X86OperandType(Enum):
    REG  = auto()
    IMM  = auto()
    MEM  = auto()

class X86Operand:
    def __init__(self, op_type: X86OperandType, value, scale=1, index=None, disp=0):
        self.op_type = op_type
        self.value   = value   # could be a register enum or immediate int
        self.scale   = scale   # used for MEM with an index register
        self.index   = index   # another register for MEM addressing
        self.disp    = disp    # displacement

    def __repr__(self):
        if self.op_type == X86OperandType.REG:
            return self._format_register(self.value)
        elif self.op_type == X86OperandType.IMM:
            return f"${self.value}"
        elif self.op_type == X86OperandType.MEM:
            return self._format_mem()
        return "<invalid_operand>"

    def _format_register(self, reg_enum):
        # Just convert register name to lowercase, e.g. "rax", "xmm0"
        return reg_enum.name.lower()

    def _format_mem(self):
        parts = []
        # base register
        base_str = ""
        if self.value:
            base_str = self.value.name.lower()
        # index register
        index_str = ""
        if self.index:
            index_str = f"{self.index.name.lower()}*{self.scale}"
        # displacement
        disp_str = ""
        if self.disp != 0:
            # e.g. +8 or -12
            disp_str = (str(self.disp) if self.disp < 0 else f"+{self.disp}")
        # combine
        # typical AT&T syntax: disp(base, index, scale)
        # if no base or index, might end up with ( +12 ) so handle carefully
        inside = []
        if base_str:
            inside.append(base_str)
        if index_str:
            inside.append(index_str)
        # join them with '+'
        core = "+".join(inside)
        if disp_str:
            # if there's no base/index, we still want disp alone, e.g. 8( )
            if core:
                core += disp_str
            else:
                # we only have displacement
                core = disp_str
        return f"({core})" if core else "(0x0)"

def reg(r: X86Register):
    return X86Operand(X86OperandType.REG, r)

def imm(value: int):
    return X86Operand(X86OperandType.IMM, value)

def mem(base=None, disp=0, index=None, scale=1):
    return X86Operand(X86OperandType.MEM, base, scale, index, disp)

class X86Op(Enum):
    # Basic integer ops
    MOV    = auto()
    ADD    = auto()
    SUB    = auto()
    MUL    = auto()
    IMUL   = auto()
    DIV    = auto()
    IDIV   = auto()
    INC    = auto()
    DEC    = auto()
    XOR    = auto()
    OR     = auto()
    AND    = auto()
    TEST   = auto()
    CMP    = auto()
    LEA    = auto()

    # stack ops
    PUSH   = auto()
    POP    = auto()

    # jumps/calls
    CALL   = auto()
    RET    = auto()
    JMP    = auto()
    JE     = auto()
    JNE    = auto()
    JG     = auto()
    JGE    = auto()
    JL     = auto()
    JLE    = auto()

    # SSE/AVX ops (for i9 or advanced usage)
    MOVSS  = auto()  # move scalar single
    MOVSD  = auto()  # move scalar double
    MOVUPS = auto()  # move unaligned packed single
    MOVUPD = auto()  # move unaligned packed double

    ADDSS  = auto()
    ADDSD  = auto()
    SUBSS  = auto()
    SUBSD  = auto()
    MULSS  = auto()
    MULSD  = auto()
    DIVSS  = auto()
    DIVSD  = auto()

    # AVX versions for packed ops
    VADDPS = auto()
    VSUBPS = auto()
    VMULPS = auto()
    VDIVPS = auto()
    VADDPD = auto()
    VSUBPD = auto()
    VMULPD = auto()
    VDIVPD = auto()

    # no operation
    NOP    = auto()

class X86Instruction:
    def __init__(self, op: X86Op, operands=None):
        self.op = op
        self.operands = operands if operands else []

    def __repr__(self):
        if not self.operands:
            return f"{self.op.name.lower()}"
        op_strs = ", ".join(repr(o) for o in self.operands)
        return f"{self.op.name.lower()} {op_strs}"

class X86Section(Enum):
    TEXT = auto()
    DATA = auto()
    BSS  = auto()

class X86Label:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"{self.name}:"

class X86Asm:
    """
    A container for advanced x86_64 assembly targeting modern Intel i9
    but still valid for standard x86_64. Includes SSE/AVX instructions
    plus general usage. Typically you'd have .text, .data, .bss sections.
    """
    def __init__(self):
        self.sections = {
            X86Section.TEXT: [],
            X86Section.DATA: [],
            X86Section.BSS:  []
        }

    def add_instruction(self, sec: X86Section, instr: X86Instruction):
        self.sections[sec].append(instr)

    def add_label(self, sec: X86Section, label: X86Label):
        self.sections[sec].append(label)

    def __repr__(self):
        lines = []
        # data
        if self.sections[X86Section.DATA]:
            lines.append(".data")
            for item in self.sections[X86Section.DATA]:
                lines.append(f"  {item}")
        # bss
        if self.sections[X86Section.BSS]:
            lines.append(".bss")
            for item in self.sections[X86Section.BSS]:
                lines.append(f"  {item}")
        # text
        if self.sections[X86Section.TEXT]:
            lines.append(".text")
            for item in self.sections[X86Section.TEXT]:
                lines.append(f"  {item}")
        return "\n".join(lines)
