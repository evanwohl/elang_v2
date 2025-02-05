import sys
from typing import Optional, Dict
from ir import (
    IRModule, IRFunction, IRBlock, IRType, IRTemp, IRConst, IRGlobalRef,
    MoveInstr, BinOpInstr, UnOpInstr, LoadInstr, StoreInstr, AtomicLoadInstr,
    AtomicStoreInstr, AcquireLockInstr, ReleaseLockInstr, JumpInstr, CJumpInstr,
    ReturnInstr, CallInstr, RequestInstr, SpawnInstr, ThreadForkInstr,
    ThreadJoinInstr, KillInstr, DetachInstr, SleepInstr, PrintInstr,
    CreateDictInstr, DictSetInstr, CreateArrayInstr, ArrayPushInstr
)
from ir_utils import IRBuilder
from parser.parser import (
    ProgramNode, FunctionNode, ClassNode, VarDeclNode, RequestNode,
    IfNode, WhileNode, ForNode, TryNode, ReturnNode, BreakNode, ContinueNode,
    ThrowNode, SpawnNode, ChannelNode, ThreadNode, LockNode, UnlockNode,
    SleepNode, KillNode, DetachNode, JoinNode, YieldNode, AwaitNode,
    BinaryOpNode, UnaryOpNode, LiteralNode, IdentifierNode, AssignNode,
    BlockNode, ExprStmtNode, DictLiteralNode, ArrayLiteralNode,
    PrintStmtNode, CallNode, MemberAccessNode, IndexAccessNode,
    ArrowFunctionNode
)

class ASTToIR:
    def __init__(self):
        self.module = IRModule()
        self.current_function: Optional[IRFunction] = None
        self.builder: Optional[IRBuilder] = None
        self.function_map: Dict[str, IRTemp] = {}
        self.block_count = 0
        self.main_created = False
        self.main_fn = None
        self.type_map = {
            "int": IRType("int"),
            "float": IRType("float"),
            "bool": IRType("bool"),
            "string": IRType("string"),
            "any": IRType("any"),
            "void": IRType("void"),
            "thread": IRType("thread"),
            "lock": IRType("lock"),
            "channel": IRType("channel")
        }

    def generate_ir(self, ast):
        self.visit_program(ast)
        return self.module

    def new_block(self, label_prefix="block"):
        self.block_count += 1
        blk = IRBlock(f"{label_prefix}{self.block_count}")
        self.current_function.add_block(blk)
        return blk

    def set_builder_block(self, blk):
        self.builder.set_block(blk)

    def visit_program(self, node: ProgramNode):
        for stmt in node.body:
            if isinstance(stmt, FunctionNode):
                fn = self.visit_function_decl(stmt)
                self.module.add_function(fn)
            elif isinstance(stmt, ClassNode):
                self.visit_class_decl(stmt)
            else:
                if not self.main_created:
                    f = IRFunction("main", [], IRType("void"))
                    self.module.add_function(f)
                    self.main_fn = f
                    self.main_created = True
                self.current_function = self.main_fn
                self.builder = IRBuilder(self.current_function)
                if not self.current_function.blocks:
                    blk = self.new_block("entry")
                else:
                    blk = self.current_function.blocks[-1]
                self.set_builder_block(blk)
                self.visit_statement(stmt)

    def visit_function_decl(self, node: FunctionNode):
        param_types = []
        for (pname, ptype) in node.params:
            t = self.type_map.get(ptype, IRType("any")) if ptype else IRType("any")
            param_types.append(t)
        ret_ty = self.type_map.get(node.return_type, IRType("void")) if node.return_type else IRType("void")

        fn = IRFunction(node.name, param_types, ret_ty)
        self.current_function = fn
        self.builder = IRBuilder(fn)
        entry_blk = IRBlock("entry0")
        fn.add_block(entry_blk)
        self.set_builder_block(entry_blk)

        self.function_map.clear()

        for i, (pname, ptype) in enumerate(node.params):
            real_ty = self.type_map.get(ptype, IRType("any")) if ptype else IRType("any")
            ptemp = fn.create_temp(real_ty)
            self.function_map[pname] = ptemp

        self.visit_block(node.body)

        if fn.blocks:
            last_blk = fn.blocks[-1]
            if not last_blk.instructions or not isinstance(last_blk.instructions[-1], ReturnInstr):
                last_blk.add_instr(ReturnInstr(None))

        self.function_map.clear()
        return fn

    def visit_class_decl(self, node: ClassNode):
        pass

    def visit_block(self, node: BlockNode):
        for stmt in node.statements:
            self.visit_statement(stmt)

    def visit_statement(self, node):
        t = node.__class__.__name__
        if t == "VarDeclNode":
            self.visit_var_decl(node)
        elif t == "IfNode":
            self.visit_if(node)
        elif t == "WhileNode":
            self.visit_while(node)
        elif t == "ForNode":
            self.visit_for(node)
        elif t == "TryNode":
            self.visit_try(node)
        elif t == "ReturnNode":
            self.visit_return(node)
        elif t == "BreakNode":
            pass
        elif t == "ContinueNode":
            pass
        elif t == "ThrowNode":
            self.visit_throw(node)
        elif t == "SpawnNode":
            self.visit_spawn(node)
        elif t == "ChannelNode":
            self.visit_channel(node)
        elif t == "ThreadNode":
            self.visit_thread(node)
        elif t == "LockNode":
            self.visit_lock(node)
        elif t == "UnlockNode":
            self.visit_unlock(node)
        elif t == "SleepNode":
            self.visit_sleep(node)
        elif t == "KillNode":
            self.visit_kill(node)
        elif t == "DetachNode":
            self.visit_detach(node)
        elif t == "JoinNode":
            self.visit_join(node)
        elif t == "YieldNode":
            pass
        elif t == "AwaitNode":
            self.visit_await(node)
        elif t == "RequestNode":
            self.visit_request_stmt(node)
        elif t == "ExprStmtNode":
            self.visit_expression(node.expr)
        elif t == "BlockNode":
            self.visit_block(node)
        elif t == "PrintStmtNode":
            self.visit_print(node)
        else:
            pass

    def create_var_temp(self, name: str, t: str):
        real_ty = self.type_map.get(t, IRType("any"))
        tmp = self.current_function.create_temp(real_ty)
        self.function_map[name] = tmp
        return tmp

    def get_var_temp(self, name: str) -> Optional[IRTemp]:
        return self.function_map.get(name, None)

    def visit_var_decl(self, node: VarDeclNode):
        t = node.var_type if node.var_type else "any"
        if node.init_expr:
            val = self.visit_expression(node.init_expr)
            dst = self.create_var_temp(node.var_name, t)
            self.builder.move(dst, val, self.type_map[t])
        else:
            self.create_var_temp(node.var_name, t)

    def visit_if(self, node: IfNode):
        cond_val = self.visit_expression(node.condition)
        then_blk = self.new_block("then")
        else_blk = self.new_block("else")
        end_blk = self.new_block("endif")

        self.builder.emit(CJumpInstr(cond_val, then_blk.label, else_blk.label))
        self.set_builder_block(then_blk)
        self.visit_statement(node.then_block)
        if not self.ends_with_jump(then_blk):
            self.builder.emit(JumpInstr(end_blk.label))
        self.set_builder_block(else_blk)
        if node.else_block:
            self.visit_statement(node.else_block)
        if not self.ends_with_jump(else_blk):
            self.builder.emit(JumpInstr(end_blk.label))
        self.set_builder_block(end_blk)

    def ends_with_jump(self, blk: IRBlock):
        if not blk.instructions:
            return False
        last = blk.instructions[-1]
        return isinstance(last, (JumpInstr, CJumpInstr, ReturnInstr))

    def visit_while(self, node: WhileNode):
        cond_blk = self.new_block("while_cond")
        body_blk = self.new_block("while_body")
        end_blk = self.new_block("while_end")

        self.builder.emit(JumpInstr(cond_blk.label))
        self.set_builder_block(cond_blk)
        cond_val = self.visit_expression(node.condition)
        self.builder.emit(CJumpInstr(cond_val, body_blk.label, end_blk.label))

        self.set_builder_block(body_blk)
        self.visit_statement(node.body)
        if not self.ends_with_jump(body_blk):
            self.builder.emit(JumpInstr(cond_blk.label))

        self.set_builder_block(end_blk)

    def visit_for(self, node: ForNode):
        init_blk = self.new_block("for_init")
        cond_blk = self.new_block("for_cond")
        body_blk = self.new_block("for_body")
        incr_blk = self.new_block("for_incr")
        end_blk = self.new_block("for_end")

        self.builder.emit(JumpInstr(init_blk.label))
        self.set_builder_block(init_blk)
        if node.init:
            self.visit_statement(node.init)
        self.builder.emit(JumpInstr(cond_blk.label))

        self.set_builder_block(cond_blk)
        if node.condition:
            cval = self.visit_expression(node.condition)
        else:
            cval = self.builder.ensure_value(True, IRType("bool"))
        self.builder.emit(CJumpInstr(cval, body_blk.label, end_blk.label))

        self.set_builder_block(body_blk)
        self.visit_statement(node.body)
        if not self.ends_with_jump(body_blk):
            self.builder.emit(JumpInstr(incr_blk.label))

        self.set_builder_block(incr_blk)
        if node.increment:
            self.visit_expression(node.increment)
        self.builder.emit(JumpInstr(cond_blk.label))

        self.set_builder_block(end_blk)

    def visit_try(self, node: TryNode):
        self.visit_block(node.try_block)
        if node.catch_var:
            self.visit_block(node.catch_block)
        if node.finally_block:
            self.visit_block(node.finally_block)

    def visit_return(self, node: ReturnNode):
        val = None
        if node.expr:
            val = self.visit_expression(node.expr)
        self.builder.emit(ReturnInstr(val))
        blk = self.new_block("after_return")

    def visit_throw(self, node: ThrowNode):
        self.visit_expression(node.expr)

    def visit_spawn(self, node: SpawnNode):
        expr_val = self.visit_expression(node.expr)
        self.builder.spawn(expr_val, IRType("thread"))

    def visit_channel(self, node: ChannelNode):
        pass

    def visit_thread(self, node: ThreadNode):
        pass

    def visit_lock(self, node: LockNode):
        lv = self.get_var_temp(node.lock_var)
        if lv:
            self.builder.emit(AcquireLockInstr(lv))

    def visit_unlock(self, node: UnlockNode):
        lv = self.get_var_temp(node.lock_var)
        if lv:
            self.builder.emit(ReleaseLockInstr(lv))

    def visit_sleep(self, node: SleepNode):
        dur = self.visit_expression(node.duration_expr)
        self.builder.emit(SleepInstr(dur))

    def visit_kill(self, node: KillNode):
        thrv = self.get_var_temp(node.thread_name)
        if thrv:
            self.builder.emit(KillInstr(thrv))

    def visit_detach(self, node: DetachNode):
        thrv = self.get_var_temp(node.thread_name)
        if thrv:
            self.builder.emit(DetachInstr(thrv))

    def visit_join(self, node: JoinNode):
        thrv = self.get_var_temp(node.thread_name)
        if thrv:
            self.builder.emit(ThreadJoinInstr(thrv))

    def visit_await(self, node: AwaitNode):
        self.visit_expression(node.expr)

    def visit_request_stmt(self, node: RequestNode):
        method = node.method
        url = self.visit_expression(node.url_expr)
        h = None
        if node.headers:
            h = self.visit_expression(node.headers)
        b = None
        if node.body_expr:
            b = self.visit_expression(node.body_expr)
        self.builder.request(method, url, h, b, IRType("any"))

    def visit_print(self, node: PrintStmtNode):
        val = self.visit_expression(node.expr)
        self.builder.emit(PrintInstr(val))

    def visit_expression(self, node):
        tname = node.__class__.__name__
        if tname == "BinaryOpNode":
            left = self.visit_expression(node.left)
            right = self.visit_expression(node.right)
            return self.builder.binop(node.op, left, right, IRType("any"))
        elif tname == "UnaryOpNode":
            sub = self.visit_expression(node.expr)
            return self.builder.unop(node.op, sub, IRType("any"))
        elif tname == "LiteralNode":
            return IRConst(node.value, self.map_literal_type(node.value))
        elif tname == "IdentifierNode":
            vtemp = self.get_var_temp(node.name)
            if vtemp:
                return vtemp
            return IRConst(None, IRType("any"))
        elif tname == "CallNode":
            c = self.visit_expression(node.callee)
            args = []
            for a in node.args:
                args.append(self.visit_expression(a))
            return self.builder.call(c, args, IRType("any"))
        elif tname == "MemberAccessNode":
            base = self.visit_expression(node.obj)
            return base
        elif tname == "IndexAccessNode":
            arr = self.visit_expression(node.arr_expr)
            idx = self.visit_expression(node.index_expr)
            return arr
        elif tname == "AssignNode":
            return self.visit_assign_expr(node)
        elif tname == "ArrowFunctionNode":
            arrow_lab = f"arrow_{len(node.params)}"
            return IRConst(arrow_lab, IRType("function"))
        elif tname == "SpawnNode":
            subexpr = self.visit_expression(node.expr)
            return self.builder.spawn(subexpr, IRType("thread"))
        elif tname == "AwaitNode":
            subexpr = self.visit_expression(node.expr)
            return subexpr
        elif tname == "RequestNode":
            meth = node.method
            url = self.visit_expression(node.url_expr)
            hdr = self.visit_expression(node.headers) if node.headers else None
            bod = self.visit_expression(node.body_expr) if node.body_expr else None
            return self.builder.request(meth, url, hdr, bod, IRType("any"))
        elif tname == "DictLiteralNode":
            return self.build_dict_literal(node)
        elif tname == "ArrayLiteralNode":
            return self.build_array_literal(node)
        else:
            return IRConst("unknown_expr", IRType("any"))

    def build_dict_literal(self, node: DictLiteralNode):
        # 1) Create a dict
        dtemp = self.current_function.create_temp(IRType("any"))
        self.builder.emit(CreateDictInstr(dtemp))
        # 2) For each (key_expr, val_expr), do:
        for (knode, vnode) in node.pairs:
            kval = self.visit_expression(knode)
            vval = self.visit_expression(vnode)
            self.builder.emit(DictSetInstr(dtemp, kval, vval))
        return dtemp

    def build_array_literal(self, node: ArrayLiteralNode):
        arrtemp = self.current_function.create_temp(IRType("any"))
        self.builder.emit(CreateArrayInstr(arrtemp))
        for elem in node.elements:
            e = self.visit_expression(elem)
            self.builder.emit(ArrayPushInstr(arrtemp, e))
        return arrtemp

    def visit_assign_expr(self, node: AssignNode):
        val = self.visit_expression(node.expr)
        if isinstance(node.target, IdentifierNode):
            dst = self.get_var_temp(node.target.name)
            if not dst:
                dst = self.create_var_temp(node.target.name, "any")
            self.builder.move(dst, val, dst.ty)
            return dst
        elif isinstance(node.target, IndexAccessNode):
            self.visit_expression(node.target.arr_expr)
            self.visit_expression(node.target.index_expr)
            # naive ignoring store
            return IRConst(None, IRType("any"))
        elif isinstance(node.target, MemberAccessNode):
            self.visit_expression(node.target.obj)
            # naive ignoring store
            return IRConst(None, IRType("any"))
        return IRConst(None, IRType("any"))

    def map_literal_type(self, v):
        if isinstance(v, bool):
            return IRType("bool")
        if v is None:
            return IRType("any")
        if isinstance(v, int):
            return IRType("int")
        if isinstance(v, float):
            return IRType("float")
        if isinstance(v, str):
            return IRType("string")
        return IRType("any")

def ast_to_ir(ast):
    cg = ASTToIR()
    return cg.generate_ir(ast)
