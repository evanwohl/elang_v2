import sys
from typing import Optional, Dict
from ir import (
    IRModule, IRFunction, IRBlock, IRType, IRTemp, IRConst, IRGlobalRef,
    MoveInstr, BinOpInstr, UnOpInstr, LoadInstr, StoreInstr, AtomicLoadInstr,
    AtomicStoreInstr, AcquireLockInstr, ReleaseLockInstr, JumpInstr, CJumpInstr,
    ReturnInstr, CallInstr, RequestInstr, SpawnInstr, ThreadForkInstr,
    ThreadJoinInstr, KillInstr, DetachInstr, SleepInstr, PrintInstr
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

        # Clear any leftover variable mappings
        self.function_map.clear()

        # Create IR temps for each parameter
        for i, (pname, ptype) in enumerate(node.params):
            real_ty = self.type_map.get(ptype, IRType("any")) if ptype else IRType("any")
            ptemp = fn.create_temp(real_ty)
            self.function_map[pname] = ptemp

        self.visit_block(node.body)

        # Ensure there's a ReturnInstr if none present
        if fn.blocks:
            last_blk = fn.blocks[-1]
            if not last_blk.instructions or not isinstance(last_blk.instructions[-1], ReturnInstr):
                last_blk.add_instr(ReturnInstr(None))

        self.function_map.clear()
        return fn

    def visit_class_decl(self, node: ClassNode):
        pass  # Not generating IR for classes in this minimal approach

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
            cv = self.visit_expression(node.condition)
        else:
            cv = self.builder.ensure_value(True, IRType("bool"))
        self.builder.emit(CJumpInstr(cv, body_blk.label, end_blk.label))

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
        pass  # no IR for channel creation in this pass

    def visit_thread(self, node: ThreadNode):
        pass

    def visit_lock(self, node: LockNode):
        lock_temp = self.get_var_temp(node.lock_var)
        if lock_temp:
            self.builder.emit(AcquireLockInstr(lock_temp))

    def visit_unlock(self, node: UnlockNode):
        lock_temp = self.get_var_temp(node.lock_var)
        if lock_temp:
            self.builder.emit(ReleaseLockInstr(lock_temp))

    def visit_sleep(self, node: SleepNode):
        dur = self.visit_expression(node.duration_expr)
        self.builder.emit(SleepInstr(dur))

    def visit_kill(self, node: KillNode):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(KillInstr(thr))

    def visit_detach(self, node: DetachNode):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(DetachInstr(thr))

    def visit_join(self, node: JoinNode):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(ThreadJoinInstr(thr))

    def visit_await(self, node: AwaitNode):
        self.visit_expression(node.expr)

    def visit_request_stmt(self, node: RequestNode):
        method = node.method
        url_val = self.visit_expression(node.url_expr)
        hdr_val = None
        if node.headers:
            hdr_val = self.ensure_stored_literal(node.headers)
        body_val = None
        if node.body_expr:
            body_val = self.ensure_stored_literal(node.body_expr)
        self.builder.request(method, url_val, hdr_val, body_val, IRType("any"))

    def ensure_stored_literal(self, literal_node):
        # If it's dict/array, produce a separate temp for it
        # so IR shows HEADERS=%h etc. rather than inline
        val = self.visit_expression(literal_node)
        if isinstance(val, IRConst):
            tmp = self.current_function.create_temp(IRType("any"))
            self.builder.move(tmp, val, IRType("any"))
            return tmp
        return val

    def visit_print(self, node: PrintStmtNode):
        val = self.visit_expression(node.expr)
        self.builder.emit(PrintInstr(val))

    def visit_expression(self, node):
        tname = node.__class__.__name__
        if tname == "BinaryOpNode":
            l = self.visit_expression(node.left)
            r = self.visit_expression(node.right)
            return self.builder.binop(node.op, l, r, IRType("any"))
        elif tname == "UnaryOpNode":
            sub = self.visit_expression(node.expr)
            return self.builder.unop(node.op, sub, IRType("any"))
        elif tname == "LiteralNode":
            return IRConst(node.value, self.map_literal_type(node.value))
        elif tname == "IdentifierNode":
            found = self.get_var_temp(node.name)
            if found:
                return found
            return IRConst(None, IRType("any"))
        elif tname == "CallNode":
            c = self.visit_expression(node.callee)
            arg_list = []
            for a in node.args:
                arg_list.append(self.visit_expression(a))
            return self.builder.call(c, arg_list, IRType("any"))
        elif tname == "MemberAccessNode":
            base = self.visit_expression(node.obj)
            return base
        elif tname == "IndexAccessNode":
            arrv = self.visit_expression(node.arr_expr)
            idxv = self.visit_expression(node.index_expr)
            return arrv
        elif tname == "AssignNode":
            return self.visit_assign_expr(node)
        elif tname == "ArrowFunctionNode":
            arrow_name = f"arrow_{len(node.params)}"
            return IRConst(arrow_name, IRType("function"))
        elif tname == "SpawnNode":
            sp = self.visit_expression(node.expr)
            return self.builder.spawn(sp, IRType("thread"))
        elif tname == "AwaitNode":
            ev = self.visit_expression(node.expr)
            return ev
        elif tname == "RequestNode":
            m = node.method
            u = self.visit_expression(node.url_expr)
            h = self.ensure_stored_literal(node.headers) if node.headers else None
            b = self.ensure_stored_literal(node.body_expr) if node.body_expr else None
            return self.builder.request(m, u, h, b, IRType("any"))
        elif tname == "DictLiteralNode":
            return IRConst("dict_lit", IRType("any"))
        elif tname == "ArrayLiteralNode":
            return IRConst("array_lit", IRType("any"))
        else:
            return IRConst("unknown_expr", IRType("any"))

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
            return IRConst(None, IRType("any"))  # ignoring store
        elif isinstance(node.target, MemberAccessNode):
            self.visit_expression(node.target.obj)
            return IRConst(None, IRType("any"))  # ignoring store
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
