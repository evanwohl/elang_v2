import sys
from ir import IRModule, IRFunction, IRBlock, IRType, IRTemp, IRConst, IRGlobalRef
from ir import MoveInstr, BinOpInstr, UnOpInstr, LoadInstr, StoreInstr, AtomicLoadInstr, AtomicStoreInstr
from ir import AcquireLockInstr, ReleaseLockInstr, JumpInstr, CJumpInstr, ReturnInstr, CallInstr, RequestInstr
from ir import SpawnInstr, ThreadForkInstr, ThreadJoinInstr, KillInstr, DetachInstr, SleepInstr, PrintInstr
from ir import ChannelSendInstr, ChannelRecvInstr, WaitAllInstr, PhiInstr
from ir_utils import IRBuilder

class ASTToIR:
    def __init__(self):
        self.module = IRModule()
        self.current_function = None
        self.builder = None
        self.function_map = {}
        self.block_count = 0
        self.main_created = False
        self.main_fn = None
        self.type_map = {"int": IRType("int"), "float": IRType("float"), "bool": IRType("bool"), "string": IRType("string"), "any": IRType("any"), "void": IRType("void"), "thread": IRType("thread"), "lock": IRType("lock"), "channel": IRType("channel")}

    def generate_ir(self, ast):
        self.visit_program(ast)
        return self.module

    def new_block(self, label_prefix="block"):
        self.block_count += 1
        b = IRBlock(f"{label_prefix}{self.block_count}")
        self.current_function.add_block(b)
        return b

    def set_builder_block(self, blk):
        self.builder.set_block(blk)

    def visit_program(self, node):
        for stmt in node.body:
            if hasattr(stmt, "params") and hasattr(stmt, "body"):
                fn = self.visit_function_decl(stmt)
                self.module.add_function(fn)
            elif hasattr(stmt, "name") and hasattr(stmt, "body") and stmt.__class__.__name__ == "ClassNode":
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

    def visit_function_decl(self, node):
        param_types = []
        for p in node.params:
            pt = "any"
            param_types.append(self.type_map.get(pt, IRType("any")))
        rt = "void"
        if hasattr(node, "return_type") and node.return_type:
            rt = node.return_type if node.return_type in self.type_map else "any"
        fn = IRFunction(node.name, param_types, self.type_map.get(rt, IRType("void")))
        self.current_function = fn
        self.builder = IRBuilder(fn)
        blk = IRBlock("entry0")
        fn.add_block(blk)
        self.set_builder_block(blk)
        self.visit_block(node.body)
        if fn.blocks and not isinstance(fn.blocks[-1].instructions[-1] if fn.blocks[-1].instructions else None, ReturnInstr):
            fn.blocks[-1].add_instr(ReturnInstr(None))
        return fn

    def visit_class_decl(self, node):
        pass

    def visit_block(self, node):
        for stmt in node.statements:
            self.visit_statement(stmt)

    def visit_statement(self, node):
        c = node.__class__.__name__
        if c == "VarDeclNode":
            self.visit_var_decl(node)
        elif c == "IfNode":
            self.visit_if(node)
        elif c == "WhileNode":
            self.visit_while(node)
        elif c == "ForNode":
            self.visit_for(node)
        elif c == "TryNode":
            self.visit_try(node)
        elif c == "ReturnNode":
            self.visit_return(node)
        elif c == "BreakNode":
            self.visit_break(node)
        elif c == "ContinueNode":
            self.visit_continue(node)
        elif c == "ThrowNode":
            self.visit_throw(node)
        elif c == "SpawnNode":
            self.visit_spawn(node)
        elif c == "ChannelNode":
            self.visit_channel(node)
        elif c == "ThreadNode":
            self.visit_thread(node)
        elif c == "LockNode":
            self.visit_lock(node)
        elif c == "UnlockNode":
            self.visit_unlock(node)
        elif c == "SleepNode":
            self.visit_sleep(node)
        elif c == "KillNode":
            self.visit_kill(node)
        elif c == "DetachNode":
            self.visit_detach(node)
        elif c == "JoinNode":
            self.visit_join(node)
        elif c == "YieldNode":
            pass
        elif c == "AwaitNode":
            self.visit_await(node)
        elif c == "RequestNode":
            self.visit_request_stmt(node)
        elif c == "ExprStmtNode":
            self.visit_expression(node.expr)
        elif c == "BlockNode":
            self.visit_block(node)
        elif c == "PrintStmtNode":
            self.visit_print(node)
        else:
            pass

    def visit_var_decl(self, node):
        t = "any"
        if hasattr(node, "var_type") and node.var_type:
            t = node.var_type if node.var_type in self.type_map else "any"
        if node.init_expr:
            val = self.visit_expression(node.init_expr)
            self.builder.move(self.create_var_temp(node.var_name, t), val, self.type_map[t])
        else:
            self.create_var_temp(node.var_name, t)

    def create_var_temp(self, name, t):
        v = self.current_function.create_temp(self.type_map[t])
        self.function_map[name] = v
        return v

    def get_var_temp(self, name):
        return self.function_map.get(name, None)

    def visit_if(self, node):
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

    def ends_with_jump(self, blk):
        if not blk.instructions:
            return False
        last = blk.instructions[-1]
        return isinstance(last, (JumpInstr, CJumpInstr, ReturnInstr))

    def visit_while(self, node):
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

    def visit_for(self, node):
        for_init_blk = self.new_block("for_init")
        cond_blk = self.new_block("for_cond")
        body_blk = self.new_block("for_body")
        incr_blk = self.new_block("for_incr")
        end_blk = self.new_block("for_end")
        self.builder.emit(JumpInstr(for_init_blk.label))
        self.set_builder_block(for_init_blk)
        if node.init:
            self.visit_statement(node.init)
        self.builder.emit(JumpInstr(cond_blk.label))
        self.set_builder_block(cond_blk)
        cond_val = None
        if node.condition:
            cond_val = self.visit_expression(node.condition)
        else:
            cond_val = self.builder.ensure_value(True, IRType("bool"))
        self.builder.emit(CJumpInstr(cond_val, body_blk.label, end_blk.label))
        self.set_builder_block(body_blk)
        self.visit_statement(node.body)
        if not self.ends_with_jump(body_blk):
            self.builder.emit(JumpInstr(incr_blk.label))
        self.set_builder_block(incr_blk)
        if node.increment:
            self.visit_expression(node.increment)
        self.builder.emit(JumpInstr(cond_blk.label))
        self.set_builder_block(end_blk)

    def visit_try(self, node):
        self.visit_block(node.try_block)
        if node.catch_var:
            self.visit_block(node.catch_block)
        if node.finally_block:
            self.visit_block(node.finally_block)

    def visit_return(self, node):
        val = None
        if node.expr:
            val = self.visit_expression(node.expr)
        self.builder.emit(ReturnInstr(val))
        blk = self.new_block("after_return")

    def visit_break(self, node):
        pass

    def visit_continue(self, node):
        pass

    def visit_throw(self, node):
        val = self.visit_expression(node.expr)

    def visit_spawn(self, node):
        expr_val = self.visit_expression(node.expr)
        self.builder.spawn(expr_val, IRType("thread"))

    def visit_channel(self, node):
        pass

    def visit_thread(self, node):
        pass

    def visit_lock(self, node):
        lock_var = self.get_var_temp(node.lock_var)
        if lock_var:
            self.builder.emit(AcquireLockInstr(lock_var))

    def visit_unlock(self, node):
        lock_var = self.get_var_temp(node.lock_var)
        if lock_var:
            self.builder.emit(ReleaseLockInstr(lock_var))

    def visit_sleep(self, node):
        dur = self.visit_expression(node.duration_expr)
        self.builder.emit(SleepInstr(dur))

    def visit_kill(self, node):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(KillInstr(thr))

    def visit_detach(self, node):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(DetachInstr(thr))

    def visit_join(self, node):
        thr = self.get_var_temp(node.thread_name)
        if thr:
            self.builder.emit(ThreadJoinInstr(thr))

    def visit_await(self, node):
        val = self.visit_expression(node.expr)

    def visit_request_stmt(self, node):
        method = node.method
        url_val = self.visit_expression(node.url_expr)
        hdr_val = None
        if node.headers:
            hdr_val = self.visit_expression(node.headers)
        body_val = None
        if node.body_expr:
            body_val = self.visit_expression(node.body_expr)
        self.builder.request(method, url_val, hdr_val, body_val, IRType("any"))

    def visit_print(self, node):
        val = self.visit_expression(node.expr)
        self.builder.emit(PrintInstr(val))

    def visit_expression(self, node):
        c = node.__class__.__name__
        if c == "BinaryOpNode":
            left = self.visit_expression(node.left)
            right = self.visit_expression(node.right)
            return self.builder.binop(node.op, left, right, IRType("any"))
        elif c == "UnaryOpNode":
            expr_val = self.visit_expression(node.expr)
            return self.builder.unop(node.op, expr_val, IRType("any"))
        elif c == "LiteralNode":
            return IRConst(node.value, self.map_literal_type(node.value))
        elif c == "IdentifierNode":
            return self.get_var_temp(node.name)
        elif c == "CallNode":
            callee = self.visit_expression(node.callee)
            args = []
            for a in node.args:
                args.append(self.visit_expression(a))
            return self.builder.call(callee, args, IRType("any"))
        elif c == "RequestNode":
            method = node.method
            url_val = self.visit_expression(node.url_expr)
            hdr_val = None
            if node.headers:
                hdr_val = self.visit_expression(node.headers)
            body_val = None
            if node.body_expr:
                body_val = self.visit_expression(node.body_expr)
            return self.builder.request(method, url_val, hdr_val, body_val, IRType("any"))
        elif c == "MemberAccessNode":
            obj_val = self.visit_expression(node.obj)
            return obj_val
        elif c == "IndexAccessNode":
            arr_val = self.visit_expression(node.arr_expr)
            idx_val = self.visit_expression(node.index_expr)
            return arr_val
        elif c == "SpawnNode":
            expr_val = self.visit_expression(node.expr)
            return self.builder.spawn(expr_val, IRType("thread"))
        elif c == "AwaitNode":
            expr_val = self.visit_expression(node.expr)
            return expr_val
        elif c == "ArrayLiteralNode":
            return IRConst("array_lit", IRType("any"))
        elif c == "DictLiteralNode":
            return IRConst("dict_lit", IRType("any"))
        else:
            return IRConst("unknown_expr", IRType("any"))

    def map_literal_type(self, val):
        if isinstance(val, bool):
            return IRType("bool")
        if isinstance(val, (int, float)):
            return IRType("float") if isinstance(val, float) else IRType("int")
        if isinstance(val, str):
            return IRType("string")
        if val is None:
            return IRType("any")
        return IRType("any")

def ast_to_ir(ast):
    c = ASTToIR()
    return c.generate_ir(ast)
