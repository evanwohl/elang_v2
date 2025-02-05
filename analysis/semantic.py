from parser.parser import (
    ProgramNode, FunctionNode, ClassNode, VarDeclNode, RequestNode, IfNode,
    WhileNode, ForNode, TryNode, ReturnNode, BreakNode, ContinueNode, ThrowNode,
    SpawnNode, ChannelNode, ThreadNode, LockNode, UnlockNode, SleepNode, KillNode,
    DetachNode, JoinNode, YieldNode, AwaitNode, BinaryOpNode, UnaryOpNode,
    LiteralNode, IdentifierNode, AssignNode, BlockNode, ExprStmtNode,
    DictLiteralNode, ArrayLiteralNode, PrintStmtNode, CallNode, MemberAccessNode,
    IndexAccessNode, ArrowFunctionNode
)

class SemanticError(Exception):
    pass

class SymbolInfo:
    def __init__(self, name, t, mutable=True):
        self.name = name
        self.type = t
        self.mutable = mutable
        self.used = False
        self.assigned = False
        self.extra = {}

class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}
        self.is_function_scope = False

    def define(self, name, t, mutable=True, extra=None):
        if name in self.symbols:
            raise SemanticError("Duplicate symbol in same scope: " + name)
        s = SymbolInfo(name, t, mutable)
        if extra:
            s.extra = extra
        self.symbols[name] = s

    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def finalize_scope(self):
        for sym in self.symbols.values():
            if not sym.used and sym.type != "function" and sym.type != "class" and sym.name != "_":
                print(f"Warning: Variable '{sym.name}' declared but never used.")

class TypeSystem:
    def __init__(self):
        self.base_types = {"int", "float", "bool", "string", "any", "void"}

    def check_assignable(self, lhs, rhs):
        if lhs == "any" or rhs == "any":
            return True
        return lhs == rhs

    def unify_arithmetic(self, left, right):
        if "any" in (left, right):
            return "any"
        if left == "string" and right == "string":
            return "string"
        if left in ["int","float"] and right in ["int","float"]:
            if left == "float" or right == "float":
                return "float"
            return "int"
        raise SemanticError("Invalid types for arithmetic")

class SemanticAnalyzer:
    def __init__(self):
        self.type_system = TypeSystem()
        self.current_function_return_type = None
        self.must_return = False

    def analyze(self, ast):
        if not isinstance(ast, ProgramNode):
            raise SemanticError("Top-level must be a ProgramNode")
        scope = Scope()
        self.visit_program(ast, scope)
        scope.finalize_scope()

    def visit_program(self, node, scope):
        for stmt in node.body:
            self.visit(stmt, scope)

    def visit(self, node, scope):
        if isinstance(node, FunctionNode):
            return self.visit_function(node, scope)
        elif isinstance(node, ClassNode):
            return self.visit_class(node, scope)
        elif isinstance(node, VarDeclNode):
            return self.visit_var_decl(node, scope)
        elif isinstance(node, RequestNode):
            return self.visit_request(node, scope)
        elif isinstance(node, IfNode):
            return self.visit_if(node, scope)
        elif isinstance(node, WhileNode):
            return self.visit_while(node, scope)
        elif isinstance(node, ForNode):
            return self.visit_for(node, scope)
        elif isinstance(node, TryNode):
            return self.visit_try(node, scope)
        elif isinstance(node, ReturnNode):
            return self.visit_return(node, scope)
        elif isinstance(node, BreakNode):
            return
        elif isinstance(node, ContinueNode):
            return
        elif isinstance(node, ThrowNode):
            return self.visit_throw(node, scope)
        elif isinstance(node, SpawnNode):
            return self.visit_spawn(node, scope)
        elif isinstance(node, ChannelNode):
            return self.visit_channel(node, scope)
        elif isinstance(node, ThreadNode):
            return self.visit_thread(node, scope)
        elif isinstance(node, LockNode):
            return
        elif isinstance(node, UnlockNode):
            return
        elif isinstance(node, SleepNode):
            return self.visit_sleep(node, scope)
        elif isinstance(node, KillNode):
            return self.visit_kill(node, scope)
        elif isinstance(node, DetachNode):
            return self.visit_detach(node, scope)
        elif isinstance(node, JoinNode):
            return self.visit_join(node, scope)
        elif isinstance(node, YieldNode):
            return
        elif isinstance(node, AwaitNode):
            return self.visit_await(node, scope)
        elif isinstance(node, BinaryOpNode):
            return self.visit_binary_op(node, scope)
        elif isinstance(node, UnaryOpNode):
            return self.visit_unary_op(node, scope)
        elif isinstance(node, LiteralNode):
            return self.visit_literal(node)
        elif isinstance(node, IdentifierNode):
            return self.visit_identifier(node, scope)
        elif isinstance(node, AssignNode):
            return self.visit_assign(node, scope)
        elif isinstance(node, BlockNode):
            return self.visit_block(node, scope)
        elif isinstance(node, ExprStmtNode):
            return self.visit_expr_stmt(node, scope)
        elif isinstance(node, DictLiteralNode):
            return self.visit_dict_literal(node, scope)
        elif isinstance(node, ArrayLiteralNode):
            return self.visit_array_literal(node, scope)
        elif isinstance(node, PrintStmtNode):
            return self.visit_print_stmt(node, scope)
        elif isinstance(node, CallNode):
            return self.visit_call(node, scope)
        elif isinstance(node, MemberAccessNode):
            return self.visit_member_access(node, scope)
        elif isinstance(node, IndexAccessNode):
            return self.visit_index_access(node, scope)
        elif isinstance(node, ArrowFunctionNode):
            return self.visit_arrow_function(node, scope)
        else:
            raise SemanticError("Unknown node type: " + str(type(node)))

    def visit_function(self, node, scope):
        param_types = []
        for p in node.params:
            pt = p[1] if p[1] else "any"
            param_types.append(pt)
        sig = {"param_types": param_types, "return_type": node.return_type if node.return_type else "void"}

        scope.define(node.name, "function", mutable=False, extra=sig)
        fn_scope = Scope(scope)
        fn_scope.is_function_scope = True
        old_return = self.current_function_return_type
        old_must_return = self.must_return

        self.current_function_return_type = sig["return_type"]
        self.must_return = (self.current_function_return_type != "void" and self.current_function_return_type != "any")

        for p in node.params:
            pt = p[1] if p[1] else "any"
            fn_scope.define(p[0], pt)

        self.visit(node.body, fn_scope)

        if self.must_return:
            print(f"Warning: Not all code paths return in function '{node.name}' which expects {self.current_function_return_type}")

        self.current_function_return_type = old_return
        self.must_return = old_must_return

        fn_scope.finalize_scope()

    def visit_class(self, node, scope):
        scope.define(node.name, "class", mutable=False)
        cscope = Scope(scope)
        self.visit(node.body, cscope)
        cscope.finalize_scope()

    def visit_var_decl(self, node, scope):
        t = node.var_type if node.var_type else "any"
        init_assigned = False
        if node.init_expr:
            rt = self.visit(node.init_expr, scope)
            if not self.type_system.check_assignable(t, rt):
                raise SemanticError("Cannot assign " + rt + " to " + t)
            init_assigned = True
        scope.define(node.var_name, t)
        sym = scope.lookup(node.var_name)
        if sym and init_assigned:
            sym.assigned = True
        return t

    def visit_request(self, node, scope):
        self.visit(node.url_expr, scope)
        if node.headers:
            self.visit(node.headers, scope)
        if node.body_expr:
            self.visit(node.body_expr, scope)
        return "any"

    def visit_if(self, node, scope):
        self.visit(node.condition, scope)
        then_scope = Scope(scope)
        self.visit(node.then_block, then_scope)
        then_scope.finalize_scope()

        if node.else_block:
            else_scope = Scope(scope)
            self.visit(node.else_block, else_scope)
            else_scope.finalize_scope()

    def visit_while(self, node, scope):
        self.visit(node.condition, scope)
        loop_scope = Scope(scope)
        self.visit(node.body, loop_scope)
        loop_scope.finalize_scope()

    def visit_for(self, node, scope):
        for_scope = Scope(scope)
        if node.init:
            self.visit(node.init, for_scope)
        if node.condition:
            self.visit(node.condition, for_scope)
        if node.increment:
            self.visit(node.increment, for_scope)
        self.visit(node.body, for_scope)
        for_scope.finalize_scope()

    def visit_try(self, node, scope):
        try_scope = Scope(scope)
        self.visit(node.try_block, try_scope)
        try_scope.finalize_scope()
        if node.catch_var:
            cscope = Scope(scope)
            cscope.define(node.catch_var, "any")
            if node.catch_block:
                self.visit(node.catch_block, cscope)
            cscope.finalize_scope()
        if node.finally_block:
            finally_scope = Scope(scope)
            self.visit(node.finally_block, finally_scope)
            finally_scope.finalize_scope()

    def visit_return(self, node, scope):
        if node.expr:
            r = self.visit(node.expr, scope)
            if not self.type_system.check_assignable(self.current_function_return_type, r):
                raise SemanticError("Return type mismatch: cannot assign " + r + " to " + self.current_function_return_type)
        else:
            if self.current_function_return_type != "void" and self.current_function_return_type != "any":
                raise SemanticError("Return type mismatch: function expects " + self.current_function_return_type)
        self.must_return = False
        return "void"

    def visit_throw(self, node, scope):
        self.visit(node.expr, scope)

    def visit_spawn(self, node, scope):
        return self.visit(node.expr, scope)

    def visit_channel(self, node, scope):
        scope.define(node.channel_name, "channel", mutable=False)
        return "channel"

    def visit_thread(self, node, scope):
        scope.define(node.thread_name, "thread", mutable=False)
        if node.body:
            s = Scope(scope)
            self.visit(node.body, s)
            s.finalize_scope()

    def visit_sleep(self, node, scope):
        self.visit(node.duration_expr, scope)

    def visit_kill(self, node, scope):
        sym = scope.lookup(node.thread_name)
        if not sym:
            raise SemanticError("Undeclared thread " + node.thread_name)

    def visit_detach(self, node, scope):
        sym = scope.lookup(node.thread_name)
        if not sym:
            raise SemanticError("Undeclared thread " + node.thread_name)

    def visit_join(self, node, scope):
        sym = scope.lookup(node.thread_name)
        if not sym:
            raise SemanticError("Undeclared thread " + node.thread_name)

    def visit_await(self, node, scope):
        return self.visit(node.expr, scope)

    def visit_binary_op(self, node, scope):
        l = self.visit(node.left, scope)
        r = self.visit(node.right, scope)
        if node.op in ["+", "-", "*", "/", "%", "**", "<<", ">>"]:
            return self.type_system.unify_arithmetic(l, r)
        if node.op in ["<", ">", "<=", ">=", "==", "!="]:
            return "bool"
        if node.op in ["||", "&&"]:
            if l == "any" or r == "any":
                return "any"
            if l == "bool" and r == "bool":
                return "bool"
            raise SemanticError("Invalid types for " + node.op)
        return "any"

    def visit_unary_op(self, node, scope):
        e = self.visit(node.expr, scope)
        if node.op in ["-", "+"]:
            if e == "any":
                return "any"
            if e in ["int","float"]:
                return e
            raise SemanticError("Invalid unary op " + node.op + " for " + e)
        if node.op == "!":
            if e == "bool" or e == "any":
                return "bool"
            raise SemanticError("Invalid unary op ! for " + e)
        return "any"

    def visit_literal(self, node):
        v = node.value
        if isinstance(v, bool):
            return "bool"
        if v is None:
            return "any"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "string"
        return "any"

    def visit_identifier(self, node, scope):
        sym = scope.lookup(node.name)
        if not sym:
            raise SemanticError("Undeclared identifier " + node.name)
        sym.used = True
        return sym.type

    def visit_assign(self, node, scope):
        t = self.visit(node.expr, scope)
        if isinstance(node.target, IdentifierNode):
            sym = scope.lookup(node.target.name)
            if not sym:
                raise SemanticError("Undeclared identifier " + node.target.name)
            if not self.type_system.check_assignable(sym.type, t):
                raise SemanticError("Cannot assign " + t + " to " + sym.type)
            sym.used = True
            sym.assigned = True
            return sym.type
        elif isinstance(node.target, IndexAccessNode):
            self.visit(node.target.arr_expr, scope)
            self.visit(node.target.index_expr, scope)
            return "any"
        elif isinstance(node.target, MemberAccessNode):
            self.visit(node.target.obj, scope)
            return "any"
        raise SemanticError("Invalid assignment target")

    def visit_block(self, node, scope):
        s = Scope(scope)
        for st in node.statements:
            self.visit(st, s)
        s.finalize_scope()

    def visit_expr_stmt(self, node, scope):
        return self.visit(node.expr, scope)

    def visit_dict_literal(self, node, scope):
        for k, v in node.pairs:
            self.visit(k, scope)
            self.visit(v, scope)
        return "any"

    def visit_array_literal(self, node, scope):
        for e in node.elements:
            self.visit(e, scope)
        return "any"

    def visit_print_stmt(self, node, scope):
        if node.expr:
            self.visit(node.expr, scope)

    def visit_call(self, node, scope):
        ctype = self.visit(node.callee, scope)
        sym = None
        if isinstance(node.callee, IdentifierNode):
            sym = scope.lookup(node.callee.name)
        if sym and sym.type == "function":
            sig = sym.extra
            param_types = sig.get("param_types", [])
            ret_type = sig.get("return_type", "void")

            if len(node.args) != len(param_types):
                raise SemanticError(f"Function '{sym.name}' called with wrong number of args")

            for i, arg in enumerate(node.args):
                arg_type = self.visit(arg, scope)
                expected = param_types[i]
                if not self.type_system.check_assignable(expected, arg_type):
                    raise SemanticError(f"Argument {i+1} to function '{sym.name}' expected {expected}, got {arg_type}")
            return ret_type
        else:
            for a in node.args:
                self.visit(a, scope)
            return "any"

    def visit_member_access(self, node, scope):
        b = self.visit(node.obj, scope)
        return "any"

    def visit_index_access(self, node, scope):
        arr = self.visit(node.arr_expr, scope)
        i = self.visit(node.index_expr, scope)
        return "any"

    def visit_arrow_function(self, node, scope):
        s = Scope(scope)
        for p in node.params:
            s.define(p, "any")
        old_ret = self.current_function_return_type
        old_must_return = self.must_return

        self.current_function_return_type = "any"
        self.must_return = False

        if node.is_block:
            self.visit(node.body, s)
        else:
            self.visit(node.body, s)

        self.current_function_return_type = old_ret
        self.must_return = old_must_return
        s.finalize_scope()
        return "function"

def analyze_semantics(ast):
    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)