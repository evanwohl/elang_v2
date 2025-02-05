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

class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}

    def define(self, name, t, mutable=True):
        if name in self.symbols:
            raise SemanticError("Duplicate symbol " + name)
        self.symbols[name] = SymbolInfo(name, t, mutable)

    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

class TypeSystem:
    def __init__(self):
        self.base_types = {"int", "float", "bool", "string", "any", "void"}

    def check_assignable(self, lhs, rhs):
        if lhs == "any" or rhs == "any":
            return True
        if lhs == rhs:
            return True
        return False

    def unify_arithmetic(self, left, right):
        # Allow string + string => string
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

    def analyze(self, ast):
        if not isinstance(ast, ProgramNode):
            raise SemanticError("Top-level must be a ProgramNode")
        scope = Scope()
        self.visit_program(ast, scope)

    def visit_program(self, node, scope):
        for stmt in node.body:
            self.visit(stmt, scope)

    def visit(self, node, scope):
        if isinstance(node, FunctionNode):
            return self.visit_function(node, scope)
        if isinstance(node, ClassNode):
            return self.visit_class(node, scope)
        if isinstance(node, VarDeclNode):
            return self.visit_var_decl(node, scope)
        if isinstance(node, RequestNode):
            return self.visit_request(node, scope)
        if isinstance(node, IfNode):
            return self.visit_if(node, scope)
        if isinstance(node, WhileNode):
            return self.visit_while(node, scope)
        if isinstance(node, ForNode):
            return self.visit_for(node, scope)
        if isinstance(node, TryNode):
            return self.visit_try(node, scope)
        if isinstance(node, ReturnNode):
            return self.visit_return(node, scope)
        if isinstance(node, BreakNode):
            return
        if isinstance(node, ContinueNode):
            return
        if isinstance(node, ThrowNode):
            return self.visit_throw(node, scope)
        if isinstance(node, SpawnNode):
            return self.visit_spawn(node, scope)
        if isinstance(node, ChannelNode):
            return self.visit_channel(node, scope)
        if isinstance(node, ThreadNode):
            return self.visit_thread(node, scope)
        if isinstance(node, LockNode):
            return
        if isinstance(node, UnlockNode):
            return
        if isinstance(node, SleepNode):
            return self.visit_sleep(node, scope)
        if isinstance(node, KillNode):
            return self.visit_kill(node, scope)
        if isinstance(node, DetachNode):
            return self.visit_detach(node, scope)
        if isinstance(node, JoinNode):
            return self.visit_join(node, scope)
        if isinstance(node, YieldNode):
            return
        if isinstance(node, AwaitNode):
            return self.visit_await(node, scope)
        if isinstance(node, BinaryOpNode):
            return self.visit_binary_op(node, scope)
        if isinstance(node, UnaryOpNode):
            return self.visit_unary_op(node, scope)
        if isinstance(node, LiteralNode):
            return self.visit_literal(node)
        if isinstance(node, IdentifierNode):
            return self.visit_identifier(node, scope)
        if isinstance(node, AssignNode):
            return self.visit_assign(node, scope)
        if isinstance(node, BlockNode):
            return self.visit_block(node, scope)
        if isinstance(node, ExprStmtNode):
            return self.visit_expr_stmt(node, scope)
        if isinstance(node, DictLiteralNode):
            return self.visit_dict_literal(node, scope)
        if isinstance(node, ArrayLiteralNode):
            return self.visit_array_literal(node, scope)
        if isinstance(node, PrintStmtNode):
            return self.visit_print_stmt(node, scope)
        if isinstance(node, CallNode):
            return self.visit_call(node, scope)
        if isinstance(node, MemberAccessNode):
            return self.visit_member_access(node, scope)
        if isinstance(node, IndexAccessNode):
            return self.visit_index_access(node, scope)
        if isinstance(node, ArrowFunctionNode):
            return self.visit_arrow_function(node, scope)
        raise SemanticError("Unknown node type: " + str(type(node)))

    def visit_function(self, node, scope):
        scope.define(node.name, "function", False)
        fn_scope = Scope(scope)
        old_return = self.current_function_return_type
        self.current_function_return_type = node.return_type if node.return_type else "void"
        for p in node.params:
            pt = p[1] if p[1] else "any"
            fn_scope.define(p[0], pt)
        self.visit(node.body, fn_scope)
        self.current_function_return_type = old_return

    def visit_class(self, node, scope):
        scope.define(node.name, "class", False)
        cscope = Scope(scope)
        self.visit(node.body, cscope)

    def visit_var_decl(self, node, scope):
        t = node.var_type if node.var_type else "any"
        if node.init_expr:
            rt = self.visit(node.init_expr, scope)
            if not self.type_system.check_assignable(t, rt):
                raise SemanticError("Cannot assign " + rt + " to " + t)
        scope.define(node.var_name, t)
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
        self.visit(node.then_block, scope)
        if node.else_block:
            self.visit(node.else_block, scope)

    def visit_while(self, node, scope):
        self.visit(node.condition, scope)
        self.visit(node.body, scope)

    def visit_for(self, node, scope):
        if node.init:
            self.visit(node.init, scope)
        if node.condition:
            self.visit(node.condition, scope)
        if node.increment:
            self.visit(node.increment, scope)
        self.visit(node.body, scope)

    def visit_try(self, node, scope):
        self.visit(node.try_block, scope)
        if node.catch_var:
            cscope = Scope(scope)
            cscope.define(node.catch_var, "any")
            if node.catch_block:
                self.visit(node.catch_block, cscope)
        if node.finally_block:
            self.visit(node.finally_block, scope)

    def visit_return(self, node, scope):
        if node.expr:
            r = self.visit(node.expr, scope)
            if not self.type_system.check_assignable(self.current_function_return_type, r):
                raise SemanticError(
                    "Return type mismatch: cannot assign " + r +
                    " to " + self.current_function_return_type
                )
            return r
        else:
            if self.current_function_return_type != "void" and self.current_function_return_type != "any":
                raise SemanticError("Return type mismatch: function expects " + self.current_function_return_type)
        return "void"

    def visit_throw(self, node, scope):
        self.visit(node.expr, scope)

    def visit_spawn(self, node, scope):
        return self.visit(node.expr, scope)

    def visit_channel(self, node, scope):
        scope.define(node.channel_name, "channel")
        return "channel"

    def visit_thread(self, node, scope):
        scope.define(node.thread_name, "thread", False)
        if node.body:
            s = Scope(scope)
            self.visit(node.body, s)

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
        if node.op == "||" or node.op == "&&":
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
        s = scope.lookup(node.name)
        if not s:
            raise SemanticError("Undeclared identifier " + node.name)
        return s.type

    def visit_assign(self, node, scope):
        t = self.visit(node.expr, scope)
        if isinstance(node.target, IdentifierNode):
            sym = scope.lookup(node.target.name)
            if not sym:
                raise SemanticError("Undeclared identifier " + node.target.name)
            if not self.type_system.check_assignable(sym.type, t):
                raise SemanticError("Cannot assign " + t + " to " + sym.type)
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
        c = self.visit(node.callee, scope)
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
        self.current_function_return_type = "any"
        if node.is_block:
            self.visit(node.body, s)
        else:
            self.visit(node.body, s)
        self.current_function_return_type = old_ret
        return "function"

def analyze_semantics(ast):
    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)
