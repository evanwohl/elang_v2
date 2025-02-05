import re
import struct
from lexer.lexer import tokenize

############################################
# AST Node Definitions
############################################

class ASTNode: pass

class ProgramNode(ASTNode):
    def __init__(self, body):
        self.body = body

class FunctionNode(ASTNode):
    def __init__(self, name, params, return_type, body, is_async=False):
        self.name = name
        self.params = params           # list of (paramName, paramType)
        self.return_type = return_type # string or None
        self.body = body
        self.is_async = is_async

class ClassNode(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

class VarDeclNode(ASTNode):
    def __init__(self, var_name, var_type, init_expr):
        self.var_name = var_name
        self.var_type = var_type  # string or None
        self.init_expr = init_expr

class RequestNode(ASTNode):
    def __init__(self, method, url_expr, headers=None, body_expr=None):
        self.method = method
        self.url_expr = url_expr
        self.headers = headers
        self.body_expr = body_expr

class IfNode(ASTNode):
    def __init__(self, condition, then_block, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class WhileNode(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForNode(ASTNode):
    def __init__(self, init, condition, increment, body):
        self.init = init
        self.condition = condition
        self.increment = increment
        self.body = body

class TryNode(ASTNode):
    def __init__(self, try_block, catch_var, catch_block, finally_block):
        self.try_block = try_block
        self.catch_var = catch_var
        self.catch_block = catch_block
        self.finally_block = finally_block

class ReturnNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BreakNode(ASTNode):
    pass

class ContinueNode(ASTNode):
    pass

class ThrowNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class SpawnNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class ChannelNode(ASTNode):
    def __init__(self, channel_name):
        self.channel_name = channel_name

class ThreadNode(ASTNode):
    def __init__(self, thread_name, body=None):
        self.thread_name = thread_name
        self.body = body

class LockNode(ASTNode):
    def __init__(self, lock_var):
        self.lock_var = lock_var

class UnlockNode(ASTNode):
    def __init__(self, lock_var):
        self.lock_var = lock_var

class SleepNode(ASTNode):
    def __init__(self, duration_expr):
        self.duration_expr = duration_expr

class KillNode(ASTNode):
    def __init__(self, thread_name):
        self.thread_name = thread_name

class DetachNode(ASTNode):
    def __init__(self, thread_name):
        self.thread_name = thread_name

class JoinNode(ASTNode):
    def __init__(self, thread_name):
        self.thread_name = thread_name

class YieldNode(ASTNode):
    pass

class AwaitNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BinaryOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOpNode(ASTNode):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class LiteralNode(ASTNode):
    def __init__(self, value):
        self.value = value

class IdentifierNode(ASTNode):
    def __init__(self, name):
        self.name = name

class AssignNode(ASTNode):
    def __init__(self, target, expr):
        self.target = target
        self.expr = expr

class BlockNode(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class ExprStmtNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class DictLiteralNode(ASTNode):
    def __init__(self, pairs):
        self.pairs = pairs

class ArrayLiteralNode(ASTNode):
    def __init__(self, elements):
        self.elements = elements

class PrintStmtNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class CallNode(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class MemberAccessNode(ASTNode):
    def __init__(self, obj, field_name):
        self.obj = obj
        self.field_name = field_name

class IndexAccessNode(ASTNode):
    def __init__(self, arr_expr, index_expr):
        self.arr_expr = arr_expr
        self.index_expr = index_expr


############################################
# Parser
############################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current = self.tokens[0] if self.tokens else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = None

    def match(self, *types):
        self.skip_newlines()
        if self.current and self.current.type in types:
            tok = self.current
            self.advance()
            return tok
        return None

    def expect(self, ttype):
        self.skip_newlines()
        if not self.current or self.current.type != ttype:
            raise Exception(f"Expected {ttype}, got {self.current}")
        val = self.current
        self.advance()
        return val

    def skip_newlines(self):
        while self.current and self.current.type == 'NEWLINE':
            self.advance()

    def skip_extra_semicolons(self):
        while self.match('SEMICOLON'):
            pass

    def parse_program(self):
        body = []
        while self.current:
            self.skip_newlines()
            if not self.current:
                break
            node = self.parse_declaration_or_statement()
            body.append(node)
            self.skip_extra_semicolons()
        return ProgramNode(body)

    def parse_declaration_or_statement(self):
        self.skip_newlines()
        if self.match('ASYNC'):
            if self.match('FUNCTION'):
                fn = self.parse_function_decl(is_async=True)
                self.skip_extra_semicolons()
                return fn
            self.pos -= 1
            self.current = self.tokens[self.pos]
            stmt = self.parse_statement()
            self.skip_extra_semicolons()
            return stmt

        if self.match('FUNCTION'):
            fn = self.parse_function_decl()
            self.skip_extra_semicolons()
            return fn

        if self.match('CLASS'):
            cls = self.parse_class_decl()
            self.skip_extra_semicolons()
            return cls

        stmt = self.parse_statement()
        self.skip_extra_semicolons()
        return stmt

    def parse_function_decl(self, is_async=False):
        """
        function name ( param1 : type1, param2 : type2 ) : returnType {
            ...
        }
        """
        name_tok = self.expect('IDENTIFIER')
        self.expect('LPAREN')
        params = []
        if self.current and self.current.type not in ('RPAREN', None):
            params.append(self.parse_typed_param())
            while self.match('COMMA'):
                params.append(self.parse_typed_param())
        self.expect('RPAREN')

        # ---- TYPE ANNOTATIONS: optional return type
        return_type = None
        if self.match('COLON'):
            # e.g. function foo(...) : int {
            return_type = self.parse_type()

        block = self.parse_block()
        return FunctionNode(name_tok.value, params, return_type, block, is_async=is_async)

    def parse_typed_param(self):
        """
        Parse 'paramName : Type' or just 'paramName'
        """
        param_name_tok = self.expect('IDENTIFIER')
        param_type = None
        if self.match('COLON'):
            param_type = self.parse_type()
        return (param_name_tok.value, param_type)

    def parse_type(self):
        """
        Simple type grammar: just parse an IDENTIFIER as type name.
        (Could expand to generics, function types, etc.)
        """
        if not self.current or self.current.type != 'IDENTIFIER':
            raise Exception(f"Expected a type name (IDENTIFIER), got {self.current}")
        tname = self.current.value
        self.advance()
        return tname

    def parse_class_decl(self):
        name_tok = self.expect('IDENTIFIER')
        body = self.parse_block()
        return ClassNode(name_tok.value, body)

    def parse_block(self):
        self.expect('LBRACE')
        statements = []
        while self.current and self.current.type != 'RBRACE':
            stmt = self.parse_block_decl_or_stmt()
            statements.append(stmt)
            self.skip_extra_semicolons()
        self.expect('RBRACE')
        return BlockNode(statements)

    def parse_block_decl_or_stmt(self):
        self.skip_newlines()
        if self.match('ASYNC'):
            if self.match('FUNCTION'):
                fn = self.parse_function_decl(is_async=True)
                self.skip_extra_semicolons()
                return fn
            # revert if not function
            self.pos -= 1
            self.current = self.tokens[self.pos]
            stmt = self.parse_statement()
            self.skip_extra_semicolons()
            return stmt

        if self.match('FUNCTION'):
            fn = self.parse_function_decl()
            self.skip_extra_semicolons()
            return fn

        stmt = self.parse_statement()
        self.skip_extra_semicolons()
        return stmt

    def parse_statement(self):
        self.skip_newlines()

        if self.match('VAR'):
            return self.parse_var_decl()
        if self.match('IF'):
            return self.parse_if_stmt()
        if self.match('WHILE'):
            return self.parse_while_stmt()
        if self.match('FOR'):
            return self.parse_for_stmt()
        if self.match('TRY'):
            return self.parse_try_stmt()
        if self.match('THROW'):
            return self.parse_throw_stmt()
        if self.match('RETURN'):
            return self.parse_return_stmt()
        if self.match('BREAK'):
            return BreakNode()
        if self.match('CONTINUE'):
            return ContinueNode()

        # concurrency expansions (spawn is expression, so removed from statement match)
        if self.match('CHANNEL'):
            return self.parse_channel_stmt()
        if self.match('THREAD'):
            return self.parse_thread_decl()
        if self.match('LOCK'):
            return self.parse_lock_stmt()
        if self.match('UNLOCK'):
            return self.parse_unlock_stmt()
        if self.match('JOIN'):
            return self.parse_join_stmt()
        if self.match('KILL'):
            return self.parse_kill_stmt()
        if self.match('DETACH'):
            return self.parse_detach_stmt()
        if self.match('YIELD'):
            self.expect('SEMICOLON')
            return YieldNode()
        if self.match('SLEEP'):
            return self.parse_sleep_stmt()
        if self.match('PRINT'):
            return self.parse_print_stmt()

        # optional block
        if self.match('LBRACE'):
            self.pos -= 1
            self.current = self.tokens[self.pos]
            block_node = self.parse_block()
            return block_node

        return self.parse_expr_statement()

    def parse_var_decl(self):
        """
        var x : Type = someExpr;
        var y : Type;
        var z = 100;
        """
        var_name = self.expect('IDENTIFIER').value

        # ---- TYPE ANNOTATIONS
        var_type = None
        if self.match('COLON'):
            var_type = self.parse_type()

        init_expr = None
        if self.match('EQUALS'):
            init_expr = self.parse_expression()

        self.expect('SEMICOLON')
        return VarDeclNode(var_name, var_type, init_expr)

    def parse_if_stmt(self):
        self.expect('LPAREN')
        cond = self.parse_expression()
        self.expect('RPAREN')
        then_block = self.parse_statement()
        else_block = None
        if self.match('ELSE'):
            else_block = self.parse_statement()
        return IfNode(cond, then_block, else_block)

    def parse_while_stmt(self):
        self.expect('LPAREN')
        cond = self.parse_expression()
        self.expect('RPAREN')
        body = self.parse_statement()
        return WhileNode(cond, body)

    def parse_for_stmt(self):
        self.expect('LPAREN')
        init = self.parse_for_init()
        self.expect('SEMICOLON')
        cond = None
        if self.current and self.current.type != 'SEMICOLON':
            cond = self.parse_expression()
        self.expect('SEMICOLON')
        increment = None
        if self.current and self.current.type != 'RPAREN':
            increment = self.parse_expression()
        self.expect('RPAREN')
        body = self.parse_statement()
        return ForNode(init, cond, increment, body)

    def parse_for_init(self):
        if self.match('VAR'):
            return self.parse_var_decl_no_semicolon()
        elif self.current and self.current.type != 'SEMICOLON':
            return self.parse_expression()
        return None

    def parse_var_decl_no_semicolon(self):
        """
        Helper for 'for' loops, e.g. for (var i=0; i<10; i=i+1)
        We want the VarDeclNode but we don't consume a semicolon here.
        """
        var_name = self.expect('IDENTIFIER').value
        var_type = None
        if self.match('COLON'):
            var_type = self.parse_type()
        init_expr = None
        if self.match('EQUALS'):
            init_expr = self.parse_expression()
        # don't expect semicolon here
        return VarDeclNode(var_name, var_type, init_expr)

    def parse_try_stmt(self):
        try_block = self.parse_block()
        catch_var = None
        catch_block = None
        finally_block = None
        if self.match('CATCH'):
            self.expect('LPAREN')
            catch_var = self.expect('IDENTIFIER').value
            self.expect('RPAREN')
            catch_block = self.parse_block()
        if self.match('FINALLY'):
            finally_block = self.parse_block()
        return TryNode(try_block, catch_var, catch_block, finally_block)

    def parse_throw_stmt(self):
        expr = self.parse_expression()
        self.expect('SEMICOLON')
        return ThrowNode(expr)

    def parse_return_stmt(self):
        expr = None
        if self.current and self.current.type not in ('SEMICOLON','RBRACE'):
            expr = self.parse_expression()
        self.expect('SEMICOLON')
        return ReturnNode(expr)

    #########################################
    # Concurrency expansions
    #########################################

    def parse_channel_stmt(self):
        chan_name = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return ChannelNode(chan_name)

    def parse_thread_decl(self):
        thr_name = self.expect('IDENTIFIER').value
        body = None
        if self.match('LBRACE'):
            block_stmts = []
            while self.current and self.current.type != 'RBRACE':
                s = self.parse_block_decl_or_stmt()
                block_stmts.append(s)
                self.skip_extra_semicolons()
            self.expect('RBRACE')
            body = BlockNode(block_stmts)
        self.expect('SEMICOLON')
        return ThreadNode(thr_name, body)

    def parse_lock_stmt(self):
        lock_var = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return LockNode(lock_var)

    def parse_unlock_stmt(self):
        lock_var = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return UnlockNode(lock_var)

    def parse_join_stmt(self):
        thr = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return JoinNode(thr)

    def parse_kill_stmt(self):
        thr = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return KillNode(thr)

    def parse_detach_stmt(self):
        thr = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return DetachNode(thr)

    def parse_sleep_stmt(self):
        duration_expr = self.parse_expression()
        self.expect('SEMICOLON')
        return SleepNode(duration_expr)

    def parse_print_stmt(self):
        expr = None
        if self.current and self.current.type not in ('SEMICOLON','RBRACE'):
            expr = self.parse_expression()
        self.expect('SEMICOLON')
        return PrintStmtNode(expr)

    #########################################
    # Expressions
    #########################################

    def parse_expr_statement(self):
        expr = self.parse_expression()
        self.expect('SEMICOLON')
        return ExprStmtNode(expr)

    def parse_expression(self):
        self.skip_newlines()
        # Allow 'await' as an expression
        if self.match('AWAIT'):
            inner_expr = self.parse_expression()
            return AwaitNode(inner_expr)

        # ALLOW 'spawn' as an expression
        if self.match('SPAWN'):
            inner_expr = self.parse_expression()
            return SpawnNode(inner_expr)

        return self.parse_assignment()

    def parse_assignment(self):
        node = self.parse_logical_or()
        if self.match('EQUALS'):
            rhs = self.parse_assignment()
            if isinstance(node, IdentifierNode):
                return AssignNode(node, rhs)
            raise Exception("Left side of assignment is not an identifier")
        return node

    def parse_logical_or(self):
        node = self.parse_logical_and()
        while True:
            if self.match('BARBAR'):
                right = self.parse_logical_and()
                node = BinaryOpNode(node, '||', right)
            else:
                break
        return node

    def parse_logical_and(self):
        node = self.parse_equality()
        while True:
            if self.match('AMPAMP'):
                right = self.parse_equality()
                node = BinaryOpNode(node, '&&', right)
            else:
                break
        return node

    def parse_equality(self):
        node = self.parse_relational()
        while True:
            if self.match('EQEQ'):
                node = BinaryOpNode(node, '==', self.parse_relational())
            elif self.match('NEQ'):
                node = BinaryOpNode(node, '!=', self.parse_relational())
            else:
                break
        return node

    def parse_relational(self):
        node = self.parse_shift()
        while True:
            if self.match('LT'):
                node = BinaryOpNode(node, '<', self.parse_shift())
            elif self.match('GT'):
                node = BinaryOpNode(node, '>', self.parse_shift())
            elif self.match('LTE'):
                node = BinaryOpNode(node, '<=', self.parse_shift())
            elif self.match('GTE'):
                node = BinaryOpNode(node, '>=', self.parse_shift())
            else:
                break
        return node

    def parse_shift(self):
        node = self.parse_term()
        while True:
            if self.match('LSHIFT'):
                node = BinaryOpNode(node, '<<', self.parse_term())
            elif self.match('RSHIFT'):
                node = BinaryOpNode(node, '>>', self.parse_term())
            else:
                break
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            if self.match('PLUS'):
                node = BinaryOpNode(node, '+', self.parse_factor())
            elif self.match('MINUS'):
                node = BinaryOpNode(node, '-', self.parse_factor())
            else:
                break
        return node

    def parse_factor(self):
        node = self.parse_exponent()
        while True:
            if self.match('STAR'):
                node = BinaryOpNode(node, '*', self.parse_exponent())
            elif self.match('SLASH'):
                node = BinaryOpNode(node, '/', self.parse_exponent())
            elif self.match('MOD'):
                node = BinaryOpNode(node, '%', self.parse_exponent())
            else:
                break
        return node

    def parse_exponent(self):
        node = self.parse_unary()
        while self.match('DBLSTAR'):
            node = BinaryOpNode(node, '**', self.parse_unary())
        return node

    def parse_unary(self):
        if self.match('BANG'):
            return UnaryOpNode('!', self.parse_unary())
        if self.match('MINUS'):
            return UnaryOpNode('-', self.parse_unary())
        if self.match('PLUS'):
            return UnaryOpNode('+', self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.match('DOT'):
                # obj.field
                field = self.expect('IDENTIFIER').value
                node = MemberAccessNode(node, field)
            elif self.match('LBRACK'):
                # arr[expr]
                idx_expr = self.parse_expression()
                self.expect('RBRACK')
                node = IndexAccessNode(node, idx_expr)
            elif self.match('LPAREN'):
                # call
                args = []
                if self.current and self.current.type not in ('RPAREN', None):
                    args.append(self.parse_expression())
                    while self.match('COMMA'):
                        args.append(self.parse_expression())
                self.expect('RPAREN')
                node = CallNode(node, args)
            else:
                break
        return node

    def parse_primary(self):
        self.skip_newlines()

        if self.match('LPAREN'):
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr

        if self.match('LBRACK'):
            return self.parse_array_literal()

        if self.match('LBRACE'):
            return self.parse_dict_literal()

        if self.current and self.current.type == 'NUMBER':
            val = self.current.value
            self.advance()
            return LiteralNode(val)

        if self.current and self.current.type == 'STRING':
            val = self.current.value
            self.advance()
            return LiteralNode(val)

        # booleans/null
        if self.match('TRUE'):
            return LiteralNode(True)
        if self.match('FALSE'):
            return LiteralNode(False)
        if self.match('NULL'):
            return LiteralNode(None)

        # handle request calls
        if self.match('GET','POST','PUT','DELETE','HEAD','OPTIONS','PATCH','CONNECT','TRACE'):
            method = self.tokens[self.pos - 1].type
            url_expr = self.parse_expression()
            headers_node = None
            body_expr = None
            if self.match('HEADERS'):
                headers_node = self.parse_expression()
            if self.match('BODY'):
                body_expr = self.parse_expression()
            return RequestNode(method, url_expr, headers_node, body_expr)

        if self.current and self.current.type == 'IDENTIFIER':
            name = self.current.value
            self.advance()
            return IdentifierNode(name)

        raise Exception(f"Unexpected token {self.current}")

    def parse_array_literal(self):
        elements = []
        while self.current and self.current.type != 'RBRACK':
            elem = self.parse_expression()
            elements.append(elem)
            if self.match('COMMA'):
                continue
            else:
                break
        self.expect('RBRACK')
        return ArrayLiteralNode(elements)

    def parse_dict_literal(self):
        pairs = []
        while True:
            self.skip_newlines()
            if self.match('RBRACE'):
                break
            key_expr = self.parse_expression()
            self.expect('COLON')
            val_expr = self.parse_expression()
            pairs.append((key_expr, val_expr))
            if self.match('COMMA'):
                continue
            elif self.match('RBRACE'):
                break
            else:
                self.skip_newlines()
                if not self.current or self.current.type == 'RBRACE':
                    break
                raise Exception(f"Expected comma or }} in dict literal, got {self.current}")
        return DictLiteralNode(pairs)


############################################
# parse_code entry
############################################

def parse_code(source):
    tok_list = tokenize(source)
    parser = Parser(tok_list)
    return parser.parse_program()

if __name__ == "__main__":
    sample = r'''
    // Example usage: strongly typed
    function doMath(a: int, b: float): float {
        var x: int = a + 10;
        var y: float = spawn someOtherCall();
        return b + x;  
    }

    var z: bool;
    z = doMath(2, 3.5) > 10;
    '''
    ast = parse_code(sample)
    print(ast)
