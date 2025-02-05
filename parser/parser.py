import re
import struct
from lexer.lexer import tokenize

class ASTNode: pass

class ProgramNode(ASTNode):
    def __init__(self, body):
        self.body = body

class FunctionNode(ASTNode):
    def __init__(self, name, params, return_type, body, is_async=False):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.is_async = is_async

class ClassNode(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

class VarDeclNode(ASTNode):
    def __init__(self, var_name, var_type, init_expr):
        self.var_name = var_name
        self.var_type = var_type
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

class BreakNode(ASTNode): pass
class ContinueNode(ASTNode): pass

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

class YieldNode(ASTNode): pass

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

class ArrowFunctionNode(ASTNode):
    def __init__(self, params, body, is_block=False):
        self.params = params
        self.body = body
        self.is_block = is_block

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
            t = self.current
            self.advance()
            return t
        return None

    def match_arrow(self):
        self.skip_newlines()
        if self.current and self.current.type == 'EQUALS':
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'GT':
                self.advance()
                self.advance()
                return True
        return False

    def expect(self, ttype):
        self.skip_newlines()
        if not self.current or self.current.type != ttype:
            raise Exception("Expected " + str(ttype) + ", got " + str(self.current))
        v = self.current
        self.advance()
        return v

    def skip_newlines(self):
        while self.current and self.current.type == 'NEWLINE':
            self.advance()

    def skip_extra_semicolons(self):
        while self.match('SEMICOLON'):
            pass

    def parse_program(self):
        b = []
        while self.current:
            self.skip_newlines()
            if not self.current:
                break
            n = self.parse_declaration_or_statement()
            b.append(n)
            self.skip_extra_semicolons()
        return ProgramNode(b)

    def parse_declaration_or_statement(self):
        self.skip_newlines()
        if self.match('ASYNC'):
            if self.match('FUNCTION'):
                f = self.parse_function_decl(True)
                self.skip_extra_semicolons()
                return f
            self.pos -= 1
            self.current = self.tokens[self.pos]
            s = self.parse_statement()
            self.skip_extra_semicolons()
            return s
        if self.match('FUNCTION'):
            f = self.parse_function_decl(False)
            self.skip_extra_semicolons()
            return f
        if self.match('CLASS'):
            c = self.parse_class_decl()
            self.skip_extra_semicolons()
            return c
        s = self.parse_statement()
        self.skip_extra_semicolons()
        return s

    def parse_function_decl(self, is_async):
        n = self.expect('IDENTIFIER')
        self.expect('LPAREN')
        ps = []
        if self.current and self.current.type not in ('RPAREN', None):
            ps.append(self.parse_typed_param())
            while self.match('COMMA'):
                ps.append(self.parse_typed_param())
        self.expect('RPAREN')
        rt = None
        if self.match('COLON'):
            rt = self.parse_type()
        b = self.parse_block()
        return FunctionNode(n.value, ps, rt, b, is_async)

    def parse_typed_param(self):
        i = self.expect('IDENTIFIER')
        t = None
        if self.match('COLON'):
            t = self.parse_type()
        return (i.value, t)

    def parse_type(self):
        if not self.current or self.current.type != 'IDENTIFIER':
            raise Exception("Expected a type name, got " + str(self.current))
        t = self.current.value
        self.advance()
        return t

    def parse_class_decl(self):
        n = self.expect('IDENTIFIER')
        b = self.parse_block()
        return ClassNode(n.value, b)

    def parse_block(self):
        self.expect('LBRACE')
        s = []
        while self.current and self.current.type != 'RBRACE':
            d = self.parse_block_decl_or_stmt()
            s.append(d)
            self.skip_extra_semicolons()
        self.expect('RBRACE')
        return BlockNode(s)

    def parse_block_decl_or_stmt(self):
        self.skip_newlines()
        if self.match('ASYNC'):
            if self.match('FUNCTION'):
                f = self.parse_function_decl(True)
                self.skip_extra_semicolons()
                return f
            self.pos -= 1
            self.current = self.tokens[self.pos]
            st = self.parse_statement()
            self.skip_extra_semicolons()
            return st
        if self.match('FUNCTION'):
            f = self.parse_function_decl(False)
            self.skip_extra_semicolons()
            return f
        st = self.parse_statement()
        self.skip_extra_semicolons()
        return st

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
        if self.match('LBRACE'):
            self.pos -= 1
            self.current = self.tokens[self.pos]
            return self.parse_block()
        return self.parse_expr_statement()

    def parse_var_decl(self):
        n = self.expect('IDENTIFIER').value
        t = None
        if self.match('COLON'):
            t = self.parse_type()
        i = None
        if self.match('EQUALS'):
            i = self.parse_expression()
        self.expect('SEMICOLON')
        return VarDeclNode(n, t, i)

    def parse_if_stmt(self):
        self.expect('LPAREN')
        c = self.parse_expression()
        self.expect('RPAREN')
        th = self.parse_statement()
        el = None
        if self.match('ELSE'):
            el = self.parse_statement()
        return IfNode(c, th, el)

    def parse_while_stmt(self):
        self.expect('LPAREN')
        c = self.parse_expression()
        self.expect('RPAREN')
        b = self.parse_statement()
        return WhileNode(c, b)

    def parse_for_stmt(self):
        self.expect('LPAREN')
        init = self.parse_for_init()
        self.expect('SEMICOLON')
        c = None
        if self.current and self.current.type != 'SEMICOLON':
            c = self.parse_expression()
        self.expect('SEMICOLON')
        inc = None
        if self.current and self.current.type != 'RPAREN':
            inc = self.parse_expression()
        self.expect('RPAREN')
        b = self.parse_statement()
        return ForNode(init, c, inc, b)

    def parse_for_init(self):
        if self.match('VAR'):
            return self.parse_var_decl_no_semicolon()
        elif self.current and self.current.type != 'SEMICOLON':
            return self.parse_expression()
        return None

    def parse_var_decl_no_semicolon(self):
        n = self.expect('IDENTIFIER').value
        t = None
        if self.match('COLON'):
            t = self.parse_type()
        i = None
        if self.match('EQUALS'):
            i = self.parse_expression()
        return VarDeclNode(n, t, i)

    def parse_try_stmt(self):
        tb = self.parse_block()
        cv = None
        cb = None
        fb = None
        if self.match('CATCH'):
            self.expect('LPAREN')
            cv = self.expect('IDENTIFIER').value
            self.expect('RPAREN')
            cb = self.parse_block()
        if self.match('FINALLY'):
            fb = self.parse_block()
        return TryNode(tb, cv, cb, fb)

    def parse_throw_stmt(self):
        e = self.parse_expression()
        self.expect('SEMICOLON')
        return ThrowNode(e)

    def parse_return_stmt(self):
        e = None
        if self.current and self.current.type not in ('SEMICOLON','RBRACE'):
            e = self.parse_expression()
        self.expect('SEMICOLON')
        return ReturnNode(e)

    def parse_channel_stmt(self):
        c = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return ChannelNode(c)

    def parse_thread_decl(self):
        t = self.expect('IDENTIFIER').value
        b = None
        if self.match('LBRACE'):
            stmts = []
            while self.current and self.current.type != 'RBRACE':
                s = self.parse_block_decl_or_stmt()
                stmts.append(s)
                self.skip_extra_semicolons()
            self.expect('RBRACE')
            b = BlockNode(stmts)
        self.expect('SEMICOLON')
        return ThreadNode(t, b)

    def parse_lock_stmt(self):
        v = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return LockNode(v)

    def parse_unlock_stmt(self):
        v = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return UnlockNode(v)

    def parse_join_stmt(self):
        t = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return JoinNode(t)

    def parse_kill_stmt(self):
        t = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return KillNode(t)

    def parse_detach_stmt(self):
        t = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return DetachNode(t)

    def parse_sleep_stmt(self):
        d = self.parse_expression()
        self.expect('SEMICOLON')
        return SleepNode(d)

    def parse_print_stmt(self):
        e = None
        if self.current and self.current.type not in ('SEMICOLON','RBRACE'):
            e = self.parse_expression()
        self.expect('SEMICOLON')
        return PrintStmtNode(e)

    def parse_expr_statement(self):
        e = self.parse_expression()
        self.expect('SEMICOLON')
        return ExprStmtNode(e)

    def parse_expression(self):
        self.skip_newlines()
        if self.match('AWAIT'):
            i = self.parse_expression()
            return AwaitNode(i)
        if self.match('SPAWN'):
            i = self.parse_expression()
            return SpawnNode(i)
        return self.parse_assignment()

    def parse_assignment(self):
        n = self.parse_logical_or()
        if self.match('EQUALS'):
            r = self.parse_assignment()
            if isinstance(n, IdentifierNode) or isinstance(n, IndexAccessNode) or isinstance(n, MemberAccessNode):
                return AssignNode(n, r)
            raise Exception("Left side of assignment is not an identifier/index/member")
        return n

    def parse_logical_or(self):
        n = self.parse_logical_and()
        while True:
            if self.match('BARBAR'):
                r = self.parse_logical_and()
                n = BinaryOpNode(n, '||', r)
            else:
                break
        return n

    def parse_logical_and(self):
        n = self.parse_equality()
        while True:
            if self.match('AMPAMP'):
                r = self.parse_equality()
                n = BinaryOpNode(n, '&&', r)
            else:
                break
        return n

    def parse_equality(self):
        n = self.parse_relational()
        while True:
            if self.match('EQEQ'):
                n = BinaryOpNode(n, '==', self.parse_relational())
            elif self.match('NEQ'):
                n = BinaryOpNode(n, '!=', self.parse_relational())
            else:
                break
        return n

    def parse_relational(self):
        n = self.parse_shift()
        while True:
            if self.match('LT'):
                n = BinaryOpNode(n, '<', self.parse_shift())
            elif self.match('GT'):
                n = BinaryOpNode(n, '>', self.parse_shift())
            elif self.match('LTE'):
                n = BinaryOpNode(n, '<=', self.parse_shift())
            elif self.match('GTE'):
                n = BinaryOpNode(n, '>=', self.parse_shift())
            else:
                break
        return n

    def parse_shift(self):
        n = self.parse_term()
        while True:
            if self.match('LSHIFT'):
                n = BinaryOpNode(n, '<<', self.parse_term())
            elif self.match('RSHIFT'):
                n = BinaryOpNode(n, '>>', self.parse_term())
            else:
                break
        return n

    def parse_term(self):
        n = self.parse_factor()
        while True:
            if self.match('PLUS'):
                n = BinaryOpNode(n, '+', self.parse_factor())
            elif self.match('MINUS'):
                n = BinaryOpNode(n, '-', self.parse_factor())
            else:
                break
        return n

    def parse_factor(self):
        n = self.parse_exponent()
        while True:
            if self.match('STAR'):
                n = BinaryOpNode(n, '*', self.parse_exponent())
            elif self.match('SLASH'):
                n = BinaryOpNode(n, '/', self.parse_exponent())
            elif self.match('MOD'):
                n = BinaryOpNode(n, '%', self.parse_exponent())
            else:
                break
        return n

    def parse_exponent(self):
        n = self.parse_unary()
        while self.match('DBLSTAR'):
            n = BinaryOpNode(n, '**', self.parse_unary())
        return n

    def parse_unary(self):
        if self.match('BANG'):
            return UnaryOpNode('!', self.parse_unary())
        if self.match('MINUS'):
            return UnaryOpNode('-', self.parse_unary())
        if self.match('PLUS'):
            return UnaryOpNode('+', self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        n = self.parse_primary()
        while True:
            if self.match('DOT'):
                f = self.expect('IDENTIFIER').value
                n = MemberAccessNode(n, f)
            elif self.match('LBRACK'):
                i = self.parse_expression()
                self.expect('RBRACK')
                n = IndexAccessNode(n, i)
            elif self.match('LPAREN'):
                a = []
                if self.current and self.current.type not in ('RPAREN', None):
                    a.append(self.parse_expression())
                    while self.match('COMMA'):
                        a.append(self.parse_expression())
                self.expect('RPAREN')
                n = CallNode(n, a)
            else:
                break
        return n

    def parse_primary(self):
        self.skip_newlines()
        if self.match('LPAREN'):
            return self.parse_arrow_or_parenthesized()
        if self.match('LBRACK'):
            return self.parse_array_literal()
        if self.match('LBRACE'):
            return self.parse_dict_literal()
        if self.current and self.current.type == 'NUMBER':
            v = self.current.value
            self.advance()
            if isinstance(v, int):
                return LiteralNode(v)
            return LiteralNode(float(v))
        if self.current and self.current.type == 'STRING':
            v = self.current.value
            self.advance()
            return LiteralNode(v)
        if self.match('TRUE'):
            return LiteralNode(True)
        if self.match('FALSE'):
            return LiteralNode(False)
        if self.match('NULL'):
            return LiteralNode(None)
        if self.match('GET','POST','PUT','DELETE','HEAD','OPTIONS','PATCH','CONNECT','TRACE'):
            m = self.tokens[self.pos - 1].type
            u = self.parse_expression()
            h = None
            b = None
            if self.match('HEADERS'):
                h = self.parse_expression()
            if self.match('BODY'):
                b = self.parse_expression()
            return RequestNode(m, u, h, b)
        if self.current and self.current.type == 'IDENTIFIER':
            n = self.current.value
            self.advance()
            return IdentifierNode(n)
        raise Exception("Unexpected token " + str(self.current))

    def parse_arrow_or_parenthesized(self):
        if self.match('RPAREN'):
            if self.match_arrow():
                return self.finish_arrow_function([])
            raise Exception("Empty parentheses without arrow is invalid")
        sp = self.pos
        first_id = self.match('IDENTIFIER')
        if not first_id:
            self.pos = sp
            self.current = self.tokens[self.pos]
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        params = [first_id.value]
        while self.match('COMMA'):
            nxt = self.match('IDENTIFIER')
            if not nxt:
                self.pos = sp
                self.current = self.tokens[self.pos]
                expr = self.parse_expression()
                self.expect('RPAREN')
                return expr
            params.append(nxt.value)
        if not self.match('RPAREN'):
            self.pos = sp
            self.current = self.tokens[self.pos]
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        if self.match_arrow():
            return self.finish_arrow_function(params)
        self.pos = sp
        self.current = self.tokens[self.pos]
        expr = self.parse_expression()
        self.expect('RPAREN')
        return expr

    def finish_arrow_function(self, p):
        if self.match('LBRACE'):
            stmts = []
            while self.current and self.current.type != 'RBRACE':
                s = self.parse_block_decl_or_stmt()
                stmts.append(s)
                self.skip_extra_semicolons()
            self.expect('RBRACE')
            return ArrowFunctionNode(p, BlockNode(stmts), True)
        b = self.parse_expression()
        return ArrowFunctionNode(p, b, False)

    def parse_array_literal(self):
        elems = []
        while self.current and self.current.type != 'RBRACK':
            e = self.parse_expression()
            elems.append(e)
            if self.match('COMMA'):
                continue
            else:
                break
        self.expect('RBRACK')
        return ArrayLiteralNode(elems)

    def parse_dict_literal(self):
        ps = []
        while True:
            self.skip_newlines()
            if self.match('RBRACE'):
                break
            k = self.parse_expression()
            self.expect('COLON')
            v = self.parse_expression()
            ps.append((k, v))
            if self.match('COMMA'):
                continue
            elif self.match('RBRACE'):
                break
            else:
                self.skip_newlines()
                if not self.current or self.current.type == 'RBRACE':
                    break
                raise Exception("Expected comma or } in dict literal, got " + str(self.current))
        return DictLiteralNode(ps)

def parse_code(source):
    tl = tokenize(source)
    p = Parser(tl)
    return p.parse_program()
