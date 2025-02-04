import re
import struct
from lexer.lexer import tokenize

class ASTNode: pass

class ProgramNode(ASTNode):
    def __init__(self, body):
        self.body = body

class FunctionNode(ASTNode):
    def __init__(self, name, params, body, is_async=False):
        self.name = name
        self.params = params
        self.body = body
        self.is_async = is_async

class ClassNode(ASTNode):
    def __init__(self, name, body):
        self.name = name
        self.body = body

class VarDeclNode(ASTNode):
    def __init__(self, var_name, init_expr):
        self.var_name = var_name
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
    def __init__(self, thread_name):
        self.thread_name = thread_name

class LockNode(ASTNode):
    def __init__(self, lock_var):
        self.lock_var = lock_var

class UnlockNode(ASTNode):
    def __init__(self, lock_var):
        self.lock_var = lock_var

class SleepNode(ASTNode):
    def __init__(self, duration_expr):
        self.duration_expr = duration_expr

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

class PrintStmtNode(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class CallNode(ASTNode):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

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

    def parse_program(self):
        body = []
        while self.current:
            self.skip_newlines()
            if not self.current:
                break
            stmt = self.parse_declaration_or_statement()
            body.append(stmt)
            self.skip_extra_semicolons()
        return ProgramNode(body)

    def skip_extra_semicolons(self):
        while self.match('SEMICOLON'):
            pass

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

    def parse_class_decl(self):
        name_tok = self.expect('IDENTIFIER')
        body = self.parse_block()
        return ClassNode(name_tok.value, body)

    def parse_function_decl(self, is_async=False):
        name_tok = self.expect('IDENTIFIER')
        self.expect('LPAREN')
        params = []
        if self.current and self.current.type not in ('RPAREN', None):
            params.append(self.expect('IDENTIFIER').value)
            while self.match('COMMA'):
                params.append(self.expect('IDENTIFIER').value)
        self.expect('RPAREN')
        block = self.parse_block()
        return FunctionNode(name_tok.value, params, block, is_async=is_async)

    def parse_block(self):
        self.expect('LBRACE')
        stmts = []
        while self.current and self.current.type != 'RBRACE':
            stmt = self.parse_block_declaration_or_statement()
            stmts.append(stmt)
            self.skip_extra_semicolons()
        self.expect('RBRACE')
        return BlockNode(stmts)

    def parse_block_declaration_or_statement(self):
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

        # parse optional blocks, then skip semicolons
        if self.match('LBRACE'):
            self.pos -= 1
            self.current = self.tokens[self.pos]
            block_node = self.parse_block()
            return block_node

        if self.match('SPAWN'):
            return self.parse_spawn_stmt()
        if self.match('CHANNEL'):
            return self.parse_channel_stmt()
        if self.match('THREAD'):
            return self.parse_thread_stmt()
        if self.match('LOCK'):
            return self.parse_lock_stmt()
        if self.match('UNLOCK'):
            return self.parse_unlock_stmt()
        if self.match('SLEEP'):
            return self.parse_sleep_stmt()
        if self.match('PRINT'):
            return self.parse_print_stmt()

        return self.parse_expr_statement()

    def parse_var_decl(self):
        var_name = self.expect('IDENTIFIER').value
        init_expr = None
        if self.match('EQUALS'):
            init_expr = self.parse_expression()
        self.expect('SEMICOLON')
        return VarDeclNode(var_name, init_expr)

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
            var_name = self.expect('IDENTIFIER').value
            init_expr = None
            if self.match('EQUALS'):
                init_expr = self.parse_expression()
            return VarDeclNode(var_name, init_expr)
        elif self.current and self.current.type != 'SEMICOLON':
            return self.parse_expression()
        return None

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

    def parse_spawn_stmt(self):
        expr = self.parse_expression()
        self.expect('SEMICOLON')
        return SpawnNode(expr)

    def parse_channel_stmt(self):
        chan_name = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return ChannelNode(chan_name)

    def parse_thread_stmt(self):
        thread_name = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return ThreadNode(thread_name)

    def parse_lock_stmt(self):
        lock_var = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return LockNode(lock_var)

    def parse_unlock_stmt(self):
        lock_var = self.expect('IDENTIFIER').value
        self.expect('SEMICOLON')
        return UnlockNode(lock_var)

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

    def parse_expr_statement(self):
        expr = self.parse_expression()
        self.expect('SEMICOLON')
        return ExprStmtNode(expr)

    def parse_expression(self):
        self.skip_newlines()
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
            if self.match('LPAREN'):
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

def parse_code(source):
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()

if __name__ == "__main__":
    # Quick test
    code = r'''
    class DemoClass {
        function loopStuff() {
            for (var i = 0; i < 10; i = i + 1) {
                if (i == 5) break;
            };
            var j = 0;
            while (j < 3) {
                j = j + 1;
            };
        }
    }
    '''
    ast = parse_code(code)
    print(ast)
