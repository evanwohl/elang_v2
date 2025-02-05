import ply.lex as lex
import re
import struct
from .tokens import tokens

keyword_map = {
    # concurrency
    'async':'ASYNC', 'await':'AWAIT','spawn':'SPAWN','channel':'CHANNEL','thread':'THREAD',
    'lock':'LOCK','unlock':'UNLOCK','join':'JOIN','yield':'YIELD','sleep':'SLEEP',
    'kill':'KILL','detach':'DETACH',

    # http
    'get':'GET','post':'POST','put':'PUT','delete':'DELETE','head':'HEAD','options':'OPTIONS',
    'patch':'PATCH','connect':'CONNECT','trace':'TRACE','websocket':'WEBSOCKET',
    'request':'REQUEST','response':'RESPONSE','headers':'HEADERS','body':'BODY',
    'cookie':'COOKIE','session':'SESSION',

    # structure
    'function':'FUNCTION','class':'CLASS','var':'VAR','print':'PRINT','return':'RETURN',
    'if':'IF','else':'ELSE','for':'FOR','while':'WHILE','break':'BREAK','continue':'CONTINUE',
    'try':'TRY','catch':'CATCH','finally':'FINALLY','throw':'THROW',

    # json booleans/null
    'true':'TRUE','false':'FALSE','null':'NULL'
}

t_ignore = ' \t\x0c'

t_LBRACE     = r'\{'
t_RBRACE     = r'\}'
t_LBRACK     = r'\['
t_RBRACK     = r'\]'
t_LPAREN     = r'\('
t_RPAREN     = r'\)'
t_COMMA      = r','
t_COLON      = r':'
t_SEMICOLON  = r';'
t_DOT        = r'\.'
t_EQUALS     = r'='
t_PLUS       = r'\+'
t_MINUS      = r'-'
t_STAR       = r'\*'
t_SLASH      = r'/'
t_MOD        = r'%'
t_AMPAMP     = r'&&'
t_BARBAR     = r'\|\|'
t_BANG       = r'!'
t_EQEQ       = r'=='
t_NEQ        = r'!='
t_GT         = r'>'
t_LT         = r'<'
t_GTE        = r'>='
t_LTE        = r'<='

t_AMP        = r'&'
t_BAR        = r'\|'
t_CARET      = r'\^'
t_TILDE      = r'~'
t_LSHIFT     = r'<<'
t_RSHIFT     = r'>>'
t_DBLSTAR    = r'\*\*'

t_PLUSEQ     = r'\+='
t_MINUSEQ    = r'-='
t_STAREQ     = r'\*='
t_SLASHEQ    = r'/='
t_MODEQ      = r'%='

def t_STRING(t):
    r'("([^"\\]|\\.)*")|(\'([^\'\\]|\\.)*\')'
    raw = t.value
    if raw.startswith('"'):
        inner = raw[1:-1]
    else:
        inner = raw[1:-1]
    t.value = _unescape_string(inner)
    return t

def _unescape_string(s):
    return (s
        .replace(r'\"', '"')
        .replace(r"\'", "'")
        .replace(r'\\', '\\')
        .replace(r'\n', '\n')
        .replace(r'\t', '\t')
        .replace(r'\r', '\r'))

def t_NUMBER(t):
    r'(\d+(\.\d+)?([eE][+\-]?\d+)?)|(0[xX][0-9A-Fa-f]+)'
    val = t.value
    if val.lower().startswith('0x'):
        t.value = int(val, 16)
    else:
        if '.' in val or 'e' in val.lower():
            t.value = float(val)
        else:
            t.value = int(val)
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    lower_val = t.value.lower()
    if lower_val in keyword_map:
        t.type = keyword_map[lower_val]
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    t.type = 'NEWLINE'
    return t

def t_comment_singleline(t):
    r'//[^\n]*'
    pass

def t_comment_hash(t):
    r'\#[^\n]*'
    pass

def t_comment_multiline(t):
    r'/\*[\s\S]*?\*/'
    pass

def t_error(t):
    t.lexer.skip(1)

lexer = lex.lex()

def tokenize(data):
    lexer.input(data)
    tokens_list = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens_list.append(tok)
    return tokens_list
