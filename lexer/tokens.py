import re
import struct

tokens = (
    # Concurrency
    'ASYNC',
    'AWAIT',
    'SPAWN',
    'CHANNEL',
    'THREAD',
    'LOCK',
    'UNLOCK',
    'JOIN',
    'YIELD',
    'SLEEP',
    'KILL',
    'DETACH',

    # HTTP
    'GET','POST','PUT','DELETE','HEAD','OPTIONS','PATCH','CONNECT','TRACE','WEBSOCKET',
    'REQUEST','RESPONSE','HEADERS','BODY','COOKIE','SESSION',

    # Java-like structure
    'FUNCTION','CLASS','VAR','PRINT','RETURN','IF','ELSE','FOR','WHILE','BREAK','CONTINUE',
    'TRY','CATCH','FINALLY','THROW',

    # JSON-related
    'TRUE','FALSE','NULL',

    # Core lexical tokens
    'IDENTIFIER','STRING','NUMBER',

    # Punctuation / operators
    'LBRACE','RBRACE','LBRACK','RBRACK','LPAREN','RPAREN','COMMA','COLON','SEMICOLON','DOT',
    'EQUALS','PLUS','MINUS','STAR','SLASH','MOD','AMPAMP','BARBAR','BANG','EQEQ','NEQ','GT',
    'LT','GTE','LTE','AMP','BAR','CARET','TILDE','LSHIFT','RSHIFT','DBLSTAR',
    'PLUSEQ','MINUSEQ','STAREQ','SLASHEQ','MODEQ',

    # Optional newline token
    'NEWLINE'
)
