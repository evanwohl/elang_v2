import re
import struct
from lexer import tokenize


def test_lexer():
    code = r'''
    var jdata = {
      "arr": [1, 2, 3, {"inner": true}],
      "flag": false,
      "missing": null
    };
    '''
    tokens_list = tokenize(code)
    for tok in tokens_list:
        print(tok)


if __name__ == "__main__":
    test_lexer()
