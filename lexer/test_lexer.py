from lexer import tokenize


def test_lexer():
    code = r'''
    // single-line comment
    /* multi
       line
       comment */
    # also a hash comment

    async function DemoClass() {
        var hexVal = 0xFF;
        var floatVal = 3.14;
        var bigExp = 1.2e+10;
        var triple = """multi
line
string""";
        var single = "hello\nworld";
        spawn thread t1
        get "https://example.com"
        /* nested comment */
        if (t1 != null) {
            print t1;
        }
    }
    '''
    tokens_list = tokenize(code)
    for tok in tokens_list:
        print(tok)


if __name__ == "__main__":
    test_lexer()
