import re
import struct
import traceback
from parser import parse_code


def test_parser_black_box():
    samples = [
        # 1. Minimal empty program
        (r'', "Empty program"),

        # 2. Single function, no body
        (r'function emptyFunc() {}', "Single empty function"),

        # 3. Simple async function with if/else, var, print, return
        (r'''
        async function testFunc(a, b) {
            var x = 42;
            if (b == null) {
                print "b is null";
            } else {
                print "b is not null";
            }
            return x;
        }
        ''', "Async function with if/else, var, and return"),

        # 4. Class with function that includes for loop and while loop
        (r'''
        class Demo {
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
        ''', "Class with function containing for loop and while loop"),

        # 5. Try/catch/finally with throw
        (r'''
        function tryStuff() {
            try {
                throw "Err";
            } catch(e) {
                print e;
            } finally {
                print "Cleanup";
            }
        }
        ''', "Try/catch/finally with throw"),

        # 6. Using concurrency: spawn, thread, lock, unlock, sleep
        (r'''
        function concurrencyTest() {
            spawn doTask;
            thread t1;
            lock myLock;
            unlock myLock;
            sleep 3000;
        }
        ''', "Concurrency constructs"),

        # 7. Requests with HEADERS, BODY, plus dict literal
        (r'''
        function requestTest() {
            var resp = post "https://api.example.com" HEADERS { "Auth": "ABC123" } BODY { "param": 42 };
            print resp;
        }
        ''', "HTTP request with HEADERS/BODY, dict literal"),

        # 8. Nested blocks and function declarations within a class block
        (r'''
        class NestedStuff {
            function outer() {
                function inner(a) {
                    return a + 1;
                }
                var x = inner(10);
                print x;
            }
        }
        ''', "Nested function declaration within class block"),

        # 9. Large combined script with multiple semicolons
        (r'''
        async function main() {
            var data = get "https://example.com";
            print data;;;;
            for (var i = 0; i < 5; i = i + 1) {
                if (i == 2) break;
            };;;
            function nested() {
                return 123;
            };;;
        }

        function otherStuff() {
            while (true) {
                break;
            };
        }
        ''', "Multiple statements with extra semicolons everywhere")
    ]

    for i, (code, desc) in enumerate(samples, start=1):
        print(f"\n--- Black-Box Test #{i}: {desc} ---")
        try:
            ast_root = parse_code(code)
            print("Parsed AST:", ast_root)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


def test_parser_white_box():
    # Carefully crafted inputs that target specific grammar branches/edge cases

    # 1. Zero or multiple parameters in function
    code1 = r'''
    function noParams() {}
    function multiParams(a,b,c) {
        return a + b + c;
    }
    '''

    # 2. Complex expression with nested unary ops and exponent
    code2 = r'''
    function mathHeavy() {
        var x = !!(3 ** -2);
        var y = -(x ** 2);
        return y;
    }
    '''

    # 3. For-loop initialization as expression instead of var
    code3 = r'''
    function forAlt() {
        for (1+2; true; i = i+1) {}
    }
    '''

    # 4. Handling optional block semicolons
    code4 = r'''
    function semicolonBlocks() {
        {
            print "Inside block";
        };
        {}
    }
    '''

    # 5. Checking nested tries
    code5 = r'''
    function nestedTries() {
        try {
            try {
                throw "Deep error";
            } catch(inner) {
                print inner;
            }
        } catch(outer) {
            print outer;
        }
    }
    '''

    # 6. Request with HEADERS but no BODY
    code6 = r'''
    function partialRequest() {
        var r = get "https://site.com" HEADERS { "UA": "Bot" };
        print r;
    }
    '''

    # 7. Function calls with multiple arguments + nested calls
    code7 = r'''
    function callTest() {
        var res = doSomething(1, 2, nestedCall(3,4));
        print res;
    }
    '''

    # 8. Channel, thread usage in block
    code8 = r'''
    function concurrency2() {
        channel c1;
        thread th;
    }
    '''

    # 9. Large dictionary with trailing commas
    code9 = r'''
    function dictTrailing() {
        var data = { 
            "k1": 1,
            "k2": 2,
            "k3": "str",
        };
        print data;
    }
    '''

    test_cases = [
        (code1, "No or multiple params in function"),
        (code2, "Complex expression with unary ops and exponent"),
        (code3, "For-loop initialization as expression"),
        (code4, "Optional semicolon after block"),
        (code5, "Nested tries"),
        (code6, "Request HEADERS but no BODY"),
        (code7, "Nested function calls in arguments"),
        (code8, "Channel & thread usage in block"),
        (code9, "Trailing comma in dictionary literal")
    ]

    for i, (code, desc) in enumerate(test_cases, start=1):
        print(f"\n--- White-Box Test #{i}: {desc} ---")
        try:
            ast_root = parse_code(code)
            print("Parsed AST:", ast_root)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


def main():
    print("==== BLACK-BOX TESTS ====")
    test_parser_black_box()
    print("\n==== WHITE-BOX TESTS ====")
    test_parser_white_box()


if __name__ == "__main__":
    main()
