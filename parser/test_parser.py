import traceback
from parser import parse_code

def test_parser_black_box():
    """
    Black-box tests: general usage scenarios
    a typical user might write. This covers:
      - typed variables/functions
      - concurrency features (spawn, thread, lock, etc.)
      - HTTP requests with HEADERS, BODY
      - JSON-like dictionaries/arrays
      - conditionals, loops, etc.
    Each test case is fairly "normal" usage.
    """
    test_cases = [
        # 1: Minimal empty program
        (r'', "Empty input"),

        # 2: Simple typed function with var, if/else, return
        (r'''
        function testFunc(a: int, b: int): int {
            var x: int = 10;
            if (a == b) {
                return a;
            } else {
                return b;
            }
        }
        ''', "Typed function with if/else, var, return"),

        # 3: Class with a typed function and loops
        (r'''
        class Demo {
            function loopStuff(): void {
                for (var i: int = 0; i < 5; i = i + 1) {
                    if (i == 2) break;
                }
                var j: int = 0;
                while (j < 3) {
                    j = j + 1;
                }
            }
        }
        ''', "Class with function, typed for/while loops"),

        # 4: Concurrency usage (spawn expression, thread, kill, detach, yield)
        (r'''
        function concurrencyTest(): void {
            var handle = spawn someTask();
            thread t1 {
                print "Thread block";
            };
            kill t1;
            detach t1;
            yield;
        }
        ''', "Concurrency: spawn as expr, thread with block, kill, detach, yield"),

        # 5: Lock/unlock, channel, sleep
        (r'''
        function syncTest(): void {
            channel c1;
            lock myLock;
            unlock myLock;
            sleep 2000;
        }
        ''', "Channel, lock/unlock, sleep"),

        # 6: HTTP requests with HEADERS, BODY, dictionary
        (r'''
        function requestStuff(): void {
            var resp = post "https://api.example.com" 
                         HEADERS { "Authorization": "Bearer xyz" }
                         BODY { "param": 42 };
            print resp;
        }
        ''', "HTTP request with dictionary HEADERS/BODY"),

        # 7: Arrays, dictionaries, booleans, null
        (r'''
        function dataStruct(): void {
            var data: any = {
                "list": [1, 2, 3, null],
                "flag": true,
                "nested": { "sub": false }
            };
            print data.list[2];
        }
        ''', "JSON arrays/objects, booleans, null"),

        # 8: Dot/index chaining
        (r'''
        function chainTest(): void {
            var obj = { "inner": [10,20,30] };
            print obj.inner[1];
            print obj.inner[1].toString();
        }
        ''', "Chained postfix: obj.inner[1].method()"),

        # 9: Try/catch/finally, throw
        (r'''
        function errorStuff(): void {
            try {
                throw "SomethingWrong";
            } catch(e) {
                print e;
            } finally {
                print "Cleanup";
            }
        }
        ''', "Try/catch/finally plus throw"),

        # 10: Combining concurrency + JSON + request + typed usage
        (r'''
        async function main(): void {
            var data = get "https://example.com" HEADERS { "User-Agent": "eLangBot" };
            if (!data) {
                throw "No data!";
            }
            thread tWorker {
                var arr: any = [1, 2, { "inside": true }];
                print arr[2].inside;
            };
            detach tWorker;
        }
        ''', "Complex usage: async, get request, concurrency, array/dict usage")
    ]

    for i, (code, desc) in enumerate(test_cases, start=1):
        print(f"\n=== BLACK-BOX Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            print("Parsed AST:", ast)
        except Exception as e:
            print(f"ERROR in test #{i}: {e}")
            traceback.print_exc()


def test_parser_white_box():
    """
    White-box tests: specifically crafted inputs
    that stress tricky grammar edges, typed usage,
    concurrency combos, etc.
    """
    test_cases = [
        # 1: Function with no params vs multiple typed params
        (r'''
        function noParams(): void {}
        function multiParams(a: int, b: string, c: bool): int {
            return a;
        }
        ''', "No vs multiple typed params, typed return"),

        # 2: Complex expression mixing unary ops, exponent, indexing, type usage
        (r'''
        function mathExpr(): float {
            var x: bool = !!(3 ** -2);   // double bang, exponent with unary minus
            var y = x ** 2;             // intentionally weird: x is bool, exponent?
            print y[0];  // force index on y?
            return 0.0;
        }
        ''', "Mixed unary, exponent, indexing, typing"),

        # 3: For loop with expression initialization (no var), typed mismatch
        (r'''
        function forAlt(): void {
            var x: float = 0.0;
            for (x = 0; x < 10; x = x + 1) {
                if (x == 5) break;
            }
        }
        ''', "For initialization as assignment expr, typed mismatch?"),

        # 4: Nested concurrency: thread within thread
        (r'''
        function nestedThreads(): void {
            thread outer {
                thread innerThread;
            };
        }
        ''', "Nested concurrency constructs"),

        # 5: Inline function declaration in a block
        (r'''
        {
            function innerFn(a: int): int {
                return a+1;
            }
            var res = innerFn(10);
            print res;
        }
        ''', "Function inside block with typed param/return"),

        # 6: Array/dict with trailing commas
        (r'''
        function trailingCommas(): void {
            var arr = [1,2,3,];
            var obj = { "k1": 1, "k2": 2, };
        }
        ''', "Trailing commas in array/dict"),

        # 7: Dictionary keys that are expressions (like [keyName]) vs. strings
        (r'''
        function oddKeys(): void {
            var keyName: string = "foo";
            var obj = {
                [keyName]: 123,
                "bar": false
            };
            print obj[keyName];
        }
        ''', "Dictionary keys as expressions and strings"),

        # 8: Deeply chained postfix combos (dot chaining vs calls vs index)
        (r'''
        function chainCombos(): void {
            var data = { 
                "arr": [ { "something": (x) => x+42 }, [0,1,2], 99 ]
            };
            data.arr[0].something(42).nested[2].finalCall();
        }
        ''', "Deeply chained postfix combos: calls, indexing, dot"),

        # 9: async + spawn + kill + typed concurrency usage
        (r'''
        async function concurrencyAwaitTest(): void {
            var result: any = spawn doWork();
            await result;
            kill result;
        }
        ''', "Await + spawn + kill in same function, typed var"),

        # 10: Syntax pitfalls like empty dict, empty array, typed var with no init
        (r'''
        function emptyStructures(): void {
            var d: any = {};
            var a: any = [];
            var nested = [ {}, [] ];
            var x: int;
        }
        ''', "Empty dict & array, nested empty, typed var with no init")
    ]

    for i, (code, desc) in enumerate(test_cases, start=1):
        print(f"\n=== WHITE-BOX Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            print("Parsed AST:", ast)
        except Exception as e:
            print(f"ERROR in test #{i}: {e}")
            traceback.print_exc()


def main():
    print("======== BLACK-BOX TESTS ========")
    test_parser_black_box()
    print("\n======== WHITE-BOX TESTS ========")
    test_parser_white_box()


if __name__ == "__main__":
    main()
