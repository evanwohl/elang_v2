import traceback
from parser.parser import parse_code
from analysis.semantic import analyze_semantics, SemanticError

def test_semantic_black_box():
    cases = [
        (
            """
            var x: int = 10;
            var y: float = 2.5;
            function sum(a: int, b: float): float {
                return a + b;
            }
            var z: float = sum(x, y);
            """,
            "Valid typed usage, function call with matching arguments",
            False
        ),
        (
            """
            function greet(name: string): string {
                return "Hello, " + name;
            }
            var s = greet("Alice");
            """,
            "String + string arithmetic, valid return type, everything typed properly",
            False
        ),
        (
            """
            function failArgs(a: int, b: bool): void {}
            failArgs(10); 
            """,
            "Invalid call: not enough args",
            True
        ),
        (
            """
            function failArgs2(a: int, b: bool): void {}
            failArgs2(1, true, 42);
            """,
            "Invalid call: too many args",
            True
        ),
        (
            """
            function failType(a: int, b: float): float {
                return a + b;
            }
            var c: bool = failType(2, 3.0);
            """,
            "Assigning float to bool, invalid assignment",
            True
        ),
        (
            """
            var fl = 2.2;
            fl = fl + 1.0;
            var anything;
            anything = "string";
            anything = 123;
            """,
            "Implicit any usage is valid, arithmetic with floats is valid",
            False
        ),
        (
            """
            var x: bool = true;
            spawn x;
            """,
            "Spawn expression referencing a bool, fine",
            False
        ),
        (
            """
            var x: int = "NotInt";
            """,
            "Invalid assignment: string to int",
            True
        ),
        (
            """
            function nonsense(a: bool, b: float): bool {
                return a && (b > 2.0);
            }
            var r: bool = nonsense(true, 3.14);
            """,
            "Binary ops with bool and float usage is valid, correct usage",
            False
        ),
        (
            """
            kill threadNotDeclared;
            """,
            "Killing an undeclared identifier for thread",
            True
        ),
        (
            """
            var x: string;
            x = 123;
            """,
            "Int assigned to string",
            True
        ),
        (
            """
            class Simple {
                function hello(name: string): string {
                    return "Hi " + name;
                }
            }
            var c: any = Simple;
            var x = c.hello("Alice");
            print x;
            """,
            "Class with a method, partial usage (no real new call in grammar, just a test scenario)",
            False
        ),
        (
            """
            var shadow: int = 10;
            {
                var shadow: float = 2.5;
            }
            """,
            "Shadowing the same var name in nested block, we disallow => error",
            True
        ),
        (
            """
            function allPathsReturn(a: bool): int {
                if (a) {
                    return 1;
                } else {
                    return 0;
                }
            }
            """,
            "All code paths return int, valid usage",
            False
        ),
        (
            """
            function notAllPaths(a: bool): int {
                if (a) {
                    return 5;
                }
            }
            """,
            "Not all code paths return => warning or error. We consider it a warning in our code, but won't fail the test",
            False
        ),
        (
            """
            function paramMismatch(a: int, b: int): void {}
            paramMismatch("str", false);
            """,
            "Passing invalid argument types",
            True
        ),
    ]

    for i, (code, desc, should_error) in enumerate(cases, 1):
        print(f"\n=== Black-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            if should_error:
                print("FAIL: Expected error but got none.")
            else:
                print("PASS: No error, as expected.")
        except SemanticError as e:
            if should_error:
                print(f"PASS: Caught expected error: {e}")
            else:
                print(f"FAIL: Unexpected SemanticError: {e}")
            # We won't print traceback for expected errors
        except Exception as e:
            print(f"FAIL: Unexpected exception: {e}")
            traceback.print_exc()


def test_semantic_white_box():
    cases = [
        (
            """
            var arr = [1, 2, false];
            arr[1] = "test";
            """,
            "Array with mismatched types, typed as any by default, no error",
            False
        ),
        (
            """
            var dict = {
                "key": 123,
                "val": true
            };
            dict["val"] = 3.14;
            """,
            "Dict usage, also typed any by default, no error",
            False
        ),
        (
            """
            function arrowInt(): void {
                var f = (x) => x + 1;
                var y = f(10);
            }
            """,
            "Arrow function returning int, no type mismatch",
            False
        ),
        (
            """
            function arrowInvalid(): void {
                var f = (x) => x + 1;
                var s: bool = f("lol");
            }
            """,
            "Arrow function called with string, plus 1 => invalid arithmetic",
            True
        ),
        (
            """
            {
                function inner(a: string): string {
                    return a + " world";
                }
                var res = inner("hello");
            }
            """,
            "Function inside block, string usage, valid string + string => string",
            False
        ),
        (
            """
            var a: bool;
            var b = a && 1;
            """,
            "bool && int mismatch",
            True
        ),
        (
            """
            var notDeclared;
            notDeclared = missingOne;
            """,
            "Assign from undeclared var missingOne",
            True
        ),
        (
            """
            thread t1 {
                var localVar: float = 0.1;
            };
            join t1;
            localVar = 2.0;
            """,
            "Accessing localVar outside thread scope => error",
            True
        ),
        (
            """
            function multiArgCheck(a: int, b: string, c: bool): int {
                print a; print b; print c;
                return a;
            }
            multiArgCheck(42, "test", true);
            """,
            "Proper call with correct arguments, success",
            False
        ),
        (
            """
            function multiArgCheck2(a: int, b: string, c: bool): bool {
                return c;
            }
            var r: bool = multiArgCheck2(10, 999, false);
            """,
            "Wrong param type: passing int in place of string => error",
            True
        ),
        (
            """
            var unusedOne: int;
            var usedOne: float = 2.0;
            print usedOne;
            """,
            "Check unused variable => warning. Not an error, so test pass",
            False
        ),
        (
            """
            function parseUrl(u: string): string {
                if (u == null) {
                    return "https://default.example.com";
                };
                return u;
            };
            
            function handleErrors(e: any): void {
                print "Error: ";
                print e;
            };
            
            class HttpBot {
                function doRequest(method: string, endpoint: string, hdrs: any, bdy: any): any {
                    if (method == "GET") {
                        var r: any = GET endpoint HEADERS hdrs BODY bdy;
                        return r;
                    } else {
                        var p: any = POST endpoint HEADERS hdrs BODY bdy;
                        return p;
                    };
                };
            
                function runSequence(): void {
                    var i: int = 0;
                    while (i < 3) {
                        print "Sequence step:";
                        print i;
                        i = i + 1;
                    };
                };
            };
            
            channel requestChan;
            channel responseChan;
            
            var globalLock: any = null;
            
            thread networkThread {
                var count: int = 0;
                while (count < 5) {
                    lock globalLock;
                    print "networkThread iteration:";
                    print count;
                    unlock globalLock;
                    sleep 200;
                    count = count + 1;
                };
            };
            
            async function doAllRequests(): void {
                try {
                    var url: string = parseUrl(null);
                    var hdrs: any = {
                        "User-Agent": "ExampleBot",
                        "Accept": "application/json"
                    };
                    var bdy: any = {
                        "data": [1, 2, 3]
                    };
            
                    // Multiple HTTP methods:
                    var resp1: any = await (GET url HEADERS hdrs BODY null);
                    print resp1;
            
                    var resp2: any = await (POST url HEADERS hdrs BODY bdy);
                    print resp2;
            
                    var resp3: any = await (PUT (url + "/update") HEADERS hdrs BODY bdy);
                    print resp3;
            
                    var condition: bool = true;
                    if (condition) {
                        print "Conditional branch executed.";
                    } else {
                        print "Should not happen.";
                    };
                } catch (err) {
                    handleErrors(err);
                } finally {
                    print "All requests done.";
                };
            };
            
            // Demonstrates an arrow function storing a numeric result (x is "any", so result is "any"):
            var arrowFunc: any = (x) => {
                return x + 1;
            };
            
            async function main(): void {
                // Spawn the async function:
                spawn doAllRequests();
            
                print "Main is doing other work...";
            
                // A simple for loop:
                for (var i: int = 0; i < 3; i = i + 1) {
                    print i;
                };
            
                // Wait for the thread to finish:
                join networkThread;
            
                print "networkThread joined.";
                print "Main done.";
            };

            """,
            "Multiple return paths, all good, typed float, no error",
            False
        ),
        (
            """
            function missingReturn(c: bool): int {
                if (c) return 1;
                print "no return if c is false!";
            }
            """,
            "Missing return in else path => warning for not all code paths returning",
            False
        ),
        (
            """
            function overArgs(a: int): void {}
            overArgs(1, 2, 3, 4);
            """,
            "Too many arguments => error",
            True
        ),
    ]

    for i, (code, desc, should_error) in enumerate(cases, 1):
        print(f"\n=== White-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            if should_error:
                print("FAIL: Expected error but got none.")
            else:
                print("PASS: No error, as expected.")
        except SemanticError as e:
            if should_error:
                print(f"PASS: Caught expected error: {e}")
            else:
                print(f"FAIL: Unexpected SemanticError: {e}")
        except Exception as e:
            print(f"FAIL: Unexpected exception: {e}")
            traceback.print_exc()


def main():
    print("=== SEMANTIC BLACK-BOX TESTS ===")
    test_semantic_black_box()
    print("\n=== SEMANTIC WHITE-BOX TESTS ===")
    test_semantic_white_box()

if __name__ == "__main__":
    main()
