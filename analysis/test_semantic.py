import traceback
from parser.parser import parse_code
from analysis.semantic import analyze_semantics, SemanticError

def test_semantic_black_box():
    cases = [
        (
            """
            var x: int = 10;
            var y: float = 2.5;
            function test(a: int, b: float): float {
                var z: float = a + b;
                return z;
            }
            """,
            "Simple valid usage, typed variables/functions",
            False
        ),
        (
            """
            function main(): void {
                var s: string = "Hello";
                if (s == "Hello") {
                    print s;
                }
            }
            """,
            "If statement, string usage",
            False
        ),
        (
            """
            var fl = 2.2;
            fl = fl + 1.0;
            """,
            "Implicit any typed usage, valid float arithmetic",
            False
        ),
        (
            """
            var x: bool = true;
            spawn x;
            """,
            "Spawn expression, referencing a bool",
            False
        ),
        (
            """
            function nonsense(a: bool, b: float): bool {
                return a && (b > 2.0);
            }
            """,
            "Binary ops, bool and float usage",
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
            var x: float;
            x = true;
            """,
            "Invalid assignment: bool to float",
            True
        ),
        (
            """
            var flag: bool = false;
            flag = 5;
            """,
            "Invalid assignment: number to bool",
            True
        ),
        (
            """
            kill threadNotDeclared;
            """,
            "Killing an undeclared identifier",
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
                # Do not print traceback if it was expected
            else:
                print(f"FAIL: Unexpected SemanticError: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"FAIL: Unexpected exception: {e}")
            traceback.print_exc()

def test_semantic_white_box():
    cases = [
        (
            """
            var a: int = 1;
            var b: int = a + 2;
            b = b + 3;
            """,
            "Chained arithmetic, int usage",
            False
        ),
        (
            """
            function testFun(a: int, b: int): bool {
                return a < b;
            }
            """,
            "Function with return bool, relational op",
            False
        ),
        (
            """
            function weirdReturn(): float {
                return "text";
            }
            """,
            "Returning string in function typed float",
            True
        ),
        (
            """
            function arrowTest(): void {
                var f = (x) => x+1;
                var y = f(10);
            }
            """,
            "Arrow function with int usage",
            False
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
            "Function inside block, string usage",
            False
        ),
        (
            """
            var arr = [1, 2, false];
            arr[1] = "test";
            """,
            "Array with mismatched types, but typed any by default, no error",
            False
        ),
        (
            """
            var dict = {
                "key": 123,
                "val": true
            };
            dict["key"] = false;
            """,
            "Dict usage, also typed any by default, no error",
            False
        ),
        (
            """
            var done: bool;
            done = done && 1;
            """,
            "bool && int mismatch",
            True
        ),
        (
            """
            var notDeclared;
            notDeclared = missingOne;
            """,
            "Assign from undeclared var",
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
            "Accessing localVar outside thread scope",
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
                # Do not print traceback if it was expected
            else:
                print(f"FAIL: Unexpected SemanticError: {e}")
                traceback.print_exc()
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
