import sys
import traceback
from parser.parser import parse_code
from analysis.semantic import analyze_semantics, SemanticError
from codegen.ast_to_ir import ast_to_ir
from ir_utils import (
    create_big_pass_manager,
    print_ir_module,
    PassManager,
    ConstFoldingPass,
    DeadCodeEliminationPass
)

def test_ir_black_box():
    """
    Black-Box tests: We feed relatively 'normal' or user-level code snippets
    through the entire pipeline (parser -> semantic -> IR) and check the IR output.
    """
    cases = [
        (
            """
            function sum(a:int, b:int): int {
                return a + b;
            }
            function runSum() {
                var a: int = 2;
                var b: int = 2;
                var result = sum(a, b);
                print result;  // Expect: 4
            }
            runSum();
            """,
            "1) Basic typed function + usage"
        ),
        (
            """
            function fib(n:int): int {
                if (n < 2) {
                    return n;
                }
                return fib(n-1) + fib(n-2);
            }
            function runFib() {
                print fib(5);  // Expect: 5
            }
            runFib();
            """,
            "2) Recursive function (Fibonacci)"
        ),
        (
            """
            function testWhileLoop() {
                var i:int = 0;
                while (i < 3) {
                    print i;
                    i = i + 1;
                }
            }
            testWhileLoop();
            """,
            "3) While-loop test"
        ),
        (
            """
            function testForLoop() {
                for (var i=0; i < 4; i = i + 1) {
                    print i;
                }
            }
            testForLoop();
            """,
            "4) For-loop test"
        ),
        (
            """
            function concurrencyExample() {
                spawn function() {
                    print 123;
                };
                print 456;
            }
            concurrencyExample();
            """,
            "5) Spawn concurrency test"
        ),
        (
            """
            function requestExample() {
                var url = "https://example.com";
                request("GET", url, null, null);
            }
            requestExample();
            """,
            "6) Simple Request test"
        ),
        (
            """
            function dictAndArray() {
                var d = { "x": 1, "y": 2 };
                var arr = [10, 20, 30];
                print d;
                print arr;
            }
            dictAndArray();
            """,
            "7) Dictionary/Array literal test"
        ),
        (
            """
            function arrowUsage() {
                var add = (x, y) => x + y;
                var result = add(3,4);
                print result;  // Expect 7
            }
            arrowUsage();
            """,
            "8) Arrow function usage test"
        ),
        (
            """
            function multiReturns(n:int): int {
                if (n < 0) {
                    return 0;
                } else if (n == 0) {
                    return 100;
                }
                return 999;
            }
            function runMulti() {
                print multiReturns(-5);   // expect 0
                print multiReturns(0);    // expect 100
                print multiReturns(10);   // expect 999
            }
            runMulti();
            """,
            "9) Multiple return paths test"
        ),
    ]

    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== IR Black-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir_mod = ast_to_ir(ast)

            # Create a pass manager with typical optimization passes
            pm = create_big_pass_manager()

            print_ir_module(ir_mod, f"Before Passes (Test {i})")
            pm.run(ir_mod)
            print_ir_module(ir_mod, f"After Passes (Test {i})")

        except SemanticError as e:
            print(f"SemanticError in Test #{i}: {e}")
            traceback.print_exc()
        except Exception as ex:
            print(f"Unexpected Exception in Test #{i}: {ex}")
            traceback.print_exc()

def test_ir_white_box():
    """
    White-Box tests: Typically smaller snippets designed to trigger or test
    specific IR transformations (e.g., constant folding, dead code elimination, etc.).
    We manually check that the IR transformations work as expected.
    """
    cases = [
        (
            """
            // Constant Folding: (2+3) => 5
            function testConstFold() {
                var x = 2 + 3;
                // There's no real 'print x' or usage, but IR pass should fold the constant
                return x;
            }
            """,
            "1) Simple constant folding test"
        ),
        (
            """
            // Dead Code Elimination: 'var y = 5; y = 6;' if never used, can be removed
            function testDeadCode() {
                var a = 5;
                var b = a + 2;
                var unused = 100;  // never used
                b = b + 1; // used
                return b;
            }
            """,
            "2) Dead code elimination test"
        ),
        (
            """
            // Nested folding & partial usage
            function testNestedFolding() {
                var x = (1 + 2) * (3 + 4);
                var y = x + 100;   // used
                var z = y - 0;     // might fold out sub-zero
                return z;
            }
            """,
            "3) Nested constant folding"
        ),
        (
            """
            // Both passes: some code is folded, some is removed
            function testBothPasses() {
                var doPrint = false;
                var val = 10 * (2 + 3); // = 50
                if (doPrint) {
                    print val; // unreachable => might get removed with advanced passes
                }
                return val;
            }
            """,
            "4) Combined passes test (fold + dce on unreachable branch)"
        ),
    ]

    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== IR White-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir_mod = ast_to_ir(ast)

            # Minimal pass manager to test specific passes
            pm = PassManager()
            pm.add_pass(ConstFoldingPass())
            pm.add_pass(DeadCodeEliminationPass())

            print_ir_module(ir_mod, f"Before White-Box Passes (Test {i})")
            pm.run(ir_mod)
            print_ir_module(ir_mod, f"After White-Box Passes (Test {i})")

        except SemanticError as e:
            print(f"SemanticError in Test #{i}: {e}")
            traceback.print_exc()
        except Exception as ex:
            print(f"Unexpected Exception in Test #{i}: {ex}")
            traceback.print_exc()

def main():
    print("=== IR BLACK-BOX TESTS ===")
    test_ir_black_box()

    print("\n=== IR WHITE-BOX TESTS ===")
    test_ir_white_box()

if __name__ == "__main__":
    main()
