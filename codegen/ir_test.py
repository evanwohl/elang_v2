import sys
import traceback
from parser.parser import parse_code
from analysis.semantic import analyze_semantics, SemanticError
from codegen.ast_to_ir import ast_to_ir
from ir_utils import create_big_pass_manager, print_ir_module, PassManager, ConstFoldingPass, DeadCodeEliminationPass

def test_ir_black_box():
    cases = [
        (
            """
            function sum(a:int, b:int): int {
                return a + b;
            }
            function runSum() {
                var result = sum(2, 3);
                print result;
            }
            runSum();
            """,
            "1) Typed int sum function + usage"
        ),
        (
            """
            function workerFn(i:any): void {
                print i;
            }
            function concurrencySpawn() {
                for (var n=0; n<3; n=n+1) {
                    spawn workerFn(n);
                }
            }
            concurrencySpawn();
            """,
            "2) Concurrency with a declared workerFn"
        ),
        (
            """
            function arrowDemo(): void {
                var f = (x) => x + 10;
                var r = f(5);
                print r;
            }
            arrowDemo();
            """,
            "3) Arrow function returning x+10, called with 5"
        ),
        (
            """
            function requestDemo(): void {
                var resp = get "https://example.com";
                print resp;
            }
            requestDemo();
            """,
            "4) Basic HTTP GET request, printing response"
        ),
        (
            """
            function typedIfElse(flag: bool): string {
                if(flag) {
                    return "Flag True";
                } else {
                    return "Flag False";
                }
            }
            var msg = typedIfElse(true);
            print msg;
            """,
            "5) If/else usage with typed bool param, returning string"
        )
    ]
    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== IR Black-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir_mod = ast_to_ir(ast)
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
    cases = [
        (
            """
            function nestedConcurrency(): void {
                function innerWorker(x:int) {
                    if(x > 2) print "Big";
                    else print "Small";
                }
                function outerWorker() {
                    spawn innerWorker(10);
                    spawn innerWorker(1);
                }
                spawn outerWorker();
            }
            nestedConcurrency();
            """,
            "1) Nested concurrency with declared functions"
        ),
        (
            """
            function arrowTyped(): float {
                var f = (z) => z * 3.14;
                var val = f(2);
                return val; 
            }
            var result = arrowTyped();
            print result;
            """,
            "2) Arrow function returning float usage"
        ),
        (
            """
            function loopBreak(): void {
                var i = 0;
                while(true) {
                    i = i + 1;
                    if(i == 3) break;
                }
                print i;
            }
            loopBreak();
            """,
            "3) While loop with break, prints final i"
        ),
        (
            """
            function requestWithHeaders(): any {
                var r = post "https://api.test" HEADERS {"X-Check": "yes", "test": "test"} BODY {"data":123};
                return r;
            }
            var resp = requestWithHeaders();
            print resp;
            """,
            "4) Request with headers/body, returning any"
        ),
        (
            """
            function tryCatchFinally() {
                try {
                    var x = 10;
                    if(x == 10) throw "X was 10";
                } catch(e) {
                    print e;
                } finally {
                    print "done";
                }
            }
            tryCatchFinally();
            var r: int = 172;
            while(r > 0) {
                r = r - 1;
            }
            """,
            "5) try/catch/finally usage"
        )
    ]
    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== IR White-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir_mod = ast_to_ir(ast)
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
