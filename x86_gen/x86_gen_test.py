# test_x86_gen.py

import sys
import traceback
from parser.parser import parse_code
from analysis.semantic import analyze_semantics, SemanticError
from codegen.ast_to_ir import ast_to_ir
from lower_concurrency import ConcurrencyLoweringPass
from lower_requests import AdvancedRequestsLoweringPass
from x86_codegen import X86Codegen

def test_x86_black_box():
    cases = [
        (
            """
            function basicMath(a:int, b:int): int {
                return a*b + b - a/2;
            }
            function main() {
                var r = basicMath(6, 3);
                print r;
            }
            main();
            """,
            "1) Basic integer math function"
        ),
        (
            """
            function spawnTest() {
                function worker(x:int) {
                    print x;
                }
                for(var i=0; i<3; i=i+1) {
                    spawn worker(i);
                }
            }
            spawnTest();
            """,
            "2) Concurrency spawn with local worker function"
        ),
        (
            """
            function arrowStuff(): float {
                var f = (z)=>z*3.14;
                var val = f(2);
                return val;
            }
            function main() {
                var x = arrowStuff();
                print x;
            }
            main();
            """,
            "3) Arrow function returning float usage"
        ),
        (
            """
            function requestDemo(): any {
                var x = get "http://example.com";
                return x;
            }
            var resp = requestDemo();
            print resp;
            """,
            "4) Basic HTTP request usage"
        ),
        (
            """
            function typedIfElse(flag: bool): string {
                if(flag) return "YES";
                return "NO";
            }
            var ans = typedIfElse(true);
            print ans;
            """,
            "5) Simple if/else returning string"
        ),
        (
            """
            function lockExample() {
                lock myLock;
                unlock myLock;
            }
            lockExample();
            """,
            "6) Lock/unlock usage"
        ),
        (
            """
            function advancedRequests(): any {
                var r = post "https://api.test" HEADERS {"X-Val":"someVal","Mode":"FAST"} BODY {"data":123};
                print "Request done";
                return r;
            }
            var out = advancedRequests();
            print out;
            """,
            "7) Advanced request with HEADERS/BODY"
        ),
        (
            """
            function killTest() {
                thread t1;
                kill t1; 
                detach t1; 
            }
            killTest();
            """,
            "8) Kill/detach thread usage"
        ),
        (
            """
            function mathFloat(a:float, b:float): float {
                return (a+b)*(a-b);
            }
            function main() {
                var res = mathFloat(2.5, 1.5);
                print res;
            }
            main();
            """,
            "9) Float arithmetic in function"
        ),
        (
            """
            function complexSpawn() {
                function doWork(n:int) {
                    if(n>5) print "Bigger"; else print "Smaller";
                }
                spawn doWork(10);
                spawn doWork(3);
            }
            complexSpawn();
            """,
            "10) Another concurrency spawn with if"
        )
    ]
    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== X86 Black-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir = ast_to_ir(ast)
            ConcurrencyLoweringPass().run_on_module(ir)
            AdvancedRequestsLoweringPass().run_on_module(ir)
            asm = X86Codegen().run_on_module(ir)
            print("Generated Assembly:\n", asm)
        except SemanticError as e:
            print("SemanticError:", e)
            traceback.print_exc()
        except Exception as ex:
            print("Unexpected exception:", ex)
            traceback.print_exc()

def test_x86_white_box():
    cases = [
        (
            """
            function nestedThreads() {
                function inner(x:int) {
                    if(x==1) print "One"; else print "NotOne";
                }
                thread thr1;
                spawn inner(1);
                join thr1;
            }
            nestedThreads();
            """,
            "1) Nested concurrency with thread usage"
        ),
        (
            """
            function multiRequest() {
                var r1 = get "http://example.com/api";
                var r2 = post "http://example.com/data" HEADERS {"Content":"json"} BODY {"val":42};
                return r1 + r2;
            }
            var combined = multiRequest();
            print combined;
            """,
            "2) Multiple requests with HEADERS/BODY"
        ),
        (
            """
            function breaksWhile() {
                var i=0;
                while(true) {
                    if(i==2) break;
                    i=i+1;
                }
                print i;
            }
            breaksWhile();
            """,
            "3) While loop with break"
        ),
        (
            """
            function arrowMixed(a:int): int {
                var f1 = (z) => z+a;
                var f2 = (q) => q*2;
                var x = f1(5);
                var y = f2(x);
                return y;
            }
            var ans = arrowMixed(10);
            print ans;
            """,
            "4) Multiple arrow functions referencing outer scope"
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
                    print "Cleanup";
                }
            }
            tryCatchFinally();
            """,
            "5) Try/catch/finally plus throw usage"
        )
    ]
    for i, (code, desc) in enumerate(cases, 1):
        print(f"\n=== X86 White-Box Test #{i}: {desc} ===")
        try:
            ast = parse_code(code)
            analyze_semantics(ast)
            ir = ast_to_ir(ast)
            ConcurrencyLoweringPass().run_on_module(ir)
            AdvancedRequestsLoweringPass().run_on_module(ir)
            asm = X86Codegen().run_on_module(ir)
            print("Generated Assembly:\n", asm)
        except SemanticError as e:
            print("SemanticError:", e)
            traceback.print_exc()
        except Exception as ex:
            print("Unexpected exception:", ex)
            traceback.print_exc()

def main():
    print("=== X86 CODEGEN BLACK-BOX TESTS ===")
    test_x86_black_box()
    print("\n=== X86 CODEGEN WHITE-BOX TESTS ===")
    test_x86_white_box()

if __name__ == "__main__":
    main()

