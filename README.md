eLang is an open-source ahead-of-time compiler for a custom grammar. The language is designed for simple large-scale data collection, manipulation, and analysis. It is a type-safe language with extensive semantic analysis pre-compilation. It compiles eLang code directly into x86_64 machine code packed into a PE stub for 64 bit windows machines. 

Syntax example:

```
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
```
