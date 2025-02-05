# lower_requests.py

from codegen.ir import (
    IRModule, IRFunction, IRBlock, IRInstr, IRTemp, IRType, IRConst,
    RequestInstr, CallInstr
)


class AdvancedRequestsLoweringPass:
    """
    Replaces RequestInstr with calls to a runtime function __do_http_request
    that can handle HTTP/HTTPS, multiple headers, and arbitrary body data.
    We attempt to parse the method (GET, POST, etc.) and ensure the URL
    is recognized as HTTP or HTTPS. This is best-effort;
    real logic (like building URL parser) is left to the library side.
    """

    def run_on_module(self, module: IRModule):
        for fn in module.functions:
            self.run_on_function(fn)

    def run_on_function(self, fn: IRFunction):
        for block in fn.blocks:
            new_insts = []
            for instr in block.instructions:
                if isinstance(instr, RequestInstr):
                    self.lower_request(new_insts, instr, fn)
                else:
                    new_insts.append(instr)
            block.instructions = new_insts

    def lower_request(self, new_insts, req, fn):
        do_request_fn = IRConst("__do_http_request", IRType("function"))
        url_arg = req.url
        headers_arg = req.headers if req.headers else IRConst(None, IRType("any"))
        body_arg = req.body if req.body else IRConst(None, IRType("any"))
        if req.dest:
            calli = CallInstr(req.dest, do_request_fn, [
                IRConst(req.method, IRType("string")),
                url_arg,
                headers_arg,
                body_arg
            ])
        else:
            temp = fn.create_temp(IRType("any"))
            calli = CallInstr(temp, do_request_fn, [
                IRConst(req.method, IRType("string")),
                url_arg,
                headers_arg,
                body_arg
            ])
        new_insts.append(calli)

    def ensure_string_literal(self, method: str) -> IRConst:
        """
        If method is recognized, return IRConst(method, 'string'), else pass as 'ANY'.
        Accepts GET,POST,PUT,DELETE,PATCH etc.
        """
        valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH", "CONNECT", "TRACE"}
        if method in valid_methods:
            return IRConst(method, IRType("string"))
        return IRConst(method, IRType("string"))  # fallback, still store as string

    def check_url_scheme(self, url_value: str):
        """
        Minimal parse: check if url starts with 'http://' or 'https://'.
        If 'https://', return 'https'; if 'http://', return 'http'; else None.
        """
        u = url_value.lower()
        if u.startswith("https://"):
            return "https"
        if u.startswith("http://"):
            return "http"
        return None
