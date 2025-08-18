#!/usr/bin/env python3
"""
Add trailing commas to:
- function call argument lists
- function definition parameter lists (def/async def)

Notes:
- Lambda parameter lists are intentionally not modified (a trailing comma is invalid in lambdas).
- Formatting is preserved using LibCST.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List
import io
import tokenize
import token as token_mod

import libcst as cst
from libcst import PartialParserConfig
from libcst import MaybeSentinel


class TrailingCommaTransformer(cst.CSTTransformer):
    """Adds trailing commas to function calls and function defs."""

    @staticmethod
    def _get_callee_tail_name(expr: cst.BaseExpression) -> str | None:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        return None

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        num_args = len(updated_node.args)
        if num_args == 0:
            return updated_node
        callee_tail = self._get_callee_tail_name(updated_node.func)
        is_logging_call = callee_tail in {"print", "log", "info", "warning", "error", "debug", "critical"}
        if num_args == 1:
            # Remove trailing comma for single-argument calls
            only_arg = updated_node.args[0]
            if isinstance(only_arg.comma, cst.Comma):
                only_arg = only_arg.with_changes(comma=MaybeSentinel.DEFAULT)
                return updated_node.with_changes(args=[only_arg])
            return updated_node
        # 2+ args
        last_arg = updated_node.args[-1]
        if is_logging_call:
            # For logging/print calls, do NOT keep a trailing comma
            if isinstance(last_arg.comma, cst.Comma):
                new_last_arg = last_arg.with_changes(comma=MaybeSentinel.DEFAULT)
                new_args = list(updated_node.args)
                new_args[-1] = new_last_arg
                return updated_node.with_changes(args=new_args)
        else:
            # Non-logging calls: ensure trailing comma on the last argument
            if not isinstance(last_arg.comma, cst.Comma):
                new_last_arg = last_arg.with_changes(comma=cst.Comma())
                new_args = list(updated_node.args)
                new_args[-1] = new_last_arg
                return updated_node.with_changes(args=new_args)
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return self._ensure_params_trailing_comma(updated_node)

    def leave_AsyncFunctionDef(self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef) -> cst.CSTNode:
        return self._ensure_params_trailing_comma(updated_node)

    @staticmethod
    def _ensure_params_trailing_comma(node: cst.FunctionDef | cst.AsyncFunctionDef) -> cst.CSTNode:
        params = node.params

        # Work on copies so we can replace the last element as needed
        posonly_params: List[cst.Param] = list(params.posonly_params)
        params_list: List[cst.Param] = list(params.params)
        kwonly_params: List[cst.Param] = list(params.kwonly_params)
        star_arg = params.star_arg
        star_kwarg = params.star_kwarg

        # Determine which parameter is last in the full parameter sequence.
        # Order: posonly_params, params, star_arg, kwonly_params, star_kwarg
        last_kind = None
        last_param = None

        if star_kwarg is not None and star_kwarg is not MaybeSentinel.DEFAULT:
            last_kind, last_param = "star_kwarg", star_kwarg
        elif len(kwonly_params) > 0:
            last_kind, last_param = "kwonly", kwonly_params[-1]
        elif star_arg is not None and star_arg is not MaybeSentinel.DEFAULT:
            last_kind, last_param = "star_arg", star_arg
        elif len(params_list) > 0:
            last_kind, last_param = "params", params_list[-1]
        elif len(posonly_params) > 0:
            last_kind, last_param = "posonly", posonly_params[-1]

        # Count total parameters present
        total_params = (
            len(posonly_params) + len(params_list) + len(kwonly_params)
            + (1 if (star_arg is not None and star_arg is not MaybeSentinel.DEFAULT) else 0)
            + (1 if (star_kwarg is not None and star_kwarg is not MaybeSentinel.DEFAULT) else 0)
        )

        if total_params == 0:
            return node

        if total_params == 1:
            # Single parameter: ensure NO trailing comma
            if len(posonly_params) == 1:
                p = posonly_params[0]
                if isinstance(p.comma, cst.Comma):
                    posonly_params[0] = p.with_changes(comma=MaybeSentinel.DEFAULT)
            elif len(params_list) == 1:
                p = params_list[0]
                if isinstance(p.comma, cst.Comma):
                    params_list[0] = p.with_changes(comma=MaybeSentinel.DEFAULT)
            elif star_arg is not None and star_arg is not MaybeSentinel.DEFAULT:
                p = star_arg
                if hasattr(p, "comma") and isinstance(p.comma, cst.Comma):
                    star_arg = p.with_changes(comma=MaybeSentinel.DEFAULT)  # type: ignore[assignment]
            elif len(kwonly_params) == 1:
                p = kwonly_params[0]
                if isinstance(p.comma, cst.Comma):
                    kwonly_params[0] = p.with_changes(comma=MaybeSentinel.DEFAULT)
            elif star_kwarg is not None and star_kwarg is not MaybeSentinel.DEFAULT:
                p = star_kwarg
                if hasattr(p, "comma") and isinstance(p.comma, cst.Comma):
                    star_kwarg = p.with_changes(comma=MaybeSentinel.DEFAULT)  # type: ignore[assignment]

            new_parameters = params.with_changes(
                posonly_params=tuple(posonly_params),
                params=tuple(params_list),
                kwonly_params=tuple(kwonly_params),
                star_arg=star_arg,
                star_kwarg=star_kwarg,
            )
            return node.with_changes(params=new_parameters)

        # Only modify if a trailing comma is missing
        current_comma = getattr(last_param, "comma", MaybeSentinel.DEFAULT)
        if not isinstance(current_comma, cst.Comma):
            new_last_param = last_param.with_changes(comma=cst.Comma())
            if last_kind == "star_kwarg":
                star_kwarg = new_last_param  # type: ignore[assignment]
            elif last_kind == "kwonly":
                kwonly_params[-1] = new_last_param  # type: ignore[index]
            elif last_kind == "star_arg":
                star_arg = new_last_param  # type: ignore[assignment]
            elif last_kind == "params":
                params_list[-1] = new_last_param  # type: ignore[index]
            elif last_kind == "posonly":
                posonly_params[-1] = new_last_param  # type: ignore[index]

        new_parameters = params.with_changes(
            posonly_params=tuple(posonly_params),
            params=tuple(params_list),
            kwonly_params=tuple(kwonly_params),
            star_arg=star_arg,
            star_kwarg=star_kwarg,
        )
        return node.with_changes(params=new_parameters)


def iter_python_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common virtualenvs or cache dirs just in case
        dirnames[:] = [d for d in dirnames if d not in {".venv", "venv", "__pycache__"}]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def process_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Token-based pre-pass: fixes cases even when LibCST can't parse a file
    # - Remove trailing commas for single-arg calls
    # - Remove trailing commas for logging/print calls (regardless of arg count)
    # - Remove trailing commas after single generator expressions
    def fix_trailing_commas_token_pass(code: str) -> str:
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except Exception:
            return code

        output: List[tokenize.TokenInfo] = []
        # Context stack for parentheses
        # Each context is a dict with:
        # - kind: '(' or '[' or '{'
        # - inner_depth: nested depth within this context (counts nested brackets)
        # - top_level_commas: list of output indices for commas at inner_depth == 0
        # - start_index: index in output where the '(' token was added
        # - saw_for: True if a NAME 'for' seen at inner_depth == 0 (generator expr)
        # - call_like: True if '(' follows a function/attribute/closing bracket/paren
        # - callee_tail: last NAME token before '('
        stack: List[dict] = []

        logging_tails = {"print", "log", "info", "warning", "error", "debug", "critical"}

        def is_trivial(tok: tokenize.TokenInfo) -> bool:
            return tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT) or tok.string.strip() == ""

        def prev_significant_index(idx: int) -> int:
            j = idx
            while j >= 0 and is_trivial(output[j]):
                j -= 1
            return j

        for tok in toks:
            ttype, tstr, start, end, line = tok
            if ttype == token_mod.OP and tstr in ("(", "[", "{"):
                # Determine if this is a call-like '('
                prev_idx = prev_significant_index(len(output) - 1)
                prev_tok = output[prev_idx] if prev_idx >= 0 else None
                call_like = False
                if prev_tok is not None:
                    if prev_tok.type == token_mod.NAME:
                        call_like = True
                    elif prev_tok.type == token_mod.OP and prev_tok.string in (")", "]", "}"):
                        call_like = True
                # Determine callee tail name
                callee_tail = None
                j = prev_idx
                while j >= 0:
                    t = output[j]
                    if t.type == token_mod.NAME:
                        callee_tail = t.string
                        break
                    if t.type == token_mod.OP and t.string in (".", ")", "]", "}"):
                        j -= 1
                        continue
                    break

                output.append(tok)
                stack.append({
                    "kind": tstr,
                    "inner_depth": 0,
                    "top_level_commas": [],
                    "start_index": len(output),  # tokens after '('
                    "saw_for": False,
                    "call_like": call_like if tstr == "(" else False,
                    "callee_tail": callee_tail if tstr == "(" else None,
                })
                continue
            elif ttype == token_mod.OP and tstr in (")", "]", "}"):
                # Before appending the closer, possibly fix trailing comma
                if stack:
                    ctx = stack.pop()
                    if ctx["kind"] == "(" and tstr == ")":
                        # Find last non-trivial token inside these parentheses
                        start_idx = ctx["start_index"]
                        end_idx = len(output)  # exclusive of the ')'
                        k = end_idx - 1
                        while k >= start_idx and is_trivial(output[k]):
                            k -= 1
                        last_inner_idx = k
                        last_is_comma = last_inner_idx >= start_idx and output[last_inner_idx].type == token_mod.OP and output[last_inner_idx].string == ","

                        # If trailing comma present at top level, decide whether to remove it
                        if last_is_comma:
                            top_level_commas = ctx["top_level_commas"]
                            # Ensure this last comma is a top-level one (matches last recorded)
                            if top_level_commas and top_level_commas[-1] == last_inner_idx:
                                callee_tail = ctx["callee_tail"]
                                is_logging = callee_tail in logging_tails if callee_tail else False
                                if is_logging:
                                    # Always remove trailing comma for logging/print calls
                                    output.pop(last_inner_idx)
                                else:
                                    # Single-generator-arg case
                                    if ctx["saw_for"] and len(top_level_commas) == 1:
                                        output.pop(last_inner_idx)
                                    # Single-arg call: one comma at top-level implies trailing comma after sole arg
                                    elif ctx["call_like"] and len(top_level_commas) == 1:
                                        output.pop(last_inner_idx)

                output.append(tok)
                # Update parent's inner_depth if any
                if stack:
                    # Closing a nested opener reduces depth for the parent context
                    # We only adjust when kinds match; otherwise outer handling already tracked depth
                    pass
                continue

            # While inside a '(' context, track commas and generator 'for' at top-level
            if stack:
                ctx = stack[-1]
                if ttype == token_mod.OP and tstr in ("(", "[", "{"):
                    ctx["inner_depth"] += 1
                elif ttype == token_mod.OP and tstr in (")", "]", "}"):
                    if ctx["inner_depth"] > 0:
                        ctx["inner_depth"] -= 1
                elif ttype == token_mod.OP and tstr == "," and ctx["inner_depth"] == 0:
                    output.append(tok)
                    ctx["top_level_commas"].append(len(output) - 1)
                    continue
                elif ttype == token_mod.NAME and tstr == "for" and ctx["inner_depth"] == 0:
                    ctx["saw_for"] = True

            # Default: append token
            output.append(tok)

        try:
            return tokenize.untokenize(output)
        except Exception:
            return code

    prefixed_code = fix_trailing_commas_token_pass(original_code)
    parser_versions = ["3.12", "3.11", "3.10", "3.9", "3.8"]
    module = None
    last_error: Exception | None = None
    for ver in parser_versions:
        try:
            module = cst.parse_module(
                prefixed_code,
                config=PartialParserConfig(python_version=ver),
            )
            break
        except Exception as e:
            last_error = e
            continue
    if module is None:
        print(f"[skip] {path}: parse error: {last_error}")
        return False

    updated_module = module.visit(TrailingCommaTransformer())
    updated_code = updated_module.code
    if updated_code != original_code:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated_code)
        print(f"[updated] {path}")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Add trailing commas to calls and defs.")
    parser.add_argument("--root", default=os.path.join(os.getcwd(), "src"), help="Root directory to process")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Root directory not found: {root}", file=sys.stderr)
        return 2

    changed = 0
    total = 0
    for path in iter_python_files(root):
        total += 1
        try:
            if process_file(path):
                changed += 1
        except Exception as e:
            print(f"[error] {path}: {e}")

    print(f"Processed {total} files; modified {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


