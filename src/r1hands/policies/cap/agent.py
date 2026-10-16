"""Code-as-policy: Claude writes short Python programs against the Skill API; we run them on the robot.

Loop: task -> Claude calls `run_code` -> code runs in a restricted namespace that holds only the skill
functions (plus any perception functions) -> printed output, return values and errors go back as the
tool_result -> Claude continues until it calls `done`.

Restrictions: no imports, no dunder or private attribute access, small whitelist of builtins.
All motion still passes through the HAL safety guards, so the sandbox only has to keep the code
inside the skill API, not make motion safe.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import math
import traceback
from typing import Callable

from r1hands.control.skills import Skills

MODEL = "claude-opus-5-5"

SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ("abs", "min", "max", "range", "len", "round", "print", "enumerate", "zip", "list",
                           "dict", "float", "int", "bool", "str", "sum", "any", "all", "sorted", "isinstance")}

SYSTEM = """You control a Unitree R1 humanoid with two BrainCo Revo2 dexterous hands by writing short Python programs.

Call the `run_code` tool with Python that uses ONLY the functions listed below (no imports; `math` is available).
Each call blocks until the motion finishes and returns a dict; print what you need to see.
Work in small steps: move, check the result, then continue. Check `blocked_fingers` after grasps, which
usually means the fingers touched the object. If something fails, read the error and adapt; do not repeat
the same call blindly. When the task is complete or impossible, call `done` with a one-line summary.

Frames: pelvis frame, x forward, y left, z up, metres. Hand positions are normalized 0 (open) .. 1 (closed)
in the order [thumb, thumb_aux, index, middle, ring, pinky]. thumb_aux rotates the thumb toward the palm
(needed for pinch and tripod).

Available functions (capability level {level}):
{api}
"""


def _api_doc(funcs: dict[str, Callable]) -> str:
    out = []
    for name, f in funcs.items():
        sig = str(inspect.signature(f))
        doc = inspect.getdoc(f) or ""
        out.append(f"- {name}{sig}\n    {doc.replace(chr(10), chr(10) + '    ')}")
    return "\n".join(out)


class UnsafeCode(Exception):
    pass


def check_code(src: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise UnsafeCode("imports/global are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeCode(f"access to private attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCode(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.While):
            raise UnsafeCode("while loops are not allowed; use for-loops with a fixed bound")
    return tree


def run_sandboxed(src: str, funcs: dict[str, Callable]) -> str:
    check_code(src)
    ns = {"__builtins__": SAFE_BUILTINS, "math": math, **funcs}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<policy>", "exec"), ns)
    except Exception:
        tb = traceback.format_exc(limit=2)
        return buf.getvalue() + "\nERROR:\n" + tb[-1500:]
    return buf.getvalue() or "(ran, no output; print results you need)"


TOOLS = [
    {
        "name": "run_code",
        "description": "Execute a Python program on the robot using only the documented skill functions. Returns stdout or an error.",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}},
                         "required": ["code"], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "done",
        "description": "Finish the episode. success=false if the task is impossible or unsafe.",
        "input_schema": {"type": "object",
                         "properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}},
                         "required": ["success", "summary"], "additionalProperties": False},
        "strict": True,
    },
]


class CodeAsPolicy:
    def __init__(self, skills: Skills, perception: dict[str, Callable] | None = None,
                 model: str = MODEL, effort: str = "high", max_turns: int = 20, confirm: Callable[[str], bool] | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.skills = skills
        self.funcs = {n: getattr(skills, n) for n in skills.available()} | (perception or {})
        self.model, self.effort, self.max_turns = model, effort, max_turns
        self.confirm = confirm  # e.g. lambda code: input(code + "\nrun? [y/N] ") == "y"  (operator gate on hardware)
        self.system = SYSTEM.format(level=skills.level, api=_api_doc(self.funcs))
        self.transcript: list[dict] = []

    def run(self, task: str) -> dict:
        messages = [{"role": "user", "content": task}]
        for _ in range(self.max_turns):
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
            if resp.stop_reason == "refusal":
                return {"success": False, "summary": "model refused", "transcript": self.transcript}
            messages.append({"role": "assistant", "content": resp.content})  # keep thinking blocks intact
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                if resp.stop_reason == "max_tokens":
                    messages.append({"role": "user", "content": "Output was cut off; continue with a shorter program."})
                    continue
                return {"success": False, "summary": "model stopped without calling done", "transcript": self.transcript}
            results = []
            finished = None
            for u in uses:
                if u.name == "done":
                    finished = dict(u.input)
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": "ok"})
                    continue
                code = u.input.get("code", "")
                if self.confirm is not None and not self.confirm(code):
                    out, err = "Operator rejected this program. Propose something safer or call done.", True
                else:
                    try:
                        out, err = run_sandboxed(code, self.funcs), False
                    except (UnsafeCode, SyntaxError) as e:
                        out, err = f"Rejected: {e}", True
                self.transcript.append({"code": code, "output": out})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out, "is_error": err})
            messages.append({"role": "user", "content": results})
            if finished is not None:
                return {**finished, "transcript": self.transcript, "skill_log": self.skills.log}
        return {"success": False, "summary": f"hit max_turns={self.max_turns}", "transcript": self.transcript}


def main():
    import argparse

    from r1hands.hal.factory import make_robot
    ap = argparse.ArgumentParser(description="Code-as-policy episode")
    ap.add_argument("task")
    ap.add_argument("--backend", default="mock", choices=["mock", "sim", "real"])
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--confirm", action="store_true", help="ask the operator before running each program")
    a = ap.parse_args()
    with make_robot(a.backend, level=a.level) as robot:
        confirm = (lambda c: input(f"\n----\n{c}\n----\nrun? [y/N] ").strip().lower() == "y") if a.confirm else None
        res = CodeAsPolicy(Skills(robot, level=a.level), confirm=confirm).run(a.task)
    print(json.dumps({k: v for k, v in res.items() if k != "transcript"}, indent=1, default=str))


if __name__ == "__main__":
    main()
