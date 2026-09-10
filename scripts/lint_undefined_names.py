#!/usr/bin/env python
"""Report global names that a script's functions load but never define.

A stand-in for pyflakes' undefined-name check, which none of the shared envs
carry. It catches the one class of bug that `py_compile` and a `--dry-run`
both miss: a name that is only touched on a code path past the dry-run's
return (a renamed variable with one stale call site, a helper that was
deleted, an import that only happens under a flag), which then raises
`NameError` after the expensive part of a job has already run.

Usage:
    <runtime python> scripts/lint_undefined_names.py scripts/<file>.py [more.py ...]

Run it with the interpreter the sbatch activates, since it IMPORTS each file
to learn its module-level names (nothing runs: `__main__` guards are not
taken). Exit status is the number of files with at least one hit.

Limits: it sees bytecode `LOAD_GLOBAL`/`LOAD_NAME` only, so names bound by
`global` statements inside functions, `exec`, or star-imports at runtime are
not modelled; a clean report is not a proof, a hit is always real.
"""
import builtins
import dis
import importlib.util
import sys
import types
from pathlib import Path


def undefined_names(path: Path):
    spec = importlib.util.spec_from_file_location(f"_lint_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    saved_argv, sys.argv = sys.argv, [str(path)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv
    known = set(vars(mod)) | set(dir(builtins))
    hits = []

    def walk(code, owner):
        for ins in dis.get_instructions(code):
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and ins.argval not in known:
                hits.append((ins.positions.lineno, ins.argval, owner))
            if isinstance(ins.argval, types.CodeType):
                walk(ins.argval, owner)

    for name, obj in vars(mod).items():
        if isinstance(obj, types.FunctionType) and obj.__module__ == mod.__name__:
            walk(obj.__code__, name)
    return sorted(hits)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for arg in argv:
        path = Path(arg)
        hits = undefined_names(path)
        for lineno, name, owner in hits:
            print(f"{path}:{lineno}: undefined name '{name}' in {owner}()")
        print(f"{path}: {len(hits)} undefined global name(s)")
        bad += bool(hits)
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
