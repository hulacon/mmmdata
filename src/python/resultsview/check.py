"""Check results against mmmview's page contract without rendering anything.

    mmmview check <path> [--strict] [--format text|json|github]
    python -m resultsview.check <path> ...        (no mmmdata venv needed)

*path* is a ``<stem>.vl.json`` spec, a ``<stem>.tsv`` table, or a directory
searched recursively (hidden directories skipped). Every spec found is read
with :func:`resultsview.page.load_spec`, the loader mmmview itself renders
from, so a spec that passes here draws there.

Each spec is also read for provenance, the ``usermeta.mmmview.provenance``
block that :func:`mmmget.write_result` stamps:

- ``standard``: written by ``write_result`` and every getter that fed it
  stayed on the standard path;
- ``non_standard``: written by ``write_result``, but at least one getter
  took an escape hatch (the reasons are listed);
- ``absent``: no provenance block, so nothing records which inputs fed it.

Exit status: 0 when every spec meets the contract; 1 when any does not, or,
with ``--strict``, when any provenance is not ``standard``; 2 when *path*
does not exist or holds no spec. Only *path* is read: no config, no dataset
root, no catalog, so the check runs the same on Talapas, on a copy of the
data, and in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import page

REPORT_SCHEMA_VERSION = 1
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def find_specs(path: Path) -> list[Path]:
    """The specs *path* names: itself, its table's specs, or a recursive search."""
    path = Path(path)
    if path.is_file():
        if path.name.endswith(page.SPEC_EXT):
            return [path]
        if path.name.endswith(page.TABLE_EXT):
            return list(page.specs_for_table(path))
        return []
    found = []
    for p in sorted(path.rglob("*" + page.SPEC_EXT)):
        rel = p.relative_to(path).parts[:-1]
        if not any(part.startswith(".") for part in rel):
            found.append(p)
    return found


def _provenance(spec: dict) -> tuple[str, list[str]]:
    prov = ((spec.get("usermeta") or {}).get("mmmview") or {}).get("provenance")
    if not isinstance(prov, dict):
        return "absent", ["not written by mmmget.write_result; nothing records "
                          "which inputs fed it"]
    if prov.get("standard") is True:
        return "standard", []
    return "non_standard", [str(r) for r in prov.get("non_standard") or []]


def check_spec(spec_path: Path) -> dict:
    """One spec's verdict: contract (``ok``/``error``) and provenance."""
    entry = {"spec": str(spec_path), "table": None, "ok": True, "error": None,
             "provenance": None, "reasons": []}
    try:
        loaded = page.load_spec(spec_path)
    except page.SpecError as exc:
        entry.update(ok=False, error=str(exc))
        try:
            spec = json.loads(Path(spec_path).read_text())
        except (OSError, ValueError):
            return entry
        if not isinstance(spec, dict):
            return entry
    else:
        entry["table"] = str(loaded["table"])
        spec = loaded["spec"]
    entry["provenance"], entry["reasons"] = _provenance(spec)
    return entry


def check(path: Path | str, strict: bool = False) -> dict:
    """Check every spec under *path*; a self-describing report."""
    path = Path(path)
    results = [check_spec(p) for p in find_specs(path)] if path.exists() else []
    invalid = sum(not r["ok"] for r in results)
    non_standard = sum(r["provenance"] not in (None, "standard") for r in results)
    if not path.exists() or not results:
        status = EXIT_USAGE
    elif invalid or (strict and non_standard):
        status = EXIT_FAIL
    else:
        status = EXIT_OK
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_schema_version": page.SCHEMA_VERSION,
        "path": str(path),
        "strict": strict,
        "exit_status": status,
        "summary": {"specs": len(results), "invalid": invalid,
                    "non_standard_provenance": non_standard},
        "results": results,
    }


def _gh(text: str) -> str:
    # workflow-command values: escape %, CR and LF
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def render(report: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2)
    lines = []
    for r in report["results"]:
        prov = r["provenance"]
        prov_note = f"{prov}: {'; '.join(r['reasons'])}" if r["reasons"] else prov
        if fmt == "github":
            if not r["ok"]:
                lines.append(f"::error file={r['spec']},title=mmmview contract::{_gh(r['error'])}")
            if prov not in (None, "standard"):
                level = "error" if report["strict"] else "warning"
                lines.append(f"::{level} file={r['spec']},title=mmmview provenance::"
                             f"{_gh(prov_note)}")
        else:
            verdict = "ok  " if r["ok"] else "FAIL"
            lines.append(f"{verdict} {r['spec']}")
            if not r["ok"]:
                lines.append(f"     {r['error']}")
            if prov not in (None, "standard"):
                lines.append(f"     provenance {prov_note}")
    s = report["summary"]
    if not Path(report["path"]).exists():
        tail = f"mmmview check: no such file or directory: {report['path']}"
    elif not s["specs"]:
        tail = (f"mmmview check: no {page.SPEC_EXT} spec found under {report['path']}; "
                "a result is <stem>.tsv + <stem>.vl.json (mmmget.write_result writes both)")
    else:
        tail = (f"mmmview check: {s['specs']} spec(s), {s['invalid']} invalid, "
                f"{s['non_standard_provenance']} without standard provenance")
    lines.append(tail if fmt == "text" else f"::notice title=mmmview check::{_gh(tail)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mmmview check",
        description="check results (<stem>.tsv + <stem>.vl.json) against "
                    "mmmview's page contract and report their provenance; "
                    "renders nothing")
    ap.add_argument("path", type=Path, help="a spec, a table, or a directory")
    ap.add_argument("--strict", action="store_true",
                    help="fail on provenance that is not standard, too")
    ap.add_argument("--format", choices=("text", "json", "github"), default="text",
                    help="github = workflow-command annotations for Actions")
    args = ap.parse_args(argv)
    report = check(args.path, strict=args.strict)
    print(render(report, args.format))
    return report["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
