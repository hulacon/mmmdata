#!/usr/bin/env python3
"""Generate Jekyll-compatible markdown documentation from Python packages.

Walks a source tree, discovers packages and modules, extracts docstrings
from functions and classes, and produces .md files for a Jekyll site using
the just-the-docs theme.

Hierarchy (3 levels, matching just-the-docs sidebar limits):

    Code Documentation  (level 1 – code_index.md)
    └── Package         (level 2 – <package>.md)
        └── Module      (level 3 – <package>_<module>.md, items inline)
"""

import ast
import re
from pathlib import Path
from typing import Any, List, Dict, Optional
import argparse


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

class DocExtractor:
    """Extract documentation from a Python module."""

    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.module_name = source_file.stem
        with open(source_file) as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def module_docstring(self) -> str:
        """Return the module-level docstring."""
        return ast.get_docstring(self.tree) or ""

    def functions(self) -> List[Dict[str, Any]]:
        """Top-level public functions (excludes private and ``main``)."""
        return [
            self._func_info(node)
            for node in ast.iter_child_nodes(self.tree)
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
            and node.name != "main"
        ]

    def classes(self) -> List[Dict[str, Any]]:
        """Top-level public classes."""
        results = []
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                results.append(self._class_info(node))
        return results

    # -- internal helpers --------------------------------------------------

    def _func_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        raw = ast.get_docstring(node) or ""
        parsed = _parse_docstring(raw)
        return {
            "name": node.name,
            "signature": self._signature(node),
            "docstring_raw": raw,
            "line": node.lineno,
            **parsed,
        }

    def _class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        raw = ast.get_docstring(node) or ""
        parsed = _parse_docstring(raw)

        # Detect @dataclass decorator
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            or (
                isinstance(d, ast.Call)
                and (
                    (isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                    or (
                        isinstance(d.func, ast.Attribute)
                        and d.func.attr == "dataclass"
                    )
                )
            )
            for d in node.decorator_list
        )

        # Class-level annotated fields (dataclass fields or typed attrs)
        fields: List[Dict[str, str]] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(
                item.target, ast.Name
            ):
                field: Dict[str, str] = {
                    "name": item.target.id,
                    "type": ast.unparse(item.annotation),
                }
                if item.value is not None:
                    field["default"] = ast.unparse(item.value)
                fields.append(field)

        # Public methods (skip dunder and private)
        methods = [
            self._func_info(m)
            for m in node.body
            if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
        ]

        return {
            "name": node.name,
            "is_dataclass": is_dataclass,
            "fields": fields,
            "methods": methods,
            "line": node.lineno,
            "docstring_raw": raw,
            **parsed,
        }

    def _signature(self, node: ast.FunctionDef) -> str:
        """Build a human-readable signature string.

        Handles positional, keyword-only, *args, **kwargs, and skips
        ``self``/``cls`` for methods.
        """
        parts: List[str] = []
        all_args = node.args.args
        defaults = node.args.defaults
        n_defaults = len(defaults)

        # Skip self/cls
        start = 0
        if all_args and all_args[0].arg in ("self", "cls"):
            start = 1

        for i in range(start, len(all_args)):
            arg = all_args[i]
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            default_idx = i - (len(all_args) - n_defaults)
            if default_idx >= 0:
                s += f" = {ast.unparse(defaults[default_idx])}"
            parts.append(s)

        # *args
        if node.args.vararg:
            v = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                v += f": {ast.unparse(node.args.vararg.annotation)}"
            parts.append(v)
        elif node.args.kwonlyargs:
            parts.append("*")

        # keyword-only args
        kw_defaults = node.args.kw_defaults
        for i, arg in enumerate(node.args.kwonlyargs):
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            if kw_defaults[i] is not None:
                s += f" = {ast.unparse(kw_defaults[i])}"
            parts.append(s)

        # **kwargs
        if node.args.kwarg:
            k = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                k += f": {ast.unparse(node.args.kwarg.annotation)}"
            parts.append(k)

        ret = ""
        if node.returns:
            ret = f" -> {ast.unparse(node.returns)}"

        return f"{node.name}({', '.join(parts)}){ret}"


# ---------------------------------------------------------------------------
# Docstring parsing (NumPy style)
# ---------------------------------------------------------------------------

_SECTION_HEADERS = frozenset(
    {
        "Parameters",
        "Returns",
        "Examples",
        "Raises",
        "Notes",
        "See Also",
        "Attributes",
        "Yields",
        "Warnings",
    }
)


def _parse_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a NumPy-style docstring into structured fields."""
    if not docstring:
        return {
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": "",
            "raises": "",
            "notes": "",
        }

    lines = docstring.split("\n")
    desc_lines: List[str] = []
    for line in lines:
        if line.strip() in _SECTION_HEADERS:
            break
        desc_lines.append(line)

    sections = _extract_sections(docstring)
    return {
        "description": "\n".join(desc_lines).strip(),
        "parameters": sections.get("Parameters", []),
        "returns": sections.get("Returns", ""),
        "examples": sections.get("Examples", ""),
        "raises": sections.get("Raises", ""),
        "notes": sections.get("Notes", ""),
    }


def _extract_sections(docstring: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {}
    lines = docstring.split("\n")
    current: Optional[str] = None
    content: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in _SECTION_HEADERS:
            if current:
                sections[current] = _process_section(current, content)
            current = stripped
            content = []
        elif current is not None:
            content.append(line)

    if current:
        sections[current] = _process_section(current, content)
    return sections


def _process_section(name: str, content: List[str]) -> Any:
    if name == "Parameters":
        return _parse_parameters(content)
    # Strip separator lines (------) that follow the section header
    filtered = [l for l in content if not re.match(r"^\s*-+\s*$", l)]
    return "\n".join(filtered).strip()


def _parse_parameters(lines: List[str]) -> List[Dict[str, str]]:
    params: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None

    for line in lines:
        # Skip separator lines (-------)
        if re.match(r"^-+$", line.strip()):
            continue

        # Parameter definition: "name : type"
        m = re.match(r"^\s{0,4}(\w+)\s*:\s*(.+)$", line)
        if m:
            if current:
                params.append(current)
            current = {
                "name": m.group(1),
                "type": m.group(2).strip(),
                "description": "",
            }
        elif current and line.strip():
            if current["description"]:
                current["description"] += " "
            current["description"] += line.strip()

    if current:
        params.append(current)
    return params


# ---------------------------------------------------------------------------
# Jekyll markdown generation
# ---------------------------------------------------------------------------

# Human-friendly display names for known packages
_DISPLAY_OVERRIDES = {
    "dcm2bids_config": "DCM2BIDS Config",
    "raw2bids_converters": "Raw-to-BIDS Converters",
    "core": "Core",
}


def _pkg_display_name(name: str) -> str:
    return _DISPLAY_OVERRIDES.get(name, name.replace("_", " ").title())


class JekyllGenerator:
    """Produce Jekyll-compatible markdown files."""

    def __init__(self, output_dir: Path, base_nav_order: int = 50):
        self.output_dir = output_dir
        self.base_nav = base_nav_order
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- top-level index ---------------------------------------------------

    def write_main_index(self, packages: List[Dict[str, Any]]) -> Path:
        p = self.output_dir / "code_index.md"
        with open(p, "w") as f:
            f.write("---\n")
            f.write("title: Code Documentation\n")
            f.write(f"nav_order: {self.base_nav}\n")
            f.write("has_children: true\n")
            f.write("---\n\n")
            f.write("# Code Documentation\n\n")
            f.write(
                "API reference for Python packages in the MMMData project.\n"
                "This documentation is auto-generated from source docstrings.\n\n"
            )
            for pkg in packages:
                display = _pkg_display_name(pkg["name"])
                f.write(f"### [{display}]({pkg['name']})\n\n")
                for mod in pkg["modules"]:
                    f.write(
                        f"- [{mod['name']}]({pkg['name']}_{mod['name']})\n"
                    )
                f.write("\n")
        return p

    # -- package index -----------------------------------------------------

    def write_package_index(
        self,
        pkg_name: str,
        modules: List[Dict[str, Any]],
        nav_order: int,
    ) -> Path:
        display = _pkg_display_name(pkg_name)
        p = self.output_dir / f"{pkg_name}.md"
        with open(p, "w") as f:
            f.write("---\n")
            f.write(f"title: {display}\n")
            f.write("parent: Code Documentation\n")
            f.write(f"nav_order: {nav_order}\n")
            f.write("has_children: true\n")
            f.write("---\n\n")
            f.write(f"# {display}\n\n")
            f.write(f"Modules in the `{pkg_name}` package.\n\n")
            f.write("| Module | Description |\n")
            f.write("|--------|-------------|\n")
            for mod in modules:
                short = mod.get("module_docstring", "").split("\n")[0][:80]
                f.write(
                    f"| [{mod['name']}]({pkg_name}_{mod['name']}) "
                    f"| {short} |\n"
                )
        return p

    # -- module page (functions + classes inline) --------------------------

    def write_module_page(
        self,
        pkg_name: str,
        mod: Dict[str, Any],
        nav_order: float,
    ) -> Path:
        display = _pkg_display_name(pkg_name)
        fname = f"{pkg_name}_{mod['name']}.md"
        p = self.output_dir / fname
        with open(p, "w") as f:
            f.write("---\n")
            f.write(f"title: {mod['name']}\n")
            f.write(f"parent: {display}\n")
            f.write("grand_parent: Code Documentation\n")
            f.write(f"nav_order: {nav_order}\n")
            f.write("---\n\n")
            f.write(f"# {mod['name']}\n\n")

            if mod.get("module_docstring"):
                f.write(f"{mod['module_docstring']}\n\n")

            f.write(
                f"**Source:** `src/python/{pkg_name}/{mod['name']}.py`\n"
            )
            f.write("{: .fs-3 .text-grey-dk-000 }\n\n")

            has_classes = bool(mod.get("classes"))
            has_functions = bool(mod.get("functions"))

            if has_classes or has_functions:
                f.write("---\n\n")

            if has_classes:
                f.write("## Classes\n\n")
                for cls in mod["classes"]:
                    self._write_class(f, cls)

            if has_functions:
                f.write("## Functions\n\n")
                for func in mod["functions"]:
                    self._write_function(f, func)

        return p

    # -- renderers ---------------------------------------------------------

    def _write_function(self, f, func: Dict[str, Any]):
        f.write(f"### `{func['name']}`\n\n")
        if func["description"]:
            f.write(f"{func['description']}\n\n")
        f.write("```python\n")
        f.write(func["signature"])
        f.write("\n```\n\n")
        self._write_params_returns_examples(f, func)
        f.write("---\n\n")

    def _write_class(self, f, cls: Dict[str, Any]):
        label = "dataclass" if cls.get("is_dataclass") else "class"
        f.write(f"### `{cls['name']}` ({label})\n\n")
        if cls["description"]:
            f.write(f"{cls['description']}\n\n")

        if cls.get("fields"):
            f.write("**Fields**\n\n")
            for fld in cls["fields"]:
                default = (
                    f" = `{fld['default']}`" if "default" in fld else ""
                )
                f.write(
                    f"- **`{fld['name']}`** (`{fld['type']}`){default}\n"
                )
            f.write("\n")

        if cls.get("methods"):
            f.write("**Methods**\n\n")
            for m in cls["methods"]:
                f.write(f"#### `{m['name']}`\n\n")
                if m["description"]:
                    f.write(f"{m['description']}\n\n")
                f.write("```python\n")
                f.write(m["signature"])
                f.write("\n```\n\n")
                self._write_params_returns_examples(f, m)
        f.write("---\n\n")

    @staticmethod
    def _write_params_returns_examples(f, item: Dict[str, Any]):
        if item.get("parameters"):
            f.write("**Parameters**\n\n")
            for p in item["parameters"]:
                desc = p["description"] or ""
                f.write(f"- **`{p['name']}`** (`{p['type']}`) — {desc}\n")
            f.write("\n")

        if item.get("returns"):
            f.write(f"**Returns**\n\n{item['returns']}\n\n")

        if item.get("examples"):
            f.write("**Examples**\n\n```python\n")
            f.write(item["examples"])
            f.write("\n```\n\n")

        if item.get("notes"):
            f.write(f"**Notes**\n\n{item['notes']}\n\n")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_packages(src_root: Path) -> List[Dict[str, Any]]:
    """Find packages (directories with ``__init__.py``) under *src_root*."""
    packages = []
    for init in sorted(src_root.glob("*/__init__.py")):
        pkg_dir = init.parent
        modules = sorted(
            p
            for p in pkg_dir.glob("*.py")
            if p.name != "__init__.py" and not p.name.startswith("_")
        )
        if modules:
            packages.append(
                {"name": pkg_dir.name, "dir": pkg_dir, "files": modules}
            )
    return packages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate Jekyll documentation from Python packages",
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Root directory containing Python packages (e.g. src/python)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/doc/code"),
        help="Output directory (default: docs/doc/code)",
    )
    parser.add_argument(
        "--nav-order",
        type=int,
        default=50,
        help="Base navigation order (default: 50)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing .md files from output dir before generating",
    )

    args = parser.parse_args()
    gen = JekyllGenerator(args.output_dir, args.nav_order)

    # Optionally clean stale output
    if args.clean:
        for old in args.output_dir.glob("*.md"):
            old.unlink()
            print(f"  removed {old.name}")

    packages = discover_packages(args.source_root)
    if not packages:
        print(f"No packages found under {args.source_root}")
        return

    all_pkg_info: List[Dict[str, Any]] = []

    for pkg_idx, pkg in enumerate(packages):
        pkg_nav = args.nav_order + pkg_idx + 1
        print(f"\nPackage: {pkg['name']}")

        mod_infos: List[Dict[str, Any]] = []

        for mod_idx, mod_file in enumerate(pkg["files"]):
            ext = DocExtractor(mod_file)
            funcs = ext.functions()
            classes = ext.classes()

            if not funcs and not classes:
                continue

            mod_info = {
                "name": mod_file.stem,
                "module_docstring": ext.module_docstring(),
                "functions": funcs,
                "classes": classes,
            }
            mod_nav = pkg_nav + (mod_idx + 1) * 0.01
            gen.write_module_page(pkg["name"], mod_info, mod_nav)
            mod_infos.append(mod_info)

            n_items = len(funcs) + len(classes)
            print(f"  {mod_file.stem}: {n_items} items")

        if mod_infos:
            gen.write_package_index(pkg["name"], mod_infos, pkg_nav)
            all_pkg_info.append(
                {"name": pkg["name"], "modules": mod_infos}
            )

    if all_pkg_info:
        gen.write_main_index(all_pkg_info)
        total = sum(
            len(m["functions"]) + len(m["classes"])
            for p in all_pkg_info
            for m in p["modules"]
        )
        print(
            f"\nDone — {len(all_pkg_info)} packages, {total} documented items"
        )
    else:
        print("No documentable items found")


if __name__ == "__main__":
    main()
