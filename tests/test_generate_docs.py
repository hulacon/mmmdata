"""Tests for scripts/generate_docs.py (the Pages code-docs generator)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_docs.py"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _front_matter(page):
    lines = page.read_text().split("\n")
    assert lines[0] == "---"
    end = lines.index("---", 1)
    return dict(line.split(": ", 1) for line in lines[1:end])


def _run(src, out, *extra):
    return subprocess.run(
        [sys.executable, str(GENERATOR), str(src), "--output-dir", str(out),
         "--clean", *map(str, extra)],
        capture_output=True, text=True, check=True,
    )


def _tree(tmp_path):
    src = tmp_path / "src"
    _write(src / "pkg" / "__init__.py", "")
    _write(src / "pkg" / "top.py", '"""Top module under <root>/x and ``<kept>/y``."""\n\ndef f(x):\n    """Do f."""\n')
    _write(src / "pkg" / "sub" / "__init__.py", "")
    _write(src / "pkg" / "sub" / "inner.py", '"""Inner."""\n\ndef g():\n    """Do g."""\n')
    # a directory without __init__.py is not a package, nor anything below it
    _write(src / "pkg" / "data" / "deep" / "__init__.py", "")
    _write(src / "pkg" / "data" / "deep" / "hidden.py", "def h():\n    pass\n")
    tool = tmp_path / "scripts" / "tool.py"
    _write(tool, '"""\ntool.py — one line summary.\n\nWrites to <sub-##>/viz/ and `<kept>`.\n\n    tool PATH <arg>\n"""\n\ndef main():\n    pass\n')
    return src, tool


def test_subpackages_are_flattened_to_level_two(tmp_path):
    src, _ = _tree(tmp_path)
    out = tmp_path / "out"
    _run(src, out)

    sub_index = _front_matter(out / "pkg_sub.md")
    assert sub_index["title"] == "Pkg / Sub"
    assert sub_index["parent"] == "Code Documentation"

    inner = out / "pkg_sub_inner.md"
    fm = _front_matter(inner)
    assert fm["parent"] == "Pkg / Sub"
    assert fm["grand_parent"] == "Code Documentation"
    assert "`src/python/pkg/sub/inner.py`" in inner.read_text()

    top = (out / "pkg_top.md").read_text()
    assert "under &lt;root>/x" in top          # module prose escaped too
    assert "``<kept>/y``" in top               # RST double-backtick span untouched
    assert not any("deep" in p.name or "hidden" in p.name for p in out.iterdir())
    assert "Tools" not in (out / "code_index.md").read_text()   # no tools, no pointer


def test_tools_section_renders_docstring(tmp_path):
    src, tool = _tree(tmp_path)
    out = tmp_path / "out"
    _run(src, out, "--tools", tool)

    index = out / "tools_index.md"
    assert "parent" not in _front_matter(index)
    assert "| [tool](tools_tool) | one line summary. |" in index.read_text()

    assert "under [Tools](tools_index)" in (out / "code_index.md").read_text()

    page = out / "tools_tool.md"
    assert _front_matter(page)["parent"] == "Tools"
    text = page.read_text()
    assert "&lt;sub-##>/viz/" in text          # prose escaped
    assert "`<kept>`" in text                  # code span untouched
    assert "    tool PATH <arg>" in text       # indented code block untouched
    assert "def main" not in text              # a tool's functions are not listed


def test_missing_tool_is_a_loud_error(tmp_path):
    src, _ = _tree(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), str(src), "--output-dir",
         str(tmp_path / "out"), "--tools", str(tmp_path / "nope.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "nope.py" in proc.stderr
