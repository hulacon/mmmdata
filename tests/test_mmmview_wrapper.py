"""bin/mmmview: the wrapper every user runs it through."""

from pathlib import Path

WRAPPER = Path(__file__).resolve().parent.parent / "bin" / "mmmview"


def test_the_wrapper_sets_a_group_writable_umask_before_running_python():
    """Bundles are rebuilt in place in the shared derivatives tree by whoever
    runs mmmview, so a file one member writes must stay writable by the rest."""
    lines = [line.strip() for line in WRAPPER.read_text().splitlines()]
    assert "umask 0002" in lines
    run = next(i for i, line in enumerate(lines) if line.startswith("exec "))
    assert lines.index("umask 0002") < run
