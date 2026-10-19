"""Errors mmmget raises where the answer is not its to give."""

from __future__ import annotations


class OpenDecisionError(NotImplementedError):
    """A getter needs a lab decision that has not been made yet.

    Raised instead of a default guess, so code written against a getter
    fails loudly until the standard exists and then starts following it.
    ``decision`` is the decision's label (``"D3"``, ``"D6"``) so callers
    and tests can tell which one is blocking.
    """

    def __init__(self, decision: str, message: str):
        super().__init__(f"open decision {decision}: {message}")
        self.decision = decision


class SourceNotConfigured(LookupError):
    """The data a getter would read has no configured or catalogued source.

    The message names what is missing (a config key, a catalog row), so the
    fix is a configuration change rather than a code change.
    """
