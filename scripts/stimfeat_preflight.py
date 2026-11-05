#!/usr/bin/env python
"""Contract B §4.1 pre-flight: assert feature-column prefixes are unique
across the whole *2psy family before an extraction campaign spends GPU time.

§4.1 requires every feature column to start with a declared prefix (the
model's registry name by default, plus any `extra_prefixes` the class
declares), and requires those prefixes to be globally unique across the
family -- "no prefix may equal or nest under another model's prefix except
by the documented greedy longest-match rule (`clip_text_` vs `clip_`)".

That rule has never been exercised with every model running together. A
collision discovered at psytwill-aggregation time means re-extraction, so
this runs first.

Checks, in order:
  1. duplicate prefixes across packages (hard failure)
  2. prefixes that nest under another prefix (reported; a failure unless
     the pair is in KNOWN_NESTED, i.e. resolvable by longest-match)
  3. prefixes colliding with psytwill's reserved non-feature columns
     (hard failure -- a consumer would sweep the column into a space)
  4. that psytwill's EMBEDDING_RE actually parses each declared prefix
  5. Contract B 1.2 labels (hard failure): every model class declares
     `labels`; values are from the vocabularies; `derived_from` is present
     exactly where the derivation needs it (checked on the default and on
     each column entry merged over it). Per-column completeness needs the
     emitted inventory, so it is each producer's own test, not this one.
     Also reported, never a failure: `derived_from` values that equal
     another family model's `checkpoint` (columns psytwill will flag
     `circular_with` when both sit in one space). Only class-level
     `checkpoint` attributes are visible here, so the preview can miss a
     parent whose checkpoint is set at load time.

Usage:  python stimfeat_preflight.py [--json]
Exit:   0 clean, 1 on any hard failure.
Env:    /gpfs/projects/hulacon/shared/envs/stimfeat (needs all four repos)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

PACKAGES = ("viz2psy", "aud2psy", "word2psy")

# Nesting pairs Contract B documents as intentional and resolvable by the
# greedy longest-match rule. (child, parent) -- child nests under parent.
KNOWN_NESTED = {
    ("clip_text", "clip"),
    ("clap_text", "clap"),
    ("ebind_text", "ebind"),
    ("ebind_audio", "ebind"),
    # Found by this pre-flight's first run (2026-08-20) and verified to
    # resolve: aud2psy `speech` emits only `speech_prob`, `speech_emotion`
    # emits `speech_emotion_{valence,arousal,dominance}`, and psytwill's
    # `_model_prefixes` sorts longest-first. Added to contracts §4.1.
    ("speech_emotion", "speech"),
    # Caught by this pre-flight before the speech_rate extraction ran
    # (2026-08-25) — resolves for the same reason as speech_emotion.
    # Added to contracts §4.1.
    ("speech_rate", "speech"),
}


def collect_prefixes() -> tuple[dict[str, list[str]], list[str]]:
    """{prefix: [owner, ...]} over the family, plus any import errors."""
    owners: dict[str, list[str]] = {}
    errors: list[str] = []
    for pkg in PACKAGES:
        registry = importlib.import_module(f"{pkg}.cli").MODEL_REGISTRY
        for name, (module_path, class_name, _desc) in registry.items():
            prefixes = [name]
            try:
                cls = getattr(importlib.import_module(module_path), class_name)
                prefixes += list(getattr(cls, "extra_prefixes", ()))
            except Exception as exc:  # noqa: BLE001 - report, don't abort
                errors.append(f"{pkg}.{name}: {type(exc).__name__}: {exc}")
            for p in prefixes:
                owners.setdefault(p, []).append(f"{pkg}.{name}")
    return owners, errors


INTERPRETABILITY = {"nameable", "opaque"}
DERIVATIONS = {
    "formula", "formula-on-model", "trained-on-labels", "probe",
    "zero-shot", "embedding", "bookkeeping",
}
NEEDS_PARENT = {"formula-on-model", "probe", "zero-shot"}


def _entry_problems(entry: dict, where: str, complete: bool) -> list[str]:
    """Vocabulary and derived_from rules for one (merged) label entry.

    ``complete`` demands both required keys; a model default may leave
    them to its column entries, so it is checked with ``complete=False``.
    """
    out = []
    interp, deriv = entry.get("interpretability"), entry.get("derivation")
    if interp is not None and interp not in INTERPRETABILITY:
        out.append(f"{where}: interpretability {interp!r}")
    if deriv is not None and deriv not in DERIVATIONS:
        out.append(f"{where}: derivation {deriv!r}")
    if complete:
        for key in ("interpretability", "derivation"):
            if entry.get(key) is None:
                out.append(f"{where}: missing {key}")
    if deriv is not None:
        has_parent = bool(entry.get("derived_from"))
        if deriv in NEEDS_PARENT and not has_parent:
            out.append(f"{where}: derivation {deriv} needs derived_from")
        if deriv not in NEEDS_PARENT and has_parent:
            out.append(f"{where}: derived_from not allowed for {deriv}")
    return out


def check_labels() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """(problems, models without labels, {derived_from: [family owners]})."""
    problems: list[str] = []
    unlabelled: list[str] = []
    checkpoints: dict[str, list[str]] = {}
    parents: dict[str, set[str]] = {}
    for pkg in PACKAGES:
        registry = importlib.import_module(f"{pkg}.cli").MODEL_REGISTRY
        for name, (module_path, class_name, _desc) in registry.items():
            owner = f"{pkg}.{name}"
            try:
                cls = getattr(importlib.import_module(module_path), class_name)
            except Exception:  # noqa: BLE001 - already reported by collect_prefixes
                continue
            ckpt = getattr(cls, "checkpoint", None)
            if isinstance(ckpt, str) and ckpt:
                checkpoints.setdefault(ckpt, []).append(owner)
            labels = getattr(cls, "labels", None)
            if not labels:
                unlabelled.append(owner)
                continue
            default = dict(labels.get("default", {}))
            columns = labels.get("columns", {})
            # A default without a derivation is allowed only when column
            # entries supply it (e.g. a model mixing probe and formula columns).
            default_complete = bool(default.get("derivation")) or not columns
            problems += _entry_problems(
                default, f"{owner}[default]", complete=default_complete
            )
            for col, over in columns.items():
                merged = {**default, **over}
                problems += _entry_problems(merged, f"{owner}[{col}]", complete=True)
                if merged.get("derived_from"):
                    parents.setdefault(merged["derived_from"], set()).add(owner)
            if default.get("derived_from"):
                parents.setdefault(default["derived_from"], set()).add(owner)
    # A model deriving columns from its own checkpoint (a formula over its
    # own network's output) is not circular with another member.
    circular = {}
    for parent, users in parents.items():
        others = [o for o in checkpoints.get(parent, []) if o not in users]
        if others:
            circular[parent] = {"derived_by": sorted(users), "checkpoint_of": others}
    return problems, unlabelled, circular


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    owners, errors = collect_prefixes()

    duplicates = {p: o for p, o in owners.items() if len(o) > 1}

    nested, undocumented = [], []
    for child in owners:
        for parent in owners:
            if child != parent and child.startswith(parent + "_"):
                nested.append((child, parent))
                if (child, parent) not in KNOWN_NESTED:
                    undocumented.append((child, parent))

    from psytwill.spaces import EMBEDDING_RE, INDEX_COLUMNS

    reserved_hits = sorted(set(owners) & set(INDEX_COLUMNS))

    # A declared prefix must survive round-tripping through the consumer's
    # embedding pattern: "<prefix>_000" must parse back to <prefix>.
    unparsable = []
    for p in owners:
        m = EMBEDDING_RE.match(f"{p}_000")
        if not m or m.group("prefix") != p:
            unparsable.append(p)

    label_problems, unlabelled, circular = check_labels()

    hard_fail = bool(
        duplicates or undocumented or reserved_hits or unparsable
        or label_problems or unlabelled
    )

    report = {
        "n_models": sum(
            len(importlib.import_module(f"{p}.cli").MODEL_REGISTRY) for p in PACKAGES
        ),
        "n_prefixes": len(owners),
        "duplicates": duplicates,
        "nested_documented": sorted(set(nested) & KNOWN_NESTED),
        "nested_undocumented": sorted(undocumented),
        "reserved_collisions": reserved_hits,
        "unparsable_by_embedding_re": sorted(unparsable),
        "import_errors": errors,
        "labels_missing": sorted(unlabelled),
        "label_problems": label_problems,
        "circular_preview": circular,
        "clean": not hard_fail,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"models: {report['n_models']}   prefixes: {report['n_prefixes']}")
        print(f"duplicates:            {duplicates or 'none'}")
        print(f"nested (documented):   {report['nested_documented'] or 'none'}")
        print(f"nested (UNDOCUMENTED): {report['nested_undocumented'] or 'none'}")
        print(f"reserved collisions:   {reserved_hits or 'none'}")
        print(f"EMBEDDING_RE failures: {report['unparsable_by_embedding_re'] or 'none'}")
        print(f"models without labels: {sorted(unlabelled) or 'none'}")
        print(f"label problems:        {label_problems or 'none'}")
        print("circular preview (derived_from = a family checkpoint):")
        for parent, info in sorted(circular.items()):
            print(f"  {parent}: {info['derived_by']} <- checkpoint of {info['checkpoint_of']}")
        if not circular:
            print("  none")
        if errors:
            print("import errors:")
            for e in errors:
                print(f"  {e}")
        print("CLEAN" if not hard_fail else "FAIL")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
