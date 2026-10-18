"""Neutralize gendered third-person pronouns in free-text TSV columns.

Every gendered pronoun is replaced; what can be uncertain is only the
grammar around it, and every uncertain choice is listed in the review
report rather than silently guessed. The report carries every changed cell,
before and after, so a person can check the whole transform.

Mapping
-------
he, she                   -> they       (+ verb agreement, below)
he's/she's                -> they're, or they've before been/got/had  [review]
he'd/she'd, he'll/she'll  -> they'd, they'll
him                       -> them
himself, herself          -> themself
hers                      -> theirs
his                       -> their before a noun; theirs at a phrase end [review]
her                       -> them / their by the next word (below)
"he or she", "he/she"     -> they

Verb agreement after a subject he/she (skipping up to two adverbs such as
``also``, ``still`` or any ``-ly`` word): is/was/has/does and their ``n't``
forms change to are/were/have/do. A third-person ``-s`` verb loses its ending
(``likes`` -> ``like``, ``tries`` -> ``try``, ``watches`` -> ``watch``); a
``-ses``/``-zes`` ending (``uses`` but ``focuses``) is changed by dropping
the ``s`` and flagged. Words ending ``ss``/``us``/``is``/``ous`` and a short
list of adverbs and function words are never treated as verbs.

``her``: ``them`` before punctuation, the end of the cell, or a function word
(preposition, conjunction, determiner, pronoun, common adverb); ``their``
before anything else. Flagged: after a causative/perception verb (made, let,
helped, saw ...: ``made her nervous`` vs ``made her day``) the choice is
``them``; before an ``-ing`` word the choice is ``their``.

Known limits (the report is the safety net): a second verb in a conjoined
predicate (``she likes and enjoys``) keeps its ``-s``; inverted questions
(``was she``) are flagged, not rewritten; gendered nouns and honorifics
(``man``, ``girl``, ``Mr``) are flagged but not changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .stage import Stage
from .tsvio import Table, is_id_column, row_key, text_columns

DEBRIEFING = "phenotype/final_debriefing.tsv"

_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

ADVERBS = {
    "also", "still", "just", "even", "never", "always", "often", "sometimes",
    "then", "now", "already", "soon", "almost", "ever", "too", "rarely",
    "seldom", "once", "again",
}
LY_NOT_ADVERB = {
    "family", "belly", "ally", "rally", "jelly", "bully", "lily", "early",
    "only", "daily", "weekly", "monthly", "yearly", "lovely", "friendly",
    "likely", "ugly", "holy", "silly", "lonely", "elderly", "fly", "reply",
    "apply", "supply", "rely", "july", "italy", "sally", "emily", "kelly",
    "molly", "polly", "billy", "holly",
}
AGREE = {
    "is": "are", "was": "were", "has": "have", "does": "do",
    "isn't": "aren't", "wasn't": "weren't", "hasn't": "haven't",
    "doesn't": "don't",
}
INVERTED_VERBS = set(AGREE)
S_NOT_VERB = {
    "always", "sometimes", "perhaps", "afterwards", "towards", "besides",
    "thus", "as", "us", "its", "yes", "less", "unless", "across", "this",
    "his", "hers", "plus", "whereas", "versus", "nowadays", "overseas",
    "upstairs", "downstairs", "outdoors", "indoors", "backwards", "forwards",
    "sideways", "anyways", "themselves", "ourselves", "yourselves", "was",
    "is", "has", "does",
}
OBJECT_FOLLOWERS = {
    # prepositions / particles
    "to", "about", "at", "by", "for", "from", "in", "into", "of", "off", "on",
    "onto", "out", "over", "through", "up", "down", "with", "without", "back",
    "away", "around", "after", "before", "during", "like", "as", "than",
    "along", "toward", "towards", "across", "behind", "under", "upon",
    # conjunctions / relativizers
    "and", "or", "but", "so", "because", "if", "when", "while", "that",
    "which", "who", "whether", "though", "although", "since", "until", "nor",
    "where", "how", "why", "what",
    # determiners / pronouns / quantifiers
    "a", "an", "the", "this", "these", "those", "some", "any", "every",
    "each", "no", "it", "them", "him", "her", "me", "us", "you", "something",
    "anything", "nothing", "everything", "all", "both",
    # adverbs / misc function words
    "too", "again", "very", "also", "then", "now", "there", "here", "yet",
    "still", "not", "enough", "more", "less", "much", "anymore", "either",
    "is", "was", "were", "are", "be", "been", "would", "could", "should",
    "will", "can", "might", "must", "did", "do", "does", "had", "has",
    "have", "feel", "feels", "felt", "think", "know",
}
CAUSATIVE = {
    "made", "make", "makes", "making", "let", "lets", "letting", "helped",
    "help", "helps", "helping", "saw", "see", "sees", "seeing", "watched",
    "watch", "watches", "heard", "hear", "hears", "kept", "keep", "keeps",
    "found", "find", "finds", "got", "get", "gets", "left", "leave",
    "leaves",
}
GENDERED_TERMS = {
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "guy",
    "guys", "gal", "gals", "lady", "ladies", "gentleman", "gentlemen",
    "male", "female", "mr", "mrs", "ms", "sir", "madam", "ma'am",
}


@dataclass
class Edit:
    start: int
    end: int
    new: str
    note: str | None = None        # set when the choice needs review


@dataclass
class CellResult:
    before: str
    after: str
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _norm(word: str) -> str:
    return word.lower().replace("’", "'")


def _case(src: str, repl: str) -> str:
    if len(src) > 1 and src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _apos(src: str, repl: str) -> str:
    return repl.replace("'", "’") if "’" in src else repl


def _strip_s(lw: str) -> tuple[str | None, bool]:
    """Bare form of a third-person -s verb, and whether it needs review.
    ``(None, False)`` means: not treated as a verb."""
    if not lw.isalpha() or len(lw) < 3 or not lw.endswith("s"):
        return None, False
    if lw in S_NOT_VERB or lw.endswith(("ss", "us", "is", "ous")):
        return None, False
    if lw.endswith("ies"):
        return (lw[:-3] + "y" if len(lw) > 4 else lw[:-1]), False
    if lw.endswith(("sses", "shes", "ches", "xes", "zzes", "oes")):
        return lw[:-2], False
    if lw.endswith(("ses", "zes")):
        return lw[:-1], True
    return lw[:-1], False


class _Cell:
    def __init__(self, text: str):
        self.text = text
        self.toks = list(_WORD.finditer(text))
        self.edits: dict[int, Edit] = {}
        self.notes: list[str] = []

    def word(self, i: int) -> str:
        return self.toks[i].group()

    def plain_gap(self, i: int, j: int) -> bool:
        """Only whitespace between token i and token j (i < j)."""
        return self.text[self.toks[i].end():self.toks[j].start()].strip() == ""

    def next_in_phrase(self, i: int) -> int | None:
        j = i + 1
        if j < len(self.toks) and self.plain_gap(i, j):
            return j
        return None

    def prev_in_phrase(self, i: int) -> int | None:
        j = i - 1
        if j >= 0 and self.plain_gap(j, i):
            return j
        return None

    def put(self, i: int, new: str, note: str | None = None, end_tok: int | None = None):
        src = self.word(i)
        end = self.toks[end_tok if end_tok is not None else i].end()
        self.edits[i] = Edit(self.toks[i].start(), end, _apos(src, _case(src, new)), note)
        if note:
            self.notes.append(note)

    # rules -----------------------------------------------------------------
    def agree_after(self, i: int) -> None:
        """Verb agreement for the word(s) after a subject pronoun at token i."""
        j = self.next_in_phrase(i)
        hops = 0
        while j is not None and hops < 2:
            lw = _norm(self.word(j))
            if lw in ADVERBS or (lw.endswith("ly") and lw not in LY_NOT_ADVERB):
                j, hops = self.next_in_phrase(j), hops + 1
                continue
            break
        if j is None or j in self.edits:
            return
        w = self.word(j)
        lw = _norm(w)
        if w[:1].isupper() and len(w) > 1 and not w.isupper():
            return                              # proper noun; leave it
        if lw in AGREE:
            self.put(j, AGREE[lw])
            return
        bare, review = _strip_s(lw)
        if bare is None:
            return
        note = (f"verb '{w}' -> '{bare}': -ses/-zes ending, check the bare form"
                if review else None)
        self.put(j, bare, note)

    def run(self) -> None:
        n = len(self.toks)
        i = 0
        while i < n:
            if i in self.edits:
                i += 1
                continue
            w = self.word(i)
            lw = _norm(w)
            if lw in ("he", "she"):
                # "he or she", "he and she", "he/she"
                end = i
                j = i + 1
                if j < n:
                    gap = self.text[self.toks[i].end():self.toks[j].start()]
                    if gap.strip() == "/" and _norm(self.word(j)) in ("he", "she"):
                        end = j
                    elif (gap.strip() == "" and _norm(self.word(j)) in ("or", "and")
                          and j + 1 < n and self.plain_gap(j, j + 1)
                          and _norm(self.word(j + 1)) in ("he", "she")):
                        end = j + 1
                self.put(i, "they", end_tok=end)
                for k in range(i + 1, end + 1):
                    self.edits[k] = Edit(0, 0, "", None)   # swallowed
                p = self.prev_in_phrase(i)
                if p is not None and _norm(self.word(p)) in INVERTED_VERBS:
                    self.notes.append(
                        f"'{self.word(p)} {w}': verb before the pronoun may need "
                        f"agreement (inverted order); left unchanged")
                self.agree_after(end)
                i = end + 1
                continue
            if lw in ("he's", "she's"):
                j = self.next_in_phrase(i)
                nxt = _norm(self.word(j)) if j is not None else ""
                new = "they've" if nxt in ("been", "got", "gotten", "had") else "they're"
                self.put(i, new, f"'{w}' -> '{new}': contraction of is/has, check")
            elif lw in ("he'd", "she'd"):
                self.put(i, "they'd")
            elif lw in ("he'll", "she'll"):
                self.put(i, "they'll")
            elif lw == "him":
                self.put(i, "them")
            elif lw in ("himself", "herself"):
                self.put(i, "themself")
            elif lw == "hers":
                self.put(i, "theirs")
            elif lw == "his":
                j = self.next_in_phrase(i)
                if j is None or _norm(self.word(j)) in OBJECT_FOLLOWERS:
                    self.put(i, "theirs", f"'{w}' -> 'theirs' (no noun follows), check")
                else:
                    self.put(i, "their")
            elif lw == "her":
                self.her(i)
            elif lw in GENDERED_TERMS:
                self.notes.append(f"gendered term '{w}' left unchanged; review")
            i += 1

    def her(self, i: int) -> None:
        w = self.word(i)
        j = self.next_in_phrase(i)
        p = self.prev_in_phrase(i)
        nxt = _norm(self.word(j)) if j is not None else None
        prv = _norm(self.word(p)) if p is not None else None
        if nxt is None or nxt in OBJECT_FOLLOWERS:
            self.put(i, "them")
        elif nxt.endswith("ly") and nxt not in LY_NOT_ADVERB:
            self.put(i, "them")
        elif prv in CAUSATIVE:
            self.put(i, "them", f"'{self.word(p)} {w} {self.word(j)}': chose 'them' "
                                f"after '{self.word(p)}'; could be 'their'")
        elif nxt.endswith("ing"):
            self.put(i, "their", f"'{w} {self.word(j)}': chose 'their' before an "
                                 f"-ing word; could be 'them'")
        else:
            self.put(i, "their")

    def result(self) -> str:
        out, pos = [], 0
        for e in sorted((e for e in self.edits.values() if e.end > e.start),
                        key=lambda e: e.start):
            out.append(self.text[pos:e.start])
            out.append(e.new)
            pos = e.end
        out.append(self.text[pos:])
        return "".join(out)


def neutralize(text: str) -> CellResult:
    """Neutralize one cell of text."""
    cell = _Cell(text)
    cell.run()
    return CellResult(before=text, after=cell.result(), notes=cell.notes)


@dataclass
class PronounChange:
    file: str
    row: str            # "#N"
    key: str            # ID columns, e.g. participant_id=sub-##
    column: str
    before: str
    after: str
    notes: list[str]

    @property
    def status(self) -> str:
        if self.before == self.after:
            return "review"
        return "changed-review" if self.notes else "changed"


@dataclass
class PronounResult:
    file: str
    columns: list[str]
    changes: list[PronounChange]

    @property
    def n_changed(self) -> int:
        return sum(c.before != c.after for c in self.changes)

    @property
    def n_review(self) -> int:
        return sum(bool(c.notes) for c in self.changes)


def plan_pronouns(stage: Stage, rel: str = DEBRIEFING,
                  columns: list[str] | None = None) -> PronounResult:
    """Neutralize the free-text columns of ``rel``; queue the rewrite."""
    if not stage.exists(rel):
        raise FileNotFoundError(f"{stage.path(rel)} not found (pronoun step)")
    table = Table(stage.read_text(rel), name=rel)
    if columns is None:
        columns = text_columns(table)
    else:
        for c in columns:
            table.col(c)                                  # KeyError if absent
            if is_id_column(c):
                raise ValueError(f"{rel}: refusing to rewrite ID column {c!r}")
    changes = []
    for i in range(len(table.rows)):
        for c in columns:
            res = neutralize(table.get(i, c))
            if res.changed or res.notes:
                changes.append(PronounChange(rel, f"#{i + 1}", row_key(table, i), c,
                                             res.before, res.after, res.notes))
            if res.changed:
                table.set(i, c, res.after)
    if table.changed:
        stage.queue_write(rel, table.to_text())
    return PronounResult(rel, columns, changes)


def report_tsv(result: PronounResult) -> str:
    """Review report: every changed or flagged cell, before and after.
    Plain tab-separated, like the TSV it describes (cells hold no tabs)."""
    lines = ["file\trow\tkey\tcolumn\tstatus\tbefore\tafter\tnotes"]
    for c in result.changes:
        lines.append("\t".join([c.file, c.row, c.key, c.column, c.status, c.before,
                                c.after, " | ".join(c.notes)]))
    return "\n".join(lines) + "\n"
