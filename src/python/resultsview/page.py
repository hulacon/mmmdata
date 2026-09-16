"""
page.py — one self-contained HTML page from result tables + Vega-Lite specs.

The input contract (schema_version 1) is a pair of files side by side:

    <stem>.tsv        long-format table, one row per plotted value; BIDS
                      ``n/a`` and empty cells are missing
    <stem>.vl.json    a Vega-Lite spec for it, with no data of its own

The spec carries mmmview's fields under Vega-Lite's own free-form slot,
``usermeta.mmmview``, so the file stays a valid spec any Vega tool opens:

    "usermeta": {"mmmview": {
        "schema_version": 1,
        "err_over": "trials",          # REQUIRED: what the error bars range
                                       # over (trials, sessions, subjects,
                                       # pairs, ...), or "none" when the
                                       # chart draws no error
        "table": "<stem>.tsv",         # optional; default: same stem
        "caption": "..."               # optional prose shown under the chart
    }}

Rules, each refused with a message naming the fix:

- the table is the single source of the numbers: no ``data.values`` and no
  ``data.url`` anywhere in the spec (constants go through ``datum``);
- ``err_over`` is required, and cannot be ``"none"`` when the spec draws an
  errorbar/errorband mark or an xError/yError channel;
- every ``field`` the spec names is a table column or something one of its
  transforms creates (a typo otherwise renders as an empty chart).

Columns are typed here, not in the browser: a column is numeric when every
value parses as a number and none carries a leading zero, so subject labels
like ``03`` stay strings.

The page inlines the pinned Vega builds (``vendor/VENDOR.md``) and the data,
so it opens over file:// with no network.
"""

import datetime
import json
import re
from html import escape as _escape
from pathlib import Path

SPEC_EXT = ".vl.json"
TABLE_EXT = ".tsv"
SCHEMA_VERSION = 1
PAGE_VERSION = 2          # bump when the page shell or the theme changes

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
VENDOR_JS = ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js")
VEGA_VERSIONS = {"vega": "6.4.0", "vega-lite": "6.4.3", "vega-embed": "7.2.0"}

ERROR_MARKS = {"errorbar", "errorband"}
ERROR_CHANNELS = {"xError", "yError", "xError2", "yError2"}
MISSING = {"", "n/a", "NA", "nan", "NaN"}

# Provisional house theme: mmmview's dark page, the categorical hues of the
# dataviz reference palette. Harmonizing it with the dataviz skill is the
# authoring skill's job (workbench mmmview-browse, 2026-09-16 decision).
THEME = {
    "background": "transparent",
    "font": "system-ui, sans-serif",
    "view": {"stroke": "transparent"},
    "axis": {"domainColor": "#44445a", "gridColor": "#2c2c38",
             "tickColor": "#44445a", "labelColor": "#9a9aae",
             "titleColor": "#c8c8d4", "titleFontWeight": 500},
    "mark": {"color": "#9a9aae"},   # unencoded marks + shape legends
    "legend": {"labelColor": "#9a9aae", "titleColor": "#c8c8d4",
               "symbolFillColor": "#9a9aae"},  # shape legends; color
                                                # legends set their own fill
    "header": {"labelColor": "#c8c8d4", "titleColor": "#c8c8d4"},
    "title": {"color": "#e8e8ee", "anchor": "start", "fontWeight": 600},
    "range": {"category": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                           "#e87ba4", "#8f6fd6", "#6b9e2f", "#c2573a"]},
    "rule": {"color": "#6a6a7e"},
    "text": {"color": "#c8c8d4"},
}


class SpecError(ValueError):
    """A spec or its table breaks the contract; the message names the fix."""


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_LEADING_ZERO = re.compile(r"^[+-]?0\d")


def read_table(path):
    """(columns, rows) from a TSV, rows as dicts with typed values."""
    path = Path(path)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SpecError(f"{path} is empty; a results table needs a header "
                        "row and at least one data row")
    columns = lines[0].split("\t")
    if len(set(columns)) != len(columns):
        raise SpecError(f"{path.name} repeats a column name in its header")
    raw = []
    for n, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise SpecError(f"{path.name} line {n} has {len(cells)} cells, "
                            f"the header has {len(columns)}")
        raw.append(cells)
    numeric = []
    for i in range(len(columns)):
        vals = [r[i] for r in raw if r[i] not in MISSING]
        numeric.append(bool(vals)
                       and all(_NUMBER.match(v) for v in vals)
                       and not any(_LEADING_ZERO.match(v) for v in vals))
    rows = []
    for r in raw:
        row = {}
        for col, v, is_num in zip(columns, r, numeric):
            if v in MISSING:
                row[col] = None
            elif is_num:
                f = float(v)
                row[col] = int(f) if f.is_integer() and re.match(
                    r"^[+-]?\d+$", v) else f
            else:
                row[col] = v
        rows.append(row)
    return columns, rows


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

def spec_stem(path):
    name = Path(path).name
    return name[: -len(SPEC_EXT)] if name.endswith(SPEC_EXT) else Path(path).stem


def _walk(node, fn, key=None):
    fn(node, key)
    if isinstance(node, dict):
        for k, v in node.items():
            _walk(v, fn, k)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn, key)


# output names a transform creates when its `as` is omitted
_DEFAULT_AS = {"fold": ("key", "value"), "density": ("value", "density"),
               "quantile": ("prob", "value")}


def _fields_used(spec):
    """(fields the spec reads, names its transforms create, checkable).
    checkable is False when a pivot is present: its new columns come from
    data values, so no static check can know them."""
    used, made, checkable = set(), set(), [True]

    def fn(node, key):
        if not isinstance(node, dict):
            return
        f = node.get("field")
        if isinstance(f, str):
            used.add(f)
        v = node.get("as")
        made.update([v] if isinstance(v, str) else
                    [x for x in (v if isinstance(v, list) else [])
                     if isinstance(x, str)])
        for t, names in _DEFAULT_AS.items():
            if t in node and "as" not in node:
                made.update(names)
        if "pivot" in node:
            checkable[0] = False

    _walk(spec, fn)
    return used, made, checkable[0]


def _top_field(name):
    """'a.b' reads field a (nested); 'a\\.b' is the literal column 'a.b'."""
    parts = re.split(r"(?<!\\)\.", name)
    return parts[0].replace("\\.", ".").replace("\\[", "[")


def _has_error_marks(spec):
    found = []

    def fn(node, key):
        if not isinstance(node, dict):
            return
        if key == "mark":
            return
        mark = node.get("mark")
        mtype = mark.get("type") if isinstance(mark, dict) else mark
        if mtype in ERROR_MARKS:
            found.append(mtype)
        enc = node.get("encoding")
        if isinstance(enc, dict):
            found.extend(c for c in enc if c in ERROR_CHANNELS)

    _walk(spec, fn)
    return found


def load_spec(spec_path):
    """Validate one spec against its table. Returns a dict with the spec
    (data not yet attached), its mmmview block, the table path, columns and
    rows. Raises SpecError naming the fix."""
    spec_path = Path(spec_path)
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, ValueError) as exc:
        raise SpecError(f"cannot read {spec_path.name} as JSON: {exc}")
    if not isinstance(spec, dict):
        raise SpecError(f"{spec_path.name} is not a Vega-Lite spec object")
    meta = (spec.get("usermeta") or {}).get("mmmview")
    if not isinstance(meta, dict):
        raise SpecError(
            f"{spec_path.name} has no usermeta.mmmview block; add "
            '"usermeta": {"mmmview": {"schema_version": 1, "err_over": '
            '"<trials|sessions|subjects|...|none>"}}')
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(
            f"{spec_path.name}: usermeta.mmmview.schema_version is "
            f"{meta.get('schema_version')!r}; this mmmview reads "
            f"{SCHEMA_VERSION}")
    err_over = meta.get("err_over")
    if not isinstance(err_over, str) or not err_over.strip():
        raise SpecError(
            f"{spec_path.name}: usermeta.mmmview.err_over is required — "
            "name what the error bars range over (trials, sessions, "
            'subjects, pairs, ...), or "none" when the chart draws none')

    offenders = []

    def no_data(node, key):
        if key == "data" and isinstance(node, dict) and (
                "values" in node or "url" in node):
            offenders.append("url" if "url" in node else "values")

    _walk(spec, no_data)
    if offenders:
        raise SpecError(
            f"{spec_path.name} carries its own data.{offenders[0]}; the "
            "table is the single source — delete it (mmmview attaches the "
            "table), and draw constants with datum encodings instead")

    errs = _has_error_marks(spec)
    if errs and err_over.strip().lower() == "none":
        raise SpecError(
            f"{spec_path.name} draws error ({', '.join(sorted(set(errs)))}) "
            'but usermeta.mmmview.err_over is "none"; name what it ranges '
            "over")

    table = spec_path.parent / meta.get("table",
                                        spec_stem(spec_path) + TABLE_EXT)
    if not table.is_file():
        raise SpecError(
            f"{spec_path.name} reads {table.name}, which does not exist "
            f"beside it; write the table or set usermeta.mmmview.table")
    columns, rows = read_table(table)
    used, made, checkable = _fields_used(spec)
    unknown = sorted({_top_field(f) for f in used}
                     - set(columns) - {_top_field(m) for m in made}) \
        if checkable else []
    if unknown:
        raise SpecError(
            f"{spec_path.name} names field(s) {', '.join(unknown)} that "
            f"{table.name} does not have (columns: {', '.join(columns)})")
    return {"path": spec_path, "spec": spec, "meta": meta, "table": table,
            "columns": columns, "rows": rows}


def specs_for_table(table):
    """Specs in the table's directory that read it (by default stem or by
    an explicit usermeta.mmmview.table). Unreadable specs are skipped here;
    load_spec reports them when they are asked for directly."""
    table = Path(table)
    out = []
    for p in sorted(table.parent.glob("*" + SPEC_EXT)):
        try:
            meta = (json.loads(p.read_text()).get("usermeta") or {}) \
                .get("mmmview") or {}
        except (OSError, ValueError, AttributeError):
            meta = {}
        want = meta.get("table", spec_stem(p) + TABLE_EXT)
        if (p.parent / want).resolve() == table.resolve():
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<!--
__NOTES__
-->
<style>
  :root { --bg:#101014; --panel:#1a1a22; --ink:#e8e8ee; --edge:#2c2c38;
          --dim:#9a9aae; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 system-ui, sans-serif; padding:0 0 40px; }
  header { padding:10px 18px; border-bottom:1px solid var(--edge);
           background:var(--panel); }
  header h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  header p { margin:2px 0; color:var(--dim); font-size:12px; }
  nav { padding:8px 18px; font-size:13px; display:flex; flex-wrap:wrap;
        gap:4px 14px; border-bottom:1px solid var(--edge); }
  a { color:#8ab4f8; text-decoration:none; }
  a:hover { text-decoration:underline; }
  main { padding:0 18px; }
  section { margin:18px 0; padding:12px 14px; background:var(--panel);
            border:1px solid var(--edge); border-radius:8px; }
  section h2 { font-size:14px; margin:0 0 4px; font-weight:600; }
  .meta { color:var(--dim); font-size:12px; margin:0 0 8px; }
  .meta b { color:var(--ink); font-weight:600; }
  .caption { margin:8px 0 0; font-size:13px; max-width:80ch; }
  .chart { overflow-x:auto; }
  .err { color:#f28b82; font-family:ui-monospace, monospace; font-size:12px; }
  details { margin-top:8px; font-size:12px; }
  summary { cursor:pointer; color:var(--dim); }
  .tbl { overflow-x:auto; max-height:360px; margin-top:6px; }
  table { border-collapse:collapse; font:12px/1.3 ui-monospace, monospace; }
  th, td { padding:2px 8px; border-bottom:1px solid var(--edge);
           text-align:right; white-space:nowrap; }
  th { position:sticky; top:0; background:var(--panel); color:var(--dim); }
  code { background:var(--bg); border:1px solid var(--edge);
         border-radius:4px; padding:1px 5px; color:var(--dim);
         font:12px/1.4 ui-monospace, monospace; }
</style>
</head>
<body>
<header>
<h1>__TITLE__</h1>
<p>regenerate with: <code>__COMMAND__</code></p>
<p>built __DATE__ by mmmview · Vega __VEGA__, Vega-Lite __VEGALITE__</p>
__HEADER_EXTRA__
</header>
__NAV__
<main>
__SECTIONS__
</main>
<script type="application/json" id="mmm-charts">__CHARTS__</script>
<script type="application/json" id="mmm-theme">__THEME__</script>
<script>__VENDOR__</script>
<script>
"use strict";
const charts = JSON.parse(document.getElementById("mmm-charts").textContent);
const theme = JSON.parse(document.getElementById("mmm-theme").textContent);
for (const c of charts) {
  const el = document.getElementById(c.id);
  vegaEmbed(el, c.spec, {config: theme, renderer: "svg",
      actions: {export: true, source: false, compiled: false, editor: false}})
    .catch(err => {
      el.innerHTML = "";
      const p = document.createElement("p");
      p.className = "err";
      p.textContent = "Vega-Lite could not draw this spec: " + err.message;
      el.appendChild(p);
    });
}
</script>
</body>
</html>
"""


def _json_for_script(obj):
    """JSON safe inside a <script> element."""
    return (json.dumps(obj, allow_nan=False)
            .replace("</", "<\\/").replace("<!--", "<\\!--"))


def _cell(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _section(i, loaded):
    spec, meta = loaded["spec"], loaded["meta"]
    title = spec.get("title")
    if isinstance(title, dict):
        title = title.get("text")
    if isinstance(title, list):
        title = " ".join(map(str, title))
    heading = title or spec_stem(loaded["path"])
    cols, rows = loaded["columns"], loaded["rows"]
    head = "".join(f"<th>{_escape(c)}</th>" for c in cols)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_escape(_cell(r[c]))}</td>" for c in cols)
        + "</tr>" for r in rows)
    caption = meta.get("caption")
    return (
        f'<section id="s{i}">\n<h2>{_escape(str(heading))}</h2>\n'
        f'<p class="meta">error bars over: <b>{_escape(meta["err_over"])}</b>'
        f' · table <code>{_escape(loaded["table"].name)}</code> '
        f'({len(rows)} rows) · spec <code>{_escape(loaded["path"].name)}'
        f'</code></p>\n'
        f'<div class="chart" id="c{i}"></div>\n'
        + (f'<p class="caption">{_escape(caption)}</p>\n' if caption else "")
        + f'<details><summary>table</summary><div class="tbl"><table>'
          f'<thead><tr>{head}</tr></thead><tbody>\n{body}\n</tbody></table>'
          f'</div></details>\n</section>')


def vendor_js():
    missing = [n for n in VENDOR_JS if not (VENDOR_DIR / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"vendored Vega build(s) missing from {VENDOR_DIR}: "
            f"{', '.join(missing)} — see vendor/VENDOR.md to restore them")
    return "\n;\n".join((VENDOR_DIR / n).read_text() for n in VENDOR_JS)


def build_results_page(spec_paths, out, title="results", notes="",
                       command="", header_extra=""):
    """Validate every spec, then write one page with a chart section each.
    Raises SpecError before writing anything if any spec fails.
    header_extra is trusted HTML placed in the header (mmmview passes its
    Refresh widget)."""
    loaded = [load_spec(p) for p in spec_paths]
    if not loaded:
        raise SpecError("no specs to draw")
    charts, sections = [], []
    for i, item in enumerate(loaded):
        spec = dict(item["spec"])
        spec.pop("title", None)     # the section heading shows it
        spec["data"] = {"values": item["rows"]}
        charts.append({"id": f"c{i}", "spec": spec})
        sections.append(_section(i, item))
    nav = ""
    if len(loaded) > 1:
        links = []
        for i, item in enumerate(loaded):
            t = item["spec"].get("title")
            t = t.get("text") if isinstance(t, dict) else t
            links.append(f'<a href="#s{i}">'
                         f'{_escape(str(t or spec_stem(item["path"])))}</a>')
        nav = "<nav>" + "".join(links) + "</nav>"
    html = _TEMPLATE
    for token, value in (
            ("__NOTES__", notes.replace("--", "- -")),
            ("__TITLE__", _escape(title)),
            ("__COMMAND__", _escape(command)),
            ("__DATE__", datetime.date.today().isoformat()),
            ("__VEGA__", VEGA_VERSIONS["vega"]),
            ("__VEGALITE__", VEGA_VERSIONS["vega-lite"]),
            ("__HEADER_EXTRA__", header_extra),
            ("__NAV__", nav),
            ("__SECTIONS__", "\n".join(sections)),
            ("__CHARTS__", _json_for_script(charts)),
            ("__THEME__", _json_for_script(THEME))):
        html = html.replace(token, value)
    # the vendor JS goes in last: its text must never be token-substituted
    html = html.replace("__VENDOR__", vendor_js().replace("</script",
                                                          "<\\/script"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
