"""Results pages (src/python/resultsview/page.py): the table + Vega-Lite spec
contract, and the self-contained page it writes. Workbench mmmview-browse,
increment 6."""

import hashlib
import json
import re
from pathlib import Path

import pytest

# src/python is on sys.path via the repo-root conftest
from resultsview import page  # noqa: E402

VENDOR_SHA256 = {
    "vega.min.js": "8f6a3587cf8d4f42c7e08120e3eb05d067e746d554e39d2dcf52acc0bd5ba28f",
    "vega-lite.min.js": "35a9821df838825b05a6a73e9414b58747a1b18321583858ed903c66393a5c7e",
    "vega-embed.min.js": "b69eac2846a0061683b7e03501790fb0bcbdb851c797c6baf3417c9d8852819e",
}

TABLE = ("subject\tcondition\taccuracy\tse\n"
         "03\tsingle\t0.61\t0.020\n"
         "04\tsingle\t0.72\t0.030\n"
         "05\tsingle\tn/a\t\n")


def spec(err_over="trials", **extra):
    s = {"$schema": "https://vega.github.io/schema/vega-lite/v6.json",
         "title": "accuracy by condition",
         "usermeta": {"mmmview": {"schema_version": 1, "err_over": err_over}},
         "mark": "point",
         "encoding": {"x": {"field": "condition", "type": "nominal"},
                      "y": {"field": "accuracy", "type": "quantitative"},
                      "color": {"field": "subject", "type": "nominal"}}}
    s.update(extra)
    return s


def write_pair(d, stem="acc", table=TABLE, sp=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.tsv").write_text(table)
    p = d / f"{stem}.vl.json"
    p.write_text(json.dumps(sp if sp is not None else spec()))
    return p


# ---------------------------------------------------------------------------
# vendor pin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(VENDOR_SHA256))
def test_vendored_vega_pinned(name):
    digest = hashlib.sha256((page.VENDOR_DIR / name).read_bytes()).hexdigest()
    assert digest == VENDOR_SHA256[name], (
        f"{name} changed; update the hash here, VEGA_VERSIONS in page.py and "
        "vendor/VENDOR.md together, deliberately")


def test_versions_match_vendor_md():
    md = (page.VENDOR_DIR / "VENDOR.md").read_text()
    for pkg, version in page.VEGA_VERSIONS.items():
        assert f"`{pkg}` {version}" in md


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

class TestReadTable:
    def test_types_and_missing(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text(TABLE)
        cols, rows = page.read_table(p)
        assert cols == ["subject", "condition", "accuracy", "se"]
        assert rows[0] == {"subject": "03", "condition": "single",
                           "accuracy": 0.61, "se": 0.02}
        assert rows[2]["accuracy"] is None and rows[2]["se"] is None

    def test_leading_zero_labels_stay_strings(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("subject\tn\n03\t12\n10\t7\n")
        _, rows = page.read_table(p)
        assert [r["subject"] for r in rows] == ["03", "10"]
        assert [r["n"] for r in rows] == [12, 7]

    def test_signed_and_zero_values_are_numbers(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("delta\n-0.0000\n+0.0033\n0.5\n")
        _, rows = page.read_table(p)
        assert [r["delta"] for r in rows] == [0.0, 0.0033, 0.5]

    def test_ragged_row_is_refused(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("a\tb\n1\n")
        with pytest.raises(page.SpecError, match="line 2 has 1 cells"):
            page.read_table(p)

    def test_empty_table_is_refused(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("")
        with pytest.raises(page.SpecError, match="empty"):
            page.read_table(p)


# ---------------------------------------------------------------------------
# spec contract
# ---------------------------------------------------------------------------

class TestLoadSpec:
    def test_valid_pair(self, tmp_path):
        got = page.load_spec(write_pair(tmp_path))
        assert got["table"] == tmp_path / "acc.tsv"
        assert got["meta"]["err_over"] == "trials"
        assert len(got["rows"]) == 3

    def test_missing_usermeta_names_the_block(self, tmp_path):
        s = spec()
        del s["usermeta"]
        with pytest.raises(page.SpecError, match="usermeta.mmmview"):
            page.load_spec(write_pair(tmp_path, sp=s))

    def test_wrong_schema_version(self, tmp_path):
        s = spec()
        s["usermeta"]["mmmview"]["schema_version"] = 2
        with pytest.raises(page.SpecError, match="schema_version"):
            page.load_spec(write_pair(tmp_path, sp=s))

    @pytest.mark.parametrize("err", [None, "", "  "])
    def test_err_over_is_required(self, tmp_path, err):
        s = spec()
        if err is None:
            del s["usermeta"]["mmmview"]["err_over"]
        else:
            s["usermeta"]["mmmview"]["err_over"] = err
        with pytest.raises(page.SpecError, match="err_over is required"):
            page.load_spec(write_pair(tmp_path, sp=s))

    def test_error_marks_cannot_claim_none(self, tmp_path):
        s = spec(err_over="none")
        s.pop("mark")
        s.pop("encoding")
        s["layer"] = [{"mark": {"type": "errorbar"},
                       "encoding": {"y": {"field": "accuracy",
                                          "type": "quantitative"},
                                    "yError": {"field": "se"}}}]
        with pytest.raises(page.SpecError, match="draws error"):
            page.load_spec(write_pair(tmp_path, sp=s))

    def test_none_is_fine_without_error_marks(self, tmp_path):
        assert page.load_spec(write_pair(tmp_path, sp=spec("none")))

    @pytest.mark.parametrize("data", [{"values": [{"a": 1}]},
                                      {"url": "elsewhere.csv"}])
    def test_spec_data_is_refused(self, tmp_path, data):
        with pytest.raises(page.SpecError, match="single source"):
            page.load_spec(write_pair(tmp_path, sp=spec(data=data)))

    def test_nested_data_is_refused_too(self, tmp_path):
        s = spec()
        s["layer"] = [{"data": {"values": [{"x": 0.5}]}, "mark": "rule"}]
        with pytest.raises(page.SpecError, match="single source"):
            page.load_spec(write_pair(tmp_path, sp=s))

    def test_unknown_field_is_named(self, tmp_path):
        s = spec()
        s["encoding"]["y"]["field"] = "acuracy"
        with pytest.raises(page.SpecError, match="acuracy"):
            page.load_spec(write_pair(tmp_path, sp=s))

    def test_transform_outputs_count_as_fields(self, tmp_path):
        s = spec(transform=[{"calculate": "datum.accuracy - datum.se",
                             "as": "lo"},
                            {"fold": ["accuracy", "se"]}])
        s["encoding"]["y"]["field"] = "lo"
        s["encoding"]["size"] = {"field": "value", "type": "quantitative"}
        assert page.load_spec(write_pair(tmp_path, sp=s))

    def test_pivot_skips_the_field_check(self, tmp_path):
        s = spec(transform=[{"pivot": "condition", "value": "accuracy"}])
        s["encoding"]["y"]["field"] = "single"
        assert page.load_spec(write_pair(tmp_path, sp=s))

    def test_missing_table_names_the_file(self, tmp_path):
        p = write_pair(tmp_path)
        (tmp_path / "acc.tsv").unlink()
        with pytest.raises(page.SpecError, match="acc.tsv"):
            page.load_spec(p)

    def test_explicit_table(self, tmp_path):
        s = spec()
        s["usermeta"]["mmmview"]["table"] = "shared.tsv"
        (tmp_path / "shared.tsv").write_text(TABLE)
        p = tmp_path / "view.vl.json"
        p.write_text(json.dumps(s))
        assert page.load_spec(p)["table"].name == "shared.tsv"

    def test_bad_json(self, tmp_path):
        p = tmp_path / "x.vl.json"
        p.write_text("{nope")
        with pytest.raises(page.SpecError, match="as JSON"):
            page.load_spec(p)


def test_specs_for_table_by_stem_and_by_name(tmp_path):
    write_pair(tmp_path, stem="acc")
    s = spec()
    s["usermeta"]["mmmview"]["table"] = "acc.tsv"
    (tmp_path / "acc_bars.vl.json").write_text(json.dumps(s))
    (tmp_path / "other.vl.json").write_text(json.dumps(spec()))
    got = [p.name for p in page.specs_for_table(tmp_path / "acc.tsv")]
    assert got == ["acc.vl.json", "acc_bars.vl.json"]


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

def _charts(html):
    m = re.search(r'<script type="application/json" id="mmm-charts">(.*?)'
                  r'</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


class TestPage:
    def test_self_contained_with_data_inlined(self, tmp_path):
        out = page.build_results_page([write_pair(tmp_path)],
                                      tmp_path / "viz" / "p.html",
                                      title="t", notes="mmmview-key: abc")
        html = out.read_text()
        assert 'src="http' not in html and "<link" not in html
        assert "vegaEmbed" in html
        charts = _charts(html)
        assert len(charts) == 1
        assert charts[0]["spec"]["data"]["values"][0]["subject"] == "03"
        assert "title" not in charts[0]["spec"]      # shown as the heading

    def test_key_sits_in_the_head_before_the_vendor_js(self, tmp_path):
        """reap reads only the first MiB; the vendored JS is ~830 KB."""
        out = page.build_results_page([write_pair(tmp_path)],
                                      tmp_path / "p.html",
                                      notes="mmmview-key: abc123")
        html = out.read_text()
        assert html.index("mmmview-key: abc123") < 4096

    def test_err_over_and_caption_are_shown(self, tmp_path):
        s = spec()
        s["usermeta"]["mmmview"]["caption"] = "chance is 0.5 <here>"
        out = page.build_results_page([write_pair(tmp_path, sp=s)],
                                      tmp_path / "p.html")
        html = out.read_text()
        assert "error bars over: <b>trials</b>" in html
        assert "chance is 0.5 &lt;here&gt;" in html

    def test_several_specs_get_a_nav_and_sections_in_order(self, tmp_path):
        a = write_pair(tmp_path, stem="a")
        b = write_pair(tmp_path, stem="b", sp=spec(title="second"))
        html = page.build_results_page([a, b], tmp_path / "p.html").read_text()
        assert "<nav>" in html
        assert html.index('id="s0"') < html.index('id="s1"')
        assert len(_charts(html)) == 2

    def test_script_breakout_is_escaped(self, tmp_path):
        table = "label\tv\n</script><b>x\t1\n"
        s = spec()
        s["encoding"] = {"x": {"field": "label", "type": "nominal"},
                         "y": {"field": "v", "type": "quantitative"}}
        html = page.build_results_page(
            [write_pair(tmp_path, table=table, sp=s)],
            tmp_path / "p.html").read_text()
        body = html.split('id="mmm-charts">', 1)[1].split("</script>", 1)[0]
        assert "</script>" not in body
        assert _charts(html)[0]["spec"]["data"]["values"][0]["label"] \
            == "</script><b>x"

    def test_a_bad_spec_writes_nothing(self, tmp_path):
        good = write_pair(tmp_path, stem="a")
        bad = write_pair(tmp_path, stem="b", sp=spec(err_over=""))
        out = tmp_path / "p.html"
        with pytest.raises(page.SpecError):
            page.build_results_page([good, bad], out)
        assert not out.exists()
