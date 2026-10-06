"""NAT events carry the registry's canonical movie_name and its stimulus_id.

The PsychoPy CSVs spell one film two ways (a casing variant). Before
2026-10-06 events kept the source spelling, every consumer was meant to
reconcile it, and several matched the exact string instead. The converters now
resolve each title through stimuli/stimulus_registry/movies.tsv, and
build_stimulus_registry.py --validate-events rejects any non-canonical name.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
SCRIPTS = REPO_ROOT / "scripts"
for p in (CONVERTERS, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The converters import their helpers by bare name, so the module they share
# state with is `common`, not `raw2bids_converters.common`.
import common  # noqa: E402
import psychopy_encoding  # noqa: E402
import build_stimulus_registry as bsr  # noqa: E402

# Synthetic films: the logic does not depend on real titles.
REGISTRY = (
    "stimulus_id\tmovie_name\tmovie_name_variants\tvideo_file\tcue_file\t"
    "annotation_file\tstyle\tduration_s\n"
    "film-one\tFilm One\tfilm one|FILM 1\tv\tc\ta\tAnimated, speech\t100\n"
    "film-two\tFilm Two\t\tv\tc\ta\tLive-action, speech\t120\n"
)


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "movies.tsv"
    path.write_text(REGISTRY)
    return common.load_movie_index(path)


@pytest.mark.parametrize("spelling", ["Film One", "film one", "FILM 1", " Film One "])
def test_every_declared_spelling_resolves_to_the_canonical(index, spelling):
    assert common.resolve_movie(spelling, index) == ("film-one", "Film One")


def test_an_undeclared_title_raises(index):
    with pytest.raises(ValueError, match="TITLE_VARIANTS"):
        common.resolve_movie("Film Three", index)


def test_a_missing_registry_names_its_path(tmp_path):
    missing = tmp_path / "nope" / "movies.tsv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        common.load_movie_index(missing)


def _encoding_csv(tmp_path, titles):
    """A minimal PsychoPy free-recall encoding CSV, one trial row per title."""
    rows = []
    for i, title in enumerate(titles):
        t0 = 20.0 + 150 * i
        rows.append({
            "movie_loop.thisN": i, "wait.started": 8.0,
            "movie_title.started": t0, "movie_title.stopped": t0 + 2,
            "fixation.started": t0 + 2, "fixation.stopped": t0 + 2.5,
            "movies.started": t0 + 2.5, "movies.stopped": t0 + 102.5,
            "condition": 1, "movie_name": title, "mov_len": 100.0,
            "style": "Animated, speech", "free_recall_position": i + 1,
            "blank_20.started": None, "stop_eyetracking.stopped": None,
        })
    rows.append({"blank_20.started": 400.0, "stop_eyetracking.stopped": 420.0})
    path = tmp_path / "98_1_1_mem_search_recall_YYYY-MM-DD_00h00.00.000.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_encoding_writes_canonical_name_and_stimulus_id(tmp_path, index, monkeypatch):
    monkeypatch.setattr(common, "_MOVIE_INDEX", index)
    src = _encoding_csv(tmp_path, ["film one", "Film Two"])
    out = tmp_path / "sub-##_ses-##_task-NATencoding_run-01_events.tsv"
    assert psychopy_encoding.convert_file(str(src), str(out))

    ev = pd.read_csv(out, sep="\t", keep_default_na=False)
    movies = ev[ev.trial_type == "movie"]
    assert movies.movie_name.tolist() == ["Film One", "Film Two"]
    assert movies.stimulus_id.tolist() == ["film-one", "film-two"]
    others = ev[ev.trial_type != "movie"]
    assert (others.stimulus_id == "n/a").all()
    assert "" not in ev.to_numpy()


def test_encoding_refuses_an_unknown_title(tmp_path, index, monkeypatch):
    monkeypatch.setattr(common, "_MOVIE_INDEX", index)
    src = _encoding_csv(tmp_path, ["Film Three"])
    with pytest.raises(ValueError, match="Film Three"):
        psychopy_encoding.convert_file(str(src), str(tmp_path / "x_events.tsv"))


def _tables():
    return {
        "movies": REGISTRY,
        "shared1000": "stimulus_id\tmmmId\n",
        "twp1000": "stimulus_id\n",
    }


def _events(root, rows):
    func = root / "sub-##" / "ses-##" / "func"
    func.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(func / "sub-##_ses-##_task-NATencoding_run-01_events.tsv",
                              sep="\t", index=False)


def test_validate_events_accepts_canonical_rows(tmp_path):
    _events(tmp_path, [{"onset": 1, "movie_name": "Film One", "stimulus_id": "film-one"},
                       {"onset": 2, "movie_name": "n/a", "stimulus_id": "n/a"}])
    bsr.validate_events(_tables(), tmp_path)


@pytest.mark.parametrize("row", [
    {"onset": 1, "movie_name": "film one", "stimulus_id": "film-one"},   # declared variant
    {"onset": 1, "movie_name": "Film One", "stimulus_id": "film-two"},   # wrong id
    {"onset": 1, "movie_name": "Film Three", "stimulus_id": "n/a"},      # unknown film
])
def test_validate_events_rejects_noncanonical_rows(tmp_path, row):
    _events(tmp_path, [row])
    with pytest.raises(SystemExit):
        bsr.validate_events(_tables(), tmp_path)


def test_registry_scan_reads_na_as_missing(tmp_path, monkeypatch):
    """The builder reads the live events with the csv module, so BIDS n/a is
    a string there, not NaN. It must count as missing, not as a film."""
    _events(tmp_path, [
        {"onset": 1, "trial_type": "title", "movie_name": "n/a", "movie_length": "n/a"},
        {"onset": 2, "trial_type": "movie", "movie_name": "Film One", "movie_length": 100.0},
    ])
    (tmp_path / "sub-##" / "ses-##" / "func" /
     "sub-##_ses-##_task-NATencoding_run-01_events.tsv").rename(
        tmp_path / "sub-##" / "ses-##" / "func" / "sub-##_ses-##_task-NATx_events.tsv")
    monkeypatch.setattr(bsr, "BIDS_ROOT", tmp_path)
    counts, durations = bsr.scan_nat_events()
    assert counts == {"Film One": 1}
    assert durations == {"film one": {100.0}}
