"""mmmget.runs(): catalog-backed runs with QC exclusions applied by default."""

import duckdb
import pytest

import mmmget
from neuroimaging.io import FmriprepRun
from tests.mmmget_fixtures import FMRIPREP, make_env


@pytest.fixture
def bids(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


def _keys(runs):
    return [(r.subject, r.task, r.run) for r in runs]


class TestQcExclusion:
    def test_person_exclude_is_dropped_by_default(self, bids):
        got = _keys(mmmget.runs(subject="01", task="memory"))
        assert got == [("01", "memory", "01"), ("01", "memory", "03")]

    def test_automated_recommendation_never_excludes(self, bids):
        run3 = mmmget.runs(subject="01", task="memory", run=3)
        assert len(run3) == 1 and not run3[0].excluded
        assert run3[0].qc_decision["decision"] == "pending"

    def test_old_schema_in_legacy_dir_still_excludes(self, bids):
        # {latest, history} files used to be read as empty histories
        got = _keys(mmmget.runs(subject="sub-02", task="memory"))
        assert ("02", "memory", "01") not in got

    def test_newer_domain_note_does_not_override_exclude(self, bids):
        got = _keys(mmmget.runs(subject="02", task="memory"))
        assert got == [("02", "memory", "03")]

    def test_include_excluded_returns_them_flagged(self, bids):
        runs = mmmget.runs(subject="01", task="memory", include_excluded=True)
        assert [r.excluded for r in runs] == [False, True, False]

    def test_include_excluded_is_recorded_non_standard(self, bids):
        mmmget.runs(subject="01", include_excluded=True)
        (rec,) = mmmget.provenance_records()
        assert rec["getter"] == "runs" and rec["standard"] is False
        assert rec["n_excluded"] == 1

    def test_default_call_is_recorded_standard(self, bids):
        mmmget.runs(subject="01")
        (rec,) = mmmget.provenance_records()
        assert rec["standard"] is True and rec["query"] == {"sub": "01"}


class TestFromCatalog:
    def test_paths_are_dataset_relative_under_configured_root(self, bids):
        (r,) = mmmget.runs(subject="01", task="memory", run=1)
        func = bids / FMRIPREP / "sub-01" / "ses-01" / "func"
        assert r.confounds == func / "sub-01_ses-01_task-memory_run-01_desc-confounds_timeseries.tsv"
        assert r.bold.name.endswith("_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        assert r.boldref.name.endswith("_res-2_boldref.nii.gz")
        assert r.surface_L.name.endswith("_hemi-L_space-fsaverage6_bold.func.gii")
        assert r.events == bids / "sub-01/ses-01/func/sub-01_ses-01_task-memory_run-01_events.tsv"

    def test_native_space_picks_unspaced_files_and_coreg_ref(self, bids):
        (r,) = mmmget.runs(subject="01", task="memory", run=1, space="func")
        assert r.bold.name == "sub-01_ses-01_task-memory_run-01_desc-preproc_bold.nii.gz"
        assert r.boldref.name.endswith("_desc-coreg_boldref.nii.gz")

    def test_is_an_fmriprep_run_with_run_key(self, bids):
        (r,) = mmmget.runs(subject="01", task="memory", run="run-01")
        assert isinstance(r, FmriprepRun)
        assert r.run_key == "sub-01_ses-01_task-memory_run-01_bold"

    def test_single_run_task_has_no_run_label(self, bids):
        (r,) = mmmget.runs(subject="01", task="rest")
        assert r.run is None and r.entity_prefix == "sub-01_ses-01_task-rest"

    def test_no_directory_walk(self, bids):
        # catalogued runs come back though their files are absent from disk...
        assert all(not r.bold.exists() for r in mmmget.runs(subject="01"))
        # ...and a confounds TSV on disk but not in the catalog does not
        extra = bids / FMRIPREP / "sub-01/ses-09/func/sub-01_ses-09_task-memory_run-01_desc-confounds_timeseries.tsv"
        extra.parent.mkdir(parents=True)
        extra.write_text("trans_x\n0\n")
        assert all(r.session == "01" for r in mmmget.runs(subject="01", include_excluded=True))

    def test_raw_bold_without_fmriprep_is_not_a_run(self, bids):
        with pytest.raises(LookupError, match="ses='02'"):
            mmmget.runs(subject="01", session="02")

    def test_unknown_label_is_an_error_not_an_empty_list(self, bids):
        with pytest.raises(LookupError, match="task='Memory'.*known values"):
            mmmget.runs(task="Memory")

    def test_duplicate_confounds_for_a_run_is_loud(self, tmp_path, monkeypatch):
        bids = make_env(tmp_path, monkeypatch)
        con = duckdb.connect(str(bids / "inventory" / "catalog.duckdb"))
        con.execute("""INSERT INTO files SELECT dataset_relpath, sub, ses, datatype, task, run,
                       space, res, hemi, "desc", suffix, ext, replace(path, '_desc', '_acq-x_desc')
                       FROM files WHERE sub = '01' AND task = 'rest' AND "desc" = 'confounds'""")
        con.close()
        with pytest.raises(ValueError, match="2 confounds TSV files"):
            mmmget.runs(subject="01", task="rest")

    def test_missing_catalog_names_the_fix(self, tmp_path, monkeypatch):
        with pytest.raises(FileNotFoundError, match="duckbrain.catalog rebuild"):
            mmmget.runs(bids_root=tmp_path)

    def test_unknown_variant(self, bids):
        with pytest.raises(ValueError, match="variant"):
            mmmget.runs(variant="fmriprep_x")
