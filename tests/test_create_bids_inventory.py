"""Tests for create_bids_inventory script functionality."""

import pytest
from pathlib import Path
import tempfile
import sys

# Add scripts directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from create_bids_inventory import (
    parse_bids_filename,
    create_shorthand_label,
    auto_discover_subjects
)


class TestParseBidsFilename:
    """Tests for parse_bids_filename function."""

    def test_parse_anat_t1w(self):
        """Test parsing anatomical T1w file."""
        filepath = "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['subject'] == '01'
        assert entities['session'] == '01'
        assert entities['datatype'] == 'anat'
        assert entities['suffix'] == 'T1w'
        assert entities['extension'] == '.nii.gz'

    def test_parse_func_bold(self):
        """Test parsing functional BOLD file with task."""
        filepath = "sub-02/ses-baseline/func/sub-02_ses-baseline_task-rest_bold.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['subject'] == '02'
        assert entities['session'] == 'baseline'
        assert entities['task'] == 'rest'
        assert entities['datatype'] == 'func'
        assert entities['suffix'] == 'bold'

    def test_parse_task_with_hyphens(self):
        """Test parsing task names with hyphens."""
        filepath = "sub-01/func/sub-01_task-rest-eyes-open_bold.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['task'] == 'rest-eyes-open'

    def test_parse_alphanumeric_session(self):
        """Test parsing alphanumeric session identifiers."""
        filepath = "sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['session'] == 'pre'

    def test_parse_run_number(self):
        """Test parsing run numbers."""
        filepath = "sub-01/func/sub-01_task-rest_run-02_bold.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['run'] == '02'

    def test_parse_direction(self):
        """Test parsing phase encoding direction."""
        filepath = "sub-01/fmap/sub-01_dir-AP_epi.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['direction'] == 'AP'
        assert entities['datatype'] == 'fmap'

    def test_parse_acquisition(self):
        """Test parsing acquisition parameter."""
        filepath = "sub-01/anat/sub-01_acq-highres_T1w.nii.gz"
        entities = parse_bids_filename(filepath)

        assert entities['acquisition'] == 'highres'

    def test_parse_json_sidecar(self):
        """Test parsing JSON sidecar file."""
        filepath = "sub-01/anat/sub-01_T1w.json"
        entities = parse_bids_filename(filepath)

        assert entities['extension'] == '.json'
        assert entities['suffix'] == 'T1w'

    def test_parse_dwi_bval(self):
        """Test parsing diffusion bval file."""
        filepath = "sub-01/dwi/sub-01_dwi.bval"
        entities = parse_bids_filename(filepath)

        assert entities['datatype'] == 'dwi'
        assert entities['extension'] == '.bval'

    def test_missing_entities(self):
        """Test that missing entities don't appear in result."""
        filepath = "sub-01/anat/sub-01_T1w.nii.gz"
        entities = parse_bids_filename(filepath)

        assert 'session' not in entities
        assert 'task' not in entities
        assert 'run' not in entities


class TestCreateShorthandLabel:
    """Tests for create_shorthand_label function."""

    def test_anat_label(self):
        """Test creating shorthand label for anatomical file."""
        entities = {'datatype': 'anat', 'suffix': 'T1w'}
        label = create_shorthand_label(entities, '.nii.gz')

        assert label == 'anat_T1w_nii'

    def test_func_with_task(self):
        """Test creating shorthand label for functional file with task."""
        entities = {'datatype': 'func', 'suffix': 'bold', 'task': 'rest'}
        label = create_shorthand_label(entities, '.nii.gz')

        assert 'func' in label
        assert 'bold' in label
        assert 'task-rest' in label
        assert 'nii' in label

    def test_json_extension(self):
        """Test label creation for JSON file."""
        entities = {'datatype': 'anat', 'suffix': 'T1w'}
        label = create_shorthand_label(entities, '.json')

        assert label == 'anat_T1w_json'

    def test_with_direction_and_acquisition(self):
        """Test label with multiple optional entities."""
        entities = {
            'datatype': 'func',
            'suffix': 'bold',
            'task': 'rest',
            'direction': 'AP',
            'acquisition': 'multiband'
        }
        label = create_shorthand_label(entities, '.nii.gz')

        assert 'func' in label
        assert 'bold' in label
        assert 'task-rest' in label
        assert 'dir-AP' in label
        assert 'acq-multiband' in label


class TestAutoDiscoverSubjects:
    """Tests for auto_discover_subjects function."""

    def test_discover_subjects(self, tmp_path):
        """Test auto-discovery of subjects in BIDS directory."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        # Create subject directories
        (bids_dir / "sub-01").mkdir()
        (bids_dir / "sub-02").mkdir()
        (bids_dir / "sub-03").mkdir()

        # Create a non-subject directory (should be ignored)
        (bids_dir / "derivatives").mkdir()
        (bids_dir / "README").touch()

        subjects = auto_discover_subjects(bids_dir)

        assert len(subjects) == 3
        assert '01' in subjects
        assert '02' in subjects
        assert '03' in subjects
        assert subjects == sorted(subjects)  # Should be sorted

    def test_empty_directory(self, tmp_path):
        """Test empty BIDS directory returns empty list."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        subjects = auto_discover_subjects(bids_dir)
        assert subjects == []

    def test_alphanumeric_subjects(self, tmp_path):
        """Test discovery of alphanumeric subject IDs."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        (bids_dir / "sub-control01").mkdir()
        (bids_dir / "sub-patient01").mkdir()

        subjects = auto_discover_subjects(bids_dir)

        assert 'control01' in subjects
        assert 'patient01' in subjects


# Pytest configuration
@pytest.fixture
def sample_bids_structure(tmp_path):
    """Create a sample BIDS directory structure for testing."""
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    # Create subjects
    for subj in ['01', '02']:
        subj_dir = bids_dir / f"sub-{subj}"
        subj_dir.mkdir()

        # Anatomical
        anat_dir = subj_dir / "anat"
        anat_dir.mkdir()
        (anat_dir / f"sub-{subj}_T1w.nii.gz").touch()
        (anat_dir / f"sub-{subj}_T1w.json").touch()

        # Functional
        func_dir = subj_dir / "func"
        func_dir.mkdir()
        (func_dir / f"sub-{subj}_task-rest_bold.nii.gz").touch()
        (func_dir / f"sub-{subj}_task-rest_bold.json").touch()

    return bids_dir


# Run tests with pytest if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
