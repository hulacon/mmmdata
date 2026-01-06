"""Pytest configuration and shared fixtures for tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_bids_dir(tmp_path):
    """
    Create a minimal BIDS directory structure for testing.

    Returns:
        Path: Path to the temporary BIDS directory
    """
    bids_dir = tmp_path / "bids_dataset"
    bids_dir.mkdir()

    # Create dataset_description.json
    dataset_desc = bids_dir / "dataset_description.json"
    dataset_desc.write_text('''{
    "Name": "Test Dataset",
    "BIDSVersion": "1.6.0"
}''')

    # Create participants.tsv
    participants_tsv = bids_dir / "participants.tsv"
    participants_tsv.write_text('''participant_id\tage\tsex
sub-01\t25\tM
sub-02\t30\tF
''')

    return bids_dir


@pytest.fixture
def sample_config_dir(tmp_path):
    """
    Create a sample configuration directory with base.toml.

    Returns:
        Path: Path to the temporary config directory
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create base.toml
    base_toml = config_dir / "base.toml"
    base_toml.write_text('''[paths]
bids_project_dir = "/test/bids"
code_root = "/test/code"
singularity_dir = "/test/singularity"
venv_dir = "/test/venv"
output_dir = "/test/output"

[slurm]
partition = "compute"
time = "12:00:00"
memory = "16G"
cpus = "4"
email = "test@example.com"
''')

    return config_dir
