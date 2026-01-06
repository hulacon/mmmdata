# Tests

This directory contains unit tests for the mmmdata utilities.

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov tomli_w
```

### Run All Tests

```bash
# From the repository root
pytest tests/

# With coverage report
pytest tests/ --cov=src/python/core --cov=scripts --cov-report=html

# Verbose output
pytest tests/ -v
```

### Run Specific Test Files

```bash
pytest tests/test_config.py
pytest tests/test_create_bids_inventory.py
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_config.py::TestDeepUpdate

# Run a specific test method
pytest tests/test_config.py::TestLoadConfig::test_load_base_config_only
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_config.py` - Tests for configuration loading utilities
- `test_create_bids_inventory.py` - Tests for BIDS inventory creation script
- `__init__.py` - Package marker

## Writing New Tests

When adding new functionality to the codebase, please add corresponding tests:

1. Create a new test file named `test_<module_name>.py`
2. Use pytest fixtures from `conftest.py` for common setup
3. Follow the existing test structure with test classes and methods
4. Use descriptive test names that explain what is being tested
5. Include docstrings explaining the test purpose

Example:
```python
def test_my_function_handles_edge_case():
    """Test that my_function correctly handles empty input."""
    result = my_function([])
    assert result == expected_value
```

## Continuous Integration

These tests can be integrated into a CI/CD pipeline (e.g., GitHub Actions) to run automatically on commits and pull requests.
