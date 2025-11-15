# Testing Documentation

This document describes the comprehensive testing setup for the blur-gui project.

## Overview

The project uses pytest as the main testing framework with extensive coverage reporting and multiple test categories.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── anonymizer/              # Tests for anonymizer module
│   ├── test_config.py      # Configuration system tests
│   ├── test_core.py        # Core anonymizer tests
│   ├── test_blurring.py    # Blurring functionality tests
│   ├── test_detection.py   # Detection system tests
│   ├── io/                 # I/O operation tests
│   ├── tracking/           # Tracking system tests
│   └── utils/              # Utility function tests
├── blur_api/               # API server tests
│   └── test_serve.py      # FastAPI endpoint tests
└── blur_gui/               # GUI/CLI tests
    ├── test_cli.py        # Command-line interface tests
    └── test_logging_setup.py # Logging configuration tests
```

## Test Categories

Pytest markers are only used for integration coverage today:

### Integration Tests (`@pytest.mark.integration`)
- Exercise component boundaries and real data flows
- Useful for features that require model files or multiple services
- Expect them to run slower than the default unit-style tests

All other tests run without markers and should stay fast and isolated.

## Running Tests

Install the test dependencies once per virtual environment:

```bash
uv sync --group test
```

Common commands:

```bash
# Run the complete suite with coverage (default pyproject options)
uv run pytest

# Focus on integration tests only
uv run pytest -m integration

# Stop on first failure or increase verbosity when chasing bugs
uv run pytest -x
uv run pytest -vv

# Target a specific module or test function
uv run pytest tests/anonymizer/test_config.py
uv run pytest tests/anonymizer/test_config.py::TestModelConfig::test_model_config_defaults

# Use xdist for parallel execution when the host has spare cores
uv run pytest -n auto
```

## Coverage Requirements

The project maintains high test coverage:

- **Minimum coverage**: 80%
- **Target coverage**: 90%+
- **Coverage reports**: Generated in HTML, XML, and terminal formats

### Viewing Coverage

After running tests with coverage:

```bash
# View in browser
open htmlcov/index.html

# View in terminal
uv run coverage report --show-missing
```

## Test Configuration

### pytest Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80"
]
markers = [
    "integration: marks integration tests that exercise multiple components"
]
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/.*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
show_missing = true
precision = 2
```

## Continuous Integration

### GitHub Actions Workflows

1. **Main CI** (`.github/workflows/ci.yml`)
   - Runs on every push/PR
   - Tests across multiple OS and Python versions
   - Includes linting, type checking, security checks
   - Uploads coverage to Codecov

2. **Release Workflow** (`.github/workflows/release.yml`)
   - Comprehensive testing before release
   - Builds and publishes packages
   - Creates GitHub releases with binaries

### Pre-commit Hooks

Tests can be run automatically before commits:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files

# Run tests manually before commit
uv run pre-commit run pytest-check --hook-stage manual
```

## Writing Tests

### Test Structure

```python
"""
Tests for module functionality.
"""

import pytest
from unittest.mock import Mock, patch

from your_module import YourClass


class TestYourClass:
    """Test YourClass functionality."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        instance = YourClass()
        result = instance.method()
        assert result == expected_result

    @pytest.mark.integration
    def test_integration_with_other_module(self):
        """Test integration with other modules."""
        # Integration test implementation
        pass
```

### Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_with_fixtures(sample_config, temp_dir, mock_progress_callback):
    """Test using shared fixtures."""
    # Use fixtures in your test
    pass
```

### Mocking

Use mocking for external dependencies:

```python
@patch('your_module.external_dependency')
def test_with_mock(mock_dependency):
    """Test with mocked dependency."""
    mock_dependency.return_value = expected_value
    # Your test implementation
    mock_dependency.assert_called_once()
```

## Debugging Tests

### Running Individual Tests

```bash
# Run with debugging output
uv run pytest -s -vv tests/specific/test_file.py::test_function

# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger on first failure
uv run pytest -x --pdb
```

### Test Output

```bash
# Show local variables on failure
uv run pytest -l

# Show full diff on assertion failures
uv run pytest --tb=long

# Capture and show print statements
uv run pytest -s
```

## Best Practices

1. **Test Organization**
   - Group related tests in classes
   - Use descriptive test names
   - Follow AAA pattern (Arrange, Act, Assert)

2. **Test Coverage**
   - Aim for high coverage but focus on meaningful tests
   - Test edge cases and error conditions
   - Don't test trivial code just for coverage

3. **Mock Usage**
   - Mock external dependencies
   - Don't mock the code under test
   - Verify mock interactions when relevant

4. **Performance**
   - Keep default (unmarked) tests fast
   - Reserve the integration marker for scenarios that truly need it
   - Consider parallel execution with `pytest -n auto`

5. **Maintenance**
   - Keep tests simple and readable
   - Update tests when changing functionality
   - Remove obsolete tests

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure test dependencies are installed: `uv sync --group test`
   - Check Python path and module imports

2. **Slow Tests**
   - Use `-k` to run specific tests
   - Run only fast tests: `pytest -m "not integration"`
   - Use parallel execution: `pytest -n auto`

3. **Coverage Issues**
   - Check coverage configuration in `pyproject.toml`
   - Ensure source paths are correct
   - Verify test files are being discovered

4. **CI Failures**
   - Check GitHub Actions logs
   - Reproduce locally with same Python version
   - Check for environment-specific issues

### Getting Help

- Check the GitHub Issues for known problems
- Review the test logs for specific error messages
- Run tests with increased verbosity: `pytest -vv`
- Use `pytest --collect-only` to see discovered tests
