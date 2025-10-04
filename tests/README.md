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

The tests are organized into several categories using pytest markers:

### Unit Tests (`@pytest.mark.unit`)
- Fast, isolated tests
- Test individual functions and classes
- Mock external dependencies
- Should run in milliseconds

### Integration Tests (`@pytest.mark.integration`)
- Test interactions between components
- May use real files/data
- Test end-to-end workflows
- Slower than unit tests

### End-to-End Tests (`@pytest.mark.e2e`)
- Test complete user workflows
- Use real video files and models
- Test CLI commands and API endpoints
- Slowest tests, run less frequently

### Slow Tests (`@pytest.mark.slow`)
- Performance tests
- Memory leak tests
- Stress tests with large datasets

## Running Tests

### Using Development Script

```bash
# Run all tests with coverage
./dev.sh test

# Run tests quickly without coverage
./dev.sh test-fast

# Run only unit tests
./dev.sh test-unit

# Run only integration tests
./dev.sh test-integration

# Run tests in parallel (faster)
./dev.sh test-parallel

# Generate detailed coverage report
./dev.sh coverage
```

### Using pytest directly

```bash
# Install test dependencies
uv sync --group test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not slow"

# Run tests in parallel
uv run pytest -n auto

# Run with verbose output
uv run pytest -v

# Stop at first failure
uv run pytest -x

# Run specific test file
uv run pytest tests/anonymizer/test_config.py

# Run specific test function
uv run pytest tests/anonymizer/test_config.py::TestModelConfig::test_model_config_defaults
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
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "e2e: marks tests as end-to-end tests"
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

2. **Nightly Tests** (`.github/workflows/nightly.yml`)
   - Runs extended test suite daily
   - Includes stress tests and memory profiling
   - Tests compatibility with different dependency versions

3. **Release Workflow** (`.github/workflows/release.yml`)
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

    @pytest.mark.slow
    def test_performance(self):
        """Test performance characteristics."""
        # Performance test implementation
        pass

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

## Performance Testing

### Memory Testing

```bash
# Install memory profiler
uv add --group test memray

# Run memory profiling
uv run pytest --memray tests/
```

### Benchmark Testing

```bash
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
   - Keep unit tests fast
   - Use appropriate markers for slow tests
   - Consider parallel execution

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
   - Run unit tests only: `pytest -m unit`
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
