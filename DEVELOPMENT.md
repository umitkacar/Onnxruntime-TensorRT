# Development Guide

This guide provides detailed information for developers working on the ONNX Runtime TensorRT project.

## Project Structure

```
Onnxruntime-TensorRT/
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml           # CI/CD pipeline
│       ├── release.yml      # Release automation
│       └── docs.yml         # Documentation deployment
│
├── src/                     # Source code
│   └── onnxruntime_tensorrt/
│       ├── __init__.py      # Package initialization
│       ├── core/            # Core functionality
│       │   ├── __init__.py
│       │   └── session.py   # TensorRT session management
│       └── utils/           # Utility functions
│           ├── __init__.py
│           └── logger.py    # Logging utilities
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_session.py     # Session tests
│   ├── test_logger.py      # Logger tests
│   └── test_version.py     # Version tests
│
├── examples/                # Example scripts
│   ├── yolov10_inference.py
│   ├── llm_inference.py
│   └── sam2_segmentation.py
│
├── benchmark/              # Benchmarking tools
│   └── benchmark_trt.py
│
├── config/                 # Configuration files
│   └── trt_config.yaml
│
├── docs/                   # Documentation (to be added)
│
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── .pre-commit-config.yaml # Pre-commit hooks
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
├── README.md              # Project README
├── CONTRIBUTING.md        # Contribution guidelines
└── DEVELOPMENT.md         # This file
```

## Development Tools

### Build System: Hatch

We use [Hatch](https://hatch.pypa.io/) as our build backend and environment manager.

#### Installation

```bash
pip install hatch
```

#### Common Commands

```bash
# Create development environment
hatch env create

# Run tests
hatch run test

# Run with coverage
hatch run test-cov

# Lint code
hatch run lint

# Format code
hatch run format

# Type check
hatch run type-check

# Run all checks
hatch run check-all

# Build package
hatch build

# Clean build artifacts
hatch clean
```

#### Environment Management

```bash
# Show all environments
hatch env show

# Remove environment
hatch env remove default

# Run in specific environment
hatch -e docs run mkdocs serve
```

### Linting: Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter.

#### Configuration

See `pyproject.toml` under `[tool.ruff]`.

#### Usage

```bash
# Lint all code
ruff check src tests examples

# Auto-fix issues
ruff check --fix src

# Format code
ruff format src

# Check formatting
ruff format --check src
```

### Formatting: Black

[Black](https://github.com/psf/black) is the uncompromising code formatter.

#### Configuration

```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
```

#### Usage

```bash
# Format code
black src tests examples

# Check formatting
black --check src
```

### Type Checking: MyPy

[MyPy](https://mypy-lang.org/) is a static type checker for Python.

#### Configuration

See `pyproject.toml` under `[tool.mypy]`. We use strict mode:

```toml
[tool.mypy]
strict = true
warn_return_any = true
disallow_untyped_defs = true
```

#### Usage

```bash
# Type check source code
mypy src

# Type check with verbose output
mypy --verbose src

# Generate HTML report
mypy --html-report ./mypy-report src
```

### Testing: Pytest

[Pytest](https://pytest.org/) is our testing framework.

#### Configuration

See `pyproject.toml` under `[tool.pytest.ini_options]`.

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/onnxruntime_tensorrt --cov-report=html

# Run specific test file
pytest tests/test_session.py

# Run specific test
pytest tests/test_session.py::test_function_name

# Run with markers
pytest -m "not slow"        # Skip slow tests
pytest -m "unit"            # Only unit tests
pytest -m "integration"     # Only integration tests

# Run in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Run with print statements
pytest -s
```

#### Test Markers

- `@pytest.mark.unit` - Fast unit tests (< 1s)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (> 5s)
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.tensorrt` - Requires TensorRT

#### Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def sample_input() -> np.ndarray:
    """Create sample input data."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)
```

### Coverage

We aim for >90% test coverage.

```bash
# Generate coverage report
coverage run -m pytest
coverage report

# Generate HTML report
coverage html
python -m http.server 8000 -d htmlcov

# Generate XML report (for CI)
coverage xml
```

## Pre-commit Hooks

Pre-commit hooks ensure code quality before commits.

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Install commit-msg hook
pre-commit install --hook-type commit-msg
```

### Usage

```bash
# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks (not recommended)
git commit --no-verify
```

### Available Hooks

1. **File checks**: trailing whitespace, EOF, merge conflicts
2. **Python quality**: ruff, black, mypy
3. **Security**: bandit, detect-secrets
4. **Documentation**: pydocstyle, interrogate
5. **Commit messages**: commitizen

## Continuous Integration

### GitHub Actions Workflows

#### CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push and PR:

1. **Code Quality**: Pre-commit hooks
2. **Linting**: Ruff checks
3. **Type Checking**: MyPy strict mode
4. **Testing**: Multi-platform, multi-version tests
5. **Security**: Bandit security scan
6. **Build**: Package building and verification
7. **Benchmark**: Performance benchmarks (PR only)

#### Release Pipeline (`.github/workflows/release.yml`)

Runs on version tags (e.g., `v1.0.0`):

1. **Build**: Build package
2. **Test**: Run full test suite
3. **Release**: Create GitHub release
4. **Publish**: Upload to PyPI

#### Documentation (`.github/workflows/docs.yml`)

Runs on main branch updates:

1. **Build**: Build documentation
2. **Deploy**: Deploy to GitHub Pages

### Running CI Locally

```bash
# Simulate CI checks
hatch run check-all

# Run tests like CI
pytest --cov=src --cov-report=term --cov-report=xml

# Build package
hatch build
twine check dist/*
```

## Code Style Guidelines

### Python Style

1. **PEP 8**: Follow Python style guide
2. **Line length**: 100 characters
3. **Quotes**: Double quotes for strings
4. **Imports**: Sort with ruff/isort
5. **Trailing commas**: Use in multi-line structures

### Type Hints

Always use type hints:

```python
from typing import List, Optional, Dict, Any

def process_data(
    data: np.ndarray,
    config: Dict[str, Any],
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """Process input data with configuration."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short description.

    Longer description with more details about what
    the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    ...
```

### Error Handling

```python
# Good: Specific exceptions with messages
if value < 0:
    raise ValueError(f"Value must be non-negative, got {value}")

# Bad: Generic exceptions
if value < 0:
    raise Exception("Invalid value")
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.exception("Exception with traceback")
```

## Release Process

### Version Bumping

We use semantic versioning: `MAJOR.MINOR.PATCH`

1. Update version in `src/onnxruntime_tensorrt/__init__.py`
2. Update `CHANGELOG.md`
3. Commit changes
4. Create and push tag

```bash
# Update version
vim src/onnxruntime_tensorrt/__init__.py

# Commit
git add .
git commit -m "chore: bump version to 1.1.0"

# Tag
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main --tags
```

### Automated Release

When you push a tag, GitHub Actions will:

1. Run full test suite
2. Build package
3. Create GitHub release
4. Publish to PyPI (if configured)

## Debugging

### VS Code Configuration

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v", "${file}"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

### IPython/IPdb

For interactive debugging:

```python
# Add breakpoint
import ipdb; ipdb.set_trace()

# Or use built-in
breakpoint()
```

## Performance Profiling

### cProfile

```bash
python -m cProfile -o output.prof script.py
python -m pstats output.prof
```

### line_profiler

```bash
pip install line_profiler

# Add @profile decorator
kernprof -l -v script.py
```

### memory_profiler

```bash
pip install memory_profiler

# Add @profile decorator
python -m memory_profiler script.py
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure package is installed in editable mode
   ```bash
   pip install -e .
   ```

2. **Pre-commit fails**: Update hooks
   ```bash
   pre-commit autoupdate
   pre-commit run --all-files
   ```

3. **Type checking errors**: Install type stubs
   ```bash
   pip install types-PyYAML types-requests
   ```

4. **Tests fail**: Clear pytest cache
   ```bash
   pytest --cache-clear
   ```

## Resources

- [Hatch Documentation](https://hatch.pypa.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
