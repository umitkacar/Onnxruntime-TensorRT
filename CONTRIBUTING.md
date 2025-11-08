# Contributing to ONNX Runtime TensorRT

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review](#code-review)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA Toolkit (for GPU support)
- TensorRT (optional, for TensorRT features)

### Setup Steps

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Onnxruntime-TensorRT.git
cd Onnxruntime-TensorRT
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or using hatch
pip install hatch
hatch env create
```

4. **Install pre-commit hooks**

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

5. **Verify installation**

```bash
# Run tests
hatch run test

# Run linters
hatch run lint

# Run type checker
hatch run type-check
```

## Code Style

We use modern Python tooling to maintain code quality:

### Formatting

- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting and import sorting

```bash
# Format code
hatch run format

# Check formatting
hatch run format-check
```

### Linting

```bash
# Run linter
hatch run lint

# Auto-fix issues
ruff check --fix src tests
```

### Type Checking

We use MyPy with strict mode:

```bash
# Run type checker
hatch run type-check
```

### Pre-commit Hooks

All code must pass pre-commit hooks before committing:

```bash
# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

## Testing

### Writing Tests

- Use `pytest` for all tests
- Place tests in the `tests/` directory
- Name test files with `test_*.py` pattern
- Use descriptive test names: `test_<what>_<condition>_<expected>`

Example:

```python
def test_session_init_raises_error_for_invalid_path() -> None:
    """Test that FileNotFoundError is raised for non-existent model."""
    with pytest.raises(FileNotFoundError):
        TensorRTSession("nonexistent.onnx")
```

### Running Tests

```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Run specific test file
pytest tests/test_session.py

# Run specific test
pytest tests/test_session.py::TestTensorRTSession::test_session_init

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Only unit tests
pytest -m "gpu"       # Only GPU tests
```

### Test Categories

Use pytest markers to categorize tests:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.tensorrt` - Requires TensorRT

### Coverage

Maintain high test coverage (aim for >90%):

```bash
# Generate coverage report
hatch run test-cov

# View HTML report
hatch run cov-html
# Opens browser at http://localhost:8000
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation if needed
- Follow the code style guidelines

### 3. Commit Changes

We use conventional commits:

```bash
# Feature
git commit -m "feat: add YOLOv10 support"

# Bug fix
git commit -m "fix: resolve TensorRT cache issue"

# Documentation
git commit -m "docs: update installation guide"

# Tests
git commit -m "test: add SAM2 segmentation tests"
```

Commit types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Build/tooling changes
- `perf:` - Performance improvements

### 4. Run Quality Checks

```bash
# Run all checks
hatch run check-all

# Or individually
hatch run format-check
hatch run lint
hatch run type-check
hatch run test-cov
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference related issues
- Include test results
- Add screenshots/examples if applicable

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
```

## Code Review

### Review Criteria

Reviewers will check for:

1. **Correctness**: Code works as intended
2. **Tests**: Adequate test coverage
3. **Style**: Follows project conventions
4. **Documentation**: Clear and complete
5. **Performance**: No obvious inefficiencies
6. **Security**: No security vulnerabilities

### Addressing Feedback

- Respond to all comments
- Make requested changes
- Mark conversations as resolved
- Re-request review after updates

## Development Workflow

### Using Hatch

Hatch provides isolated environments and scripts:

```bash
# List available environments
hatch env show

# Run commands in default environment
hatch run test
hatch run lint
hatch run type-check

# Run commands in specific environment
hatch run docs:build
hatch run docs:serve
```

### Available Scripts

```bash
# Testing
hatch run test              # Run tests
hatch run test-cov          # Run with coverage
hatch run cov-report        # Coverage report
hatch run cov-html          # HTML coverage report

# Code Quality
hatch run lint              # Run linter
hatch run format            # Format code
hatch run format-check      # Check formatting
hatch run type-check        # Type checking
hatch run check-all         # Run all checks

# Documentation
hatch run docs:build        # Build docs
hatch run docs:serve        # Serve docs locally
```

## Best Practices

### Code Quality

1. **Write type hints** for all functions
2. **Add docstrings** using Google style
3. **Keep functions small** and focused
4. **Avoid magic numbers** - use constants
5. **Handle errors** appropriately
6. **Log important events**

### Testing

1. **Test edge cases** and error conditions
2. **Use fixtures** for common setup
3. **Mock external dependencies**
4. **Keep tests independent**
5. **Test one thing** per test
6. **Use descriptive names**

### Documentation

1. **Update README** for user-facing changes
2. **Add docstrings** to all public APIs
3. **Include examples** in docstrings
4. **Document breaking changes**
5. **Keep CHANGELOG** updated

### Git

1. **Commit often** with clear messages
2. **Keep commits atomic**
3. **Rebase before merging**
4. **Squash related commits**
5. **Write descriptive** PR descriptions

## Getting Help

- 📖 Read the [README](README.md)
- 🐛 Check [existing issues](https://github.com/umitkacar/Onnxruntime-TensorRT/issues)
- 💬 Ask questions in discussions
- 📧 Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
