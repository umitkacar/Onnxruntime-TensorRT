.PHONY: help install dev-install clean test lint format type-check check-all build publish docs

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==================== Installation ====================

install: ## Install package
	pip install -e .

dev-install: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

# ==================== Cleaning ====================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

clean-all: clean ## Clean all including virtual environments
	rm -rf venv/
	rm -rf env/
	rm -rf .venv/

# ==================== Testing ====================

test: ## Run tests
	hatch run test

test-cov: ## Run tests with coverage
	hatch run test-cov

test-html: ## Run tests and open HTML coverage report
	hatch run test-cov
	hatch run cov-html

test-fast: ## Run fast tests only (skip slow tests)
	pytest -m "not slow"

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests only
	pytest -m integration

# ==================== Code Quality ====================

lint: ## Run linter (ruff)
	hatch run lint

format: ## Format code with black and ruff
	hatch run format
	ruff check --fix src tests examples benchmark

format-check: ## Check code formatting
	hatch run format-check

type-check: ## Run type checker (mypy)
	hatch run type-check

check-all: ## Run all quality checks
	hatch run check-all

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

security: ## Run security checks
	bandit -r src -c pyproject.toml
	safety check || true

# ==================== Building ====================

build: clean ## Build package
	hatch build

build-check: build ## Build and check package
	twine check dist/*

# ==================== Publishing ====================

publish-test: build-check ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build-check ## Publish to PyPI
	twine upload dist/*

# ==================== Documentation ====================

docs-build: ## Build documentation
	hatch run docs:build

docs-serve: ## Serve documentation locally
	hatch run docs:serve

docs-deploy: docs-build ## Deploy documentation to GitHub Pages
	@echo "Deploying to GitHub Pages..."
	# Add deployment commands here

# ==================== Development ====================

dev: ## Start development environment
	@echo "Starting development environment..."
	@echo "Run 'hatch shell' to activate environment"

bump-patch: ## Bump patch version
	@echo "Current version: $$(grep -E '^__version__' src/onnxruntime_tensorrt/__init__.py | cut -d'"' -f2)"
	@echo "Bump version manually in src/onnxruntime_tensorrt/__init__.py"

bump-minor: ## Bump minor version
	@echo "Current version: $$(grep -E '^__version__' src/onnxruntime_tensorrt/__init__.py | cut -d'"' -f2)"
	@echo "Bump version manually in src/onnxruntime_tensorrt/__init__.py"

bump-major: ## Bump major version
	@echo "Current version: $$(grep -E '^__version__' src/onnxruntime_tensorrt/__init__.py | cut -d'"' -f2)"
	@echo "Bump version manually in src/onnxruntime_tensorrt/__init__.py"

# ==================== Benchmarking ====================

benchmark: ## Run benchmarks
	@echo "Running benchmarks..."
	@echo "Add benchmark commands here"

# ==================== Utility ====================

todo: ## Show TODO/FIXME/XXX comments in code
	@grep -rn "TODO\|FIXME\|XXX" src/ tests/ || echo "No TODOs found"

loc: ## Count lines of code
	@echo "Lines of code:"
	@find src -name '*.py' | xargs wc -l | tail -n 1
	@echo "Lines of tests:"
	@find tests -name '*.py' | xargs wc -l | tail -n 1

deps-update: ## Update dependencies
	pip install --upgrade pip
	pip install --upgrade hatch
	pre-commit autoupdate

# ==================== Quick Commands ====================

qa: format lint type-check test ## Run quick quality assurance (format, lint, type-check, test)

ci: check-all ## Simulate CI pipeline locally

release-check: clean check-all build-check ## Check if ready for release

# ==================== Info ====================

info: ## Show project information
	@echo "Project: ONNX Runtime TensorRT"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Version: $$(grep -E '^__version__' src/onnxruntime_tensorrt/__init__.py | cut -d'"' -f2)"
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E 'onnx|ruff|black|mypy|pytest|hatch'
