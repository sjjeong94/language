# Python to MLIR Compiler - Makefile

.PHONY: help test test-unit test-integration test-parametrized test-all test-coverage clean install dev-install format lint check

# Default Python executable
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

# Directories
SRC_DIR := src
TEST_DIR := tests
EXAMPLES_DIR := $(TEST_DIR)/examples

help:  ## Show this help message
	@echo "Python to MLIR Compiler - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PIP) install -r requirements.txt

dev-install:  ## Install development dependencies
	$(PIP) install pytest pytest-cov pytest-xdist black flake8 mypy

test:  ## Run all tests
	$(PYTEST) $(TEST_DIR) -v

test-unit:  ## Run unit tests only
	$(PYTEST) $(TEST_DIR) -v -m unit

test-integration:  ## Run integration tests only
	$(PYTEST) $(TEST_DIR) -v -m integration

test-parametrized:  ## Run parametrized tests only
	$(PYTEST) $(TEST_DIR)/test_parametrized.py -v

test-fast:  ## Run fast tests (exclude slow tests)
	$(PYTEST) $(TEST_DIR) -v -m "not slow"

test-slow:  ## Run slow tests only
	$(PYTEST) $(TEST_DIR) -v -m slow

test-coverage:  ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v

test-parallel:  ## Run tests in parallel
	$(PYTEST) $(TEST_DIR) -v -n auto

test-all:  ## Run all tests with coverage and parallel execution
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v -n auto

format:  ## Format code with black
	black $(SRC_DIR) $(TEST_DIR) --line-length 88

lint:  ## Run linting with flake8
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=88 --exclude=__pycache__

type-check:  ## Run type checking with mypy
	mypy $(SRC_DIR) --ignore-missing-imports

check: format lint type-check  ## Run all code quality checks

compile-example:  ## Compile the sample.py example
	$(PYTHON) main.py $(EXAMPLES_DIR)/sample.py -o $(EXAMPLES_DIR)/sample.mlir

compile-complex:  ## Compile the complex.py example
	$(PYTHON) main.py $(EXAMPLES_DIR)/complex.py -o $(EXAMPLES_DIR)/complex.mlir

compile-debug:  ## Compile sample.py with debug output
	$(PYTHON) main.py $(EXAMPLES_DIR)/sample.py -o $(EXAMPLES_DIR)/sample_debug.mlir --debug

clean:  ## Clean generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.mlir" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

benchmark:  ## Run simple benchmarks
	@echo "Running compilation benchmarks..."
	@time $(PYTHON) main.py $(EXAMPLES_DIR)/sample.py -o /tmp/sample_bench.mlir
	@time $(PYTHON) main.py $(EXAMPLES_DIR)/complex.py -o /tmp/complex_bench.mlir
	@echo "Benchmark completed. Check /tmp/ for output files."

validate-examples:  ## Validate all example files compile successfully
	@echo "Validating example files..."
	@for file in $(EXAMPLES_DIR)/*.py; do \
		echo "Compiling $$file..."; \
		$(PYTHON) main.py "$$file" -o "/tmp/$$(basename $$file .py).mlir" || exit 1; \
	done
	@echo "All examples validated successfully!"

# Development workflow commands
dev-setup: dev-install  ## Set up development environment
	@echo "Development environment setup complete!"

dev-test: format lint test-fast  ## Quick development test cycle

dev-full: format lint type-check test-coverage  ## Full development check

# CI/CD commands
ci-test:  ## Run tests suitable for CI environment
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=xml --cov-report=term -v

# Documentation
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

# Package commands
package:  ## Build package (placeholder)
	@echo "Package building not implemented yet"

# Show test markers
show-markers:  ## Show available pytest markers
	$(PYTEST) --markers

# Run specific test file
test-file:  ## Run specific test file (usage: make test-file FILE=test_compiler.py)
	$(PYTEST) $(TEST_DIR)/$(FILE) -v

# Run specific test function
test-func:  ## Run specific test function (usage: make test-func FUNC=test_function_name)
	$(PYTEST) $(TEST_DIR) -v -k $(FUNC)

# Debug failed tests
test-debug:  ## Run tests with debug on failure
	$(PYTEST) $(TEST_DIR) -v --pdb

# List all available targets
list:  ## List all Makefile targets
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
