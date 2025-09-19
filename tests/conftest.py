"""
Pytest configuration and shared fixtures for the compiler tests.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oven.compiler import PythonToMLIRCompiler
from oven.mlir_generator import MLIRGenerator
from oven.ast_visitor import PythonToMLIRASTVisitor


@pytest.fixture
def compiler():
    """Create a compiler instance for testing."""
    return PythonToMLIRCompiler(debug=False, optimize=True)


@pytest.fixture
def debug_compiler():
    """Create a debug compiler instance for testing."""
    return PythonToMLIRCompiler(debug=True, optimize=False)


@pytest.fixture
def mlir_generator():
    """Create an MLIR generator instance for testing."""
    return MLIRGenerator()


@pytest.fixture
def ast_visitor():
    """Create an AST visitor instance for testing."""
    return PythonToMLIRASTVisitor()


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing file compilation."""

    def _create_temp_file(content):
        fd, path = tempfile.mkstemp(suffix=".py", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            return path
        except:
            os.close(fd)
            raise

    files_to_cleanup = []

    def cleanup():
        for path in files_to_cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass

    def create_file(content):
        path = _create_temp_file(content)
        files_to_cleanup.append(path)
        return path

    yield create_file
    cleanup()


@pytest.fixture
def sample_python_code():
    """Provide sample Python code snippets for testing."""
    return {
        "simple_function": """
def add(a, b):
    return a + b
""",
        "function_with_constants": """
def get_answer():
    return 42
""",
        "function_with_if": """
def max_value(a, b):
    if a > b:
        return a
    else:
        return b
""",
        "function_with_while": """
def countdown(n):
    while n > 0:
        n = n - 1
    return n
""",
        "recursive_function": """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
""",
        "complex_example": """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    result = fibonacci(5)
    return result
""",
    }


@pytest.fixture
def expected_mlir_patterns():
    """Provide expected MLIR patterns for validation."""
    return {
        "function_declaration": "func.func @",
        "function_return": "func.return",
        "integer_constant": "arith.constant",
        "integer_addition": "arith.addi",
        "integer_multiplication": "arith.muli",
        "integer_subtraction": "arith.subi",
        "integer_comparison": "arith.cmpi",
        "conditional_branch": "cf.cond_br",
        "unconditional_branch": "cf.br",
        "function_call": "func.call",
    }


@pytest.mark.unit
class TestFixtures:
    """Test the fixtures themselves."""

    def test_compiler_fixture(self, compiler):
        """Test that compiler fixture works."""
        assert isinstance(compiler, PythonToMLIRCompiler)
        assert not compiler.debug
        assert compiler.optimize

    def test_debug_compiler_fixture(self, debug_compiler):
        """Test that debug compiler fixture works."""
        assert isinstance(debug_compiler, PythonToMLIRCompiler)
        assert debug_compiler.debug
        assert not debug_compiler.optimize

    def test_mlir_generator_fixture(self, mlir_generator):
        """Test that MLIR generator fixture works."""
        assert isinstance(mlir_generator, MLIRGenerator)

    def test_ast_visitor_fixture(self, ast_visitor):
        """Test that AST visitor fixture works."""
        assert isinstance(ast_visitor, PythonToMLIRASTVisitor)

    def test_temp_python_file_fixture(self, temp_python_file):
        """Test that temporary file fixture works."""
        content = "def test(): return 42"
        path = temp_python_file(content)

        assert os.path.exists(path)
        assert path.endswith(".py")

        with open(path, "r") as f:
            assert f.read() == content

    def test_sample_python_code_fixture(self, sample_python_code):
        """Test that sample code fixture provides expected codes."""
        expected_keys = {
            "simple_function",
            "function_with_constants",
            "function_with_if",
            "function_with_while",
            "recursive_function",
            "complex_example",
        }
        assert set(sample_python_code.keys()) == expected_keys

        # Verify each sample is valid Python
        import ast

        for key, code in sample_python_code.items():
            try:
                ast.parse(code)
            except SyntaxError:
                pytest.fail(f"Sample code '{key}' has syntax errors")

    def test_expected_mlir_patterns_fixture(self, expected_mlir_patterns):
        """Test that MLIR patterns fixture provides expected patterns."""
        expected_keys = {
            "function_declaration",
            "function_return",
            "integer_constant",
            "integer_addition",
            "integer_multiplication",
            "integer_subtraction",
            "integer_comparison",
            "conditional_branch",
            "unconditional_branch",
            "function_call",
        }
        assert set(expected_mlir_patterns.keys()) == expected_keys
