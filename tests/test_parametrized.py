"""
Parametrized tests for Python to MLIR Compiler
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import PythonToMLIRCompiler
from src.utils.mlir_utils import MLIRUtils


@pytest.mark.unit
class TestParametrizedMLIRUtils:
    """Parametrized tests for MLIR utilities."""

    @pytest.mark.parametrize(
        "python_type,expected_mlir",
        [
            ("int", "i32"),
            ("float", "f32"),
            ("bool", "i1"),
            ("str", "!llvm.ptr<i8>"),
            ("unknown", "i32"),  # default case
        ],
    )
    def test_python_type_to_mlir_parametrized(self, python_type, expected_mlir):
        """Test Python to MLIR type conversion with various types."""
        assert MLIRUtils.python_type_to_mlir(python_type) == expected_mlir

    @pytest.mark.parametrize(
        "ssa_name,expected_valid",
        [
            ("%0", True),
            ("%result", True),
            ("%temp_var", True),
            ("0", False),
            ("%", False),
            ("result", False),
            ("", False),
        ],
    )
    def test_ssa_name_validation_parametrized(self, ssa_name, expected_valid):
        """Test SSA name validation with various inputs."""
        assert MLIRUtils.is_valid_ssa_name(ssa_name) == expected_valid

    @pytest.mark.parametrize(
        "block_label,expected_valid",
        [
            ("^bb0", True),
            ("^label", True),
            ("^then_block", True),
            ("bb0", False),
            ("^", False),
            ("label", False),
            ("", False),
        ],
    )
    def test_block_label_validation_parametrized(self, block_label, expected_valid):
        """Test block label validation with various inputs."""
        assert MLIRUtils.is_valid_block_label(block_label) == expected_valid

    @pytest.mark.parametrize(
        "identifier,expected_sanitized",
        [
            ("valid_name", "valid_name"),
            ("invalid-name", "invalid_name"),
            ("123invalid", "_123invalid"),
            ("name with spaces", "name_with_spaces"),
            ("name@special#chars", "name_special_chars"),
            ("", "_unnamed"),
            ("123", "_123"),
        ],
    )
    def test_sanitize_identifier_parametrized(self, identifier, expected_sanitized):
        """Test identifier sanitization with various inputs."""
        assert MLIRUtils.sanitize_identifier(identifier) == expected_sanitized


@pytest.mark.unit
class TestParametrizedCompiler:
    """Parametrized tests for compiler functionality."""

    @pytest.fixture
    def compiler(self):
        return PythonToMLIRCompiler(debug=False)

    @pytest.mark.parametrize(
        "source,expected_patterns",
        [
            (
                "def add(a, b): return a + b",
                ["func.func @add", "arith.addi", "func.return"],
            ),
            (
                "def sub(a, b): return a - b",
                ["func.func @sub", "arith.subi", "func.return"],
            ),
            (
                "def mul(a, b): return a * b",
                ["func.func @mul", "arith.muli", "func.return"],
            ),
            (
                "def div(a, b): return a / b",
                ["func.func @div", "arith.divsi", "func.return"],
            ),
        ],
    )
    def test_arithmetic_operations_parametrized(
        self, compiler, source, expected_patterns
    ):
        """Test compilation of various arithmetic operations."""
        mlir_code = compiler.compile_source(source)
        for pattern in expected_patterns:
            assert pattern in mlir_code

    @pytest.mark.parametrize(
        "source,expected_patterns",
        [
            (
                "def eq(a, b): return a == b",
                ["func.func @eq", "arith.cmpi eq", "func.return"],
            ),
            (
                "def ne(a, b): return a != b",
                ["func.func @ne", "arith.cmpi ne", "func.return"],
            ),
            (
                "def lt(a, b): return a < b",
                ["func.func @lt", "arith.cmpi slt", "func.return"],
            ),
            (
                "def gt(a, b): return a > b",
                ["func.func @gt", "arith.cmpi sgt", "func.return"],
            ),
        ],
    )
    def test_comparison_operations_parametrized(
        self, compiler, source, expected_patterns
    ):
        """Test compilation of various comparison operations."""
        mlir_code = compiler.compile_source(source)
        for pattern in expected_patterns:
            assert pattern in mlir_code

    @pytest.mark.parametrize(
        "constant_value,expected_pattern",
        [
            (0, "arith.constant 0 : i32"),
            (42, "arith.constant 42 : i32"),
            (1000, "arith.constant 1000 : i32"),
            # Note: negative constants are handled as unary minus, not direct constants
        ],
    )
    def test_integer_constants_parametrized(
        self, compiler, constant_value, expected_pattern
    ):
        """Test compilation of various integer constants."""
        source = f"def get_value(): return {constant_value}"
        mlir_code = compiler.compile_source(source)
        assert expected_pattern in mlir_code

    def test_negative_constant_special_case(self, compiler):
        """Test that negative constants are handled as unary minus operations."""
        source = "def get_negative(): return -1"
        mlir_code = compiler.compile_source(source)
        # Negative constants become unary minus operations
        assert "arith.constant 0 : i32" in mlir_code  # zero for subtraction
        assert "arith.constant 1 : i32" in mlir_code  # the positive value
        assert "arith.subi" in mlir_code  # subtraction operation

    @pytest.mark.parametrize(
        "invalid_source",
        [
            "def broken_func(:",  # Missing closing parenthesis
            "def func\n    return 42",  # Missing colon
            # Note: "def func(): return" is actually valid Python (returns None)
            "if True:\npass",  # Missing function definition
            "def func(: return 42",  # Invalid parameter syntax
        ],
    )
    def test_syntax_errors_parametrized(self, compiler, invalid_source):
        """Test handling of various syntax errors."""
        with pytest.raises(SyntaxError):
            compiler.compile_source(invalid_source)

    def test_valid_empty_return(self, compiler):
        """Test that empty return statements are handled correctly."""
        source = "def func(): return"
        mlir_code = compiler.compile_source(source)
        assert "func.func @func" in mlir_code
        assert "func.return" in mlir_code


@pytest.mark.integration
class TestParametrizedIntegration:
    """Parametrized integration tests."""

    @pytest.fixture
    def compiler(self):
        return PythonToMLIRCompiler(debug=False)

    @pytest.mark.parametrize(
        "function_name,args,body,expected_patterns",
        [
            ("simple", ["a"], "return a", ["func.func @simple", "func.return %arg0"]),
            (
                "double",
                ["x"],
                "return x + x",
                ["func.func @double", "arith.addi", "func.return"],
            ),
            (
                "triple",
                ["y"],
                "return y * 3",
                ["func.func @triple", "arith.constant 3", "arith.muli"],
            ),
        ],
    )
    def test_function_patterns_parametrized(
        self, compiler, function_name, args, body, expected_patterns
    ):
        """Test compilation of various function patterns."""
        args_str = ", ".join(args)
        source = f"def {function_name}({args_str}): {body}"
        mlir_code = compiler.compile_source(source)

        for pattern in expected_patterns:
            assert pattern in mlir_code

    @pytest.mark.parametrize(
        "control_structure,expected_patterns",
        [
            (
                "if True:\n        return 1\n    else:\n        return 0",
                ["cf.cond_br", "func.return"],
            ),
            (
                "while n > 0:\n        n = n - 1\n    return n",
                ["cf.cond_br", "cf.br", "arith.subi"],
            ),
        ],
    )
    def test_control_structures_parametrized(
        self, compiler, control_structure, expected_patterns
    ):
        """Test compilation of various control structures."""
        source = f"def test_func(n):\n    {control_structure}"
        mlir_code = compiler.compile_source(source)

        for pattern in expected_patterns:
            assert pattern in mlir_code

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "complexity_level,source",
        [
            (
                "simple_recursive",
                """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
""",
            ),
            (
                "multiple_functions",
                """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def combined(x, y):
    return multiply(add(x, y), 2)
""",
            ),
            (
                "nested_control_flow",
                """
def complex_func(n):
    result = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            result = result + i
        i = i + 1
    return result
""",
            ),
        ],
    )
    def test_complex_compilation_parametrized(self, compiler, complexity_level, source):
        """Test compilation of complex code patterns."""
        mlir_code = compiler.compile_source(source)

        # Basic checks for valid MLIR
        assert "func.func" in mlir_code
        assert "func.return" in mlir_code

        # Verify no obvious syntax errors in generated code
        lines = mlir_code.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("//"):
                # Should not have obvious syntax issues
                assert not line.endswith(",")  # No trailing commas
                assert line.count("(") == line.count(")")  # Balanced parentheses


if __name__ == "__main__":
    # Run parametrized tests
    pytest.main([__file__, "-v"])
