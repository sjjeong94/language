"""
Unit tests for Python to MLIR Compiler
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oven.compiler import PythonToMLIRCompiler, CompilationError, compile_python_string
from oven.ast_visitor import PythonToMLIRASTVisitor
from oven.mlir_generator import MLIRGenerator
from oven.utils.mlir_utils import MLIRUtils


@pytest.mark.unit
class TestMLIRUtils:
    """Test MLIR utility functions."""

    def setup_method(self):
        self.utils = MLIRUtils()

    def test_python_type_to_mlir(self):
        """Test Python to MLIR type conversion."""
        assert MLIRUtils.python_type_to_mlir("int") == "i32"
        assert MLIRUtils.python_type_to_mlir("float") == "f32"
        assert MLIRUtils.python_type_to_mlir("bool") == "i1"
        assert MLIRUtils.python_type_to_mlir("str") == "!llvm.ptr<i8>"

    def test_ssa_name_validation(self):
        """Test SSA value name validation."""
        assert MLIRUtils.is_valid_ssa_name("%0") is True
        assert MLIRUtils.is_valid_ssa_name("%result") is True
        assert MLIRUtils.is_valid_ssa_name("0") is False
        assert MLIRUtils.is_valid_ssa_name("%") is False

    def test_block_label_validation(self):
        """Test block label validation."""
        assert MLIRUtils.is_valid_block_label("^bb0") is True
        assert MLIRUtils.is_valid_block_label("^label") is True
        assert MLIRUtils.is_valid_block_label("bb0") is False
        assert MLIRUtils.is_valid_block_label("^") is False

    def test_sanitize_identifier(self):
        """Test identifier sanitization."""
        assert MLIRUtils.sanitize_identifier("valid_name") == "valid_name"
        assert MLIRUtils.sanitize_identifier("invalid-name") == "invalid_name"
        assert MLIRUtils.sanitize_identifier("123invalid") == "_123invalid"


@pytest.mark.unit
class TestMLIRGenerator:
    """Test MLIR code generation."""

    def test_constant_generation(self, mlir_generator):
        """Test constant value generation."""
        ssa_val = mlir_generator.add_constant_int(42)
        assert ssa_val.startswith("%")
        code = mlir_generator.get_code()
        assert "arith.constant 42 : i32" in code

    def test_binary_operation(self, mlir_generator):
        """Test binary operation generation."""
        val1 = mlir_generator.add_constant_int(10)
        val2 = mlir_generator.add_constant_int(20)
        result = mlir_generator.add_binary_op("arith.addi", val1, val2)

        code = mlir_generator.get_code()
        assert "arith.addi" in code
        assert result.startswith("%")

    def test_function_definition(self, mlir_generator):
        """Test function definition generation."""
        mlir_generator.start_function("test_func", ["i32", "i32"], "i32")
        mlir_generator.add_return("%0")
        mlir_generator.end_function()

        code = mlir_generator.get_code()
        assert "func.func @test_func" in code
        assert "func.return %0" in code


@pytest.mark.unit
class TestASTVisitor:
    """Test AST visitor functionality."""

    def test_simple_function(
        self, ast_visitor, sample_python_code, expected_mlir_patterns
    ):
        """Test compilation of a simple function."""
        import ast

        tree = ast.parse(sample_python_code["simple_function"])
        ast_visitor.visit(tree)

        mlir_code = ast_visitor.get_mlir_code()
        assert expected_mlir_patterns["function_declaration"] + "add" in mlir_code
        assert expected_mlir_patterns["integer_addition"] in mlir_code
        assert expected_mlir_patterns["function_return"] in mlir_code


@pytest.mark.unit
class TestCompiler:
    """Test main compiler functionality."""

    def test_compile_simple_function(
        self, compiler, sample_python_code, expected_mlir_patterns
    ):
        """Test compiling a simple function."""
        mlir_code = compiler.compile_source(sample_python_code["simple_function"])
        assert expected_mlir_patterns["function_declaration"] + "add" in mlir_code
        assert expected_mlir_patterns["integer_addition"] in mlir_code

    def test_compile_with_constants(
        self, compiler, sample_python_code, expected_mlir_patterns
    ):
        """Test compiling code with constants."""
        mlir_code = compiler.compile_source(
            sample_python_code["function_with_constants"]
        )
        assert expected_mlir_patterns["integer_constant"] + " 42" in mlir_code

    def test_compile_with_if_statement(
        self, compiler, sample_python_code, expected_mlir_patterns
    ):
        """Test compiling code with if statement."""
        mlir_code = compiler.compile_source(sample_python_code["function_with_if"])
        assert expected_mlir_patterns["integer_comparison"] in mlir_code
        assert expected_mlir_patterns["conditional_branch"] in mlir_code

    def test_compile_with_while_loop(
        self, compiler, sample_python_code, expected_mlir_patterns
    ):
        """Test compiling code with while loop."""
        mlir_code = compiler.compile_source(sample_python_code["function_with_while"])
        assert expected_mlir_patterns["conditional_branch"] in mlir_code
        assert expected_mlir_patterns["unconditional_branch"] in mlir_code

    def test_syntax_error_handling(self, compiler):
        """Test handling of Python syntax errors."""
        source = "def broken_func(:\n    return 42"

        with pytest.raises(SyntaxError):
            compiler.compile_source(source)

    def test_file_compilation(self, compiler, temp_python_file, expected_mlir_patterns):
        """Test compiling from file."""
        content = """
def hello():
    return 1337
"""
        temp_file = temp_python_file(content)

        mlir_code = compiler.compile_file(temp_file)
        assert expected_mlir_patterns["function_declaration"] + "hello" in mlir_code
        assert expected_mlir_patterns["integer_constant"] + " 1337" in mlir_code

    def test_convenience_function(self, expected_mlir_patterns):
        """Test convenience compilation function."""
        source = "def test(): return 1 + 2"
        mlir_code = compile_python_string(source)
        assert expected_mlir_patterns["function_declaration"] + "test" in mlir_code
        assert expected_mlir_patterns["integer_addition"] in mlir_code


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete compiler pipeline."""

    def test_complex_example(self, sample_python_code, expected_mlir_patterns):
        """Test compilation of a more complex example."""
        compiler = PythonToMLIRCompiler(debug=False)
        mlir_code = compiler.compile_source(sample_python_code["complex_example"])

        # Check that all expected elements are present
        assert expected_mlir_patterns["function_declaration"] + "fibonacci" in mlir_code
        assert expected_mlir_patterns["function_declaration"] + "main" in mlir_code
        assert expected_mlir_patterns["function_call"] + " @fibonacci" in mlir_code
        assert expected_mlir_patterns["integer_comparison"] in mlir_code
        assert expected_mlir_patterns["conditional_branch"] in mlir_code

    def test_sample_file_compilation(self, expected_mlir_patterns):
        """Test compilation of the sample file."""
        sample_file = os.path.join(os.path.dirname(__file__), "examples", "sample.py")

        if os.path.exists(sample_file):
            compiler = PythonToMLIRCompiler(debug=False)
            mlir_code = compiler.compile_file(sample_file)

            # Check that all functions are compiled
            assert expected_mlir_patterns["function_declaration"] + "add" in mlir_code
            assert (
                expected_mlir_patterns["function_declaration"] + "factorial"
                in mlir_code
            )
            assert expected_mlir_patterns["function_declaration"] + "main" in mlir_code

    @pytest.mark.slow
    def test_recursive_compilation(self, sample_python_code, expected_mlir_patterns):
        """Test compilation of recursive functions."""
        compiler = PythonToMLIRCompiler(debug=False)
        mlir_code = compiler.compile_source(sample_python_code["recursive_function"])

        # Check recursive call
        assert expected_mlir_patterns["function_call"] + " @factorial" in mlir_code
        assert expected_mlir_patterns["integer_multiplication"] in mlir_code


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
