"""
Tests for mathematical function compilation
"""

import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oven.compiler import PythonToMLIRCompiler


@pytest.mark.integration
class TestMathFunctionCompilation:
    """Test mathematical function specific compilation features."""

    def test_exp_function(self, compiler):
        """Test that exp function is correctly compiled."""
        source = """
def test_exp(x):
    result = exp(x)
    return result
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types
        assert "func.func @test_exp(%arg0: f32) -> f32" in mlir_code

        # Check exp operation is present
        assert "math.exp %arg0 : f32" in mlir_code

        # Check return type is f32 (flexible SSA numbering)
        assert "func.return %" in mlir_code and ": f32" in mlir_code

    def test_sigmoid_function(self, compiler):
        """Test that sigmoid function is correctly compiled."""
        source = """
def test_sigmoid(x):
    result = sigmoid(x)
    return result
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types
        assert "func.func @test_sigmoid(%arg0: f32) -> f32" in mlir_code

        # Check sigmoid operation is present
        assert "oven.sigmoid %arg0 : f32 -> f32" in mlir_code

        # Check return type is f32 (flexible SSA numbering)
        assert "func.return %" in mlir_code and ": f32" in mlir_code

    def test_multiple_math_functions(self, compiler):
        """Test compilation of file with multiple mathematical functions."""
        source = """
def test_exp(x):
    return exp(x)

def test_sigmoid(x):
    return sigmoid(x)

def combined_math(x):
    a = exp(x)
    b = sigmoid(a)
    return b
"""
        mlir_code = compiler.compile_source(source)

        # Check all functions have f32 signatures
        assert "func.func @test_exp(%arg0: f32) -> f32" in mlir_code
        assert "func.func @test_sigmoid(%arg0: f32) -> f32" in mlir_code
        assert "func.func @combined_math(%arg0: f32) -> f32" in mlir_code

        # Check both operations are present
        assert "math.exp" in mlir_code
        assert "oven.sigmoid" in mlir_code

    def test_mixed_math_and_regular_functions(self, compiler):
        """Test compilation of file with both math and regular functions."""
        source = """
def regular_function(x, y):
    return x + y

def math_function(x):
    return exp(x)

def another_regular(n):
    if n > 0:
        return n * 2
    else:
        return 0
"""
        mlir_code = compiler.compile_source(source)

        # Check regular functions have i32 signatures
        assert "func.func @regular_function(%arg0: i32, %arg1: i32) -> i32" in mlir_code
        assert "func.func @another_regular(%arg0: i32) -> i32" in mlir_code

        # Check math function has f32 signature
        assert "func.func @math_function(%arg0: f32) -> f32" in mlir_code

        # Check mathematical operation
        assert "math.exp" in mlir_code

    def test_mixed_math_gpu_and_regular_functions(self, compiler):
        """Test compilation of file with math, GPU, and regular functions."""
        source = """
def regular_add(x, y):
    return x + y

def gpu_kernel(a, b):
    tid = get_tid_x()
    val = load(a, tid)
    store(val, b, tid)
    return

def math_exp(x):
    return exp(x)

def math_sigmoid(x):
    return sigmoid(x)
"""
        mlir_code = compiler.compile_source(source)

        # Check regular function has i32 signature
        assert "func.func @regular_add(%arg0: i32, %arg1: i32) -> i32" in mlir_code

        # Check GPU kernel has GPU signature
        assert "func.func @gpu_kernel(%a: !llvm.ptr, %b: !llvm.ptr)" in mlir_code

        # Check math functions have f32 signatures
        assert "func.func @math_exp(%arg0: f32) -> f32" in mlir_code
        assert "func.func @math_sigmoid(%arg0: f32) -> f32" in mlir_code

        # Check different operations are present
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code  # GPU
        assert "math.exp" in mlir_code  # Math
        assert "oven.sigmoid" in mlir_code  # Math
        assert "arith.addi" in mlir_code  # Regular

    @pytest.mark.parametrize(
        "math_func,expected_op",
        [
            ("exp", "math.exp"),
            ("exp2", "math.exp2"),
            ("log2", "math.log2"),
            ("sigmoid", "oven.sigmoid"),
        ],
    )
    def test_individual_math_functions(self, compiler, math_func, expected_op):
        """Test individual mathematical functions are correctly translated."""
        source = f"""
def test_func(x):
    result = {math_func}(x)
    return result
"""
        mlir_code = compiler.compile_source(source)
        assert expected_op in mlir_code
        assert "f32" in mlir_code  # Should use f32 types

    def test_math_function_with_multiple_args(self, compiler):
        """Test mathematical functions with multiple arguments."""
        source = """
def test_multiple_args(x, y):
    a = exp(x)
    b = sigmoid(y)
    return a
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types for all args
        assert (
            "func.func @test_multiple_args(%arg0: f32, %arg1: f32) -> f32" in mlir_code
        )

        # Check both operations are present
        assert "math.exp %arg0 : f32" in mlir_code
        assert "oven.sigmoid %arg1 : f32 -> f32" in mlir_code

    def test_math_functions_py_compilation(self):
        """Test compilation of the actual math_functions.py file."""
        math_file = Path(__file__).parent.parent / "math_functions.py"

        if math_file.exists():
            compiler = PythonToMLIRCompiler(debug=False)
            mlir_code = compiler.compile_file(str(math_file))

            # Verify the expected MLIR patterns
            expected_patterns = [
                "func.func @test_exp(%arg0: f32) -> f32",
                "math.exp %arg0 : f32",
                "func.return %0 : f32",
                "func.func @test_sigmoid(%arg0: f32) -> f32",
                "oven.sigmoid %arg0 : f32 -> f32",
                "func.return %1 : f32",
            ]

            for pattern in expected_patterns:
                assert (
                    pattern in mlir_code
                ), f"Pattern '{pattern}' not found in generated MLIR"

    def test_math_function_detection(self, compiler):
        """Test that functions are correctly identified as math functions."""
        # Function with exp should be detected as math function
        exp_source = """
def has_exp(x):
    return exp(x) + 1
"""
        mlir_code = compiler.compile_source(exp_source)
        assert "func.func @has_exp(%arg0: f32) -> f32" in mlir_code

        # Function with sigmoid should be detected as math function
        sigmoid_source = """
def has_sigmoid(x):
    y = sigmoid(x)
    return y
"""
        mlir_code = compiler.compile_source(sigmoid_source)
        assert "func.func @has_sigmoid(%arg0: f32) -> f32" in mlir_code

        # Function without math operations should use i32
        regular_source = """
def no_math(x):
    return x + 1
"""
        mlir_code = compiler.compile_source(regular_source)
        assert "func.func @no_math(%arg0: i32) -> i32" in mlir_code

    def test_nested_math_calls(self, compiler):
        """Test nested mathematical function calls."""
        source = """
def nested_math(x):
    inner = exp(x)
    outer = sigmoid(inner)
    return outer
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature
        assert "func.func @nested_math(%arg0: f32) -> f32" in mlir_code

        # Check both operations are present in sequence
        assert "math.exp %arg0 : f32" in mlir_code
        assert "oven.sigmoid" in mlir_code

        # Should have intermediate SSA values
        assert "%0 = math.exp" in mlir_code
        assert "%1 = oven.sigmoid %0" in mlir_code

    def test_exp2_function(self, compiler):
        """Test that exp2 function is correctly compiled."""
        source = """
def test_exp2(x):
    result = exp2(x)
    return result
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types
        assert "func.func @test_exp2(%arg0: f32) -> f32" in mlir_code

        # Check exp2 operation is present
        assert "math.exp2 %arg0 : f32" in mlir_code

        # Check return type is f32
        assert "func.return %" in mlir_code and ": f32" in mlir_code

    def test_log2_function(self, compiler):
        """Test that log2 function is correctly compiled."""
        source = """
def test_log2(x):
    result = log2(x)
    return result
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types
        assert "func.func @test_log2(%arg0: f32) -> f32" in mlir_code

        # Check log2 operation is present
        assert "math.log2 %arg0 : f32" in mlir_code

        # Check return type is f32
        assert "func.return %" in mlir_code and ": f32" in mlir_code

    def test_exp2_log2_composition(self, compiler):
        """Test composition of exp2 and log2 functions."""
        source = """
def test_exp2_log2(x):
    a = exp2(x)
    result = log2(a)
    return result
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses f32 types
        assert "func.func @test_exp2_log2(%arg0: f32) -> f32" in mlir_code

        # Check both operations are present in sequence
        assert "math.exp2 %arg0 : f32" in mlir_code
        assert "math.log2" in mlir_code

        # Should have intermediate SSA values
        assert "%0 = math.exp2" in mlir_code
