"""Test automatic operator type inference."""

import pytest
from oven.compiler import compile_python_string


def test_automatic_multiply_gpu_function():
    """Test that multiplication in GPU functions uses mulf (float)."""
    python_code = """
import oven.language as ol

def test_kernel():
    x = ol.load_input_x(0)
    y = ol.load_input_x(1)
    result = x * y
    ol.store_output_x(result, 0)
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.mulf" in mlir_output
    assert "arith.muli" not in mlir_output


def test_automatic_multiply_math_function():
    """Test that multiplication in math functions uses mulf (float)."""
    python_code = """
import oven.language as ol

def test_math():
    x = 3.0
    y = 2.0
    result = x * y
    return ol.exp(result)
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.mulf" in mlir_output
    assert "arith.muli" not in mlir_output


def test_automatic_multiply_regular_function():
    """Test that multiplication in regular functions uses muli (integer)."""
    python_code = """
def test_regular():
    x = 3
    y = 2
    result = x * y
    return result
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.muli" in mlir_output
    assert "arith.mulf" not in mlir_output


def test_automatic_addition_gpu_function():
    """Test that addition in GPU functions uses addf (float)."""
    python_code = """
import oven.language as ol

def test_kernel():
    x = ol.load_input_x(0)
    y = ol.load_input_x(1)
    result = x + y
    ol.store_output_x(result, 0)
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.addf" in mlir_output
    assert "arith.addi" not in mlir_output


def test_automatic_subtraction_math_function():
    """Test that subtraction in math functions uses subf (float)."""
    python_code = """
import oven.language as ol

def test_math():
    x = 5.0
    y = 2.0
    result = x - y
    return ol.sigmoid(result)
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.subf" in mlir_output
    assert "arith.subi" not in mlir_output


def test_automatic_subtraction_regular_function():
    """Test that subtraction in regular functions uses subi (integer)."""
    python_code = """
def test_regular():
    x = 5
    y = 2
    result = x - y
    return result
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.subi" in mlir_output
    assert "arith.subf" not in mlir_output


def test_automatic_division_gpu_function():
    """Test that division in GPU functions uses divf (float)."""
    python_code = """
import oven.language as ol

def test_kernel():
    x = ol.load_input_x(0)
    y = ol.load_input_x(1)
    result = x / y
    ol.store_output_x(result, 0)
"""

    mlir_output = compile_python_string(python_code)
    assert "arith.divf" in mlir_output


def test_mixed_operators():
    """Test mixed operators in different function types."""
    python_code = """
import oven.language as ol

def test_kernel():
    x = ol.load_input_x(0)
    y = ol.load_input_x(1)
    z = ol.load_input_x(2)
    
    # All should use float operations in GPU context
    sum_val = x + y
    diff_val = x - z
    prod_val = sum_val * diff_val
    div_val = prod_val / y
    
    ol.store_output_x(div_val, 0)

def regular_math(a, b):
    # Should use integer operations
    return (a + b) * (a - b)
"""

    mlir_output = compile_python_string(python_code)

    # GPU function should use float operations
    assert "arith.addf" in mlir_output
    assert "arith.subf" in mlir_output
    assert "arith.mulf" in mlir_output
    assert "arith.divf" in mlir_output

    # Regular function should use integer operations
    assert "arith.addi" in mlir_output
    assert "arith.subi" in mlir_output
    assert "arith.muli" in mlir_output


def test_y_dimension_operators():
    """Test operators in Y-dimension GPU functions."""
    python_code = """
import oven.language as ol

def test_y_kernel():
    x = ol.load_input_y(0)
    y = ol.load_input_y(1)
    result = x * y + x - y
    ol.store_output_y(result, 0)
"""

    mlir_output = compile_python_string(python_code)

    # Y-dimension GPU functions should also use float operations
    assert "arith.mulf" in mlir_output
    assert "arith.addf" in mlir_output
    assert "arith.subf" in mlir_output
