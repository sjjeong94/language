"""Test type hint support in Python function definitions."""

import pytest
from oven.compiler import compile_python_string
import oven.language as ol


def test_basic_type_hints():
    """Test basic type hints like int, f32."""
    python_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""

    result = compile_python_string(python_code)

    # Should generate function with i32 types
    assert "func.func @add_numbers(%arg0: i32, %arg1: i32) -> i32" in result
    assert "arith.addi" in result  # Integer addition


def test_float_type_hints():
    """Test f32 type hints."""
    python_code = """
def add_floats(a: f32, b: f32) -> f32:
    return a + b
"""

    result = compile_python_string(python_code)

    # Should generate function with f32 types
    assert "func.func @add_floats(%arg0: f32, %arg1: f32) -> f32" in result
    assert "arith.addf" in result  # Float addition


def test_pointer_type_hints():
    """Test ol.ptr type hints for GPU functions."""
    python_code = """
import oven.language as ol

def gpu_kernel(a_ptr: ol.ptr, b_ptr: ol.ptr):
    tid = ol.get_tid_x()
    a_val = ol.load(a_ptr, tid)
    ol.store(b_ptr, tid, a_val)
"""

    result = compile_python_string(python_code)

    # Should generate GPU function with pointer types (using actual parameter names)
    assert "func.func @gpu_kernel(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr)" in result
    assert "nvvm.read.ptx.sreg.tid.x" in result


def test_mixed_type_hints():
    """Test mixing different type hints."""
    python_code = """
import oven.language as ol

def mixed_function(ptr_arg: ol.ptr, int_arg: int, float_arg: f32) -> f32:
    tid = ol.get_tid_x()
    val = ol.load(ptr_arg, tid)
    result = val + float_arg
    return result
"""

    result = compile_python_string(python_code)

    # Should generate function with mixed types (GPU function uses actual parameter names)
    assert (
        "func.func @mixed_function(%ptr_arg: !llvm.ptr, %int_arg: i32, %float_arg: f32)"
        in result
    )
    assert "arith.addf" in result  # Float addition should be used


def test_no_type_hints_fallback():
    """Test that functions without type hints still work with context inference."""
    python_code = """
import oven.language as ol

def gpu_function_no_hints(a, b):
    tid = ol.get_tid_x()
    val = ol.load(a, tid)
    ol.store(b, tid, val)
"""

    result = compile_python_string(python_code)

    # Should fall back to context-based inference (GPU function -> pointers, using actual names)
    assert "func.func @gpu_function_no_hints(%a: !llvm.ptr, %b: !llvm.ptr)" in result


def test_math_function_with_hints():
    """Test math function with explicit type hints."""
    python_code = """
import oven.language as ol

def compute_sigmoid(x: f32) -> f32:
    return ol.sigmoid(x)
"""

    result = compile_python_string(python_code)

    # Should generate function with f32 types
    assert "func.func @compute_sigmoid(%arg0: f32) -> f32" in result
    assert "oven.sigmoid" in result  # Sigmoid operation should be present


def test_index_type_hints():
    """Test index type hints."""
    python_code = """
def use_index(idx: index) -> index:
    return idx + 1
"""

    result = compile_python_string(python_code)

    # Should generate function with index types
    assert "func.func @use_index(%arg0: index) -> index" in result


def test_partial_type_hints():
    """Test functions with some parameters having type hints and others not."""
    python_code = """
def partial_hints(typed_arg: f32, untyped_arg) -> f32:
    return typed_arg + untyped_arg
"""

    result = compile_python_string(python_code)

    # First arg should use hint, second should fall back to context
    assert "func.func @partial_hints(%arg0: f32, %arg1: i32) -> f32" in result


def test_explicit_i32_hints():
    """Test explicit i32 type hints."""
    python_code = """
import oven.language as ol

def explicit_i32(a: ol.i32, b: ol.i32) -> ol.i32:
    return a * b
"""

    result = compile_python_string(python_code)

    # Should generate function with i32 types
    assert "func.func @explicit_i32(%arg0: i32, %arg1: i32) -> i32" in result
    assert "arith.muli" in result  # Integer multiplication


if __name__ == "__main__":
    pytest.main([__file__])
