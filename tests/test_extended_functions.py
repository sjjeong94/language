"""
Tests for extended oven.language functions (Y dimension, arithmetic, etc.)
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from oven.compiler import PythonToMLIRCompiler


@pytest.mark.integration
class TestExtendedOvenFunctions:
    """Test extended oven.language functions."""

    def test_y_dimension_gpu_functions(self, compiler):
        """Test Y dimension GPU functions."""
        source = """
import oven.language as ol

def test_2d_kernel(a_ptr, b_ptr):
    tid_x = ol.get_tid_x()
    tid_y = ol.get_tid_y()
    bid_x = ol.get_bid_x()
    bid_y = ol.get_bid_y()
    return
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature for GPU kernel
        assert "func.func @test_2d_kernel(%a: !llvm.ptr, %b: !llvm.ptr)" in mlir_code

        # Check Y dimension operations are present
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "nvvm.read.ptx.sreg.tid.y" in mlir_code
        assert "nvvm.read.ptx.sreg.ctaid.x" in mlir_code
        assert "nvvm.read.ptx.sreg.ctaid.y" in mlir_code

    def test_arithmetic_operations(self, compiler):
        """Test arithmetic operation functions."""
        source = """
import oven.language as ol

def test_arithmetic(a_ptr, b_ptr):
    x = ol.get_tid_x()
    y = ol.get_tid_y()
    
    # Integer arithmetic
    mul_result = ol.muli(x, y)
    add_result = ol.addi(mul_result, x)
    
    # Float arithmetic (after loading)
    a_val = ol.load(a_ptr, add_result)
    b_val = ol.load(b_ptr, add_result)
    mul_float = ol.mulf(a_val, b_val)
    add_float = ol.addf(mul_float, a_val)
    
    ol.store(add_float, a_ptr, add_result)
"""
        mlir_code = compiler.compile_source(source)

        # Check arithmetic operations are present
        assert "arith.muli" in mlir_code
        assert "arith.addi" in mlir_code
        assert "arith.mulf" in mlir_code
        assert "arith.addf" in mlir_code

    def test_constant_operations(self, compiler):
        """Test constant creation functions."""
        source = """
import oven.language as ol

def test_constants(a_ptr):
    zero = ol.constant(0, "i32")
    offset = ol.addi(zero, ol.get_tid_x())
    value = ol.load(a_ptr, offset)
    ol.store(value, a_ptr, zero)
"""
        mlir_code = compiler.compile_source(source)

        # Check constant operations are present
        assert "arith.constant 0 : i32" in mlir_code

    @pytest.mark.parametrize(
        "func_call,expected_op",
        [
            ("ol.get_tid_y()", "nvvm.read.ptx.sreg.tid.y"),
            ("ol.get_bid_y()", "nvvm.read.ptx.sreg.ctaid.y"),
            ("ol.muli(a, b)", "arith.muli"),
            ("ol.addi(a, b)", "arith.addi"),
            ("ol.mulf(a, b)", "arith.mulf"),
            ("ol.addf(a, b)", "arith.addf"),
        ],
    )
    def test_individual_extended_functions(self, compiler, func_call, expected_op):
        """Test individual extended functions."""
        if "muli" in func_call or "addi" in func_call:
            source = f"""
import oven.language as ol

def test_func(ptr):
    a = ol.get_tid_x()
    b = ol.get_tid_y()
    result = {func_call}
    return
"""
        elif "mulf" in func_call or "addf" in func_call:
            source = f"""
import oven.language as ol

def test_func(ptr):
    a = ol.load(ptr, 0)
    b = ol.load(ptr, 1)
    result = {func_call}
    ol.store(result, ptr, 0)
"""
        else:
            source = f"""
import oven.language as ol

def test_func():
    result = {func_call}
    return
"""

        mlir_code = compiler.compile_source(source)
        assert expected_op in mlir_code

    def test_mixed_2d_gpu_with_arithmetic(self, compiler):
        """Test 2D GPU functions combined with arithmetic operations."""
        source = """
import oven.language as ol

def matrix_index_kernel(a_ptr, b_ptr, width):
    row = ol.get_tid_y()
    col = ol.get_tid_x()
    
    # Calculate linear index: row * width + col
    row_offset = ol.muli(row, width)
    linear_index = ol.addi(row_offset, col)
    
    # Load, process, and store
    value = ol.load(a_ptr, linear_index)
    processed = ol.mulf(value, ol.constant(2.0, "f32"))
    ol.store(processed, b_ptr, linear_index)
"""
        mlir_code = compiler.compile_source(source)

        # Check all operations are present
        assert "nvvm.read.ptx.sreg.tid.y" in mlir_code
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "arith.muli" in mlir_code
        assert "arith.addi" in mlir_code
        assert "arith.mulf" in mlir_code
        assert "oven.load" in mlir_code
        assert "oven.store" in mlir_code
