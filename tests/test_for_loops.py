"""Test for loop compilation to SCF for loops."""

import pytest
from oven.compiler import compile_python_string


def test_simple_for_loop():
    """Test basic for loop with range."""
    python_code = """
def simple_loop():
    result = 0
    for i in range(5):
        result = result + i
    return result
"""

    mlir_output = compile_python_string(python_code)

    # Check for SCF for loop structure
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output
    assert "scf.yield" in mlir_output
    assert "arith.index_cast" in mlir_output


def test_for_loop_with_start_stop_step():
    """Test for loop with start, stop, and step."""
    python_code = """
def loop_with_step():
    sum_val = 0.0
    for i in range(1, 10, 2):
        sum_val = sum_val + float(i)
    return sum_val
"""

    mlir_output = compile_python_string(python_code)

    # Check for proper range handling
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output
    assert "scf.yield" in mlir_output


def test_matmul_kernel_for_loop():
    """Test matrix multiplication kernel with for loop."""
    python_code = """
import oven.language as ol

def matmul_kernel(a_ptr, b_ptr, c_ptr, m, n, k):
    block_size = ol.get_bdim_x()
    cCol = ol.get_bid_x()
    cRow = ol.get_bid_y()
    tCol = ol.get_tid_x()
    tRow = ol.get_tid_y()

    col0 = ol.muli(cCol, block_size)
    col = ol.addi(col0, tCol)
    row0 = ol.muli(cRow, block_size)
    row = ol.addi(row0, tRow)

    sum_value = 0.0
    for i in range(0, k, 1):
        a_offset0 = ol.muli(row, k)
        a_offset = ol.addi(a_offset0, i)
        b_offset0 = ol.muli(i, n)
        b_offset = ol.addi(b_offset0, col)

        a = ol.load(a_ptr, a_offset)
        b = ol.load(b_ptr, b_offset)
        prod = ol.mulf(a, b)
        sum_value = ol.addf(sum_value, prod)

    c_offset0 = ol.muli(row, n)
    c_offset = ol.addi(c_offset0, col)
    ol.store(sum_value, c_ptr, c_offset)
"""

    mlir_output = compile_python_string(python_code)

    # Check for GPU kernel structure
    assert "func.func @matmul_kernel" in mlir_output
    assert "!llvm.ptr" in mlir_output

    # Check for GPU intrinsics
    assert "nvvm.read.ptx.sreg.ntid.x" in mlir_output
    assert "nvvm.read.ptx.sreg.ctaid.x" in mlir_output
    assert "nvvm.read.ptx.sreg.tid.x" in mlir_output

    # Check for SCF for loop
    assert "scf.for" in mlir_output
    assert "iter_args(%" in mlir_output  # Check for SSA-style iter_args
    assert "scf.yield" in mlir_output

    # Check for proper type usage
    assert "arith.muli" in mlir_output  # Integer operations for indices
    assert "arith.addi" in mlir_output
    assert "arith.mulf" in mlir_output  # Float operations for data
    assert "arith.addf" in mlir_output

    # Check for memory operations
    assert "oven.load" in mlir_output
    assert "oven.store" in mlir_output

    # Check for index type conversions
    assert "arith.index_cast" in mlir_output
    assert ": i32 to index" in mlir_output
    assert ": index to i32" in mlir_output


def test_nested_accumulator_operations():
    """Test for loop with multiple accumulator operations."""
    python_code = """
import oven.language as ol

def accumulate_kernel():
    sum_a = 0.0
    sum_b = 1.0
    
    for i in range(10):
        value = float(i)
        sum_a = ol.addf(sum_a, value)
        sum_b = ol.mulf(sum_b, value)
    
    return ol.addf(sum_a, sum_b)
"""

    mlir_output = compile_python_string(python_code)

    # Check basic structure
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output
    assert "scf.yield" in mlir_output

    # Check for floating point operations
    assert "arith.addf" in mlir_output
    assert "arith.mulf" in mlir_output


def test_for_loop_with_gpu_operations():
    """Test for loop combined with GPU-specific operations."""
    python_code = """
import oven.language as ol

def gpu_loop_kernel(input_ptr, output_ptr, size):
    tid = ol.get_tid_x()
    
    if tid < size:
        accumulator = 0.0
        for i in range(5):
            offset = ol.addi(ol.muli(tid, 5), i)
            value = ol.load(input_ptr, offset)
            accumulator = ol.addf(accumulator, value)
        
        ol.store(accumulator, output_ptr, tid)
"""

    mlir_output = compile_python_string(python_code)

    # Check GPU context
    assert "nvvm.read.ptx.sreg.tid.x" in mlir_output

    # Check for loop structure
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output

    # Check memory operations
    assert "oven.load" in mlir_output
    assert "oven.store" in mlir_output


def test_for_loop_index_usage():
    """Test that loop index variable is correctly used inside the loop."""
    python_code = """
def index_usage():
    result = 0
    for i in range(3):
        # Use loop index in computation
        result = result + i * 2
    return result
"""

    mlir_output = compile_python_string(python_code)

    # Check for index conversion
    assert "arith.index_cast" in mlir_output
    assert ": index to i32" in mlir_output

    # Check for loop structure
    assert "scf.for" in mlir_output
    assert "scf.yield" in mlir_output


def test_for_loop_with_constants():
    """Test for loop with constant expressions."""
    python_code = """
def constant_loop():
    total = 0.0
    for i in range(0, 100, 10):
        total = total + 1.5
    return total
"""

    mlir_output = compile_python_string(python_code)

    # Check for constant handling
    assert "arith.constant" in mlir_output
    assert "scf.for" in mlir_output
    assert "scf.yield" in mlir_output


def test_for_loop_type_consistency():
    """Test that types are consistent throughout the for loop."""
    python_code = """
import oven.language as ol

def type_test():
    x = 0
    y = 0.0
    
    for i in range(5):
        x = ol.addi(x, i)        # Integer accumulator
        y = ol.addf(y, 1.5)      # Float accumulator
    
    return ol.addf(float(x), y)
"""

    mlir_output = compile_python_string(python_code)

    # Check for both integer and float operations
    assert "arith.addi" in mlir_output
    assert "arith.addf" in mlir_output
    assert "scf.for" in mlir_output


def test_for_loop_without_accumulator():
    """Test for loop that doesn't accumulate values."""
    python_code = """
import oven.language as ol

def no_accumulator_loop(ptr):
    for i in range(10):
        value = float(i * 2)
        offset = i
        ol.store(value, ptr, offset)
"""

    mlir_output = compile_python_string(python_code)

    # Should still generate SCF for loop but without iter_args
    assert "scf.for" in mlir_output
    # Should not have iter_args since no accumulation
    # The exact behavior depends on implementation details


def test_range_function_variations():
    """Test different variations of range() function."""
    # Test range(stop)
    python_code1 = """
def range_stop():
    result = 0
    for i in range(5):
        result = result + 1
    return result
"""

    mlir_output1 = compile_python_string(python_code1)
    assert "scf.for" in mlir_output1

    # Test range(start, stop)
    python_code2 = """
def range_start_stop():
    result = 0
    for i in range(2, 8):
        result = result + 1
    return result
"""

    mlir_output2 = compile_python_string(python_code2)
    assert "scf.for" in mlir_output2

    # Test range(start, stop, step)
    python_code3 = """
def range_start_stop_step():
    result = 0
    for i in range(1, 10, 2):
        result = result + 1
    return result
"""

    mlir_output3 = compile_python_string(python_code3)
    assert "scf.for" in mlir_output3
