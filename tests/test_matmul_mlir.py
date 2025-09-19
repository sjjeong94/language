"""Test that the matmul kernel generates the expected MLIR."""

import pytest
from oven.compiler import compile_python_string


def test_matmul_kernel_mlir():
    """Test that the matmul kernel generates correct MLIR structure."""
    python_code = """
import oven.language as ol

def function(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, m: int, n: int, k: int):
    block_size = ol.get_bdim_x()
    cCol = ol.get_bid_x()
    cRow = ol.get_bid_y()
    tCol = ol.get_tid_x()
    tRow = ol.get_tid_y()

    col0 = cCol * block_size
    col = col0 + tCol
    row0 = cRow * block_size
    row = row0 + tRow

    sum_value = 0.0
    for i in range(0, k, 1):
        a_offset0 = row * k
        a_offset = a_offset0 + i
        b_offset0 = i * n
        b_offset = b_offset0 + col

        a = ol.load(a_ptr, a_offset)
        b = ol.load(b_ptr, b_offset)
        prod = a * b
        sum_value = sum_value + prod

    c_offset0 = row * n
    c_offset = c_offset0 + col
    ol.store(sum_value, c_ptr, c_offset)
"""

    result = compile_python_string(python_code)

    # Check function signature
    assert (
        "func.func @function(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %c_ptr: !llvm.ptr, %m: i32, %n: i32, %k: i32)"
        in result
    )

    # Check GPU intrinsics
    assert "nvvm.read.ptx.sreg.ntid.x : i32" in result
    assert "nvvm.read.ptx.sreg.ctaid.x : i32" in result
    assert "nvvm.read.ptx.sreg.ctaid.y : i32" in result
    assert "nvvm.read.ptx.sreg.tid.x : i32" in result
    assert "nvvm.read.ptx.sreg.tid.y : i32" in result

    # Check integer arithmetic for index calculations
    assert "arith.muli" in result and ": i32" in result
    assert "arith.addi" in result and ": i32" in result

    # Check float arithmetic for matrix operations
    assert "arith.mulf" in result and ": f32" in result
    assert "arith.addf" in result and ": f32" in result

    # Check SCF for loop
    assert "scf.for" in result
    assert "scf.yield" in result
    assert "iter_args" in result

    # Check memory operations
    assert "oven.load" in result
    assert "oven.store" in result

    # Check proper constant types
    assert "arith.constant 0.0 : f32" in result
    assert "arith.constant 0 : index" in result
    assert "arith.constant 1 : index" in result


def test_parameter_naming():
    """Test that GPU function parameters have correct names."""
    python_code = """
import oven.language as ol

def function(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, m: int, n: int, k: int):
    tid = ol.get_tid_x()
"""

    result = compile_python_string(python_code)

    # Check that parameters use their actual names instead of %a, %b, %c, etc.
    assert "%a_ptr: !llvm.ptr" in result
    assert "%b_ptr: !llvm.ptr" in result
    assert "%c_ptr: !llvm.ptr" in result
    assert "%m: i32" in result
    assert "%n: i32" in result
    assert "%k: i32" in result


def test_mixed_arithmetic_types():
    """Test that integer and float arithmetic are properly distinguished."""
    python_code = """
import oven.language as ol

def function(a_ptr: ol.ptr):
    # Integer arithmetic (from GPU intrinsics)
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    global_id = tid + bid
    
    # Float arithmetic (from loaded values)
    value1 = ol.load(a_ptr, 0)
    value2 = ol.load(a_ptr, 1)
    sum_val = value1 + value2
"""

    result = compile_python_string(python_code)

    # Should have both integer and float operations
    assert "arith.addi" in result and ": i32" in result  # For tid + bid
    assert "arith.addf" in result and ": f32" in result  # For value1 + value2
    assert "oven.load" in result and "-> f32" in result  # Loads return f32


if __name__ == "__main__":
    pytest.main([__file__])
