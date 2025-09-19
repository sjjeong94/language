"""Test edge cases for for loop compilation."""

import pytest
from oven.compiler import compile_python_string


def test_for_loop_edge_cases():
    """Test edge cases in for loop compilation."""

    # Test with single iteration
    python_code1 = """
def single_iteration():
    result = 0
    for i in range(1):
        result = result + i
    return result
"""

    mlir_output1 = compile_python_string(python_code1)
    assert "scf.for" in mlir_output1

    # Test with zero iterations
    python_code2 = """
def zero_iterations():
    result = 0
    for i in range(0):
        result = result + i
    return result
"""

    mlir_output2 = compile_python_string(python_code2)
    assert "scf.for" in mlir_output2

    # Test with negative step (should still work)
    python_code3 = """
def negative_step():
    result = 0
    for i in range(10, 0, -1):
        result = result + i
    return result
"""

    mlir_output3 = compile_python_string(python_code3)
    assert "scf.for" in mlir_output3


def test_for_loop_multiple_variables():
    """Test for loop with multiple accumulator variables."""
    python_code = """
def multiple_accumulators():
    sum_val = 0
    product = 1
    count = 0
    
    for i in range(5):
        sum_val = sum_val + i
        product = product * 2
        count = count + 1
    
    return sum_val + product + count
"""

    mlir_output = compile_python_string(python_code)
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output
    assert "scf.yield" in mlir_output


def test_for_loop_in_gpu_context():
    """Test comprehensive GPU kernel with for loop."""
    python_code = """
import oven.language as ol

def comprehensive_gpu_kernel(input_ptr, output_ptr, temp_ptr, size, iterations):
    # GPU thread information
    tid_x = ol.get_tid_x()
    tid_y = ol.get_tid_y()
    bid_x = ol.get_bid_x()
    bid_y = ol.get_bid_y()
    bdim_x = ol.get_bdim_x()
    
    # Calculate global thread ID
    global_x = ol.addi(ol.muli(bid_x, bdim_x), tid_x)
    global_y = ol.addi(ol.muli(bid_y, bdim_x), tid_y)
    global_idx = ol.addi(ol.muli(global_y, size), global_x)
    
    # Check bounds
    if global_idx < size:
        # Initialize accumulator
        accumulator = 0.0
        
        # Main computation loop
        for i in range(iterations):
            # Calculate offset for this iteration
            offset = ol.addi(ol.muli(global_idx, iterations), i)
            
            # Load input value
            input_val = ol.load(input_ptr, offset)
            
            # Apply some computation
            processed_val = ol.mulf(input_val, 2.0)
            
            # Accumulate
            accumulator = ol.addf(accumulator, processed_val)
            
            # Store intermediate result
            ol.store(processed_val, temp_ptr, offset)
        
        # Store final result
        ol.store(accumulator, output_ptr, global_idx)
"""

    mlir_output = compile_python_string(python_code)

    # Verify all expected components
    expected_elements = [
        "func.func @comprehensive_gpu_kernel",
        "nvvm.read.ptx.sreg.tid.x",
        "nvvm.read.ptx.sreg.tid.y",
        "nvvm.read.ptx.sreg.ctaid.x",
        "nvvm.read.ptx.sreg.ctaid.y",
        "nvvm.read.ptx.sreg.ntid.x",
        "scf.for",
        "iter_args",
        "scf.yield",
        "oven.load",
        "oven.store",
        "arith.muli",
        "arith.addi",
        "arith.mulf",
        "arith.addf",
    ]

    for element in expected_elements:
        assert element in mlir_output, f"Missing element: {element}"


def test_nested_operations_in_for_loop():
    """Test complex nested operations within for loop."""
    python_code = """
import oven.language as ol

def complex_loop_operations():
    result_a = 0.0
    result_b = 1.0
    
    for i in range(3):
        # Complex expression involving loop variable
        temp1 = ol.mulf(float(i), 2.5)
        temp2 = ol.addf(temp1, 1.0)
        
        # Update accumulators with different operations
        result_a = ol.addf(result_a, temp2)
        result_b = ol.mulf(result_b, temp2)
    
    # Final computation
    return ol.addf(result_a, result_b)
"""

    mlir_output = compile_python_string(python_code)

    # Check for proper structure
    assert "scf.for" in mlir_output
    assert "iter_args" in mlir_output
    assert "scf.yield" in mlir_output
    assert "arith.addf" in mlir_output
    assert "arith.mulf" in mlir_output


def test_for_loop_with_function_calls():
    """Test for loop containing function calls."""
    python_code = """
import oven.language as ol

def loop_with_math_functions():
    sum_exp = 0.0
    sum_sigmoid = 0.0
    
    for i in range(5):
        val = float(i) * 0.1
        exp_val = ol.exp(val)
        sigmoid_val = ol.sigmoid(val)
        
        sum_exp = ol.addf(sum_exp, exp_val)
        sum_sigmoid = ol.addf(sum_sigmoid, sigmoid_val)
    
    return ol.addf(sum_exp, sum_sigmoid)
"""

    mlir_output = compile_python_string(python_code)

    # Check for math functions within loop
    assert "scf.for" in mlir_output
    assert "math.exp" in mlir_output
    assert "oven.sigmoid" in mlir_output or "math.sigmoid" in mlir_output
    assert "scf.yield" in mlir_output
