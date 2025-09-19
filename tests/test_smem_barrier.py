"""Test shared memory (smem) and barrier functionality."""

import pytest
from oven.compiler import compile_python_string
import oven.language as ol


def test_smem_allocation():
    """Test shared memory allocation."""
    python_code = """
import oven.language as ol

def test_smem():
    smem_ptr = ol.smem()
    return smem_ptr
"""

    result = compile_python_string(python_code)

    # Should generate shared memory allocation
    assert "oven.smem : !llvm.ptr<3>" in result


def test_barrier_sync():
    """Test barrier synchronization."""
    python_code = """
import oven.language as ol

def test_barrier():
    ol.barrier()
"""

    result = compile_python_string(python_code)

    # Should generate barrier synchronization
    assert "nvvm.barrier0" in result


def test_smem_and_barrier_together():
    """Test using shared memory with barriers."""
    python_code = """
import oven.language as ol

def gpu_kernel_with_smem(a_ptr: ol.ptr, b_ptr: ol.ptr):
    smem = ol.smem()
    tid = ol.get_tid_x()
    
    # Load from global memory to shared memory
    value = ol.load(a_ptr, tid)
    ol.store(value, smem, tid)
    
    # Synchronize threads
    ol.barrier()
    
    # Load from shared memory and store to output
    shared_value = ol.load(smem, tid)
    ol.store(shared_value, b_ptr, tid)
"""

    result = compile_python_string(python_code)

    # Should contain all the expected operations
    assert "oven.smem : !llvm.ptr<3>" in result
    assert "nvvm.barrier0" in result
    assert "oven.load" in result
    assert "oven.store" in result


def test_multiple_smem_allocations():
    """Test multiple shared memory allocations."""
    python_code = """
import oven.language as ol

def multi_smem_kernel():
    smem1 = ol.smem()
    smem2 = ol.smem()
    tid = ol.get_tid_x()
    return smem1, smem2
"""

    result = compile_python_string(python_code)

    # Should have two shared memory allocations
    smem_count = result.count("oven.smem : !llvm.ptr<3>")
    assert smem_count == 2


def test_smem_store_and_load():
    """Test storing to and loading from shared memory."""
    python_code = """
import oven.language as ol

def smem_operations(input_ptr: ol.ptr):
    smem = ol.smem()
    tid = ol.get_tid_x()
    
    # Load from input
    value = ol.load(input_ptr, tid)
    
    # Store to shared memory
    ol.store(value, smem, tid)
    
    # Barrier
    ol.barrier()
    
    # Load from shared memory
    result = ol.load(smem, tid)
    return result
"""

    result = compile_python_string(python_code)

    # Check for shared memory operations
    assert "oven.smem : !llvm.ptr<3>" in result
    assert "oven.store" in result and "!llvm.ptr<3>" in result
    assert "oven.load" in result and "!llvm.ptr<3>" in result
    assert "nvvm.barrier0" in result


if __name__ == "__main__":
    pytest.main([__file__])
