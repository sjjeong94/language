#!/usr/bin/env python3
"""
Example demonstrating shared memory (smem) and barrier functionality.

This example shows how to use ol.smem() and ol.barrier() for GPU kernel
optimization with shared memory and thread synchronization.
"""

from oven.compiler import compile_python_string
import oven.language as ol


def demo_smem_barrier():
    """Demonstrate shared memory and barrier usage."""

    print("=== Shared Memory and Barrier Demo ===\n")

    # Example 1: Basic shared memory allocation
    print("1. Basic shared memory allocation:")
    python_code = """
import oven.language as ol

def allocate_smem():
    smem_ptr = ol.smem()
    return smem_ptr
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 2: Simple barrier usage
    print("2. Barrier synchronization:")
    python_code = """
import oven.language as ol

def sync_threads():
    ol.barrier()
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 3: Complete shared memory kernel pattern
    print("3. Complete shared memory kernel:")
    python_code = """
import oven.language as ol

def shared_memory_kernel(input_ptr: ol.ptr, output_ptr: ol.ptr):
    # Allocate shared memory
    smem = ol.smem()
    
    # Get thread and block information
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    bdim = ol.get_bdim_x()
    
    # Calculate global index
    global_idx = bid * bdim + tid
    
    # Load from global memory to shared memory
    value = ol.load(input_ptr, global_idx)
    ol.store(value, smem, tid)
    
    # Synchronize all threads in the block
    ol.barrier()
    
    # Load from shared memory (could be from different thread's data)
    neighbor_tid = (tid + 1) % bdim
    neighbor_value = ol.load(smem, neighbor_tid)
    
    # Process and write back to global memory
    result = value + neighbor_value
    ol.store(result, output_ptr, global_idx)
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 4: Matrix transpose with shared memory
    print("4. Matrix transpose pattern:")
    python_code = """
import oven.language as ol

def transpose_kernel(input_ptr: ol.ptr, output_ptr: ol.ptr, width: int):
    # Shared memory for tile
    tile = ol.smem()
    
    # Thread coordinates
    tx = ol.get_tid_x()
    ty = ol.get_tid_y() 
    bx = ol.get_bid_x()
    by = ol.get_bid_y()
    
    # Block dimensions (assume 16x16)
    block_size = 16
    
    # Input coordinates
    in_x = bx * block_size + tx
    in_y = by * block_size + ty
    
    # Load into shared memory
    if in_x < width and in_y < width:
        input_idx = in_y * width + in_x
        value = ol.load(input_ptr, input_idx)
        shared_idx = ty * block_size + tx
        ol.store(value, tile, shared_idx)
    
    # Synchronize
    ol.barrier()
    
    # Output coordinates (transposed)
    out_x = by * block_size + tx  
    out_y = bx * block_size + ty
    
    # Store transposed data
    if out_x < width and out_y < width:
        shared_idx = tx * block_size + ty  # Transposed access
        transposed_value = ol.load(tile, shared_idx)
        output_idx = out_y * width + out_x
        ol.store(transposed_value, output_ptr, output_idx)
"""
    result = compile_python_string(python_code)
    print(result)


if __name__ == "__main__":
    demo_smem_barrier()
