#!/usr/bin/env python3
"""
Example demonstrating Python type hints support in Oven compiler.

This example shows how to use type hints with oven.language types
to generate correctly typed MLIR functions.
"""

from oven.compiler import compile_python_string
import oven.language as ol


def demo_type_hints():
    """Demonstrate various type hint features."""

    print("=== Type Hints Demo ===\n")

    # Example 1: Basic integer types
    print("1. Basic integer types:")
    python_code = """
def add_integers(a: int, b: int) -> int:
    return a + b
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 2: Float types
    print("2. Float types:")
    python_code = """
def add_floats(a: f32, b: f32) -> f32:
    return a + b
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 3: GPU function with pointer types
    print("3. GPU function with pointer types:")
    python_code = """
import oven.language as ol

def gpu_vector_add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    tid = ol.get_tid_x()
    a_val = ol.load(a_ptr, tid)
    b_val = ol.load(b_ptr, tid)
    result = a_val + b_val
    ol.store(c_ptr, tid, result)
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 4: Mixed types
    print("4. Mixed types in one function:")
    python_code = """
import oven.language as ol

def mixed_types_example(ptr_arg: ol.ptr, size: int, scale: f32) -> f32:
    tid = ol.get_tid_x()
    if tid < size:
        value = ol.load(ptr_arg, tid)
        return value * scale
    return 0.0
"""
    result = compile_python_string(python_code)
    print(result)
    print()

    # Example 5: Explicit oven language types
    print("5. Explicit oven language types:")
    python_code = """
import oven.language as ol

def explicit_types(a: ol.i32, b: ol.f32) -> ol.f32:
    return a + b  # Will promote to float
"""
    result = compile_python_string(python_code)
    print(result)


if __name__ == "__main__":
    demo_type_hints()
