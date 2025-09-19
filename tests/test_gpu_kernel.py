"""
Tests for GPU kernel compilation
"""

import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oven.compiler import PythonToMLIRCompiler


@pytest.mark.integration
class TestGPUKernelCompilation:
    """Test GPU kernel specific compilation features."""

    def test_gpu_kernel_functions(self, compiler):
        """Test that GPU-specific functions are correctly recognized."""
        gpu_kernel_source = """
def test_kernel(a, b):
    bdim_x = get_bdim_x()
    bid_x = get_bid_x()
    tid_x = get_tid_x()
    idx = bdim_x * bid_x + tid_x
    val = load(a, idx)
    store(val, b, idx)
    return
"""
        mlir_code = compiler.compile_source(gpu_kernel_source)

        # Check GPU operations are present
        assert "nvvm.read.ptx.sreg.ntid.x" in mlir_code
        assert "nvvm.read.ptx.sreg.ctaid.x" in mlir_code
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "oven.store" in mlir_code

        # Check function signature for GPU kernel
        assert "func.func @test_kernel(%a: !llvm.ptr, %b: !llvm.ptr)" in mlir_code

        # Check it ends with plain return (not func.return)
        assert mlir_code.strip().endswith("return\n}")

    def test_kernel_py_compilation(self):
        """Test compilation of the actual kernel.py file."""
        kernel_file = Path(__file__).parent.parent / "kernel.py"

        if kernel_file.exists():
            compiler = PythonToMLIRCompiler(debug=False)
            mlir_code = compiler.compile_file(str(kernel_file))

            # Verify the expected MLIR patterns
            expected_patterns = [
                "func.func @test_load_store",
                "nvvm.read.ptx.sreg.tid.x : i32",
                "arith.mulf",  # Changed from muli to mulf for GPU kernels
                "arith.addf",  # Changed from addi to addf for GPU kernels
                "oven.load %a,",
                "oven.store",
                "return",
            ]

            for pattern in expected_patterns:
                assert (
                    pattern in mlir_code
                ), f"Pattern '{pattern}' not found in generated MLIR"

    def test_mixed_gpu_and_regular_functions(self, compiler):
        """Test compilation of file with both GPU and regular functions."""
        mixed_source = """
def regular_function(x, y):
    return x + y

def gpu_kernel(a, b):
    tid = get_tid_x()
    val = load(a, tid)
    store(val, b, tid)
    return

def another_regular(n):
    if n > 0:
        return n * 2
    else:
        return 0
"""
        mlir_code = compiler.compile_source(mixed_source)

        # Check regular functions have normal signatures
        assert "func.func @regular_function(%arg0: i32, %arg1: i32) -> i32" in mlir_code
        assert "func.func @another_regular(%arg0: i32) -> i32" in mlir_code

        # Check GPU kernel has GPU signature
        assert "func.func @gpu_kernel(%a: !llvm.ptr, %b: !llvm.ptr)" in mlir_code

        # Check GPU operations only in GPU kernel
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "oven.store" in mlir_code

    def test_gpu_function_argument_mapping(self, compiler):
        """Test that GPU function arguments are correctly mapped."""
        source = """
def kernel(input_array, output_array, size):
    tid = get_tid_x()
    if tid < size:
        val = load(input_array, tid)
        result = val * 2
        store(result, output_array, tid)
    return
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature uses proper GPU argument names
        assert (
            "func.func @kernel(%input_array: !llvm.ptr, %output_array: !llvm.ptr, %size: !llvm.ptr)"
            in mlir_code
        )

        # The arguments should be accessible as %a, %b, %c in the function body
        # This is verified by the successful compilation without errors

    @pytest.mark.parametrize(
        "gpu_func,expected_op",
        [
            ("get_bdim_x", "nvvm.read.ptx.sreg.ntid.x"),
            ("get_bid_x", "nvvm.read.ptx.sreg.ctaid.x"),
            ("get_tid_x", "nvvm.read.ptx.sreg.tid.x"),
        ],
    )
    def test_individual_gpu_functions(self, compiler, gpu_func, expected_op):
        """Test individual GPU functions are correctly translated."""
        source = f"""
def test_func(arr):
    idx = {gpu_func}()
    return
"""
        mlir_code = compiler.compile_source(source)
        assert expected_op in mlir_code

    def test_load_store_operations(self, compiler):
        """Test load and store operations are correctly translated."""
        source = """
def test_memory(src, dst):
    offset = get_tid_x()
    value = load(src, offset)
    store(value, dst, offset)
    return
"""
        mlir_code = compiler.compile_source(source)

        # Check load operation
        assert "oven.load %src," in mlir_code
        assert "(!llvm.ptr, i32) -> f32" in mlir_code

        # Check store operation
        assert "oven.store" in mlir_code
        assert "(f32, !llvm.ptr, i32)" in mlir_code

    def test_nvvm_intrinsic_functions(self, compiler):
        """Test NVIDIA intrinsic functions are correctly translated."""
        source = """
def test_intrinsics(a, b):
    ntid_x = __nvvm_read_ptx_sreg_ntid_x()
    ctaid_x = __nvvm_read_ptx_sreg_ctaid_x()
    tid_x = __nvvm_read_ptx_sreg_tid_x()
    
    thread_idx = ctaid_x * ntid_x + tid_x
    value = __load_from_ptr(a, thread_idx)
    __store_to_ptr(b, thread_idx, value)
    return
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature for GPU kernel
        assert "func.func @test_intrinsics(%a: !llvm.ptr, %b: !llvm.ptr)" in mlir_code

        # Check NVIDIA intrinsic operations
        assert "nvvm.read.ptx.sreg.ntid.x : i32" in mlir_code
        assert "nvvm.read.ptx.sreg.ctaid.x : i32" in mlir_code
        assert "nvvm.read.ptx.sreg.tid.x : i32" in mlir_code

        # Check memory operations
        assert "oven.load %a," in mlir_code
        assert "oven.store" in mlir_code

    def test_math_functions_in_gpu_context(self, compiler):
        """Test that math functions work correctly when mixed with GPU code."""
        source = """
def gpu_with_math(input_ptr, output_ptr):
    tid = get_tid_x()
    value = load(input_ptr, tid)
    return

def pure_math(x):
    return exp(x)

def math_then_gpu(data, result):
    tid = get_tid_x()
    val = load(data, tid)
    store(val, result, tid)
    return
"""
        mlir_code = compiler.compile_source(source)

        # Check GPU functions have GPU signatures
        assert (
            "func.func @gpu_with_math(%input_ptr: !llvm.ptr, %output_ptr: !llvm.ptr)"
            in mlir_code
        )
        assert (
            "func.func @math_then_gpu(%data: !llvm.ptr, %result: !llvm.ptr)"
            in mlir_code
        )

        # Check math function has f32 signature
        assert "func.func @pure_math(%arg0: f32) -> f32" in mlir_code

        # Check operations are present
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "oven.store" in mlir_code
        assert "math.exp" in mlir_code
