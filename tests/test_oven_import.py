"""
Tests for oven.language import structure
"""

import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from oven.compiler import PythonToMLIRCompiler


@pytest.mark.integration
class TestOvenLanguageImport:
    """Test oven.language import structure compilation features."""

    def test_import_as_alias(self, compiler):
        """Test import oven.language as ol structure."""
        source = """
import oven.language as ol

def test_gpu_ops(a_ptr, b_ptr):
    tid = ol.get_tid_x()
    value = ol.load(a_ptr, tid)
    result = ol.exp(value)
    ol.store(result, b_ptr, tid)
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature for GPU kernel
        assert (
            "func.func @test_gpu_ops(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr)" in mlir_code
        )

        # Check GPU operations are present
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load %a_ptr," in mlir_code
        assert "math.exp" in mlir_code
        assert "oven.store" in mlir_code

        # Check it ends with plain return (GPU kernel)
        assert mlir_code.strip().endswith("return\n}")

    def test_direct_import(self, compiler):
        """Test from oven.language import structure."""
        source = """
from oven.language import load, store, exp, sigmoid, get_tid_x

def test_direct(input_ptr, output_ptr):
    tid = get_tid_x()
    value = load(input_ptr, tid)
    result = exp(value)
    final = sigmoid(result)
    store(final, output_ptr, tid)
"""
        mlir_code = compiler.compile_source(source)

        # Check function signature for GPU kernel
        assert (
            "func.func @test_direct(%input_ptr: !llvm.ptr, %output_ptr: !llvm.ptr)"
            in mlir_code
        )

        # Check operations are present
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "math.exp" in mlir_code
        assert "oven.sigmoid" in mlir_code
        assert "oven.store" in mlir_code

    def test_mixed_import_styles(self, compiler):
        """Test mixing different import styles."""
        source = """
import oven.language as ol
from oven.language import exp

def test_alias_style(a_ptr, b_ptr):
    value = ol.load(a_ptr, 0)
    result = ol.sigmoid(value)
    ol.store(result, b_ptr, 0)

def test_direct_style(x_ptr, y_ptr):
    x = ol.load(x_ptr, 0)
    y = exp(x)  # Direct import
    ol.store(y, y_ptr, 0)
"""
        mlir_code = compiler.compile_source(source)

        # Check both functions have GPU signatures
        assert (
            "func.func @test_alias_style(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr)"
            in mlir_code
        )
        assert (
            "func.func @test_direct_style(%x_ptr: !llvm.ptr, %y_ptr: !llvm.ptr)"
            in mlir_code
        )

        # Check operations are present
        assert "oven.load" in mlir_code
        assert "oven.sigmoid" in mlir_code
        assert "math.exp" in mlir_code
        assert "oven.store" in mlir_code

    def test_math_functions_with_import(self, compiler):
        """Test mathematical functions with oven.language import."""
        source = """
import oven.language as ol

def test_math_only(x):
    return ol.exp(x)

def test_mixed_math_gpu(input_ptr, output_ptr):
    tid = ol.get_tid_x()
    x = ol.load(input_ptr, tid)
    y = ol.sigmoid(ol.exp(x))
    ol.store(y, output_ptr, tid)
"""
        mlir_code = compiler.compile_source(source)

        # Check math-only function has f32 signature
        assert "func.func @test_math_only(%arg0: f32) -> f32" in mlir_code

        # Check GPU function has GPU signature
        assert (
            "func.func @test_mixed_math_gpu(%input_ptr: !llvm.ptr, %output_ptr: !llvm.ptr)"
            in mlir_code
        )

        # Check operations
        assert "math.exp" in mlir_code
        assert "oven.sigmoid" in mlir_code
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "oven.store" in mlir_code

    def test_import_backward_compatibility(self, compiler):
        """Test that old-style function calls still work alongside imports."""
        source = """
import oven.language as ol

def old_style_gpu():
    # Old style calls should still work
    tid = get_tid_x()
    return tid

def new_style_gpu(ptr):
    # New style calls
    tid = ol.get_tid_x()
    value = ol.load(ptr, tid)
    return value

def old_style_math(x):
    # Old style math
    return exp(x)

def new_style_math(x):
    # New style math
    return ol.exp(x)
"""
        mlir_code = compiler.compile_source(source)

        # Check GPU functions have GPU signatures
        assert "func.func @old_style_gpu()" in mlir_code
        assert "func.func @new_style_gpu(%ptr: !llvm.ptr)" in mlir_code

        # Check math functions have appropriate signatures
        assert "func.func @old_style_math(%arg0: f32) -> f32" in mlir_code
        assert "func.func @new_style_math(%arg0: f32) -> f32" in mlir_code

        # Check operations
        assert "nvvm.read.ptx.sreg.tid.x" in mlir_code
        assert "oven.load" in mlir_code
        assert "math.exp" in mlir_code

    @pytest.mark.parametrize(
        "import_style,func_call,expected_op",
        [
            ("import oven.language as ol", "ol.exp", "math.exp"),
            ("import oven.language as ol", "ol.sigmoid", "oven.sigmoid"),
            ("import oven.language as ol", "ol.load", "oven.load"),
            ("import oven.language as ol", "ol.store", "oven.store"),
            ("from oven.language import exp", "exp", "math.exp"),
            ("from oven.language import sigmoid", "sigmoid", "oven.sigmoid"),
            ("from oven.language import load", "load", "oven.load"),
            ("from oven.language import store", "store", "oven.store"),
        ],
    )
    def test_individual_import_functions(
        self, compiler, import_style, func_call, expected_op
    ):
        """Test individual import styles for each function."""
        if "store" in func_call:
            # Store needs special handling (3 args)
            source = f"""
{import_style}

def test_func(ptr1, ptr2):
    {func_call}(1.0, ptr1, 0)
"""
        elif "load" in func_call:
            # Load needs special handling (2 args)
            source = f"""
{import_style}

def test_func(ptr):
    return {func_call}(ptr, 0)
"""
        else:
            # Math functions and GPU functions
            source = f"""
{import_style}

def test_func(x):
    return {func_call}(x)
"""

        mlir_code = compiler.compile_source(source)
        assert expected_op in mlir_code

    def test_oven_import_files_compilation(self):
        """Test compilation of files using oven.language imports."""
        test_file = Path(__file__).parent.parent / "test_oven_import.py"

        if test_file.exists():
            compiler = PythonToMLIRCompiler(debug=False)
            mlir_code = compiler.compile_file(str(test_file))

            # Verify the expected MLIR patterns
            expected_patterns = [
                "func.func @test_exp(%a: !llvm.ptr, %b: !llvm.ptr)",
                "func.func @test_sigmoid(%a: !llvm.ptr, %b: !llvm.ptr)",
                "func.func @gpu_kernel(%a: !llvm.ptr, %b: !llvm.ptr)",
                "oven.load %a,",
                "math.exp",
                "oven.sigmoid",
                "oven.store",
                "nvvm.read.ptx.sreg.tid.x",
                "return",
            ]

            for pattern in expected_patterns:
                assert (
                    pattern in mlir_code
                ), f"Pattern '{pattern}' not found in generated MLIR"
