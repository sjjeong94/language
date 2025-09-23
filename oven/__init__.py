"""
Oven Language - Python to MLIR compilation library

This module provides GPU and mathematical operations for compilation to MLIR.
"""

from .language import *
from .compiler import PythonToMLIRCompiler

__version__ = "0.1.4"
__all__ = [
    # Core compiler
    "PythonToMLIRCompiler",
    # GPU operations
    "load",
    "store",
    "get_tid_x",
    "get_tid_y",
    "get_bid_x",
    "get_bid_y",
    "get_bdim_x",
    # Mathematical operations
    "exp",
    "sigmoid",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "log",
    # Arithmetic operations
    "muli",
    "addi",
    "mulf",
    "addf",
    # Type conversion
    "index_cast",
    # Constants and loops
    "constant",
    "for_loop",
    "yield_value",
    # NVIDIA intrinsics (for compatibility)
    "nvvm_read_ptx_sreg_ntid_x",
    "nvvm_read_ptx_sreg_ctaid_x",
    "nvvm_read_ptx_sreg_tid_x",
]
