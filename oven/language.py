"""
Oven Language Core Operations

This module provides the core operations for GPU computing and mathematical functions
that can be compiled to MLIR.
"""


# GPU Memory Operations
def load(ptr, offset):
    """
    Load a value from GPU memory at the specified offset.

    Args:
        ptr: Memory pointer
        offset: Offset index

    Returns:
        Loaded value
    """
    # This function is a placeholder - actual implementation happens during MLIR compilation
    raise NotImplementedError("This function is compiled to MLIR operations")


def store(value, ptr, offset):
    """
    Store a value to GPU memory at the specified offset.

    Args:
        value: Value to store
        ptr: Memory pointer
        offset: Offset index
    """
    # This function is a placeholder - actual implementation happens during MLIR compilation
    raise NotImplementedError("This function is compiled to MLIR operations")


# GPU Thread and Block Operations
def get_tid_x():
    """Get the current thread ID in the X dimension."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def get_bid_x():
    """Get the current block ID in the X dimension."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def get_bdim_x():
    """Get the block dimension in the X dimension."""
    raise NotImplementedError("This function is compiled to MLIR operations")


# Mathematical Operations
def exp(x):
    """
    Compute the exponential function e^x.

    Args:
        x: Input value

    Returns:
        e^x
    """
    raise NotImplementedError("This function is compiled to MLIR operations")


def sigmoid(x):
    """
    Compute the sigmoid function 1 / (1 + e^(-x)).

    Args:
        x: Input value

    Returns:
        sigmoid(x)
    """
    raise NotImplementedError("This function is compiled to MLIR operations")


def sin(x):
    """Compute the sine function."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def cos(x):
    """Compute the cosine function."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def tan(x):
    """Compute the tangent function."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def sqrt(x):
    """Compute the square root function."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def log(x):
    """Compute the natural logarithm function."""
    raise NotImplementedError("This function is compiled to MLIR operations")


# NVIDIA Intrinsics (for compatibility with existing code)
def nvvm_read_ptx_sreg_ntid_x():
    """NVIDIA intrinsic: Read block dimension X."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def nvvm_read_ptx_sreg_ctaid_x():
    """NVIDIA intrinsic: Read block ID X."""
    raise NotImplementedError("This function is compiled to MLIR operations")


def nvvm_read_ptx_sreg_tid_x():
    """NVIDIA intrinsic: Read thread ID X."""
    raise NotImplementedError("This function is compiled to MLIR operations")


# Aliases for the NVIDIA intrinsics (underscore versions)
__nvvm_read_ptx_sreg_ntid_x = nvvm_read_ptx_sreg_ntid_x
__nvvm_read_ptx_sreg_ctaid_x = nvvm_read_ptx_sreg_ctaid_x
__nvvm_read_ptx_sreg_tid_x = nvvm_read_ptx_sreg_tid_x
__load_from_ptr = load
__store_to_ptr = store
