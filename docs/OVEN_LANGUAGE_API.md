# Oven Language API Documentation

Oven Language provides specialized functions for compiling Python to MLIR. It includes functions for GPU computing and mathematical operations.

## Table of Contents

1. [Type Hints](#type-hints)
2. [GPU Memory Operations](#gpu-memory-operations)
3. [GPU Thread and Block Operations](#gpu-thread-and-block-operations)
4. [Mathematical Functions](#mathematical-functions)
5. [Arithmetic Operations](#arithmetic-operations)
6. [NVIDIA Intrinsic Functions](#nvidia-intrinsic-functions)
7. [Input/Output Operations](#inputoutput-operations)
8. [Type Conversion](#type-conversion)
9. [Constants and Loop Operations](#constants-and-loop-operations)

---

## Type Hints

Oven Language provides type hints for MLIR compilation.

### `ptr`
```python
class ptr:
    """Pointer type for MLIR compilation (!llvm.ptr)."""
```
- **Description**: Corresponds to MLIR pointer type
- **MLIR Type**: `!llvm.ptr`

### `f32`
```python
class f32:
    """32-bit floating point type (f32)."""
```
- **Description**: 32-bit floating point type
- **MLIR Type**: `f32`

### `i32`
```python
class i32:
    """32-bit integer type (i32)."""
```
- **Description**: 32-bit integer type
- **MLIR Type**: `i32`

### `index`
```python
class index:
    """Index type for MLIR (index)."""
```
- **Description**: MLIR index type
- **MLIR Type**: `index`

---

## GPU Memory Operations

### `load(ptr, offset)`
```python
def load(ptr, offset):
    """
    Load a value from GPU memory at the specified offset.
    
    Args:
        ptr: Memory pointer
        offset: Offset index
    
    Returns:
        Loaded value
    """
```
- **Description**: Load a value from GPU memory at the specified offset
- **MLIR Operation**: `oven.load`
- **Example**: `value = ol.load(memory_ptr, 0)`

### `store(value, ptr, offset)`
```python
def store(value, ptr, offset):
    """
    Store a value to GPU memory at the specified offset.
    
    Args:
        value: Value to store
        ptr: Memory pointer
        offset: Offset index
    """
```
- **Description**: Store a value to GPU memory at the specified offset
- **MLIR Operation**: `oven.store`
- **Example**: `ol.store(42.0, memory_ptr, 0)`

### `smem()`
```python
def smem():
    """
    Allocate shared memory for GPU computations.
    
    Returns:
        Shared memory pointer (!llvm.ptr<3>)
    """
```
- **Description**: Allocate GPU shared memory
- **MLIR Operation**: `oven.smem`
- **Return Type**: `!llvm.ptr<3>`
- **Example**: `shared_mem = ol.smem()`

---

## GPU Thread and Block Operations

### Thread ID Functions

#### `get_tid_x()`
```python
def get_tid_x():
    """Get the current thread ID in the X dimension."""
```
- **Description**: Returns the current thread ID in the X dimension
- **MLIR Operation**: `nvvm.read.ptx.sreg.tid.x`

#### `get_tid_y()`
```python
def get_tid_y():
    """Get the current thread ID in the Y dimension."""
```
- **Description**: Returns the current thread ID in the Y dimension
- **MLIR Operation**: `nvvm.read.ptx.sreg.tid.y`

### Block ID Functions

#### `get_bid_x()`
```python
def get_bid_x():
    """Get the current block ID in the X dimension."""
```
- **Description**: Returns the current block ID in the X dimension
- **MLIR Operation**: `nvvm.read.ptx.sreg.ctaid.x`

#### `get_bid_y()`
```python
def get_bid_y():
    """Get the current block ID in the Y dimension."""
```
- **Description**: Returns the current block ID in the Y dimension
- **MLIR Operation**: `nvvm.read.ptx.sreg.ctaid.y`

### Block Dimension Functions

#### `get_bdim_x()`
```python
def get_bdim_x():
    """Get the block dimension in the X dimension."""
```
- **Description**: Returns the block size in the X dimension
- **MLIR Operation**: `nvvm.read.ptx.sreg.ntid.x`

### Synchronization Functions

#### `barrier()`
```python
def barrier():
    """
    Synchronization barrier for GPU threads.
    Ensures all threads in a block reach this point before continuing.
    """
```
- **Description**: GPU thread synchronization barrier
- **MLIR Operation**: `nvvm.barrier0`
- **Purpose**: Wait until all threads in a block reach this point

---

## Mathematical Functions

### Exponential and Logarithmic Functions

#### `exp(x)`
```python
def exp(x):
    """
    Compute the exponential function e^x.
    
    Args:
        x: Input value
    
    Returns:
        e^x
    """
```
- **Description**: Compute the natural exponential function
- **MLIR Operation**: `math.exp`

#### `log(x)`
```python
def log(x):
    """Compute the natural logarithm function."""
```
- **Description**: Compute the natural logarithm function
- **MLIR Operation**: `math.log`

#### `sigmoid(x)`
```python
def sigmoid(x):
    """
    Compute the sigmoid function 1 / (1 + e^(-x)).
    
    Args:
        x: Input value
    
    Returns:
        sigmoid(x)
    """
```
- **Description**: Compute the sigmoid function
- **Formula**: `1 / (1 + e^(-x))`

### Trigonometric Functions

#### `sin(x)`
```python
def sin(x):
    """Compute the sine function."""
```
- **Description**: Compute the sine function
- **MLIR Operation**: `math.sin`

#### `cos(x)`
```python
def cos(x):
    """Compute the cosine function."""
```
- **Description**: Compute the cosine function
- **MLIR Operation**: `math.cos`

#### `tan(x)`
```python
def tan(x):
    """Compute the tangent function."""
```
- **Description**: Compute the tangent function
- **MLIR Operation**: `math.tan`

### Other Mathematical Functions

#### `sqrt(x)`
```python
def sqrt(x):
    """Compute the square root function."""
```
- **Description**: Compute the square root function
- **MLIR Operation**: `math.sqrt`

#### `abs(x)`
```python
def abs(x):
    """Compute the absolute value."""
```
- **Description**: Compute the absolute value function
- **MLIR Operation**: `math.absf`

#### `ceil(x)`
```python
def ceil(x):
    """Compute the ceiling function (smallest integer >= x)."""
```
- **Description**: Compute the ceiling function (smallest integer >= x)
- **MLIR Operation**: `math.ceil`

#### `floor(x)`
```python
def floor(x):
    """Compute the floor function (largest integer <= x)."""
```
- **Description**: Compute the floor function (largest integer <= x)
- **MLIR Operation**: `math.floor`

#### `rsqrt(x)`
```python
def rsqrt(x):
    """Compute the reciprocal square root (1/sqrt(x))."""
```
- **Description**: Compute the reciprocal square root function (1/sqrt(x))
- **MLIR Operation**: `math.rsqrt`

---

## Arithmetic Operations

### Integer Operations

#### `muli(a, b)`
```python
def muli(a, b):
    """Multiply two integer values."""
```
- **Description**: Multiply two integer values
- **MLIR Operation**: `arith.muli`

#### `addi(a, b)`
```python
def addi(a, b):
    """Add two integer values."""
```
- **Description**: Add two integer values
- **MLIR Operation**: `arith.addi`

### Floating-Point Operations

#### `mulf(a, b)`
```python
def mulf(a, b):
    """Multiply two floating-point values."""
```
- **Description**: Multiply two floating-point values
- **MLIR Operation**: `arith.mulf`

#### `addf(a, b)`
```python
def addf(a, b):
    """Add two floating-point values."""
```
- **Description**: Add two floating-point values
- **MLIR Operation**: `arith.addf`

---

## NVIDIA Intrinsic Functions

### Direct Intrinsic Functions

#### `nvvm_read_ptx_sreg_ntid_x()`
```python
def nvvm_read_ptx_sreg_ntid_x():
    """NVIDIA intrinsic: Read block dimension X."""
```
- **Description**: NVIDIA intrinsic function - Read X dimension block size

#### `nvvm_read_ptx_sreg_ctaid_x()`
```python
def nvvm_read_ptx_sreg_ctaid_x():
    """NVIDIA intrinsic: Read block ID X."""
```
- **Description**: NVIDIA intrinsic function - Read X dimension block ID

#### `nvvm_read_ptx_sreg_tid_x()`
```python
def nvvm_read_ptx_sreg_tid_x():
    """NVIDIA intrinsic: Read thread ID X."""
```
- **Description**: NVIDIA intrinsic function - Read X dimension thread ID

### Alias Functions

The following aliases are also available:
- `__nvvm_read_ptx_sreg_ntid_x` → `nvvm_read_ptx_sreg_ntid_x`
- `__nvvm_read_ptx_sreg_ctaid_x` → `nvvm_read_ptx_sreg_ctaid_x`
- `__nvvm_read_ptx_sreg_tid_x` → `nvvm_read_ptx_sreg_tid_x`
- `__load_from_ptr` → `load`
- `__store_to_ptr` → `store`

---

## Input/Output Operations

### Input Functions

#### `load_input_x(index)`
```python
def load_input_x(index):
    """Load value from input buffer x at specified index."""
```
- **Description**: Load value from input buffer x at specified index

#### `load_input_y(index)`
```python
def load_input_y(index):
    """Load value from input buffer y at specified index."""
```
- **Description**: Load value from input buffer y at specified index

### Output Functions

#### `store_output_x(value, index)`
```python
def store_output_x(value, index):
    """Store value to output buffer x at specified index."""
```
- **Description**: Store value to output buffer x at specified index

#### `store_output_y(value, index)`
```python
def store_output_y(value, index):
    """Store value to output buffer y at specified index."""
```
- **Description**: Store value to output buffer y at specified index

---

## Type Conversion

#### `index_cast(value, from_type, to_type)`
```python
def index_cast(value, from_type, to_type):
    """Cast between index and integer types."""
```
- **Description**: Cast between index and integer types
- **MLIR Operation**: `arith.index_cast`

---

## Constants and Loop Operations

### Constants

#### `constant(value, data_type)`
```python
def constant(value, data_type):
    """Create a constant value."""
```
- **Description**: Create a constant value
- **MLIR Operation**: `arith.constant`

### Loop Operations

#### `for_loop(start, end, step, body_func, init_args=None)`
```python
def for_loop(start, end, step, body_func, init_args=None):
    """Create a for loop with iter_args."""
```
- **Description**: Create a for loop with iter_args
- **MLIR Operation**: `scf.for`

#### `yield_value(*values)`
```python
def yield_value(*values):
    """Yield values in a loop."""
```
- **Description**: Yield values in a loop
- **MLIR Operation**: `scf.yield`

---

## Usage Examples

```python
import oven.language as ol

def gpu_kernel(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, n: int):
    # Get GPU thread and block information
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    bdim = ol.get_bdim_x()
    
    # Calculate global index
    idx = bid * bdim + tid
    
    # Allocate shared memory
    smem = ol.smem()
    
    # Load values from memory
    a_val = ol.load(a_ptr, idx)
    b_val = ol.load(b_ptr, idx)
    
    # Perform mathematical operations
    result = ol.exp(a_val) + ol.sin(b_val)
    
    # Store result to memory
    ol.store(result, c_ptr, idx)
    
    # Thread synchronization
    ol.barrier()
```

This document systematically organizes all functions in Oven Language. Each function is converted to corresponding MLIR operations during the MLIR compilation process.