# Oven Language Quick Reference Guide

## Import
```python
import oven.language as ol
```

## Type Hints
```python
ol.ptr     # Pointer type (!llvm.ptr)
ol.f32     # 32-bit floating point (f32)
ol.i32     # 32-bit integer (i32)
ol.index   # Index type (index)
```

## GPU Memory
```python
ol.load(ptr, offset)          # Load from memory
ol.store(value, ptr, offset)  # Store to memory
ol.smem()                     # Allocate shared memory
```

## GPU Thread Information
```python
ol.get_tid_x()    # Thread ID (X)
ol.get_tid_y()    # Thread ID (Y)
ol.get_bid_x()    # Block ID (X)
ol.get_bid_y()    # Block ID (Y)
ol.get_bdim_x()   # Block size (X)
ol.barrier()      # Thread synchronization
```

## Mathematical Functions
```python
# Basic functions
ol.exp(x)      # e^x
ol.log(x)      # ln(x)
ol.sqrt(x)     # √x
ol.sigmoid(x)  # 1/(1+e^(-x))

# Trigonometric functions
ol.sin(x)      # sin(x)
ol.cos(x)      # cos(x)
ol.tan(x)      # tan(x)

# Additional math functions
ol.abs(x)      # |x| (absolute value)
ol.ceil(x)     # ⌈x⌉ (ceiling)
ol.floor(x)    # ⌊x⌋ (floor)
ol.rsqrt(x)    # 1/√x (reciprocal square root)
```

## Arithmetic Operations
```python
# Integer operations
ol.muli(a, b)  # Integer multiplication
ol.addi(a, b)  # Integer addition

# Floating-point operations
ol.mulf(a, b)  # Float multiplication
ol.addf(a, b)  # Float addition
```

## Input/Output
```python
ol.load_input_x(index)       # Load from input x buffer
ol.load_input_y(index)       # Load from input y buffer
ol.store_output_x(val, idx)  # Store to output x buffer
ol.store_output_y(val, idx)  # Store to output y buffer
```

## Utilities
```python
ol.index_cast(val, from, to)  # Type conversion
ol.constant(value, type)      # Create constant
ol.for_loop(start, end, step, body, init)  # Loop
ol.yield_value(*values)       # Yield
```

## Example: Simple GPU Kernel
```python
def vector_add(a: ol.ptr, b: ol.ptr, c: ol.ptr, n: int):
    idx = ol.get_tid_x() + ol.get_bid_x() * ol.get_bdim_x()
    a_val = ol.load(a, idx)
    b_val = ol.load(b, idx)
    result = a_val + b_val  # Python operators also available
    ol.store(result, c, idx)
```

## Example: Using Shared Memory
```python
def shared_memory_example(data: ol.ptr, output: ol.ptr):
    tid = ol.get_tid_x()
    smem = ol.smem()
    
    # Load data to shared memory
    value = ol.load(data, tid)
    ol.store(value, smem, tid)
    
    # Synchronization
    ol.barrier()
    
    # Process from shared memory
    shared_val = ol.load(smem, tid)
    result = ol.exp(shared_val)
    ol.store(result, output, tid)
```