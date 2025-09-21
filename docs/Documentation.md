# Oven Language Documentation

## Quick Start

```python
import oven.language as ol

def vector_add(a: ol.ptr, b: ol.ptr, out: ol.ptr):
    idx = ol.get_tid_x()
    result = ol.load(a, idx) + ol.load(b, idx)
    ol.store(result, out, idx)
```

Compile to MLIR:
```bash
oven kernel.py
```

## Type Hints

```python
import oven.language as ol

# Basic types
int → i32           # 32-bit integer  
float → f32         # 32-bit float
ol.ptr → !llvm.ptr  # LLVM pointer

def kernel(a: ol.ptr, b: ol.ptr, size: int) -> None:
    idx: int = ol.get_tid_x()
    if idx < size:
        value: float = ol.load(a, idx)
        result: float = ol.exp(value)
        ol.store(result, b, idx)
```

## API Reference

### GPU Functions
```python
ol.get_tid_x()    # Thread ID X
ol.get_tid_y()    # Thread ID Y
ol.get_bid_x()    # Block ID X  
ol.get_bid_y()    # Block ID Y
ol.get_bdim_x()   # Block dimension X
ol.barrier()      # Thread synchronization
```

### Memory Operations
```python
ol.load(ptr, idx)         # Load value
ol.store(val, ptr, idx)   # Store value
ol.vload(ptr, idx, size)  # Load vector (size: 2, 4)
ol.vstore(vec, ptr, idx, size)  # Store vector
ol.smem()                 # Allocate shared memory
```

### Math Functions
```python
ol.exp(x)       # Exponential
ol.log(x)       # Natural logarithm
ol.sqrt(x)      # Square root
ol.sin(x)       # Sine
ol.cos(x)       # Cosine
ol.tan(x)       # Tangent
ol.sigmoid(x)   # Sigmoid function
ol.abs(x)       # Absolute value
```

## Examples

### Basic Addition
```python
def add_kernel(a: ol.ptr, b: ol.ptr, out: ol.ptr):
    idx = ol.get_tid_x()
    result = ol.load(a, idx) + ol.load(b, idx)
    ol.store(result, out, idx)
```

### Vector Operations
```python
def vector_add(a: ol.ptr, b: ol.ptr, out: ol.ptr):
    idx = ol.get_tid_x() * 4
    a_vec = ol.vload(a, idx, 4)
    b_vec = ol.vload(b, idx, 4)
    ol.vstore(a_vec + b_vec, out, idx, 4)
```

### Shared Memory
```python
def shared_sum(data: ol.ptr, output: ol.ptr):
    tid = ol.get_tid_x()
    smem = ol.smem()
    
    # Load to shared memory
    value = ol.load(data, tid)
    ol.store(value, smem, tid)
    
    # Synchronize threads
    ol.barrier()
    
    # Process and store result
    shared_val = ol.load(smem, tid)
    result = ol.exp(shared_val)
    ol.store(result, output, tid)
```