# Python Type Hints Support

The Oven compiler now supports Python type hints to specify MLIR types for function parameters and return values. This allows for more precise control over the generated MLIR code.

## Supported Type Hints

### Basic Python Types
- `int` → `i32` (32-bit integer)
- `f32` → `f32` (32-bit floating point)
- `index` → `index` (MLIR index type)

### Oven Language Types
When importing `oven.language as ol`, you can use:
- `ol.ptr` → `!llvm.ptr` (LLVM pointer type)
- `ol.i32` → `i32` (32-bit integer)
- `ol.f32` → `f32` (32-bit floating point)
- `ol.index` → `index` (MLIR index type)

## Usage Examples

### Example 1: Basic Integer Function
```python
def add_numbers(a: int, b: int) -> int:
    return a + b
```
Generates:
```mlir
func.func @add_numbers(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
```

### Example 2: Float Function
```python
def multiply_floats(x: f32, y: f32) -> f32:
    return x * y
```
Generates:
```mlir
func.func @multiply_floats(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg1 : f32
  func.return %0 : f32
}
```

### Example 3: GPU Kernel with Pointers
```python
import oven.language as ol

def gpu_kernel(input_ptr: ol.ptr, output_ptr: ol.ptr):
    tid = ol.get_tid_x()
    value = ol.load(input_ptr, tid)
    ol.store(output_ptr, tid, value * 2.0)
```
Generates:
```mlir
func.func @gpu_kernel(%a: !llvm.ptr, %b: !llvm.ptr) {
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  %1 = oven.load %a, %0 : (!llvm.ptr, i32) -> f32
  %2 = arith.constant 2.0 : f32
  %3 = arith.mulf %1, %2 : f32
  oven.store %b, %0, %3 : (f32, !llvm.ptr, i32)
  return
}
```

### Example 4: Mixed Types
```python
import oven.language as ol

def mixed_function(ptr_arg: ol.ptr, count: int, scale: f32) -> f32:
    tid = ol.get_tid_x()
    if tid < count:
        value = ol.load(ptr_arg, tid)
        return value * scale
    return 0.0
```

## Type Inference and Arithmetic Operations

The compiler automatically selects the correct arithmetic operations based on the operand types:

- **Integer operations**: `arith.addi`, `arith.muli`, `arith.subi`, `arith.divi_signed`
- **Float operations**: `arith.addf`, `arith.mulf`, `arith.subf`, `arith.divf`

When operands have different types, the compiler promotes to the more general type (e.g., `i32 + f32` becomes `arith.addf`).

## Fallback Behavior

If type hints are not provided, the compiler falls back to context-based inference:
- **GPU functions**: Use `!llvm.ptr` for parameters
- **Math functions**: Use `f32` for parameters
- **Regular functions**: Use `i32` for parameters

## Benefits

1. **Precise Control**: Specify exact MLIR types for better performance
2. **Mixed Types**: Use different types in the same function
3. **GPU Optimization**: Proper pointer types for GPU kernels
4. **Math Precision**: Explicit float types for mathematical computations
5. **Compatibility**: Backward compatible with existing code (fallback inference)

## Testing

The type hint functionality is thoroughly tested in `tests/test_type_hints.py` with examples covering:
- Basic type hints
- Mixed type functions
- GPU functions with pointers
- Fallback behavior
- Partial type hints
- Math function integration
