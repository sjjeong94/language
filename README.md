# Oven Compiler

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/oven-language.svg)](https://badge.fury.io/py/oven-language)

A Python-to-MLIR compiler that enables GPU kernel development and mathematical computing with high-level Python syntax.

## Features

- **GPU Kernel Compilation**: Write GPU kernels in Python and compile to MLIR
- **Vector Operations**: Built-in support for vectorized memory operations (`vload`, `vstore`)
- **Mathematical Functions**: Hardware-accelerated math functions (exp, sigmoid, etc.)
- **Type Safety**: Strong type inference and MLIR type system integration
- **Clean API**: Triton-inspired syntax with `import oven.language as ol`
- **Command Line Tools**: Simple compilation workflow with `oven compile`

## Installation

```bash
pip install oven-language
```

## Quick Start

### GPU Kernel Example

```python
import oven.language as ol

def vector_add_kernel(a_ptr: ol.ptr, b_ptr: ol.ptr, out_ptr: ol.ptr):
    """Element-wise vector addition on GPU."""
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    
    idx = bid * ol.get_bdim_x() + tid
    offset = idx * 4
    
    # Load vectors (4 elements each)
    vec_a = ol.vload(a_ptr, offset, 4)
    vec_b = ol.vload(b_ptr, offset, 4)
    
    # Vector addition
    result = vec_a + vec_b
    
    # Store result
    ol.vstore(result, out_ptr, offset, 4)

def math_pipeline_kernel(input_ptr: ol.ptr, output_ptr: ol.ptr):
    """Mathematical transformation pipeline."""
    tid = ol.get_tid_x()
    offset = tid * 4
    
    # Load data
    data = ol.vload(input_ptr, offset, 4)
    
    # Apply transformations
    data = ol.sigmoid(data)  # Sigmoid activation
    data = ol.exp(data)      # Exponential
    
    # Store result
    ol.vstore(data, output_ptr, offset, 4)
```

### Compilation

```bash
# Compile to MLIR
oven compile kernel.py

# Specify output file
oven compile kernel.py -o kernel.mlir

# Enable debug output
oven compile kernel.py --debug
```

### Generated MLIR

```mlir
func.func @vector_add_kernel(%a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr, %out_ptr: !llvm.ptr) {
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  %1 = nvvm.read.ptx.sreg.ctaid.x : i32
  %2 = nvvm.read.ptx.sreg.ntid.x : i32
  %3 = arith.muli %1, %2 : i32
  %4 = arith.addi %3, %0 : i32
  %5 = arith.constant 4 : i32
  %6 = arith.muli %4, %5 : i32
  %7 = oven.vload %a_ptr, %6, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %8 = oven.vload %b_ptr, %6, 4 : (!llvm.ptr, i32) -> vector<4xf32>
  %9 = arith.addf %7, %8 : vector<4xf32>
  oven.vstore %9, %out_ptr, %6, 4 : (vector<4xf32>, !llvm.ptr, i32)
  return
}
```

## API Reference

### GPU Operations

| Function                            | Description                       | MLIR Output                  |
| ----------------------------------- | --------------------------------- | ---------------------------- |
| `ol.get_tid_x()`                    | Get thread ID (X dimension)       | `nvvm.read.ptx.sreg.tid.x`   |
| `ol.get_bid_x()`                    | Get block ID (X dimension)        | `nvvm.read.ptx.sreg.ctaid.x` |
| `ol.get_bdim_x()`                   | Get block dimension (X dimension) | `nvvm.read.ptx.sreg.ntid.x`  |
| `ol.load(ptr, offset)`              | Load scalar value                 | `oven.load`                  |
| `ol.store(value, ptr, offset)`      | Store scalar value                | `oven.store`                 |
| `ol.vload(ptr, offset, size)`       | Load vector                       | `oven.vload`                 |
| `ol.vstore(vec, ptr, offset, size)` | Store vector                      | `oven.vstore`                |
| `ol.smem()`                         | Allocate shared memory            | `oven.smem`                  |
| `ol.barrier()`                      | Thread synchronization            | `nvvm.barrier0`              |

### Mathematical Functions

| Function        | Description          | MLIR Output    |
| --------------- | -------------------- | -------------- |
| `ol.exp(x)`     | Exponential function | `math.exp`     |
| `ol.sigmoid(x)` | Sigmoid activation   | `oven.sigmoid` |
| `ol.sin(x)`     | Sine function        | `math.sin`     |
| `ol.cos(x)`     | Cosine function      | `math.cos`     |
| `ol.sqrt(x)`    | Square root          | `math.sqrt`    |
| `ol.log(x)`     | Natural logarithm    | `math.log`     |

### Vector Operations

- **Supported sizes**: 2, 4 elements
- **Element types**: f32 (single precision float)
- **Operations**: Addition, subtraction, multiplication, division
- **Functions**: All mathematical functions support vector inputs

## Type System

```python
import oven.language as ol

def typed_kernel(input_ptr: ol.ptr, output_ptr: ol.ptr, n: int):
    """Example with explicit type hints."""
    tid: int = ol.get_tid_x()
    
    if tid < n:
        value: float = ol.load(input_ptr, tid)
        result: float = ol.exp(value)
        ol.store(result, output_ptr, tid)
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=oven --cov-report=html
```

### Project Structure

```
oven-language/
├── oven/                    # Main package
│   ├── language.py          # GPU and math function definitions
│   ├── compiler.py          # Main compiler interface
│   ├── ast_visitor.py       # AST processing
│   ├── mlir_generator.py    # MLIR code generation
│   └── cli.py              # Command line interface
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## Requirements

- Python 3.8+
- No runtime dependencies for basic usage
- Development: pytest, pytest-cov

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Links

- **Documentation**: [Complete API Documentation](docs/OVEN_LANGUAGE_API.md)
- **GitHub**: [https://github.com/sjjeong94/oven-language](https://github.com/sjjeong94/oven-language)
- **PyPI**: [https://pypi.org/project/oven-language/](https://pypi.org/project/oven-language/)
- **Issues**: [GitHub Issues](https://github.com/sjjeong94/language/issues)
