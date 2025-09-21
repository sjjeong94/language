# Oven Compiler Documentation

This directory contains the official documentation for the Oven Compiler.

## 📚 Documentation Index

### API Reference
- **[Oven Language API Documentation](OVEN_LANGUAGE_API.md)** - Detailed description of all Oven Language functions
- **[Oven Language Quick Reference](OVEN_LANGUAGE_QUICK_REFERENCE.md)** - Quick reference guide for key functions

### Tutorials and Guides
- **[Type Hints Guide](type_hints.md)** - How to use MLIR type hints

## 🚀 Quick Start

### 1. Basic Usage
```python
import oven.language as ol

def simple_kernel(a: ol.ptr, b: ol.ptr, result: ol.ptr):
    idx = ol.get_tid_x()
    a_val = ol.load(a, idx)
    b_val = ol.load(b, idx)
    ol.store(a_val + b_val, result, idx)
```

### 2. Compilation
```bash
# Using CLI
oven compile my_kernel.py

# Or in Python
import oven.compiler as comp
compiler = comp.PythonToMLIRCompiler()
mlir_code = compiler.compile_file("my_kernel.py")
```

## 📖 Key Concepts

### GPU Programming
- **Thread ID**: `ol.get_tid_x()`, `ol.get_tid_y()`
- **Block ID**: `ol.get_bid_x()`, `ol.get_bid_y()`
- **Shared Memory**: `ol.smem()`
- **Synchronization**: `ol.barrier()`

### Memory Operations
- **Load**: `ol.load(ptr, offset)`
- **Store**: `ol.store(value, ptr, offset)`

### Mathematical Functions
- **Exponential/Log**: `ol.exp()`, `ol.log()`
- **Trigonometric**: `ol.sin()`, `ol.cos()`, `ol.tan()`
- **Others**: `ol.sqrt()`, `ol.sigmoid()`

## 🔗 Related Links

- [GitHub Repository](https://github.com/sjjeong94/language)
- [PyPI Package](https://pypi.org/project/oven-language/)
- [MLIR Official Documentation](https://mlir.llvm.org/)

## 📝 Contributing

If you'd like to contribute to improving the documentation:
1. Open an issue to suggest problems or improvements
2. Submit a Pull Request to directly improve the documentation
3. Add examples or tutorials

Documentation is written in Markdown format, and all contributions are welcome!