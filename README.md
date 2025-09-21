# Oven Language

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/oven-language.svg)](https://badge.fury.io/py/oven-language)
[![Tests](https://github.com/sjjeong94/oven-language/workflows/Tests/badge.svg)](https://github.com/sjjeong94/oven-language/actions)
[![Build](https://github.com/sjjeong94/oven-language/workflows/Build/badge.svg)](https://github.com/sjjeong94/oven-language/actions)

A Python-to-MLIR compiler for GPU kernel development.

## Installation

```bash
pip install oven-language
```

## Quick Start

```python
import oven.language as ol

def vector_add(a_ptr: ol.ptr, b_ptr: ol.ptr, out_ptr: ol.ptr):
    tid = ol.get_tid_x()
    idx = tid * 4
    
    # Load vectors
    a = ol.vload(a_ptr, idx, 4)
    b = ol.vload(b_ptr, idx, 4)
    
    # Add and store
    ol.vstore(a + b, out_ptr, idx, 4)
```

Compile to MLIR:
```bash
oven kernel.py
```

## Key Functions

- **Thread/Block IDs**: `ol.get_tid_x()`, `ol.get_bid_x()`
- **Memory**: `ol.load()`, `ol.store()`, `ol.vload()`, `ol.vstore()`
- **Math**: `ol.exp()`, `ol.log()`, `ol.sin()`, `ol.sigmoid()`
- **GPU**: `ol.barrier()`, `ol.smem()`

## Links

- [Documentation](docs/Documentation.md)
- [GitHub](https://github.com/sjjeong94/oven-language)
- [PyPI](https://pypi.org/project/oven-language/)
