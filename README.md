# Python to MLIR Compiler

A Python compiler that converts Python source code to MLIR (Multi-Level Intermediate Representation) using Python's AST.

## Features

- **Python AST 활용**: Python의 내장 AST 모듈을 사용하여 소스 코드를 파싱
- **일반화된 구조**: 방문자 패턴을 사용하여 확장 가능한 아키텍처
- **MLIR 생성**: 표준 MLIR 연산들을 생성 (func, arith, cf, memref 다이얼렉트 지원)
- **백엔드 호환**: 생성된 MLIR은 다양한 백엔드 컴파일러에서 사용 가능
- **테스트 지원**: Pytest 기반의 포괄적인 테스트 스위트

## Project Structure

```
compiler/
├── src/
│   ├── __init__.py
│   ├── ast_visitor.py      # AST visitor pattern implementation
│   ├── mlir_generator.py   # MLIR code generation
│   ├── compiler.py         # Main compiler interface
│   └── utils/
│       ├── __init__.py
│       └── mlir_utils.py   # MLIR utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures and configuration
│   ├── test_compiler.py    # Main test suite
│   ├── test_parametrized.py # Parametrized tests
│   └── examples/
│       ├── sample.py       # Basic examples
│       └── complex.py      # Complex examples
├── pyproject.toml          # Pytest configuration
├── Makefile               # Build and test automation
├── requirements.txt
└── main.py                # CLI entry point
```

## Usage

### Command Line Interface
```bash
# Basic compilation
python main.py input.py -o output.mlir

# Debug mode
python main.py input.py --debug

# Disable optimizations
python main.py input.py --no-optimize
```

### Programming Interface
```python
from src.compiler import compile_python_string, PythonToMLIRCompiler

# Simple usage
mlir_code = compile_python_string("def add(a, b): return a + b")

# Advanced usage
compiler = PythonToMLIRCompiler(debug=True, optimize=True)
mlir_code = compiler.compile_file("input.py")
```

## Testing

이 프로젝트는 pytest 기반의 포괄적인 테스트 스위트를 제공합니다.

### Running Tests

```bash
# All tests
make test
# 또는
pytest tests/ -v

# Unit tests only
make test-unit
# 또는  
pytest tests/ -v -m unit

# Integration tests only
make test-integration
# 또는
pytest tests/ -v -m integration

# With coverage
make test-coverage
# 또는
pytest tests/ --cov=src --cov-report=html

# Fast tests (exclude slow tests)
make test-fast
# 또는
pytest tests/ -v -m "not slow"
```

### Test Structure

- **Unit Tests**: 개별 컴포넌트 테스트 (MLIRUtils, MLIRGenerator, ASTVisitor, Compiler)
- **Integration Tests**: 전체 파이프라인 테스트
- **Parametrized Tests**: 다양한 입력에 대한 체계적 테스트
- **Fixtures**: 재사용 가능한 테스트 설정 및 데이터

### Test Coverage

현재 테스트 커버리지:
- 총 69개 테스트 케이스
- 전체 코드 커버리지: 62%
- 핵심 기능 커버리지: 80%+

## Requirements

- Python 3.8+
- pytest (개발용)
- pytest-cov (커버리지 리포트용)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd compiler

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
make test
```

## Examples

### Input Python Code
```python
def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
```

### Generated MLIR Code
```mlir
// Generated MLIR code from Python source
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
func.func @factorial(%arg0: i32) -> i32 {
  %1 = arith.constant 1 : i32
  %2 = arith.cmpi sle, %arg0, %1 : i32
  cf.cond_br %2, ^then0, ^else1
then0:
  %3 = arith.constant 1 : i32
  func.return %3 : i32
  cf.br ^if_end2
else1:
  %4 = arith.constant 1 : i32
  %5 = arith.subi %arg0, %4 : i32
  %6 = func.call @factorial(%5) : (i32) -> i32
  %7 = arith.muli %arg0, %6 : i32
  func.return %7 : i32
  cf.br ^if_end2
if_end2:
}
```

## Supported Python Features

### ✅ Fully Supported
- Function definitions (def)
- Basic data types (int, float, bool, string)
- Arithmetic operations (+, -, *, /, %)
- Comparison operations (==, !=, <, <=, >, >=)
- Control flow (if/else, while)
- Function calls
- Variable assignment
- Constants

### 🔄 Partially Supported
- for loops (basic structure)
- Unary operations (+, -, not)

### ❌ Not Supported (Future Extensions)
- Classes and objects
- Complex data structures (lists, dictionaries)
- Exception handling (try/except)
- Lambda functions
- Generators

## Development

### Available Make Commands

```bash
make help           # Show all available commands
make test           # Run all tests
make test-unit      # Run unit tests only
make test-coverage  # Run tests with coverage
make clean          # Clean generated files
make compile-example # Compile sample examples
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit a pull request

## License

[Add your license information here]
