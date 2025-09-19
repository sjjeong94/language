# Python to MLIR Compiler

이 프로젝트는 Python 소스 코드를 MLIR(Multi-Level Intermediate Representation)로 변환하는 컴파일러입니다.

## 주요 특징

- **Python AST 활용**: Python의 내장 AST 모듈을 사용하여 소스 코드를 파싱
- **일반화된 구조**: 방문자 패턴을 사용하여 확장 가능한 아키텍처
- **MLIR 생성**: 표준 MLIR 연산들을 생성 (func, arith, cf, memref 다이얼렉트 지원)
- **백엔드 호환**: 생성된 MLIR은 다양한 백엔드 컴파일러에서 사용 가능

## 지원되는 Python 기능

### ✅ 완전 지원
- 함수 정의 (def)
- 기본 데이터 타입 (int, float, bool, string)
- 산술 연산 (+, -, *, /, %)
- 비교 연산 (==, !=, <, <=, >, >=)
- 제어 흐름 (if/else, while)
- 함수 호출
- 변수 할당
- 상수 값

### 🔄 부분 지원
- for 루프 (기본 구조만)
- 단항 연산 (+, -, not)

### ❌ 미지원 (향후 확장 가능)
- 클래스 및 객체
- 리스트, 딕셔너리 등 복합 자료형
- 예외 처리 (try/except)
- 람다 함수
- 제너레이터

## 사용 방법

### 명령행 인터페이스
```bash
# 기본 컴파일
python main.py input.py -o output.mlir

# 디버그 모드
python main.py input.py --debug

# 최적화 비활성화
python main.py input.py --no-optimize
```

### 프로그래밍 인터페이스
```python
from src.compiler import compile_python_string, PythonToMLIRCompiler

# 간단한 사용
mlir_code = compile_python_string("def add(a, b): return a + b")

# 고급 사용
compiler = PythonToMLIRCompiler(debug=True, optimize=True)
mlir_code = compiler.compile_file("input.py")
```

## 예제

### 입력 Python 코드
```python
def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)
```

### 생성된 MLIR 코드
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

## 프로젝트 구조

```
compiler/
├── src/
│   ├── ast_visitor.py      # AST 방문자 (Python AST → MLIR 변환)
│   ├── mlir_generator.py   # MLIR 코드 생성기
│   ├── compiler.py         # 메인 컴파일러 인터페이스
│   └── utils/
│       └── mlir_utils.py   # MLIR 유틸리티 함수들
├── tests/
│   ├── test_compiler.py    # 단위 테스트
│   └── examples/          # 테스트 예제들
├── main.py                # CLI 인터페이스
└── README.md
```

## 백엔드 연동

생성된 MLIR 코드는 다음과 같은 백엔드에서 사용할 수 있습니다:

1. **LLVM 백엔드**: MLIR → LLVM IR → 네이티브 코드
2. **커스텀 백엔드**: 직접 구현한 MLIR 기반 컴파일러
3. **GPU 백엔드**: CUDA/ROCm 등으로 변환
4. **기타 타겟**: TensorFlow, JAX 등

## 확장 가능성

이 컴파일러는 확장 가능한 구조로 설계되었습니다:

- **새로운 Python 구문 지원**: `ast_visitor.py`에 새로운 `visit_*` 메서드 추가
- **새로운 MLIR 연산**: `mlir_generator.py`에 새로운 생성 메서드 추가
- **최적화 패스**: `compiler.py`의 최적화 함수 확장
- **타입 시스템**: 더 정교한 타입 추론 및 검사 추가

## 테스트

```bash
# 모든 테스트 실행
python tests/test_compiler.py

# 샘플 파일 컴파일 테스트
python main.py tests/examples/sample.py --debug
```

이 컴파일러는 Python 코드를 MLIR로 변환하는 기본적인 기능을 제공하며, 필요에 따라 추가 기능을 확장할 수 있는 견고한 기반을 제공합니다.
