# Oven Language API 문서

Oven Language는 Python을 MLIR로 컴파일하기 위한 특수한 함수들을 제공합니다. GPU 컴퓨팅과 수학적 연산을 위한 함수들이 포함되어 있습니다.

## 목차

1. [타입 힌트](#타입-힌트)
2. [GPU 메모리 연산](#gpu-메모리-연산)
3. [GPU 스레드 및 블록 연산](#gpu-스레드-및-블록-연산)
4. [수학 함수](#수학-함수)
5. [산술 연산](#산술-연산)
6. [NVIDIA 내장 함수](#nvidia-내장-함수)
7. [입출력 연산](#입출력-연산)
8. [타입 변환](#타입-변환)
9. [상수 및 루프 연산](#상수-및-루프-연산)

---

## 타입 힌트

Oven Language는 MLIR 컴파일을 위한 타입 힌트를 제공합니다.

### `ptr`
```python
class ptr:
    """Pointer type for MLIR compilation (!llvm.ptr)."""
```
- **설명**: MLIR의 포인터 타입에 대응
- **MLIR 타입**: `!llvm.ptr`

### `f32`
```python
class f32:
    """32-bit floating point type (f32)."""
```
- **설명**: 32비트 부동소수점 타입
- **MLIR 타입**: `f32`

### `i32`
```python
class i32:
    """32-bit integer type (i32)."""
```
- **설명**: 32비트 정수 타입
- **MLIR 타입**: `i32`

### `index`
```python
class index:
    """Index type for MLIR (index)."""
```
- **설명**: MLIR의 인덱스 타입
- **MLIR 타입**: `index`

---

## GPU 메모리 연산

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
- **설명**: 지정된 오프셋에서 GPU 메모리로부터 값을 로드
- **MLIR 연산**: `oven.load`
- **예제**: `value = ol.load(memory_ptr, 0)`

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
- **설명**: 지정된 오프셋의 GPU 메모리에 값을 저장
- **MLIR 연산**: `oven.store`
- **예제**: `ol.store(42.0, memory_ptr, 0)`

### `smem()`
```python
def smem():
    """
    Allocate shared memory for GPU computations.
    
    Returns:
        Shared memory pointer (!llvm.ptr<3>)
    """
```
- **설명**: GPU 공유 메모리 할당
- **MLIR 연산**: `oven.smem`
- **반환 타입**: `!llvm.ptr<3>`
- **예제**: `shared_mem = ol.smem()`

---

## GPU 스레드 및 블록 연산

### 스레드 ID 함수

#### `get_tid_x()`
```python
def get_tid_x():
    """Get the current thread ID in the X dimension."""
```
- **설명**: X 차원의 현재 스레드 ID 반환
- **MLIR 연산**: `nvvm.read.ptx.sreg.tid.x`

#### `get_tid_y()`
```python
def get_tid_y():
    """Get the current thread ID in the Y dimension."""
```
- **설명**: Y 차원의 현재 스레드 ID 반환
- **MLIR 연산**: `nvvm.read.ptx.sreg.tid.y`

### 블록 ID 함수

#### `get_bid_x()`
```python
def get_bid_x():
    """Get the current block ID in the X dimension."""
```
- **설명**: X 차원의 현재 블록 ID 반환
- **MLIR 연산**: `nvvm.read.ptx.sreg.ctaid.x`

#### `get_bid_y()`
```python
def get_bid_y():
    """Get the current block ID in the Y dimension."""
```
- **설명**: Y 차원의 현재 블록 ID 반환
- **MLIR 연산**: `nvvm.read.ptx.sreg.ctaid.y`

### 블록 차원 함수

#### `get_bdim_x()`
```python
def get_bdim_x():
    """Get the block dimension in the X dimension."""
```
- **설명**: X 차원의 블록 크기 반환
- **MLIR 연산**: `nvvm.read.ptx.sreg.ntid.x`

### 동기화 함수

#### `barrier()`
```python
def barrier():
    """
    Synchronization barrier for GPU threads.
    Ensures all threads in a block reach this point before continuing.
    """
```
- **설명**: GPU 스레드 동기화 배리어
- **MLIR 연산**: `nvvm.barrier0`
- **용도**: 블록 내 모든 스레드가 이 지점에 도달할 때까지 대기

---

## 수학 함수

### 지수 및 로그 함수

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
- **설명**: 자연 지수 함수 계산
- **MLIR 연산**: `math.exp`

#### `log(x)`
```python
def log(x):
    """Compute the natural logarithm function."""
```
- **설명**: 자연 로그 함수 계산
- **MLIR 연산**: `math.log`

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
- **설명**: 시그모이드 함수 계산
- **공식**: `1 / (1 + e^(-x))`

### 삼각 함수

#### `sin(x)`
```python
def sin(x):
    """Compute the sine function."""
```
- **설명**: 사인 함수 계산
- **MLIR 연산**: `math.sin`

#### `cos(x)`
```python
def cos(x):
    """Compute the cosine function."""
```
- **설명**: 코사인 함수 계산
- **MLIR 연산**: `math.cos`

#### `tan(x)`
```python
def tan(x):
    """Compute the tangent function."""
```
- **설명**: 탄젠트 함수 계산
- **MLIR 연산**: `math.tan`

### 기타 수학 함수

#### `sqrt(x)`
```python
def sqrt(x):
    """Compute the square root function."""
```
- **설명**: 제곱근 함수 계산
- **MLIR 연산**: `math.sqrt`

---

## 산술 연산

### 정수 연산

#### `muli(a, b)`
```python
def muli(a, b):
    """Multiply two integer values."""
```
- **설명**: 두 정수 값 곱셈
- **MLIR 연산**: `arith.muli`

#### `addi(a, b)`
```python
def addi(a, b):
    """Add two integer values."""
```
- **설명**: 두 정수 값 덧셈
- **MLIR 연산**: `arith.addi`

### 부동소수점 연산

#### `mulf(a, b)`
```python
def mulf(a, b):
    """Multiply two floating-point values."""
```
- **설명**: 두 부동소수점 값 곱셈
- **MLIR 연산**: `arith.mulf`

#### `addf(a, b)`
```python
def addf(a, b):
    """Add two floating-point values."""
```
- **설명**: 두 부동소수점 값 덧셈
- **MLIR 연산**: `arith.addf`

---

## NVIDIA 내장 함수

### 직접 내장 함수

#### `nvvm_read_ptx_sreg_ntid_x()`
```python
def nvvm_read_ptx_sreg_ntid_x():
    """NVIDIA intrinsic: Read block dimension X."""
```
- **설명**: NVIDIA 내장 함수 - X 차원 블록 크기 읽기

#### `nvvm_read_ptx_sreg_ctaid_x()`
```python
def nvvm_read_ptx_sreg_ctaid_x():
    """NVIDIA intrinsic: Read block ID X."""
```
- **설명**: NVIDIA 내장 함수 - X 차원 블록 ID 읽기

#### `nvvm_read_ptx_sreg_tid_x()`
```python
def nvvm_read_ptx_sreg_tid_x():
    """NVIDIA intrinsic: Read thread ID X."""
```
- **설명**: NVIDIA 내장 함수 - X 차원 스레드 ID 읽기

### 별칭 함수

다음 별칭들도 사용 가능합니다:
- `__nvvm_read_ptx_sreg_ntid_x` → `nvvm_read_ptx_sreg_ntid_x`
- `__nvvm_read_ptx_sreg_ctaid_x` → `nvvm_read_ptx_sreg_ctaid_x`
- `__nvvm_read_ptx_sreg_tid_x` → `nvvm_read_ptx_sreg_tid_x`
- `__load_from_ptr` → `load`
- `__store_to_ptr` → `store`

---

## 입출력 연산

### 입력 함수

#### `load_input_x(index)`
```python
def load_input_x(index):
    """Load value from input buffer x at specified index."""
```
- **설명**: 입력 버퍼 x에서 지정된 인덱스의 값 로드

#### `load_input_y(index)`
```python
def load_input_y(index):
    """Load value from input buffer y at specified index."""
```
- **설명**: 입력 버퍼 y에서 지정된 인덱스의 값 로드

### 출력 함수

#### `store_output_x(value, index)`
```python
def store_output_x(value, index):
    """Store value to output buffer x at specified index."""
```
- **설명**: 출력 버퍼 x의 지정된 인덱스에 값 저장

#### `store_output_y(value, index)`
```python
def store_output_y(value, index):
    """Store value to output buffer y at specified index."""
```
- **설명**: 출력 버퍼 y의 지정된 인덱스에 값 저장

---

## 타입 변환

#### `index_cast(value, from_type, to_type)`
```python
def index_cast(value, from_type, to_type):
    """Cast between index and integer types."""
```
- **설명**: 인덱스와 정수 타입 간 변환
- **MLIR 연산**: `arith.index_cast`

---

## 상수 및 루프 연산

### 상수

#### `constant(value, data_type)`
```python
def constant(value, data_type):
    """Create a constant value."""
```
- **설명**: 상수 값 생성
- **MLIR 연산**: `arith.constant`

### 루프 연산

#### `for_loop(start, end, step, body_func, init_args=None)`
```python
def for_loop(start, end, step, body_func, init_args=None):
    """Create a for loop with iter_args."""
```
- **설명**: iter_args를 가진 for 루프 생성
- **MLIR 연산**: `scf.for`

#### `yield_value(*values)`
```python
def yield_value(*values):
    """Yield values in a loop."""
```
- **설명**: 루프에서 값 yield
- **MLIR 연산**: `scf.yield`

---

## 사용 예제

```python
import oven.language as ol

def gpu_kernel(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, n: int):
    # GPU 스레드 및 블록 정보 가져오기
    tid = ol.get_tid_x()
    bid = ol.get_bid_x()
    bdim = ol.get_bdim_x()
    
    # 글로벌 인덱스 계산
    idx = bid * bdim + tid
    
    # 공유 메모리 할당
    smem = ol.smem()
    
    # 메모리에서 값 로드
    a_val = ol.load(a_ptr, idx)
    b_val = ol.load(b_ptr, idx)
    
    # 수학 연산 수행
    result = ol.exp(a_val) + ol.sin(b_val)
    
    # 결과를 메모리에 저장
    ol.store(result, c_ptr, idx)
    
    # 스레드 동기화
    ol.barrier()
```

이 문서는 Oven Language의 모든 함수들을 체계적으로 정리한 것입니다. 각 함수는 MLIR 컴파일 과정에서 해당하는 MLIR 연산으로 변환됩니다.