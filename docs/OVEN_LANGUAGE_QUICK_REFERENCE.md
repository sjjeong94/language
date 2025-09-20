# Oven Language 빠른 참조 가이드

## 가져오기
```python
import oven.language as ol
```

## 타입 힌트
```python
ol.ptr     # 포인터 타입 (!llvm.ptr)
ol.f32     # 32비트 부동소수점 (f32)
ol.i32     # 32비트 정수 (i32)
ol.index   # 인덱스 타입 (index)
```

## GPU 메모리
```python
ol.load(ptr, offset)          # 메모리에서 로드
ol.store(value, ptr, offset)  # 메모리에 저장
ol.smem()                     # 공유 메모리 할당
```

## GPU 스레드 정보
```python
ol.get_tid_x()    # 스레드 ID (X)
ol.get_tid_y()    # 스레드 ID (Y)
ol.get_bid_x()    # 블록 ID (X)
ol.get_bid_y()    # 블록 ID (Y)
ol.get_bdim_x()   # 블록 크기 (X)
ol.barrier()      # 스레드 동기화
```

## 수학 함수
```python
# 기본 함수
ol.exp(x)      # e^x
ol.log(x)      # ln(x)
ol.sqrt(x)     # √x
ol.sigmoid(x)  # 1/(1+e^(-x))

# 삼각 함수
ol.sin(x)      # sin(x)
ol.cos(x)      # cos(x)
ol.tan(x)      # tan(x)
```

## 산술 연산
```python
# 정수 연산
ol.muli(a, b)  # 정수 곱셈
ol.addi(a, b)  # 정수 덧셈

# 부동소수점 연산
ol.mulf(a, b)  # 실수 곱셈
ol.addf(a, b)  # 실수 덧셈
```

## 입출력
```python
ol.load_input_x(index)       # 입력 x 버퍼에서 로드
ol.load_input_y(index)       # 입력 y 버퍼에서 로드
ol.store_output_x(val, idx)  # 출력 x 버퍼에 저장
ol.store_output_y(val, idx)  # 출력 y 버퍼에 저장
```

## 유틸리티
```python
ol.index_cast(val, from, to)  # 타입 변환
ol.constant(value, type)      # 상수 생성
ol.for_loop(start, end, step, body, init)  # 루프
ol.yield_value(*values)       # yield
```

## 예제: 간단한 GPU 커널
```python
def vector_add(a: ol.ptr, b: ol.ptr, c: ol.ptr, n: int):
    idx = ol.get_tid_x() + ol.get_bid_x() * ol.get_bdim_x()
    a_val = ol.load(a, idx)
    b_val = ol.load(b, idx)
    result = a_val + b_val  # Python 연산자도 사용 가능
    ol.store(result, c, idx)
```

## 예제: 공유 메모리 사용
```python
def shared_memory_example(data: ol.ptr, output: ol.ptr):
    tid = ol.get_tid_x()
    smem = ol.smem()
    
    # 데이터를 공유 메모리로 로드
    value = ol.load(data, tid)
    ol.store(value, smem, tid)
    
    # 동기화
    ol.barrier()
    
    # 공유 메모리에서 처리
    shared_val = ol.load(smem, tid)
    result = ol.exp(shared_val)
    ol.store(result, output, tid)
```