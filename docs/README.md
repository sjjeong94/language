# Oven Compiler 문서

이 디렉토리는 Oven Compiler의 공식 문서를 포함합니다.

## 📚 문서 목록

### API 레퍼런스
- **[Oven Language API 문서](OVEN_LANGUAGE_API.md)** - 모든 Oven Language 함수의 상세한 설명
- **[Oven Language 빠른 참조](OVEN_LANGUAGE_QUICK_REFERENCE.md)** - 주요 함수들의 간단한 참조 가이드

### 튜토리얼 및 가이드
- **[타입 힌트 가이드](type_hints.md)** - MLIR 타입 힌트 사용법

## 🚀 빠른 시작

### 1. 기본 사용법
```python
import oven.language as ol

def simple_kernel(a: ol.ptr, b: ol.ptr, result: ol.ptr):
    idx = ol.get_tid_x()
    a_val = ol.load(a, idx)
    b_val = ol.load(b, idx)
    ol.store(a_val + b_val, result, idx)
```

### 2. 컴파일
```bash
# CLI 사용
oven compile my_kernel.py

# 또는 Python에서
import oven.compiler as comp
compiler = comp.PythonToMLIRCompiler()
mlir_code = compiler.compile_file("my_kernel.py")
```

## 📖 주요 개념

### GPU 프로그래밍
- **스레드 ID**: `ol.get_tid_x()`, `ol.get_tid_y()`
- **블록 ID**: `ol.get_bid_x()`, `ol.get_bid_y()`
- **공유 메모리**: `ol.smem()`
- **동기화**: `ol.barrier()`

### 메모리 연산
- **로드**: `ol.load(ptr, offset)`
- **저장**: `ol.store(value, ptr, offset)`

### 수학 함수
- **지수/로그**: `ol.exp()`, `ol.log()`
- **삼각함수**: `ol.sin()`, `ol.cos()`, `ol.tan()`
- **기타**: `ol.sqrt()`, `ol.sigmoid()`

## 🔗 관련 링크

- [GitHub Repository](https://github.com/sjjeong94/language)
- [PyPI Package](https://pypi.org/project/oven-compiler/)
- [MLIR 공식 문서](https://mlir.llvm.org/)

## 📝 기여하기

문서 개선에 기여하고 싶으시다면:
1. 이슈를 열어 문제점이나 개선사항을 제안해주세요
2. Pull Request를 통해 직접 문서를 개선해주세요
3. 예제나 튜토리얼을 추가해주세요

문서는 Markdown 형식으로 작성되며, 모든 기여를 환영합니다!