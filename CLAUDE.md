# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RapidUDF is a high-performance SIMD-vectorized expression and script computation engine. It compiles a C-like DSL at runtime via LLVM JIT, achieving near-native performance. Key capabilities: SIMD-accelerated columnar operations (Google Highway + SLEEF), zero-cost FFI to C++/Protobuf/FlatBuffers/STL, dynamic Table APIs, and thread-safe JIT-generated functions with LRU compilation cache.

## Build Commands

### Bazel (primary)
```bash
bazel build //rapidudf/...                    # build everything
bazel test  //rapidudf/tests/...              # all tests
bazel test  //rapidudf/tests:grammar_test     # single test
bazel test  //rapidudf/tests:math_test --test_output=all  # verbose
```

### CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Formatting
```bash
clang-format -i <file>   # Google style, 120 columns (see .clang-format)
```

## Architecture

### Compilation Pipeline
```
Source text → Lexer → Parser → AST → JitCompiler → LLVM IR → Machine Code → JitFunction<>
```

- **Lexer** (`ast/lexer.h`): Hand-written tokenizer
- **Parser** (`ast/`): Recursive descent parser producing AST nodes (Function, Expression in RPN form, Statement, Block)
- **JitCompiler** (`compiler/compiler.h`): Main API entry point. Template methods: `CompileExpression<RET, Args...>()`, `CompileFunction<>()`, `CompileSource()`, `LoadFunction<>()`
- **CodeGen** (`compiler/codegen.h` + 6 specialized `.cc` files): LLVM IR generation using `llvm::orc::LLJIT`
- **JitFunction<RET, Args...>** (`compiler/function.h`): Callable wrapper around JIT-compiled function pointer
- **EvalEngine** (`exec/eval_engine.h`): Higher-level API with LRU compilation cache

### Type System
- **DType** (`meta/dtype.h`): 64-bit encoded type descriptor supporting primitives, strings, JSON, DynObject, Vector, collections
- **get_dtype<T>()**: Compile-time C++ type → DType mapping
- **Vector<T>** (`types/vector.h`): Non-owning view over contiguous data for columnar operations
- **Bit** (`types/bit.h`): Packed single-bit type for boolean vectors

### SIMD / Vectorization (`functions/simd/`)
Uses Google Highway for portable SIMD (AVX2/NEON), SLEEF for vectorized math, x86-simd-sort for sorting. All SIMD functions compiled at `-O3`.

### FFI / Reflection (`reflect/`)
- `RUDF_STRUCT_FIELDS` macro registers C++ struct fields
- Zero-cost reflection for Protobuf (`reflect/protobuf.h`), FlatBuffers (`reflect/flatbuffers.h`), STL containers (`reflect/stl.h`)

### Memory Management
- Arena allocators in `memory/` (Folly-style, LevelDB-style, ThreadCachedArena)
- **Context** (`context/context.h`): Runtime context owning an arena; provides `NewVector()`, `ArenaAllocate()`, `Own()`

### Table API (`table/`)
Columnar table with schema supporting `.filter()`, `.order_by()`, `.topk()`, `.take()`.

### Public API
Single umbrella header: `rapidudf/rapidudf.h` exposing `JitCompiler`, `JitFunction`, `Options`, `EvalEngine`, `Table`, `Context`, reflection macros.

## Key Dependencies

| Library | Purpose |
|---------|---------|
| LLVM 18+ | JIT compilation (ORC LLJIT, IRBuilder) |
| Google Highway 1.4.0 | Portable SIMD |
| SLEEF 3.9.0 | Vectorized math |
| abseil-cpp | Status, flat_hash_map, btree |
| fmt 11.0.2 + spdlog 1.14.1 | Formatting and logging |
| protobuf 3.19.2 / flatbuffers 2.0+ | Serialization FFI |
| Google Test 1.14.0 + Benchmark 1.8.3 | Testing |

Bazel uses system LLVM at `/usr`; CMake uses `find_package(LLVM)`.

## Test Structure

All tests in `rapidudf/tests/` using Google Test. Key test categories:
- **Parser**: `grammar_test`
- **Type system**: `dtype_test`, `cast_test`
- **Control flow**: `ifelse_test`, `while_test`, `ternary_test`
- **Operations**: `cmp_test`, `logic_op_test`, `math_test`, `string_test`
- **SIMD**: `simd_vector_test`, `simd_sort_test`, `simd_table_test`
- **FFI**: `ffi_func_test`, `ffi_struct_test`, `ffi_pb_test`, `ffi_fbs_test`, `ffi_stl_test`
- **Other**: `json_test`, `dyn_object_test`, `eval_engine_test`, `arena_test`

Benchmarks: `benchmark`, `parse_benchmark`, `dot_benchmark`, `string_benchmark`

## Code Style

Google C++ style with 120-column limit. Pointer alignment: left (`int* p`). C++17 standard.
