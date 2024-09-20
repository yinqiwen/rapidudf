# RapidUDF

`RapidUDF` is a C++ vectorized expression/script calculation execution engine primarily based on `vectorization` and `JIT compilation` technologies. Compared to similar implementations, it offers significant performance improvements. Users can apply it in scenarios requiring high-performance online calculations with complex and flexible needs, such as rule engines, storage UDFs, and feature computations.

## Dependencies

- [LLVM](https://llvm.org/)
- [highway](https://github.com/google/highway)
- [x86-simd-sort](https://github.com/intel/x86-simd-sort)
- [sleef](https://github.com/shibatch/sleef)
- [fmtlib](https://github.com/fmtlib/fmt)
- [spdlog](https://github.com/gabime/spdlog)
- [abseil-cpp](https://github.com/abseil/abseil-cpp)
- boost
  - [preprocessor](http://boost.org/libs/preprocessor)
  - [parser](https://github.com/tzlaine/parser)
- [protobuf](https://github.com/protocolbuffers)
- [flatbuffers](https://github.com/google/flatbuffers)
- [json](https://github.com/nlohmann/json)

## Limitations

- C++17

## Features

- JIT compilation of expressions/UDFs (C-like function DSL)
- `Thread-safe`
  - Compiled into stateless function pointers that can be repeatedly invoked
- [Built-in Operators](docs/operator.md)
  - Arithmetic operators (`+`, `-`, `*`, `/`, `%`, `^`)
  - Comparison operators (`>`, `>=`, `<`, `<=`, `!=`, `==`)
  - Logical operators (`&&`, `||`)
  - Assignment operators (`=`, `+=`, `-=``, `*=``, `/=`, `%=`)
  - Conditional operator (`?:`)
  - Negation operator (`!`)
  - Access operators (`.`, `[]`)
- Conditional control (`if-elif*-else`)
- Loop control (`while`, `continue`, `break`)
- Supported data types
  - Basic types (`bit/bool`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `string_view`)
  - Vectorized vectors (`simd_vector<u64>`, `simd_vector<f64>`, etc.)
  - `Protobuf` (requires FFI compile-time binding)
  - `Flatbuffers` (requires FFI compile-time binding)
  - `STL containers` (`string`, `vector`, `set`, `map`, `unordered_map`, `unordered_set`)
  - Any C++ class (requires FFI compile-time binding)
- `Vectorization acceleration`
  - Generates vectorized accelerated code based on `highway` (mathematical calculations, comparisons, etc.)
  - Provides sort vectorization acceleration implementation based on `x86-simd-sort`
- `FFI`
  - Use C++ data structures in expressions/UDFs
  - Call C++ methods in expressions/UDFs
    - Ordinary C-style methods `func(arg0, arg1)`
    - Object methods `obj.func(arg0, arg1)`
- [Built-in Function Implementations](docs/builtin_function.md)
  - C math library functions
- Variables
  - Parameter variables (exposed to users via generated JIT Function parameters)
  - Temporary variables (temporary variables defined in UDFs used to store intermediate results)
- Comments (supports C++ style `//` comments)

## Installation

### Bazel
```python
    git_repository(
        name = "rapidudf",
        remote = "https://git.woa.com/qiyingwang/rapidudf.git",
        commit = "...",
    )
    load("@rapidudf//:rapidudf.bzl", "rapidudf_workspace")
    rapidudf_workspace()
```
### CMake

## Usage

### Simple Expression
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. If needed, set the rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);
  // 2. Expression string
  std::string expression = "x >= 1 && y < 10";

  // 3. Compile into a Function; this Function object can be saved for subsequent repeated execution; compilation time is generally between 10ms-100ms;
  rapidudf::JitCompiler compiler;
  // The template parameters of CompileExpression support multiple values; the first template parameter is the return type, followed by the function argument types;
  // Variable names used in the expression need to be passed as a list of argument names, otherwise, compilation will fail.
  auto result = compiler.CompileExpression<bool, int, int>(expression, {"x", "y"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4. Execute the function
  rapidudf::JitFunction<bool, int, int> f = std::move(result.value());
  bool v = f(2, 3);  // true
  v = f(0, 1);       // false
  return 0;
};
```

### Simple UDF
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. If needed, set the rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);

  // 2. UDF string
  std::string source = R"(
    int fib(int n) 
    { 
       if (n <= 1){
         return n; 
       }
       return fib(n - 1) + fib(n - 2);  // recursive call
    } 
  )";

  // 3. Compile into a Function; this Function object can be saved for subsequent repeated execution; compilation time is generally between 10ms-100ms;
  rapidudf::JitCompiler compiler;
  // The template parameters of CompileExpression support multiple values; the first template parameter is the return type, followed by the function argument types.
  auto result = compiler.CompileFunction<int, int>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4. Execute the function
  rapidudf::JitFunction<int, int> f = std::move(result.value());
  int n = 9;
  int x = f(n);  // 34
  RUDF_INFO("fib({}):{}", n, x);
  return 0;
};
```

### Vectorized Acceleration with Vectors
```cpp
#include "rapidudf/rapidudf.h"

struct User {
  std::string city;
};
RUDF_STRUCT_FIELDS(User, city)  // Bind the User class, accessible via the city field in UDFs

struct Feed {
  std::string city;
  float score;
};
struct Feeds {
  rapidudf::simd::Vector<rapidudf::StringView> city;
  rapidudf::simd::Vector<float> score;
};
RUDF_STRUCT_FIELDS(Feeds, city, score)  // Bind the Feeds class, accessible via the city/score fields in UDFs

int main() {
  // 2. UDF string
  std::string source = R"(
    void boost_scores(Context ctx, User user, Feeds feeds) 
    { 
      // Note that boost is a float array
      var boost = (feeds.city == user.city ? 2.0_f32 : 1.1_f32);
      feeds.score *= boost;
    } 
  )";

  // 3. Compile into a Function; this Function object can be saved for subsequent repeated execution.
  rapidudf::JitCompiler compiler;
  // The template parameters of CompileExpression support multiple values; the first template parameter is the return type, followed by the function argument types.
  // 'rapidudf::Context' is a required parameter in SIMD implementations, involving arena memory allocation.
  auto result = compiler.CompileFunction<void, rapidudf::Context&, const User&, Feeds&>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 Test data, need to convert original data into columnar format
  User user;
  user.city = "sz";
  std::vector<Feed> feeds;
  for (size_t i = 0; i < 1024; i++) {
    Feed feed;
    feed.city = (i % 2 == 0 ? "sz" : "bj");
    feed.score = i + 1.1;
    feeds.emplace_back(feed);
  }

  // 4.2 Convert original data into columnar format
  std::vector<rapidudf::StringView> cities;
  std::vector<float> scores;
  for (auto& feed : feeds) {
    cities.emplace_back(feed.city);
    scores.emplace_back(feed.score);
  }
  Feeds column_feeds;
  column_feeds.city = cities;
  column_feeds.score = scores;

  // 5. Execute the function
  rapidudf::Context ctx;
  rapidudf::JitFunction<void, rapidudf::Context&, const User&, Feeds&> f = std::move(result.value());
  f(ctx, user, column_feeds);
  for (size_t i = 0; i < column_feeds.score.Size(); i++) {
    RUDF_INFO("{} {}/{}", cities[i], scores[i], column_feeds.score[i]);
  }

  return 0;
};
```

### More Examples and Usage
- [Using custom C++ classes in expressions/UDFs](docs/ffi.md)
- [Calling member methods of custom C++ classes in expressions/UDFs](docs/ffi.md)
- [Using protobuf objects in expressions/UDFs](docs/ffi.md)
- [Using flatbuffers objects in expressions/UDFs](docs/ffi.md)
- [Using STL objects in expressions/UDFs](docs/ffi.md)

## Performance
### Comparison with Native C++
Due to its implementation based on LLVM JIT, RapidUDF theoretically can achieve performance very close to native C++ code.     
Benchmark results comparing the fibonacci method compiled with O0 optimization:    
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22547 ns        22547 ns        31060
BM_native_fib_func        38933 ns        38933 ns        17964
```
Benchmark results comparing the fibonacci method compiled with O2 optimization:
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22552 ns        22552 ns        31041
BM_native_fib_func        19212 ns        19212 ns        36437
```
Note: The JIT implementation currently executes the same JIT compilation logic under O0/O2 compilation, theoretically generating the same code.

### Vectorized Computation Scenario (AVX2)
Computation involves executing `x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y` on a double array, compiled with -O2 optimization, where the corresponding SIMD vector operation code runs on the AVX2 instruction set.      
Note: rapidudf_expr/exprtk/native all implement non-vectorized loop calls.    
```
Benchmark                             Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_expr_func             51290 ns        51290 ns        13684
BM_rapidudf_vector_expr_func       8395 ns         8395 ns        83350
BM_exprtk_expr_func               65001 ns        65001 ns        10869
BM_native_func                    50161 ns        50160 ns        13953
```

### Comparison with Lua/WASM/...
Compared to third-party language implementations like Lua/WASM, the biggest performance advantage of RapidUDF lies in the lack of overhead for data transfer/function calls.     
The compiled UDF can access objects defined in C++ according to the C++ memory model, including but not limited to:   
- Protobuf
- Flatbufers
- Json
- STL containers
- Custom C++ classes
