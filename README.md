<h1 align="center">RapidUDF</h1>

`RapidUDF` is a high-performance SIMD vectorized expression/script computation execution engine library designed for online systems. It can be used in scenarios requiring high performance and flexibility such as rule engines, storage systems, and feature computation.


## Limitations

- C++17

## Features
- **Easy to Use**:
  - Provides support for conventional expression syntax
  - For more complex logic, supports a C-like DSL including if-elif-else* conditional control, while loop control, auto temporary variables, etc.;
  - For columnar memory data (vector<T>), provides dynamic Table APIs similar to Spark's DataFrame and operations like filter/order_by/topk/take/group_by/concat/distinct/map/flatmap/merge;
- **High Performance**:
  - Based on LLVM JIT compilation, startup and execution performance comparable to native cpp implementation;
  - For columnar memory data (vector<T>), provides SIMD vectorization acceleration implementation
- **Thread Safe**:
  - State-less JIT-generated C methods are naturally thread-safe
- **FFI**:
  - Supports zero-cost access to C++ defined class objects (custom classes/stl/protobufs/flatbuffers/...) in expressions/UDFs
  - Supports zero-cost calls to methods/class methods defined in C++ within expressions/UDFs
- **Rich Built-in Data Types, Operators, and Functions**:
  - [built-in data types](docs/dtype.md)
  - [built-in operators](docs/operator.md)
  - [built-in functions](docs/builtin_function.md)
- **Configurable Compilation**:
  - Supports configurable LLVM optimization level, fast-math, and other compilation options

## Compilation and Installation
Compilation requires a compiler that supports C++17
### Bazel
Add in WORKSPACE:
```python
    git_repository(
        name = "rapidudf",
        remote = "https://github.com/yinqiwen/rapidudf.git",
        commit = "...",
    )
    load("@rapidudf//:rapidudf.bzl", "rapidudf_workspace")
    rapidudf_workspace()
```
Add in the BUILD file for relevant code compilation rules:
```python
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = [
        "mylib.h",
    ],
    deps = [
        "@rapidudf",
    ],
)
```
### CMake
First, compile and install `rapidudf`
```bash
cd <rapidudf src dir>
mkdir build; cd build;
cmake ..
make install
```
Add the following to the CMake configuration of the related project:
```cmake
find_package(rapidudf REQUIRED)
....
# link rapidudf
target_link_libraries(mylib PRIVATE rapidudf::rapidudf)
```
[Example](rapidudf/examples/CMakeLists.txt)


## Usage Overview

### Simple Expression
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. If needed, set up rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);
  // 2. Expression string
  std::string expression = "x >= 1 && y < 10";
  // 3. Compile to generate Function, the generated Function object can be saved for subsequent repeated execution; compilation usually takes between 10ms-100ms;
  rapidudf::JitCompiler compiler;
  // CompileExpression's template parameters support multiple types, the first template parameter is the return type, the rest are function parameter types;
  // Variable names used in the expression need to be passed in as a parameter name list, otherwise compilation fails
  auto result = compiler.CompileExpression<bool, int, int>(expression, {"x", "y"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  // 4. Execute function
  rapidudf::JitFunction<bool, int, int> f = std::move(result.value());
  bool v = f(2, 3);  // true
  v = f(0, 1);       // false
  return 0;
};
```

### Simple UDF Script
Fibonacci function
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. If needed, can set up rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);
  // 2. UDF string
  std::string source = R"(
    int fib(int n)
    {
       if (n <= 1){
         return n;
       }
       // Supports cpp // comments
       return fib(n - 1) + fib(n - 2);  // Recursive call
    }
  )";
  // 3. Compile to generate Function, the generated Function object can be saved for subsequent repeated execution; compilation usually takes between 10ms-100ms;
  rapidudf::JitCompiler compiler;
  // CompileFunction's template parameters support multiple types, the first template parameter is the return type, the rest are function parameter types
  auto result = compiler.CompileFunction<int, int>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4. Execute function
  rapidudf::JitFunction<int, int> f = std::move(result.value());
  int n = 9;
  int x = f(n);  // 34
  RUDF_INFO("fib({}):{}", n, x);
  return 0;
};
```

### Vector Calculation
```cpp
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 2. UDF string
  std::string source = R"(
    simd_vector<f32> boost_scores(Context ctx, simd_vector<string_view> location, simd_vector<f32> score)
    {
      auto boost=(location=="home"?2.0_f32:0_f32);
      return score*boost;
    }
  )";

  // 3. Compile to generate Function, the generated Function object can be saved for subsequent use
  rapidudf::JitCompiler compiler;
  // CompileFunction's template parameters support multiple types, the first template parameter is the return type, the rest are function parameter types
  // 'rapidudf::Context' is a mandatory parameter involved in arena memory allocation in the simd implementation
  auto result =
      compiler.CompileFunction<Vector<float>, rapidudf::Context&, Vector<StringView>, Vector<float>>(
          source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 Test data, need to convert raw data into columnar data
  std::vector<float> scores;
  std::vector<std::string> locations;
  for (size_t i = 0; i < 4096; i++) {
    scores.emplace_back(1.1 + i);
    locations.emplace_back(i % 3 == 0 ? "home" : "other");
  }

  // 5. Execute function
  rapidudf::Context ctx;
  auto f = std::move(result.value());
  try {
    auto new_scores = f(ctx, ctx.NewVector(locations), ctx.NewVector(scores));
    for (size_t i = 0; i < new_scores.Size(); i++) {
      // RUDF_INFO("{}", new_scores[i]);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }
  return 0;
};
```

### Dynamic Vector Table
**RapidUDF** supports dynamically creating vector tables, allowing arbitrary computational operations on table columns (accelerated through SIMD) in expressions/UDFs;
The table class also provides operations similar to Spark DataFrame, such as:
- `.filter(Vector<Bit>)`   returns a new table instance filtered by condition
- `.order_by(Vector<T> column, bool descending)`   returns a new table instance sorted by condition
- `.topk(Vector<T> column, uint32_t k, bool descending)`  returns a new table instance with top k entries
- `.group_by(Vector<T> column)`   returns grouped table instances
- `.concat(Table* other)`   returns a new table instance concatenated with another table
- `.distinct(columns)`   removes duplicate rows by specified columns
- `.map(schema, func)` / `.flatmap(schema, func)`   row-wise transformation
- `.merge(other, columns, func)`   merge with conflict resolution
```cpp
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
struct Student {
  std::string name;
  uint16_t age = 0;
  float score = 0;
  bool gender = false;
};
RUDF_STRUCT_FIELDS(Student, name, age, score, gender)
int main() {
  // 1. Create table schema
  auto schema =
      table::TableSchema::GetOrCreate("Student", [](table::TableSchema* s) { std::ignore = s->AddColumns<Student>(); });

  // 2. UDF string, table<TABLE_NAME> generic format where TABLE_NAME must match the previously created table schema name
  // table supports filter/order_by/topk/take/group_by/concat/distinct/map/flatmap/merge, etc. operations
  std::string source = R"(
    table<Student> select_students(Context ctx, table<Student> x)
    {
       auto filtered = x.filter(x.score >90 && x.age<10);
       // Sort by score in descending order and take top 10
       return filtered.topk(filtered.score,10,true);
    }
  )";

  // 3. Compile to generate Function, the generated Function object can be saved for subsequent use
  rapidudf::JitCompiler compiler;
  // CompileFunction's template parameters support multiple types, the first template parameter is the return type, the rest are function parameter types
  auto result = compiler.CompileFunction<table::Table*, Context&, table::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 Test data, need to convert raw data into columnar data
  std::vector<Student> students;
  for (size_t i = 0; i < 128; i++) {
    float score = (i + 1) % 150;
    uint16_t age = i % 5 + 8;
    bool gender = i % 2 == 0;
    students.emplace_back(Student{"test_" + std::to_string(i), age, score, gender});
  }
  // 4.2 Create table instance
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(students);

  // 5. Execute function
  try {
    auto result_table = f(ctx, table.get());
    auto result_scores = result_table->Get<float>("score").value();
    auto result_names = result_table->Get<StringView>("name").value();
    auto result_ages = result_table->Get<uint16_t>("age").value();
    auto result_genders = result_table->Get<Bit>("gender").value();
    for (size_t i = 0; i < result_scores.Size(); i++) {
      RUDF_INFO("name:{},score:{},age:{},gender:{}", result_names[i], result_scores[i], result_ages[i],
                result_genders[i] ? true : false);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }
  return 0;
};
```

### Dynamic Vector Table Based on Protobuf/Flatbuffers/Struct
**RapidUDF** can also create a table from Protobuf/Flatbuffers, avoiding the tedious process of creating a TableSchema. Building table instances can be done directly from arrays of Protobuf objects such as `std::vector<T>, std::vector<const T*>, std::vector<T*>`.

Here is an example of creating a vector table based on Protobuf;
Examples based on flatbuffers can be found in [fbs_vector_table_udf](rapidudf/examples/fbs_vector_table_udf.cc);
Examples based on struct can be found in [struct_vector_table_udf](rapidudf/examples/struct_vector_table_udf.cc);
```cpp
#include "rapidudf/examples/student.pb.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 1. Create table schema
  auto schema = table::TableSchema::GetOrCreate(
      "Student", [](table::TableSchema* s) { std::ignore = s->AddColumns<examples::Student>(); });

  // 2. UDF string
  std::string source = R"(
    table<Student> select_students(Context ctx, table<Student> x)
    {
       auto filtered = x.filter(x.score >90 && x.age<10);
       // Sort in descending order
       return filtered.topk(filtered.score,10, true);
    }
  )";

  // 3. Compile to generate Function, the generated Function object can be saved for subsequent use
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<table::Table*, Context&, table::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 Test data
  std::vector<examples::Student> students;
  for (size_t i = 0; i < 150; i++) {
    examples::Student student;
    student.set_score((i + 1) % 150);
    student.set_name("test_" + std::to_string(i));
    student.set_age(i % 5 + 8);
    students.emplace_back(std::move(student));
  }
  // 4.2 Create table instance and populate data
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(students);

  // 5. Execute function
  try {
    auto result_table = f(ctx, table.get());
    // 5.1 Fetch columns
    auto result_scores = result_table->Get<float>("score").value();
    auto result_names = result_table->Get<StringView>("name").value();
    auto result_ages = result_table->Get<int32_t>("age").value();

    for (size_t i = 0; i < result_scores.Size(); i++) {
      RUDF_INFO("name:{},score:{},age:{}", result_names[i], result_scores[i], result_ages[i]);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }
  return 0;
};
```

### Compilation Cache
**RapidUDF** incorporates an LRU cache with keys as the string of expressions/UDFs. Users can use `eval_function`/`eval_expression` to compile (or cache-hit) and execute in one call, avoiding parse/compile overhead each time:
```cpp
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == 123;
    }
  )";
  // eval_function compiles on first call, then caches for subsequent calls
  auto rc = exec::eval_function<bool, const JsonObject&>(content, json);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
    return -1;
  }
  bool v = rc.value();  // true

  // Second call hits the cache, no re-compilation
  rc = exec::eval_function<bool, const JsonObject&>(content, json);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
    return -1;
  }
  v = rc.value();  // true (from cache)
  return 0;
}
```

### Compiler Options
**RapidUDF** supports configurable compilation options via the `Options` struct:
```cpp
rapidudf::Options opts;
opts.optimize_level = 2;              // LLVM optimization level (0-3, default 2)
opts.fast_math = false;               // Enable fast-math optimizations
opts.print_asm = false;               // Print generated assembly
opts.skip_auto_vectorize_passes = true; // Skip LLVM auto-vectorization (default true, UDF uses hand-written SIMD)
rapidudf::JitCompiler compiler(opts);
```

### Multi-Function Compilation
**RapidUDF** supports compiling source containing multiple function definitions, then loading individual functions by name:
```cpp
std::string source = R"(
  int add(int a, int b) { return a + b; }
  int mul(int a, int b) { return a * b; }
)";
rapidudf::JitCompiler compiler;
auto result = compiler.CompileSource(source);
if (!result.ok()) { ... }

// Load individual functions by name
auto add_func = compiler.LoadFunction<int, int, int>("add");
auto mul_func = compiler.LoadFunction<int, int, int>("mul");
```

### More Examples and Usage
- [Using Custom C++ Classes in Expressions/UDFs](docs/ffi.md)
- [Using Member Functions of Custom C++ Classes in Expressions/UDFs](docs/ffi.md)
- [Using Protobuf Objects in Expressions/UDFs](docs/ffi.md)
- [Using FlatBuffers Objects in Expressions/UDFs](docs/ffi.md)
- [Using STL Objects in Expressions/UDFs](docs/ffi.md)

There are more examples for different scenarios in the [examples](rapidudf/examples/) and [tests](rapidudf/tests/) code directories.

## Performance

### Comparison with Native C++
Since RapidUDF is based on LLVM Jit, it theoretically can achieve performance very close to native C++ code. Comparison results for compiling the Fibonacci method with `O0`:
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22547 ns        22547 ns        31060
BM_native_fib_func        38933 ns        38933 ns        17964
```
Fibonacci method GCC `O2` compilation comparison results:
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22557 ns        22555 ns        31065
BM_native_fib_func        19246 ns        19239 ns        36395
```
Note: The Jit implementation currently uses the same jit compilation logic under `O0/O2` compilation switches; theoretically, the generated code should be identical.

### Vectorized Acceleration Scenarios
The following tests were run on a CPU that supports `AVX2`, with the compilation optimization flag `O2`, and an array length of `4099`.
#### Complex Trigonometric Expression
The calculation is to execute the double array `x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y`; theoretically, the acceleration ratio should be the multiple of the `AVX2` register width to the `double` width, which is `4`.
Actual results are as follows, showing that the acceleration ratio has exceeded `4` times, reaching **6.09**:
```
Benchmark                               Time             CPU   Iterations
-------------------------------------------------------------------------
BM_rapidudf_expr_func              207713 ns       207648 ns         3362
BM_rapidudf_vector_expr_func        33962 ns        33962 ns        20594
BM_native_func                     207145 ns       207136 ns         3387
```
#### Wilson Ctr
Original function prototype:
```cpp
float  wilson_ctr(float exp_cnt, float clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * std::sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
}
```

Corresponding vector UDF script implementation:
```cpp
    simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> exp_cnt, simd_vector<f32> clk_cnt)
    {
       return log10(exp_cnt) *
         (clk_cnt / exp_cnt +  1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
    }
```
Theoretically, the acceleration ratio should be the multiple of the `AVX2` register width to the float width, which is `8`;
Actual results are as follows, showing that the acceleration ratio has exceeded `8` times, reaching **10.5**:
```
Benchmark                               Time             CPU   Iterations
-------------------------------------------------------------------------
BM_native_wilson_ctr                69961 ns        69957 ns      9960
BM_rapidudf_vector_wilson_ctr       6661 ns         6659 ns       105270
```


## Dependencies

- [LLVM](https://llvm.org/) 18+
- [highway](https://github.com/google/highway) 1.4.0
- [x86-simd-sort](https://github.com/intel/x86-simd-sort)
- [sleef](https://github.com/shibatch/sleef) 3.9.0
- [fmtlib](https://github.com/fmtlib/fmt) 11.0.2
- [spdlog](https://github.com/gabime/spdlog) 1.14.1
- [abseil-cpp](https://github.com/abseil/abseil-cpp) 20250512.2
- boost
  - [preprocessor](http://boost.org/libs/preprocessor)
  - [parser](https://github.com/tzlaine/parser)
- [protobuf](https://github.com/protocolbuffers) 3.19.2
- [flatbuffers](https://github.com/google/flatbuffers) 2.0.5
- [json](https://github.com/nlohmann/json) 3.11.3
