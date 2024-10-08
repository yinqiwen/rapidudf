<h1 align="center">RapidUDF</h1>

**RapidUDF**是一个针对在线系统设计的高性能SIMD向量化表达式/脚本计算执行引擎库, 使用者可以在类如规则引擎/存储系统/特征计算等需要高性能以及灵活多变的场景使用。

## 特性

- **易于使用**: 
  - 提供常规表达式语法支持
  - 针对较复杂逻辑, 提供类C的DSL支持，包含**if-elif*-else** 条件控制，**while**循环控制，**var**临时变量等能力；
  - 针对列式内存数据（`vector<T>`）,提供类spark的DataFrame的动态Table/Column API以及 filter/order_by/topk/take等操作;
- **高性能**: 
  - 基于LLVM JIT编译，启动和执行性能相当于native cpp；
  - 针对列式内存数据（`vector<T>`）, 提供**SIMD向量化加速**实现
- **线程安全**: 无状态的JIT生成的C方法天然线程安全
- **FFI**:
  - 支持表达式/UDFs里零开销访问在C++中定义的类对象(自定义类/stl/protobuffers/flatbuffers/...)
  - 支持表达式/UDFs里零开销调用C++中定义的方法/类方法
- **丰富的内置数据类型，运算符和函数**:
  - [内置数据类型](docs/dtype.md)
  - [内置运算符](docs/operator.md)
  - [内置函数](docs/builtin_function.md)

## 编译与安装
编译需要**C++17**支持的编译器
### Bazel
在**WORKSPACE**中添加
```python
    git_repository(
        name = "rapidudf",
        remote = "https://git.woa.com/qiyingwang/rapidudf.git",
        commit = "...",
    )
    load("@rapidudf//:rapidudf.bzl", "rapidudf_workspace")
    rapidudf_workspace()
```
在相关代码编译规则BUILD中添加：
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

### CMake(todo)

## 用法一览

### 简单表达式
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. 如果需要, 可以设置rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);
  // 2. expression string
  std::string expression = "x >= 1 && y < 10";
  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行; 编译耗时一般在10ms-100ms之间;
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型；
  // 表达式使用的变量名需要作为参数名列表传入，否则编译失败
  auto result = compiler.CompileExpression<bool, int, int>(expression, {"x", "y"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  // 4. 执行function
  rapidudf::JitFunction<bool, int, int> f = std::move(result.value());
  bool v = f(2, 3);  // true
  v = f(0, 1);       // false
  return 0;
};
```

### 简单UDF
简单fibonacci函数 
```cpp
#include "rapidudf/rapidudf.h"

int main() {
  // 1. 如果需要, 可以设置rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);
  // 2. UDF string
  std::string source = R"(
    int fib(int n) 
    { 
       if (n <= 1){
         return n; 
       }
       // 支持cpp的//注释
       return fib(n - 1) + fib(n - 2);  //递归调用
    } 
  )";
  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行; 编译耗时一般在10ms-100ms之间;
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<int, int>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4. 执行function
  rapidudf::JitFunction<int, int> f = std::move(result.value());
  int n = 9;
  int x = f(n);  // 34
  RUDF_INFO("fib({}):{}", n, x);
  return 0;
};
```

### vector计算
```cpp
#include "rapidudf/rapidudf.h"
int main() {


  return 0;
};
```

### 类spark的DataFrame的动态Table/Column

### 更多的例子与用法
- [在expression/UDFs中使用自定义的c++类](docs/ffi.md)
- [在expression/UDFs中使用自定义的c++类成员方法](docs/ffi.md)
- [在expression/UDFs中使用protobuf对象](docs/ffi.md)
- [在expression/UDFs中使用flatbuffers对象](docs/ffi.md)
- [在expression/UDFs中使用stl对象](docs/ffi.md)


## 性能
 
### 与native cpp比较
由于RapidUDF实现基于LLVM Jit，理论上可以实现非常接近原生cpp代码性能；   
fibonacci方法O0编译对比结果
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22547 ns        22547 ns        31060
BM_native_fib_func        38933 ns        38933 ns        17964
```

fibonacci方法O2编译对比结果
```
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_fib_func      22552 ns        22552 ns        31041
BM_native_fib_func        19212 ns        19212 ns        36437
```
注意：Jit实现目前在O0/O2编译下执行相同的jit编译逻辑，理论上生成的代码时一致的；

### 向量化加速计算场景(avx2)
计算为执行float数组的`x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y`，编译为加上`-O2`优化，对应的simd vector操作代码是在avx2指令集下运行                      
注：rapidudf_expr/exprtk/native都是非向量化的循环调用实现

```
Benchmark                             Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_expr_func             51290 ns        51290 ns        13684
BM_rapidudf_vector_expr_func       8395 ns         8395 ns        83350
BM_exprtk_expr_func               65001 ns        65001 ns        10869
BM_native_func                    50161 ns        50160 ns        13953
```


## 依赖
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