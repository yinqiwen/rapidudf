# RapidUDF
`RapidUDF`是一个C++的向量化表达式/脚本计算执行引擎, 主要基于`向量化`和`JIT编译`技术，相较于类似实现有较大的性能提升；   
使用者可以在类似规则引擎/存储UDF/特征计算等需要高性能+复杂灵活需求的在线计算场景使用；


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

## 限制
- C++17

## 特性
- `JIT`编译执行表达式/UDFs（类C的function DSL）
- `线程安全`
  - 编译生成的是无状态函数指针，可以重复调用
- [内置Operators](docs/operator.md)
   - 数学运算符(`+`, `-`, `*`, `/`, `%`, `^`)
   - 比较运算符(`>`, `>=`, `<`, `<=`, `!=`, `==`)
   - 逻辑运算符(`&&`, `||`)
   - 赋值运算符(`=`, `+=`, `-=`, `*=`, `/=`, `%=`)
   - 条件运算符(`?:`)
   - 非运算符(`!`)
   - 访问运算符(`.` `[]`)
- 条件控制(`if-elif*-else`)
- 循环控制(`while` `continue` `break`)
- 支持数据类型
  - 基本类型(`bit/bool` `u8`,`u16`,`u32`,`u64`,`f32`,`f64`,`i8`,`i16`,`i32`,`i64`,`string_view`)
  - `向量化vector`(`simd_vector<u64>`,`simd_vector<f64>`,....)
  - `Protobuf`(需要FFI编译期绑定)
  - `Flatbuffers`(需要FFI编译期绑定)
  - `STL容器`(`string`,`vector`,`set`,`map`,`unordered_map`,`unordered_set`)
  - 任意c++ class(需要FFI编译期绑定)
- `向量化加速`
  - 基于`highway`生成向量化加速代码(数学计算，比较等)
  - 基于`x86-simd-sort`提供sort向量化加速实现
- `FFI`
  - 在expression/UDFs中使用C++数据结构
  - 在expression/UDFs中调用C++方法
    - 普通C风格方法 `func(arg0,arg1)`
    - 对象方法 `obj.func(arg0,arg1)`
- [内置函数实现](docs/builtin_function.md)
  - C math lib functions
- 变量
  - 参数变量（通过生成的Jit Fuction参数暴露给使用者）
  - 临时变量（UDFs中定义的临时变量，用于保存中间结果） 
- 注释(支持类c++的 `//`注释)


## 安装
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

### vector向量化加速
```cpp
#include "rapidudf/rapidudf.h"

struct User {
  std::string city;
};
RUDF_STRUCT_FIELDS(User, city)  // 绑定User类，可在UDF里访问city字段

struct Feed {
  std::string city;
  float score;
};
struct Feeds {
  rapidudf::simd::Vector<rapidudf::StringView> city;
  rapidudf::simd::Vector<float> score;
};
RUDF_STRUCT_FIELDS(Feeds, city, score)  // 绑定Feeds类，可在UDF里访问city/score字段

int main() {
  // 2. UDF string
  std::string source = R"(
    void boost_scores(Context ctx, User user,Feeds feeds) 
    { 
      // 注意boost是个float数组
      var boost=(feeds.city==user.city?2.0_f32:1.1_f32);
      feeds.score*=boost;
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  // 'rapidudf::Context' 是在simd 实现中必须的参数，涉及arena内存分配
  auto result = compiler.CompileFunction<void, rapidudf::Context&, const User&, Feeds&>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 测试数据， 需要将原始数据转成列式数据
  User user;
  user.city = "sz";
  std::vector<Feed> feeds;
  for (size_t i = 0; i < 1024; i++) {
    Feed feed;
    feed.city = (i % 2 == 0 ? "sz" : "bj");
    feed.score = i + 1.1;
    feeds.emplace_back(feed);
  }

  // 4.2 将原始数据转成列式数据
  std::vector<rapidudf::StringView> citys;
  std::vector<float> scores;
  for (auto& feed : feeds) {
    citys.emplace_back(feed.city);
    scores.emplace_back(feed.score);
  }
  Feeds column_feeds;
  column_feeds.city = citys;
  column_feeds.score = scores;

  // 5. 执行function
  rapidudf::Context ctx;
  rapidudf::JitFunction<void, rapidudf::Context&, const User&, Feeds&> f = std::move(result.value());
  f(ctx, user, column_feeds);
  for (size_t i = 0; i < column_feeds.score.Size(); i++) {
    RUDF_INFO("{} {}/{}", citys[i], scores[i], column_feeds.score[i]);
  }

  return 0;
};
```

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

### 向量化计算场景(avx2)
计算为执行double数组的`x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y`，编译为加上`-O2`优化，对应的simd vector操作代码是在avx2指令集下运行                      
注：rapidudf_expr/exprtk/native都是非向量化的循环调用实现

```
Benchmark                             Time             CPU   Iterations
---------------------------------------------------------------
BM_rapidudf_expr_func             51290 ns        51290 ns        13684
BM_rapidudf_vector_expr_func       8395 ns         8395 ns        83350
BM_exprtk_expr_func               65001 ns        65001 ns        10869
BM_native_func                    50161 ns        50160 ns        13953
```

### 与lua/wasm等比较
RapidUDF 对比lua/wasm等第三方语言实现，性能上的最大优势在于数据传递/函数调用无需拷贝/转换的开销； 
编译生成的UDF可以按C++内存模型理解访问c++上定义的对象，包括不限与:
- Protobuf
- Flatbufers
- Json
- STL容器
- 自定义的C++类