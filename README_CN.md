<h1 align="center">RapidUDF</h1>

**RapidUDF**是一个针对在线系统设计的高性能SIMD向量化表达式/脚本计算执行引擎库, 使用者可以在类如规则引擎/存储系统/特征计算等需要高性能以及灵活多变的场景使用。

## 特性

- **易于使用**: 
  - 提供常规表达式语法支持
  - 针对较复杂逻辑, 提供类C的DSL支持，包含**if-elif*-else** 条件控制，**while**循环控制，**auto**临时变量等能力；
  - 针对列式内存数据（`vector<T>`）,提供类spark的DataFrame的动态Table API以及 `filter/order_by/topk/take`等操作;
- **高性能**: 
  - 基于LLVM JIT编译，启动和执行性能相当于native cpp实现；
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
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
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

### Vector计算
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

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  // 'rapidudf::Context' 是在simd 实现中必须携带的参数，涉及arena内存分配
  auto result =
      compiler.CompileFunction<simd::Vector<float>, rapidudf::Context&, simd::Vector<StringView>, simd::Vector<float>>(
          source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 测试数据， 需要将原始数据转成列式数据
  std::vector<float> scores;
  std::vector<std::string> locations;
  for (size_t i = 0; i < 4096; i++) {
    scores.emplace_back(1.1 + i);
    locations.emplace_back(i % 3 == 0 ? "home" : "other");
  }

  // 5. 执行function
  rapidudf::Context ctx;
  auto f = std::move(result.value());
  auto new_scores = f(ctx, ctx.NewSimdVector(locations), ctx.NewSimdVector(scores));
  for (size_t i = 0; i < new_scores.Size(); i++) {
    // RUDF_INFO("{}", new_scores[i]);
  }
  return 0;
};
```

### 动态Vector Table
**RapidUDF**支持动态创建vector table, 在expression/UDFs里可以针对table的column进行任意计算操作(经过simd加速)；  
table类也提供一些类Spark DataFrame的操作，如：
- `.filter(simd::Vector<Bit>)`   返回按条件过滤后的新table实例
- `.order_by(simd::Vector<T> column, bool descending)`   返回按条件排序后的新table实例
- `.topk(simd::Vector<T> column, uint32_t k, bool descending)`    返回topk后的新table实例
```cpp
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 1. 创建table schema
  auto schema = simd::TableSchema::GetOrCreate("Student", [](simd::TableSchema* s) {
    std::ignore = s->AddColumn<StringView>("name");
    std::ignore = s->AddColumn<uint16_t>("age");
    std::ignore = s->AddColumn<float>("score");
    std::ignore = s->AddColumn<Bit>("gender");
  });

  // 2. UDF string, table<TABLE_NAME> 泛型格式中TABLE_NAME需要为之前创建的table schema name
  // table 支持 filter/order_by/topk/take等操作
  std::string source = R"(
    table<Student> select_students(Context ctx, table<Student> x) 
    { 
       auto filtered = x.filter(x.score >90 && x.age<10);
       // 按score降序排列取top10
       return filtered.topk(filtered.score,10,true); 
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<simd::Table*, Context&, simd::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 测试数据， 需要将原始数据转成列式数据
  std::vector<float> scores;
  std::vector<std::string> names;
  std::vector<uint16_t> ages;
  std::vector<bool> genders;

  for (size_t i = 0; i < 128; i++) {
    float score = (i + 1) % 150;
    scores.emplace_back(score);
    names.emplace_back("test_" + std::to_string(i));
    ages.emplace_back(i % 5 + 8);
    genders.emplace_back(i % 2 == 0 ? true : false);
  }
  // 4.2创建table实例
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("score", scores);
  std::ignore = table->Set("name", names);
  std::ignore = table->Set("age", ages);
  std::ignore = table->Set("gender", genders);

  // 5. 执行function
  auto result_table = f(ctx, table.get());
  auto result_scores = result_table->Get<float>("score").value();
  auto result_names = result_table->Get<StringView>("name").value();
  auto result_ages = result_table->Get<uint16_t>("age").value();
  auto result_genders = result_table->Get<Bit>("gender").value();
  for (size_t i = 0; i < result_scores.Size(); i++) {
    RUDF_INFO("name:{},score:{},age:{},gender:{}", result_names[i], result_scores[i], result_ages[i],
              result_genders[i] ? true : false);
  }
  return 0;
};
```

### 基于Protobuf/Flatbuffers的动态Vector Table
**RapidUDF**也可以从Protobuf/Flatbuffers创建table，避免繁琐的`TableSchema`创建过程；构建table实例也可以从Protobuf数组`std::vector<T>` `std::vector<const T*>` `std::vector<T*>` 直接构建；   
以下是基于Protobuf构建vector table样例， 基于flatbuffers的样例可参考[fbs_vector_table_udf](rapidudf/examples/fbs_vector_table_udf.cc)
```cpp
#include "rapidudf/examples/student.pb.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 1. 创建table schema
  auto schema = simd::TableSchema::GetOrCreate(
      "Student", [](simd::TableSchema* s) { std::ignore = s->BuildFromProtobuf<examples::Student>(); });

  // 2. UDF string
  std::string source = R"(
    table<Student> select_students(Context ctx, table<Student> x) 
    { 
       auto filtered = x.filter(x.score >90 && x.age<10);
       // 降序排列
       return filtered.topk(filtered.score,10, true); 
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<simd::Table*, Context&, simd::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 测试数据
  std::vector<examples::Student> students;
  for (size_t i = 0; i < 150; i++) {
    examples::Student student;
    student.set_score((i + 1) % 150);
    student.set_name("test_" + std::to_string(i));
    student.set_age(i % 5 + 8);
    students.emplace_back(std::move(student));
  }
  // 4.2创建table实例并填充数据
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->BuildFromProtobufVector(students);

  // 5. 执行function
  auto result_table = f(ctx, table.get());
  // 5.1 获取列
  auto result_scores = result_table->Get<float>("score").value();
  auto result_names = result_table->Get<StringView>("name").value();
  auto result_ages = result_table->Get<int32_t>("age").value();

  for (size_t i = 0; i < result_scores.Size(); i++) {
    RUDF_INFO("name:{},score:{},age:{}", result_names[i], result_scores[i], result_ages[i]);
  }
  return 0;
};
```

### 编译Cache
**RapidUDF**内置一个lru cache, key为expression/UDFs的字符串； 使用者可以通过cache获取编译的JitFunction对象，避免每次使用时parse/compile开销；  
```cpp
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == 123;
    }
  )";
  auto rc = GlobalJitCompiler::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));
  ASSERT_FALSE(f.IsFromCache());  // 第一次编译

  rc = GlobalJitCompiler::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  f = std::move(rc.value());
  ASSERT_TRUE(f(json));
  ASSERT_TRUE(f.IsFromCache());  //后续从cache中获取

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
BM_rapidudf_fib_func      22557 ns        22555 ns        31065
BM_native_fib_func        19246 ns        19239 ns        36395
```
注意：Jit实现目前在O0/O2编译下执行相同的jit编译逻辑，理论上生成的代码一致；

### 向量化加速计算场景
以下测试在支持avx2的cpu上运行，编译优化开关`-O2`，数组长度为`4099`;
#### 复杂三角函数表达式
计算为执行double数组的`x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y`; 理论上加速比应该为avx2寄存器位宽对于double位宽的倍数4;    
实际运行结果如下，可以看到加速比已经超过了4倍，达到了`6.09` 
```
Benchmark                               Time             CPU   Iterations
-------------------------------------------------------------------------
BM_rapidudf_expr_func              207713 ns       207648 ns         3362
BM_rapidudf_vector_expr_func        33962 ns        33962 ns        20594
BM_native_func                     207145 ns       207136 ns         3387
```
注：rapidudf_expr/native_func都是非向量化的循环调用实现

#### Wilson Ctr
原始函数原型为：
```cpp
float  wilson_ctr(float exp_cnt, float clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * std::sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
}
```
对应的vector udf脚本实现:
```cpp
    simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> exp_cnt, simd_vector<f32> clk_cnt)
    {
       return log10(exp_cnt) *
         (clk_cnt / exp_cnt +  1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
    }
```
理论上加速比应该为avx2寄存器位宽对于float位宽的倍数8;    
实际运行结果如下，可以看到加速比也已经超过了8倍，达到了`10.5` 
```
Benchmark                               Time             CPU   Iterations
-------------------------------------------------------------------------
BM_native_wilson_ctr                69961 ns        69957 ns         9960
BM_rapidudf_vector_wilson_ctr       6661 ns         6659 ns       105270
```
注：native_wilson_ctr是非向量化的循环调用实现


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