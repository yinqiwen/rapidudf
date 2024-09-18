# FFI
<!-- TOC -->

- [FFI](#ffi)
    - [Use c/c++ struct/class fields in UDFs](#use-cc-structclass-fields-in-udfs)
        - [Declare all fields used in expression/UDFs](#declare-all-fields-used-in-expressionudfs)
        - [Use the dot. operator to access fields](#use-the-dot-operator-to-access-fields)
    - [Use c/c++ struct/class member functions in expression/UDFs](#use-cc-structclass-member-functions-in-expressionudfs)
        - [Declare all member functions used in expression/UDFs](#declare-all-member-functions-used-in-expressionudfs)
        - [Use the dot. operator to access method functions in expression/UDFs](#use-the-dot-operator-to-access-method-functions-in-expressionudfs)
    - [Use c/c++ functions in expression/UDFs](#use-cc-functions-in-expressionudfs)
        - [Register functions first](#register-functions-first)
        - [Use registed functions in expression/UDFs](#use-registed-functions-in-expressionudfs)
    - [Use STL vector/map/set/unordered_map/unordered_set in expression/UDFs](#use-stl-vectormapsetunordered_mapunordered_set-in-expressionudfs)
    - [Use Protobuf in UDFs](#use-protobuf-in-udfs)
        - [Declare all proto type&fields used in expression/UDFs](#declare-all-proto-typefields-used-in-expressionudfs)
        - [Use pb get/set functions in expression/UDFs](#use-pb-getset-functions-in-expressionudfs)
    - [Use FlatBuffers in UDFs](#use-flatbuffers-in-udfs)
        - [Declare all fbs type&fields used in expression/UDFs](#declare-all-fbs-typefields-used-in-expressionudfs)
        - [Use fbs functions in expression/UDFs](#use-fbs-functions-in-expressionudfs)

<!-- /TOC -->

## Use c/c++ struct/class fields in UDFs

### Declare all fields used in expression/UDFs
```cpp
#include "rapidudf/rapidudf.h"

struct TestInternal {
  int a = 0;
};
// declare all fields used in expression/UDFs
RUDF_STRUCT_FIELDS(TestInternal, a)
struct TestStruct {
  int a = 0;
  TestInternal internal;
  TestInternal* internal_ptr = nullptr;
  std::vector<int> vec;
};
// declare all fields used in expression/UDFs
RUDF_STRUCT_FIELDS(TestStruct, internal, internal_ptr, a, vec)
```

### Use the dot(`.`) operator to access fields
```cpp
JitCompiler compiler;
std::string source = "x.internal.a";
auto result = compiler.CompileExpression<int, const TestStruct&>(source, {"x"});
if (!result.ok()) {
    // handle error
}
auto f = std::move(result.value());
TestStruct test;
int a = f(test);

std::string source1 = "x.internal_ptr.a";
auto result1 = compiler.CompileExpression<int, const TestStruct&>(source1, {"x"});
if (!result1.ok()) {
    // handle error
}
auto f1 = std::move(result1.value());
TestInternal tmp;
test.internal_ptr=&tmp;  
int other_a = f1(test); 

```

## Use c/c++ struct/class member functions in expression/UDFs
### Declare all member functions used in expression/UDFs
```cpp
#include "rapidudf/rapidudf.h"

struct TestMethodStruct {
  int a;
  float b;
  int* p = nullptr;
  std::string_view c;
  int get_a() const { return a; }
  void set_a(int x) { this->a = x; }
};
// declare all member methods used in expression/UDFs
RUDF_STRUCT_MEMBER_METHODS(TestMethodStruct, get_a, set_a);
```

### Use the dot(`.`) operator to access method functions in expression/UDFs
```cpp
TestMethodStruct test;
rapidudf::JitCompiler compiler;
std::string source = R"(
    // "TestMethodStruct x" is  a pointer/reference of 'TestMethodStruct'  
    int test_func(TestMethodStruct x, int a){
      x.set_a(a);
      return x.get_a();
    }
  )";
auto result = compiler.CompileFunction<int, TestMethodStruct&, int>(source);
if (!result.ok()) {
    // handle error
}
auto f = std::move(result.value());
int a = f(test, 101); //return 101
```

## Use c/c++ functions in expression/UDFs

### Register functions first
```cpp
#include "rapidudf/rapidudf.h"

static int test_cpp_func(int x, int y) { return x * 100 + y; }
// use 'test_cpp_func' in expression/UDFs
RUDF_FUNC_REGISTER(test_cpp_func)

// use 'test_cpp_func111' in expression/UDFs
RUDF_FUNC_REGISTER_WITH_NAME("test_cpp_func111", test_cpp_func)
```

### Use registed functions in expression/UDFs
```cpp
rapidudf::JitCompiler compiler;
std::string source = R"(
    int test_func(int x, int y){
      return test_cpp_func(x, y);
    }
)";
auto rc = compiler.CompileFunction<int, int, int>(source);
auto f = std::move(rc.value());
f(2,3); // 203
```

## Use STL vector/map/set/unordered_map/unordered_set in expression/UDFs
All vector/map/set/unordered_map/unordered_set with type u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/string_view were registed.   
Users can use builtin methods in expression/UDFs
```cpp
std::vector<int> vec{1, 2, 3};
JitCompiler compiler;
auto result0 = compiler.CompileExpression<int, std::vector<int>&>("x.size()", {"x"});
auto result1 = compiler.CompileExpression<int, std::vector<int>&>("x[0]", {"x"});

std::map<std::string, std::string> map{{"t0", "v0"}, {"t1", "v1"}};
auto result2 = compiler.CompileExpression<int, std::map<std::string, std::string>&>("x.size()", {"x"});
auto result3 = compiler.CompileExpression<bool, std::map<std::string, std::string>&>(R"(x.contains("t1"))", {"x"});
auto result4 = compiler.CompileExpression<StringView, std::map<std::string, std::string>&>(R"(x["t1"])", {"x"});

```

## Use Protobuf in UDFs
```proto
syntax = "proto3";
package test;

message Item{
  int32 id = 1;
}

message PBStruct{
  string str = 1;
  int32  id = 2;
  repeated int64 ids_array = 3;
  repeated string strs_array = 4;
  repeated Item item_array = 5;
  map<string, Item> item_map = 6;
  map<string, int32> str_int_map = 7;
}
```
### Declare all proto type&fields used in expression/UDFs
```cpp
// include pb header
#include "rapidudf/rapidudf.h"

//declare all fields used in expression/UDFs
RUDF_PB_FIELDS(::test::Item, id)
RUDF_PB_FIELDS(::test::PBStruct, id, str, item_array, item_map, item_map, str_int_map)

// declare write methods used in expression/UDFs
RUDF_PB_SET_FIELDS(::test::PBStruct, id, str)
```

### Use pb get/set functions in expression/UDFs
```cpp
// include pb header
#include "rapidudf/rapidudf.h"

JitCompiler compiler;
// read repeated field      "x.item_array().get(0).id()"
// read map field str->int  "x.str_int_map().get("k1")"
// read map field str->Item "x.item_map().get(key).id()"
std::string source = "x.id()";
auto rc = compiler.CompileExpression<int, const test::PBStruct&>(source, {"x"});
auto f = std::move(rc.value());
::test::PBStruct pb;
pb.set_id(101);
f(pb); // 101


std::string content = R"(
int test_func(test::PBStruct x, int y){
    x.set_id(y);
    return x.id();
}
)";
auto result = compiler.CompileFunction<int, test::PBStruct&, int>(content);
auto f1 = std::move(result.value());
f1(pb, 101);  // return 101
```

## Use FlatBuffers in UDFs
```proto
namespace test_fbs;

table Item {
  id:uint;
}

table FBSStruct {
  id:uint;
  str:string;
  item:Item;
  strs:[string];
  items:[Item];
  ints:[uint];
}
root_type FBSStruct;
```
### Declare all fbs type&fields used in expression/UDFs
```cpp
// include fbs header
#include "rapidudf/rapidudf.h"

RUDF_STRUCT_MEMBER_METHODS(::test_fbs::Item, id)
RUDF_STRUCT_MEMBER_METHODS(::test_fbs::FBSStruct, id, str, item, strs, items, ints)
```

### Use fbs functions in expression/UDFs
```cpp
// include pb header
#include "rapidudf/rapidudf.h"

JitCompiler compiler;
// read repeated string element      "x.strs().get(0)"
// read repeated fbs struct element  "x.items().get(0).id()"
std::string source = "x.id()";
auto rc = compiler.CompileExpression<int, const test_fbs::FBSStruct*>(source, {"x"});
auto f = std::move(rc.value());
const test_fbs::FBSStruct* fbs_ptr =  ....; 
f(fbs_ptr); 
```
 