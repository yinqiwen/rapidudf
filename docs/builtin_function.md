
# Built-in Functions

<!-- TOC -->

- [Built-in Functions](#built-in-functions)
    - [Builtin C Functions](#builtin-c-functions)
        - [abs](#abs)
        - [max](#max)
        - [min](#min)
        - [pow](#pow)
        - [ceil](#ceil)
        - [floor](#floor)
        - [round](#round)
        - [rint](#rint)
        - [trunc](#trunc)
        - [erf](#erf)
        - [erfc](#erfc)
        - [exp](#exp)
        - [expm1](#expm1)
        - [exp2](#exp2)
        - [sqrt](#sqrt)
        - [log](#log)
        - [log2](#log2)
        - [log10](#log10)
        - [log1p](#log1p)
        - [sin](#sin)
        - [sin](#sin)
        - [tan](#tan)
        - [asin](#asin)
        - [acos](#acos)
        - [atan](#atan)
        - [sinh](#sinh)
        - [cosh](#cosh)
        - [tanh](#tanh)
        - [asinh](#asinh)
        - [acosh](#acosh)
        - [atanh](#atanh)
        - [atan2](#atan2)
        - [hypot](#hypot)
        - [sum](#sum)
        - [dot](#dot)
        - [iota](#iota)
        - [clamp](#clamp)
        - [fma](#fma)
        - [fms](#fms)
        - [fnma](#fnma)
        - [fnms](#fnms)
        - [sort](#sort)
        - [select](#select)
        - [topk](#topk)
        - [argsort](#argsort)
        - [argselect](#argselect)
        - [sort_kv](#sort_kv)
        - [select_kv](#select_kv)
        - [topk_kv](#topk_kv)
    - [Builtin C++ Member Functions](#builtin-c-member-functions)
        - [StringView](#stringview)
        - [std::vector](#stdvector)
        - [std::map/std::unordered_map](#stdmapstdunordered_map)
        - [std::set/std::unordered_set](#stdsetstdunordered_set)
        - [google::protobuf::RepeatedField](#googleprotobufrepeatedfield)
        - [google::protobuf::RepeatedPtrField](#googleprotobufrepeatedptrfield)
        - [google::protobuf::Map](#googleprotobufmap)
        - [flatbuffers::Vector](#flatbuffersvector)

<!-- /TOC -->


## Builtin C Functions

### [`abs`](https://en.cppreference.com/w/cpp/numeric/math/abs)
#### Format
```cpp
abs(x)
```
#### Supported Parameter Types:
- `i32` `i64` `f32` `f64` 
- `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<int64_t, int64_t>("abs(x)", {"x"});
```

### [`max`](https://en.cppreference.com/w/cpp/algorithm/max)
#### Format
```cpp
max(x,y)
```
#### Supported Parameter Types:
- `i8` `i16` `i32` `i64` `u8` `u16` `u32` `u64` `f32` `f64` 
- `simd_vector<i8>` `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`
- `simd_vector<u8>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when size mismatch
#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<int64_t, int64_t>("max(x,y)", {"x", "y"});
```

### [`min`](https://en.cppreference.com/w/cpp/algorithm/min)
#### Format
```cpp
min(x,y)
```
#### Supported Parameter Types:
- `i8` `i16` `i32` `i64` `u8` `u16` `u32` `u64` `f32` `f64` 
- `simd_vector<i8>` `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`
- `simd_vector<u8>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<int64_t, int64_t>("min(x,y)", {"x", "y"});
```

### [`pow`](https://en.cppreference.com/w/cpp/numeric/math/pow)
#### Format
```cpp
pow(x,y)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("pow(x,y)", {"x", "y"});
```

### [`ceil`](https://en.cppreference.com/w/cpp/numeric/math/ceil)
#### Format
```cpp
ceil(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("ceil(x)", {"x"});
```


### [`floor`](https://en.cppreference.com/w/cpp/numeric/math/floor)
#### Format
```cpp
floor(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("floor(x)", {"x"});
```

### [`round`](https://en.cppreference.com/w/cpp/numeric/math/round)
#### Format
```cpp
round(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("round(x)", {"x"});
```

### [`rint`](https://en.cppreference.com/w/cpp/numeric/math/rint)
#### Format
```cpp
rint(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("rint(x)", {"x"});
```

### [`trunc`](https://en.cppreference.com/w/cpp/numeric/math/trunc)
#### Format
```cpp
trunc(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("trunc(x)", {"x"});
```

### [`erf`](https://en.cppreference.com/w/cpp/numeric/math/erf)
#### Format
```cpp
erf(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("erf(x)", {"x"});
```


### [`erfc`](https://en.cppreference.com/w/cpp/numeric/math/erfc)
#### Format
```cpp
erfc(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("erfc(x)", {"x"});
```

### [`exp`](https://en.cppreference.com/w/cpp/numeric/math/exp)
#### Format
```cpp
exp(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("exp(x)", {"x"});
```

### [`expm1`](https://en.cppreference.com/w/cpp/numeric/math/expm1)
#### Format
```cpp
expm1(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("expm1(x)", {"x"});
```

### [`exp2`](https://en.cppreference.com/w/cpp/numeric/math/exp2)
#### Format
```cpp
exp2(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("exp2(x)", {"x"});
```


### [`sqrt`](https://en.cppreference.com/w/cpp/numeric/math/sqrt)
#### Format
```cpp
sqrt(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("sqrt(x)", {"x"});
```


### [`log`](https://en.cppreference.com/w/cpp/numeric/math/log)
#### Format
```cpp
log(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("log(x)", {"x"});
```

### [`log2`](https://en.cppreference.com/w/cpp/numeric/math/log2)
#### Format
```cpp
log2(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("log2(x)", {"x"});
```

### [`log10`](https://en.cppreference.com/w/cpp/numeric/math/log10)
#### Format
```cpp
log10(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("log10(x)", {"x"});
```

### [`log1p`](https://en.cppreference.com/w/cpp/numeric/math/log1p)
#### Format
```cpp
log1p(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("log1p(x)", {"x"});
```

### [`sin`](https://en.cppreference.com/w/cpp/numeric/math/sin)
#### Format
```cpp
sin(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("sin(x)", {"x"});
```

### [`sin`](https://en.cppreference.com/w/cpp/numeric/math/cos)
#### Format
```cpp
cos(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("cos(x)", {"x"});
```

### [`tan`](https://en.cppreference.com/w/cpp/numeric/math/tan)
#### Format
```cpp
tan(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("tan(x)", {"x"});
```


### [`asin`](https://en.cppreference.com/w/cpp/numeric/math/asin)
#### Format
```cpp
asin(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("asin(x)", {"x"});
```

### [`acos`](https://en.cppreference.com/w/cpp/numeric/math/acos)
#### Format
```cpp
acos(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("acos(x)", {"x"});
```

### [`atan`](https://en.cppreference.com/w/cpp/numeric/math/atan)
#### Format
```cpp
atan(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("atan(x)", {"x"});
```

### [`sinh`](https://en.cppreference.com/w/cpp/numeric/math/sinh)
#### Format
```cpp
sinh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("sinh(x)", {"x"});
```


### [`cosh`](https://en.cppreference.com/w/cpp/numeric/math/cosh)
#### Format
```cpp
cosh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("cosh(x)", {"x"});
```

### [`tanh`](https://en.cppreference.com/w/cpp/numeric/math/tanh)
#### Format
```cpp
tanh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("tanh(x)", {"x"});
```


### [`asinh`](https://en.cppreference.com/w/cpp/numeric/math/asinh)
#### Format
```cpp
asinh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("asinh(x)", {"x"});
```

### [`acosh`](https://en.cppreference.com/w/cpp/numeric/math/acosh)
#### Format
```cpp
acosh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("acosh(x)", {"x"});
```

### [`atanh`](https://en.cppreference.com/w/cpp/numeric/math/atanh)
#### Format
```cpp
atanh(x)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float>("atanh(x)", {"x"});
```


### [`atan2`](https://en.cppreference.com/w/cpp/numeric/math/atan2)
#### Format
```cpp
atan2(x,y)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("atan2(x,y)", {"x", "y"});
```

### [`hypot`](https://en.cppreference.com/w/cpp/numeric/math/hypot)
#### Format
```cpp
hypot(x,y)
```
#### Supported Parameter Types:
- `f32` `f64` 
- `simd_vector<f32>` `simd_vector<f64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("hypot(x,y)", {"x", "y"});
```

### `sum`
#### Format
```cpp
sum(x)
```
#### Return Value
Returns the sum of all 
#### Supported Parameter Types:
- `simd_vector<i8>` `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`
- `simd_vector<u8>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, simd::Vector<float>>("sum(x)", {"x"});
```

### `dot`
#### Format
```cpp
dot(x,y )
```
#### Return Value
Returns sum{x[i] * y[i]} for floating-point inputs
#### Supported Parameter Types:
- `simd_vector<f32>` `simd_vector<f64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, simd::Vector<float>,simd::Vector<float>>("dot(x,y)", {"x", "y"});
```

### `iota`
#### Format
```cpp
iota(Start, N)
```
#### Return Value
Returns vector with size `N` where the index i has the given value of `Start+i`
#### Supported Parameter Types:
- `simd_vector<i8>` `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`
- `simd_vector<u8>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<simd::Vector<uint32_t>>("iota(1_u32,100)", {});
```

### [`clamp`](https://en.cppreference.com/w/cpp/algorithm/clamp)
#### Format
```cpp
clamp(x,y,z)
```
#### Supported Parameter Types:
- `i8` `i16` `i32` `i64` `u8` `u16` `u32` `u64` `f32` `f64` 
- `simd_vector<i8>` `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<f32>` `simd_vector<f64>`
- `simd_vector<u8>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("clamp(x,y,z)", {"x", "y", "z"});
```


### [`fma`](https://en.cppreference.com/w/cpp/numeric/math/fma)
#### Format
```cpp
fma(x,y,z)
```
#### Supported Parameter Types:
-  `f32` `f64` 
-  `simd_vector<f32>` `simd_vector<f64>`
#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch
#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("fma(x,y,z)", {"x", "y", "z"});
```

### `fms`
#### Format
```cpp
fms(a,b,c)
```
#### Return Value
a[i] * b[i] - c[i]

#### Supported Parameter Types:
-  `f32` `f64` 
-  `simd_vector<f32>` `simd_vector<f64>`
#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch
#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("fms(x,y,z)", {"x", "y", "z"});
```

### `fnma`
#### Format
```cpp
fnma(a,b,c)
```
#### Return Value
-a[i] * b[i] + c[i]
#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch
#### Supported Parameter Types:
-  `f32` `f64` 
-  `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("fnma(x,y,z)", {"x", "y", "z"});
```

### `fnms`
#### Format
```cpp
fnms(a,b,c)
```
#### Return Value
-a[i] * b[i] - c[i]
#### Throws
- throw `rapidudf::SizeMismatchException` when vector size mismatch
#### Supported Parameter Types:
-  `f32` `f64` 
-  `simd_vector<f32>` `simd_vector<f64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<float, float, float>("fnms(x,y,z)", {"x", "y", "z"});
```

### `sort`
#### Format
```cpp
sort(vec,descending)
```
#### Return Value
void
#### Throws
- throw `rapidudf::ReadonlyException` when vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<float>>("sort(x,true)", {"x"});
```

### `select`
#### Format
```cpp
select(vec,k, descending)
```
#### Return Value
void
#### Throws
- throw `rapidudf::ReadonlyException` when vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<float>>("select(x,5,true)", {"x"});
```

### `topk`
#### Format
```cpp
topk(vec,k, descending)
```
#### Return Value
void
#### Throws
- throw `rapidudf::ReadonlyException` when vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<float>>("topk(x,5,true)", {"x"});
```

### `argsort`
#### Format
```cpp
argsort(vec, descending)
```
#### Return Value
`simd::Vector<uint64_t>`
#### Throws
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<simd::Vector<uint64_t>, simd::Vector<float>>("argsort(x,true)", {"x"});
```

### `argselect`
#### Format
```cpp
argselect(vec, k, descending)
```
#### Return Value
`simd::Vector<uint64_t>`
#### Throws
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i16>` `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u16>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<simd::Vector<uint64_t>, simd::Vector<float>>("argselect(x,3, true)", {"x"});
```

### `sort_kv`
#### Format
```cpp
sort_kv(key_vec,value_vec, descending)
```
#### Return Value
`simd::Vector<uint64_t>`
#### Throws
- throw `rapidudf::ReadonlyException` when key/value vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<uint64_t>, simd::Vector<float>>("sort_kv(x,y, true)", {"x"});
```

### `select_kv`
#### Format
```cpp
select_kv(key_vec,value_vec,k,descending)
```
#### Return Value
`simd::Vector<uint64_t>`
#### Throws
- throw `rapidudf::ReadonlyException` when key/value vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<uint64_t>, simd::Vector<float>>("select_kv(x,y,10,true)", {"x"});
```

### `topk_kv`
#### Format
```cpp
topk_kv(key_vec,value_vec,k,descending)
```
#### Return Value
`simd::Vector<uint64_t>`
#### Throws
- throw `rapidudf::ReadonlyException` when key/value vector is readonly
#### Supported Parameter Types:
-  `simd_vector<f32>` `simd_vector<f64>`
-  `simd_vector<i32>` `simd_vector<i64>` `simd_vector<u32>` `simd_vector<u64>`

#### Examples
```cpp
JitCompiler compiler;
auto result = compiler.CompileExpression<void, simd::Vector<uint64_t>, simd::Vector<float>>("topk_kv(x,y,10,true)", {"x"});
```


## Builtin C++ Member Functions

## StringView
- `.size()`    return container size
- `.contains(part)` return true if part exist
- `.starts_with(part)` return true if part starts_with
- `.ends_with(part)` return true if part ends_with
- `.contains_ignore_case(part)` return true if part contains_ignore_case
- `.starts_with_ignore_case(part)` return true if part starts_with_ignore_case
- `.ends_with_ignore_case(part)` return true if part ends_with_ignore_case

## std::vector
- `.get(idx)`  get element by index
- `.size()`    return container size
- `.find(v)`   return element idx, return -1 if not found
- `.contains(v)` return true if element exist

## std::map/std::unordered_map
- `.get(key)`  get element by key, return default value if not exist
- `.size()`    return container size
- `.contains(key)` return true if key exist

## std::set/std::unordered_set
- `.size()`    return container size
- `.contains(key)` return true if key exist

## google::protobuf::RepeatedField
- `.get(idx)`  get element by index
- `.size()`    return container size
- `.find(v)`   return element idx, return -1 if not found
- `.contains(v)` return true if element exist

## google::protobuf::RepeatedPtrField
- `.get(idx)`  get element by index
- `.size()`    return container size

## google::protobuf::Map
- `.get(key)`  get element by key, return default value if not exist
- `.size()`    return container size
- `.contains(key)` return true if key exist

## flatbuffers::Vector
- `.get(idx)`  get element by index
- `.size()`    return container size

## Vector(`rapidudf::simd::Vector`)
- `.find(T v)`    return first position for given value, return -1 when failed
- `.find_neq(T v)`    return first value's position not equal given value, return -1 when failed
- `.find_gt(T v)`    return first value's position greater than given value, return -1 when failed
- `.find_ge(T v)`    return first value's position greater equal than given value, return -1 when failed
- `.find_lt(T v)`    return first value's position less than given value, return -1 when failed
- `.find_le(T v)`    return first value's position less equal than given value, return -1 when failed

## Vector Table
- `.filter(simd::Vector<Bit>)`   return new table after filter
- `.order_by(simd::Vector<T> column, bool descending)`   return new table after order_by
- `.topk(simd::Vector<T> column, uint32_t k, bool descending)`    return new table after topk
- `.group_by(simd::Vector<T> column)`    return tables after group_by


