# Data Types

## Scalar Types
- `u8`, `uint8_t` in c++
- `u16`, `uint16_t` in c++
- `u32`, `uint32_t` in c++
- `u64`, `uint64_t` in c++
- `i8`,  `int8_t` in c++
- `i16`, `int16_t` in c++
- `i32`, `int32_t` in c++
- `i64`, `int64_t` in c++
- `f32`, `float` in c++
- `f64`, `double` in c++
- `f80`, `long double` in c++
- `bit`, `bool` in c++
- `string_view`,  `rapidudf::StringView` in c++
- `string`, `std::string*` in c++
- `std_string_view`, `std::string_view` in c++
- `scalar`, `rapidudf::Scalar*` in c++, a dynamic scalar object with variant type

## Vector Types
- `simd_vector<T>`, `rapidudf::simd_vector<T>` in c++
- `simd_column`, a dynamic `simd_vector<T>` object with variant type
- `simd_table`, a table with many named `simd_column`s


## STL Types
- `vector<T>`, `std::vector<T>*` in c++
- `set<T>`, `std::set<T>*` in c++
- `map<K,V>`, `std::map<K,V>*` in c++
- `unordered_map<K,V>`, `std::unordered_map<K,V>*` in c++
- `unordered_set<K,V>`, `std::unordered_set<K,V>*` in c++

## Context Type
- `Context`, `rapidudf::Context*` in c++, context object reference


