# Built-in Functions

## Builtin C Functions

|Function        | Ints  | Floats | StringView  |Bool/U8| simd_vector\<Ints\> |simd_vector\<Floats\>  |simd_vector\<Bit\> |simd_vector\<StringView\>|
|------    | ----- | ------ | ------------|-------| --------------------|-------------          |--------           |-------------------------|
|abs(x)    | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|max(x,y)  | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|min(x,y)  | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|pow(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|ceil(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|erf(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|erfc(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|exp(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|expm1(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|exp2(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|floor(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|sqrt(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|log(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|log2(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|log10(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|log1p(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|sin(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|cos(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|tan(x)    | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|asin(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|acos(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|atan(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|sinh(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|cosh(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|tanh(x)   | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|asinh(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|acosh(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|atanh(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|atan2(x)  | N     | Y      | N           | N     | N                   | Y                     |N                  |N
|hypot(x,y)| N     | Y      | N           | N     | N                   | Y                     |N                  |N
|dot(x,y)  | N     | N      | N           | N     | N                   | Y                     |N                  |N
|iota(S,N) | N     | N      | N           | N     | Y                   | Y                     |N                  |N
|clamp(x,y,z) | Y     | Y      | Y           | Y     | Y                   | Y                    |N                  |N
|fma(x,y,z) | N    |Y       | N           | N     | Y                   | Y                    |N                  |N
|fms(x,y,z) | N    |N       | N           | N     | Y                   | Y                    |N                  |N
|fnma(x,y,z) | N    |N       | N           | N     | Y                   | Y                    |N                  |N
|fnms(x,y,z) | N    |N       | N           | N     | Y                   | Y                    |N                  |N

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


