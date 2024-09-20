# Built-in Operators

## Binary Operators

|Op   | Description     | Ints  | Floats | StringView  |Bool/Bit| simd_vector\<Ints\> |simd_vector\<Floats\>  |simd_vector\<Bit\> |simd_vector\<StringView\>|
|-----|------------     | ----- | ------ | ------------|-------| --------------------|-------------          |--------           |-------------------------|
|+    | addition        | Y     | Y      | N           | N     | Y                   | Y                     |Y                  |N
|-    | subtraction     | Y     | Y      | N           | N     | Y                   | Y                     |Y                  |N
|*    | multiplication  | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|/    | division        | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|%    | mod             | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|^    | power           | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|>    | greater than    | Y     | Y      | Y           | N     | Y                   | Y                     |N                  |N
|<    | greater or equal| Y     | Y      | Y           | N     | Y                   | Y                     |N                  |Y
|>=   | less than       | Y     | Y      | Y           | N     | Y                   | Y                     |N                  |Y
|<=   | less or equal   | Y     | Y      | Y           | N     | Y                   | Y                     |N                  |Y
|==   | equal           | Y     | Y      | Y           | N     | Y                   | Y                     |N                  |Y
|!=   | not equal       | Y     | Y      | Y           | N     | Y                   | Y                     |N                  |Y
|&&   | logical and     | N     | N      | N           | Y     | N                   | N                     |Y                  |N
|\|\| | logical or      | N     | N      | N           | Y     | N                   | N                     |Y                  |N
|=    |assignement      | Y     | Y      | Y           | Y     | Y                   | Y                     |Y                  |Y
|+=   |add assign       | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|-=   |sub assign       | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|*=   |mul assign       | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|/=   |div assign       | Y     | Y      | N           | N     | Y                   | Y                     |N                  |N
|%=   |mod assign       | Y     | N      | N           | N     | Y                   | N                     |N                  |N

## Unary Operators

|Op   | Description     | Ints  | Floats | StringView  |Bool/Bit| simd_vector\<Ints\> |simd_vector\<Floats\>  |simd_vector\<Bit\> |simd_vector\<StringView\>|
|-----|------------     | ----- | ------ | ------------|-------| --------------------|-------------          |--------           |-------------------------|
|!    | not             | N     | N      | N           | Y     | N                   | N                     |Y                  |N

## Ternary Operators

|Op   | Description     | Ints  | Floats | StringView  |Bool/Bit| simd_vector\<Ints\> |simd_vector\<Floats\>  |simd_vector\<Bit\> |simd_vector\<StringView\>|
|-----|------------     | ----- | ------ | ------------|-------| --------------------|-------------          |--------           |-------------------------|
|?:   | if then else    | Y     | Y      | Y           | Y     | Y                   | Y                     |Y                  |Y

Note: the first `if` value MUST be `bool` or `simd_vector<Bit>`

## Brackets Operators

|Op   | Description                  | std::vector | std::map |std::unordered_map | json|
|-----|------------                  | ------------| ---------|------------------ | ----| 
|[]   | get one element in an object.| Y           | Y        | Y                 | Y   |

