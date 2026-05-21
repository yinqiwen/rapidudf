# AST DSL Syntax Design Review

## Overview

RapidUDF uses a hand-written recursive descent parser (Pratt-style precedence climbing) to compile a C-like DSL into LLVM IR via JIT. The parser lives in `rapidudf/ast/` and produces an arena-allocated AST.

## Current Syntax Summary

### Entry Modes

- **Expression mode**: `parse_expression_ast()` — compiles a standalone expression into a JIT function
- **Function mode**: `parse_function_ast()` / `parse_functions_ast()` — compiles one or more typed function definitions

### Grammar

```
program         = { func }
func            = type_name identifier '(' [ func_arg { ',' func_arg } ] ')' block
func_arg        = type_name identifier
block           = '{' statements '}'
statement       = 'continue' ';' | 'break' ';' | return_stmt | while_stmt | ifelse_stmt | expr_stmt
return_stmt     = 'return' [ expression ] ';'
expr_stmt       = expression ';'
while_stmt      = 'while' choice_stmt
ifelse_stmt     = 'if' choice_stmt { 'elif' choice_stmt } [ 'else' '{' statements '}' ]
choice_stmt     = '(' expression ')' '{' statements '}'
expression      = assign
assign          = logic_expr [ '?' expression ':' expression ] [ assign_op expression ]
logic_expr      = cmp_expr { logic_op cmp_expr }
cmp_expr        = additive_expr { cmp_op additive_expr }
additive_expr   = multiplicative_expr { add_op multiplicative_expr }
multiplicative_expr = power_expr { mul_op power_expr }
power_expr      = unary_expr { '^' power_expr }    (* right-associative *)
unary_expr      = [ unary_op ] operand
operand         = constant_number | bool | string | 'auto' identifier | '(' expression ')' | array | var_accessor
var_accessor    = identifier [ member_access | func_invoke_args ]
member_access   = ( field_access | dynamic_param_access )+
field_access    = '.' identifier [ func_invoke_args ]
dynamic_param_access = '[' ( string | number | identifier ) ']'
func_invoke_args = '(' [ expression { ',' expression } ] ')'
array           = '[' expression { ',' expression } ']'
type_name       = identifier [ '<' { any_token_except_gt } '>' ]
```

### Operator Precedence (lowest to highest)

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1. Assignment | `=` `+=` `-=` `*=` `/=` `%=` | Right |
| 2. Ternary | `? :` | Right |
| 3. Logic | `||` `&&` | Left |
| 4. Comparison | `==` `!=` `<` `<=` `>` `>=` | Left |
| 5. Additive | `+` `-` | Left |
| 6. Multiplicative | `*` `/` `%` | Left |
| 7. Power | `^` | Right |
| 8. Unary | `-` `!` | Prefix |

### Built-in Functions (as operators)

- **Unary math**: `sqrt`, `cbrt`, `floor`, `ceil`, `round`, `rint`, `trunc`, `erf`, `erfc`, `abs`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atanh`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `approx_recip`, `approx_recip_sqrt`
- **Binary math**: `atan2`, `max`, `min`, `pow`, `hypot`, `abs_diff`
- **Ternary**: `clamp`, `fma`, `fms`, `fnma`, `fnms`
- **Vector/misc**: `dot`, `cos_distance`, `l2_distance`, `iota`, `sort`, `select`, `topk`, `sum`, `avg`, `clone`, `filter`, `gather`

### Keywords

`return`, `if`, `elif`, `else`, `while`, `auto`, `break`, `continue`

### Type System

| Type | FundamentalType |
|------|----------------|
| `void` | DATA_VOID |
| `bool` | DATA_BIT |
| `u8` / `i8` / `u16` / `i16` / `u32` / `i32` / `u64` / `i64` | Integer types |
| `f16` / `f32` / `f64` / `f80` | Float types |
| `int` | Alias for `i32` |
| `long` | Alias for `i64` |
| `float` | Alias for `f32` |
| `double` | Alias for `f64` |
| `string` / `string_view` / `std_string_view` | String types |
| `json` | JSON pointer |
| `Context` | Runtime context pointer |
| `simd_vector<T>` | SIMD vector of T |
| `dyn_obj<schema>` | Dynamic object |
| `table<schema>` | Columnar table |

Number literals support type suffixes: `1.0_f32`, `42_i64`, `0_u8`.

### AST Node Types

- **Expression**: `BinaryExpr` (left + vector of `(OpToken, Operand)` pairs), `UnaryExpr`, `SelectExpr` (ternary)
- **Operand**: `bool`, `ConstantNumber`, `string`, `VarAccessor`, `SelectExpr`, `BinaryExpr`, `UnaryExpr`, `VarDefine`, `Array`
- **Statement**: `ReturnStatement`, `ExpressionStatement`, `WhileStatement`, `IfElseStatement`, `ContinueStatement`, `BreakStatement`
- **Function**: `Function` with return type, name, args, body block
- **Memory**: Arena-allocated via `AstPool` with `AstPtr<T>` borrowed-reference smart pointers

## Issues & Improvement Suggestions

### 1. `^` for Power is Confusing for C/C++ Users

**Severity: High** | **Effort: Medium**

`^` means XOR in C/C++/Java/JS. Using it for power will trip up every C++ developer who writes `a ^ b` expecting bitwise XOR.

**Suggestion:** Use `**` for power (Python/Ruby/Julia convention) and leave `^` for future bitwise XOR. Alternatively, deprecate `^` and keep only `pow(a, b)` function syntax (already supported).

### 2. No `for` Loop

**Severity: High** | **Effort: Medium**

Only `while` is available. Counted loops require awkward manual init/test/increment:

```
int i = 0;
while(i < n){
    ...
    i = i + 1;
}
```

**Suggestion:** Add C-style `for`:
```
for(int i = 0; i < n; i += 1){
    ...
}
```
This doesn't require `++`/`--` operators — compound assignment (`i += 1`) works. Parser change: add `TOKEN_FOR`, parse `for` `(` init `;` cond `;` update `)` block, desugar to `while` in AST or add `ForStatement` node.

### 3. No Typed Variable Declarations

**Severity: Medium** | **Effort: Low**

Only `auto x = expr;` is supported. Can't write `int x;` or `f32 y = 0;` for explicit typing. Forces type inference everywhere and makes it impossible to declare without initializing.

**Suggestion:** Allow `type_name identifier ['=' expression] ';'` as a statement:
```
int x = 0;
f32 y;
y = compute(z);
```
Parser change: in `parse_statements()`, add lookahead — when a type keyword is followed by an identifier, parse as a typed declaration instead of an expression.

### 4. `elif` vs `else if`

**Severity: Medium** | **Effort: Low**

`elif` is Python-flavored. Every C/C++/Java/JS developer will instinctively write `else if` and get a parse error.

**Suggestion:** Support `else if` as an alternative. In the parser, after consuming `else`, peek for `{` — if it's `{`, it's the else block; if it's `if`, it's an elif chain. This is how C handles it and requires no new token.

### 5. No Bitwise Operators

**Severity: Medium** | **Effort: Medium**

`&`, `|`, `~`, `<<`, `>>` are missing. For a data processing engine, bitwise ops are essential for masking, flag manipulation, and packed-bit work.

**Suggestion:** Add `TOKEN_AMP` (`&`), `TOKEN_PIPE` (`|`), `TOKEN_TILDE` (`~`), `TOKEN_LSHIFT` (`<<`), `TOKEN_RSHIFT` (`>>`). Place between comparison and additive precedence (standard C precedence).

### 6. No String Concatenation

**Severity: Medium** | **Effort: Low**

Strings support `==` comparison and `.contains()` / `.size()` member functions, but no way to concatenate strings or do interpolation.

**Suggestion:** Either add `+` for string concatenation (Python/JS convention) or template literals like `` `hello ${name}` ``. The `+` approach is simpler — check if either operand is a string at codegen time.

### 7. Error Messages Include Full Source

**Severity: Low** | **Effort: Low**

In `grammar.cc:871`, the entire source text is embedded in the error message:
```cpp
return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, err_detail));
```
For long expressions this produces unusable output.

**Suggestion:** Include a snippet around the error position with line/column info instead of the full source.

### 8. No `const` / Immutability

**Severity: Low** | **Effort: Medium**

All variables are mutable. Being able to mark bindings as immutable (`const f32 pi = 3.14159;`) prevents accidental mutation and enables optimizations.

### 9. Trailing Semicolons

**Severity: Low** | **Effort: Low**

Every statement requires `;`, including the last before `}`. Consider allowing optional trailing semicolons for the last statement in a block — a common source of parse errors for new users.

### 10. No `++`/`--` (Minor)

Not strictly necessary since `i += 1` works, but `i++` and `i--` are very common idioms. Low priority since compound assignment covers the use case.

## Priority Summary

| Issue | Severity | Effort | Impact |
|-------|----------|--------|--------|
| `^` means power not XOR | High | Medium | Prevents future XOR, confuses C++ users |
| No `for` loop | High | Medium | Verbose iteration code |
| No typed var decl | Medium | Low | Forces `auto` everywhere |
| `elif` not `else if` | Medium | Low | Familiarity barrier for C++ users |
| No bitwise ops | Medium | Medium | Limits low-level data manipulation |
| No string concat | Medium | Low | Can't build strings inline |
| Error messages verbose | Low | Low | Poor developer experience |
| No `const` | Low | Medium | No immutability guarantees |
| Trailing semicolons | Low | Low | Minor friction for new users |

The highest-value changes are `for` loops, `else if` support, and `**` for power (freeing `^` for XOR). These improve familiarity for C++ users with relatively small parser changes.
