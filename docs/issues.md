# SLEEF Integration Issues (rapidudf/functions/simd)

> Status: All P0/P1 fixes applied; `bazel build //...` and `bazel test //...`
> (28/28) pass. The historical findings below are preserved with a note on
> what was fixed and what was overturned during implementation.

## Design principle (established this round)

**SLEEF is invoked only for math functions Highway does not implement.**
Concretely:

| Op family | Highway has it? | Implementation |
|---|---|---|
| `sin`, `cos`, `sinh`, `tanh`, `atan`, `atanh`, `atan2`, `asin`, `acos`, `acosh`, `asinh`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `hypot`, `sqrt`, `floor`, `ceil`, `round`, `trunc`, `abs` | ✅ | Highway (`hwy/contrib/math/math-inl.h` + core) |
| `cosh`, `tan`, `pow`, `erf`, `erfc`, `rint` | ❌ | SLEEF |

Operations Highway already provides MUST NOT route through SLEEF. The earlier
asymmetric handling of `OP_LOG/LOG2/LOG10/LOG1P` (float → Highway, double →
SLEEF) was inconsistent with this principle; it has been removed -- see #8.

## SLEEF / ARM NEON support

* SLEEF library: NEON is fully supported (the `helperadvsimd.h` /
  `helperneon32.h` backends ship in SLEEF source). The Bazel `@sleef` rule
  uses `rules_foreign_cc` to drive SLEEF's own CMake, which auto-detects
  the host arch -- so building on ARM produces a NEON-capable
  `libsleef.a` with the same target name.
* `sleef_dispatch-inl.h` (this round): now includes a NEON branch
  (`HWY_TARGET == HWY_NEON | HWY_NEON_BF16 | HWY_NEON_WITHOUT_AES`). On
  ARM, the macros expand to `Sleef_*f4` / `Sleef_*d2` -- same names as the
  SSE base, but our extern declarations switch to `float32x4_t` /
  `float64x2_t` and pull in `<arm_neon.h>` instead of `<x86intrin.h>`.
* SVE / SVE2 are intentionally not yet wired up: SVE uses a variable-length
  vector and a different SLEEF symbol family (`Sleef_*fx_*sve`). On
  HWY_SVE / HWY_SVE2 / HWY_RVV / HWY_SCALAR / HWY_EMU128 the dispatch
  defines `RUDF_SLEEF_NO_SIMD`, and SLEEF-only ops emit a clean
  `static_assert` rather than a cryptic compile error.

## Design principle (established this round)

**SLEEF is invoked only for math functions Highway does not implement.**
Concretely:

| Op family | Highway has it? | Implementation |
|---|---|---|
| `sin`, `cos`, `sinh`, `tanh`, `atan`, `atanh`, `atan2`, `asin`, `acos`, `acosh`, `asinh`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `hypot`, `sqrt`, `floor`, `ceil`, `round`, `trunc`, `abs` | ✅ | Highway (`hwy/contrib/math/math-inl.h` + core) |
| `cosh`, `tan`, `pow`, `erf`, `erfc`, `rint` | ❌ | SLEEF |

Operations Highway already provides MUST NOT route through SLEEF. The earlier
asymmetric handling of `OP_LOG/LOG2/LOG10/LOG1P` (float → Highway, double →
SLEEF) was inconsistent with this principle; it has been removed -- see #8.

## SLEEF / ARM NEON support

* SLEEF library: NEON is fully supported (the `helperadvsimd.h` /
  `helperneon32.h` backends ship in SLEEF source). The Bazel `@sleef` rule
  uses `rules_foreign_cc` to drive SLEEF's own CMake, which auto-detects
  the host arch -- so building on ARM produces a NEON-capable
  `libsleef.a` with the same target name.
* `sleef_dispatch-inl.h` (this round): now includes a NEON branch
  (`HWY_TARGET == HWY_NEON | HWY_NEON_BF16 | HWY_NEON_WITHOUT_AES`). On
  ARM, the macros expand to `Sleef_*f4` / `Sleef_*d2` -- same names as the
  SSE base, but our extern declarations switch to `float32x4_t` /
  `float64x2_t` and pull in `<arm_neon.h>` instead of `<x86intrin.h>`.
* SVE / SVE2 are intentionally not yet wired up: SVE uses a variable-length
  vector and a different SLEEF symbol family (`Sleef_*fx_*sve`). On
  HWY_SVE / HWY_SVE2 / HWY_RVV / HWY_SCALAR / HWY_EMU128 the dispatch
  defines `RUDF_SLEEF_NO_SIMD`, and SLEEF-only ops emit a clean
  `static_assert` rather than a cryptic compile error.

## 1. Dead Code: Duplicate `OP_COSH` (vector_unary.cc:220-275) — FIXED

`OP_COSH` appeared twice. The first `if constexpr (op == OP_COSH)` matched,
making the second block unreachable dead code (identical logic). Removed.

## 2. Dead Code: Duplicate `OP_SINH` (vector_unary.cc:119 & 129) — FIXED

`OP_SINH` was handled at line 119 (`return hn::Sinh(d, lv);`) and again at
line 129. The second was dead code. Removed.

## 3. Inconsistent SLEEF Header Inclusion — REVERTED, ROOT CAUSE EXPLAINED

**Original recommendation was to unify on `#include "sleef.h"` and remove the
manual `extern "C"` declarations. Implementation revealed this is unsafe.**

`sleef.h` gates AVX-512 declarations behind `#ifdef __AVX512F__`:

```c
#ifdef __AVX512F__
SLEEF_IMPORT SLEEF_CONST __m512d Sleef_powd8_u10(__m512d, __m512d);
...
#endif
```

Highway's `foreach_target.h` re-includes the source file once per SIMD target
and applies per-function target attributes via `#pragma GCC target(...)`.
Because `sleef.h` has its own include guard, it is processed exactly once —
at the first inclusion — and at that moment no per-target pragma is yet
active, so `__AVX512F__` is **not** defined and the AVX-512 prototypes never
become visible. Subsequent re-inclusions are no-ops.

The result: replacing manual extern "C" with `#include "sleef.h"` produces
`'Sleef_powd8_u10' was not declared in this scope` errors when compiling the
AVX3 target. Verified during this round of work (build error reproduced).

**Resolution adopted**: kept the manual `extern "C"` declarations in both
files. They are the *only* portable way to bypass SLEEF's `__AVX512F__`
gate without lying to the preprocessor. Added a long comment in
`vector_unary.cc` explaining this for future maintainers.

## 4. Redundant `#include <x86intrin.h>` — REVERTED

Same root cause as #3: the manual extern "C" block uses `__m128/__m256/__m512`
types, which require `<x86intrin.h>` (and on GCC, AVX-512 types only become
visible when `__AVX512F__` is defined or — in our setup — when extern
declarations defer the type-resolution to link time). Kept the include.

## 5. Repetitive `#if HWY_TARGET == ...` Dispatch Blocks — FIXED

The 10 nearly-identical 8-10 line `#if/#elif/#else` dispatch blocks in
`vector_unary.cc` and 1 in `vector_binary.cc` were consolidated into two
token-pasting macros, hoisted into a single shared header
`rapidudf/functions/simd/sleef_dispatch-inl.h`:

```cpp
#if HWY_TARGET == HWY_AVX3 || ... || HWY_TARGET == HWY_AVX3_SPR
  #define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f16##SUF
  #define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d8##SUF
#elif HWY_TARGET == HWY_AVX2
  #define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f8##SUF
  #define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d4##SUF
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
  #define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f4##SUF
  #define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d2##SUF
#else
  #define RUDF_SLEEF_NO_SIMD 1
#endif
```

The shared header uses Highway's `HWY_TARGET_TOGGLE` guard idiom (the same
pattern used by `hwy/contrib/math/math-inl.h`) so it is re-processed exactly
once per SIMD target driven by `foreach_target.h`. This:

* eliminates duplication between `vector_unary.cc` and `vector_binary.cc`,
* keeps the `extern "C"` SLEEF declarations in one authoritative place,
* automatically `#undef`s before each redefinition to silence
  `-Wmacro-redefined`.

Each call site reduces from 8-10 lines per branch to one line, e.g.:

```cpp
} else if constexpr (op == OP_COSH) {
  if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
    return V{RUDF_SLEEF_F(cosh, _u10)(lv.raw)};
  } else {
    return V{RUDF_SLEEF_D(cosh, _u10)(lv.raw)};
  }
}
```

## 6. Potential Linker Risk: Generic vs ISA-Suffixed SLEEF Symbols — UNCHANGED

Current code calls generic SLEEF symbols (e.g. `Sleef_logd2_u10`). SLEEF also
provides ISA-suffixed variants (`Sleef_logd2_u10sse2`, `Sleef_logd4_u10avx2`,
etc.). The generic symbols dispatch correctly with the SLEEF 3.9.0 build
linked here. Low risk; not modified.

---

## 7. x86-only Assumption — FIXED, EXTENDED TO NEON

`<x86intrin.h>` and `__m128/__m256/__m512` types broke compilation on
non-x86 architectures. The dispatch helper now:

* gates x86 declarations on `HWY_ARCH_X86`,
* adds an ARM NEON branch (`HWY_ARCH_ARM` && `HWY_TARGET ==
  HWY_NEON{,_BF16,_WITHOUT_AES}`) that pulls in `<arm_neon.h>` and
  declares `float32x4_t` / `float64x2_t` SLEEF entry points,
* falls through to `RUDF_SLEEF_NO_SIMD` for everything else (SVE, RVV,
  SCALAR, EMU128).

SLEEF-only ops (`cosh`, `tan`, `pow`, `erf`, `erfc`, `rint`) emit a
`static_assert` when SIMD SLEEF is unavailable. Ops that exist in Highway
(`log`, etc.) continue to work everywhere via Highway.

## 8. Asymmetric Type Coverage for log/log2/log10/log1p — RESOLVED, FAVOR HIGHWAY

The earlier code routed `OP_LOG/LOG2/LOG10/LOG1P` to SLEEF for `double` but
to Highway for `float`. There was no compelling reason for SLEEF here:
Highway provides `Log/Log2/Log10/Log1p` for both lane types. Per the design
principle stated at the top of this doc, **the entire LOG family now uses
Highway uniformly** for both float and double. The corresponding extern
declarations (`Sleef_log{,2,10,1p}d{2,4,8}_u10`) were removed from
`sleef_dispatch-inl.h`.

## 9. Selective SLEEF Use vs Highway Built-ins — RESOLVED CORRECTLY THIS ROUND

A previous note in this doc claimed "Highway has Cosh, but we use SLEEF for
performance." That was **factually wrong** -- inspection of
`hwy/contrib/math/math-inl.h` confirms Highway has `Sinh` and `Tanh` but
**not** `Cosh`. Same story for `Tan`, `Pow`, `Erf`, `Erfc`, `Rint`. SLEEF
is required, not preferred. Comments in `vector_unary.cc` /
`vector_binary.cc` and the new header now state this correctly. As a
follow-on bug-fix: the prior non-x86 fallback for `OP_COSH` read
`return hn::Cosh(d, lv);`, which would have failed to compile on ARM
because `hn::Cosh` does not exist. Replaced with `static_assert`.

## 10. extern "C" / sleef.h Order in vector_binary.cc — RESOLVED BY #3

Was a moot point once the decision in #3 was made: `sleef.h` is no longer
included from these files. Only the explicit `extern "C"` block remains
(under `HWY_ARCH_X86` guard).

## 11. Style: missing `static` on do_simd_binary_op — FIXED

`do_simd_binary_op` in `vector_binary.cc` was missing `static`; now matches
`do_simd_unary_op`'s `static HWY_INLINE` qualification.

## 12. Reserved OP_MULTIPLY / OP_DIVIDE Branches — DOCUMENTED

The implementation of `OP_MULTIPLY` and `OP_DIVIDE` exists in
`do_simd_binary_op` but is never instantiated (the corresponding
`DEFINE_SIMD_BINARY_OP(...)` macros are commented out). Inline comment
added clarifying these are reserved for future use.

---

## Verification

```
bazel build //...                # 512 actions, completed successfully
bazel test  //...                # 28/28 tests pass (incl. simd_vector_test,
                                 # math_test, arithmetic_test, hwy_test)
```

## Files Touched

* `rapidudf/functions/simd/sleef_dispatch-inl.h` — **new**, 147 lines.
  Single source of truth for the SLEEF `extern "C"` declarations and the
  `RUDF_SLEEF_F` / `RUDF_SLEEF_D` per-target dispatch macros, gated by
  Highway's `HWY_TARGET_TOGGLE` toggle-guard idiom so it is re-processed
  exactly once per SIMD target.
* `rapidudf/functions/simd/vector_unary.cc` — 466 → 265 lines (-43%).
* `rapidudf/functions/simd/vector_binary.cc` — 151 → 128 lines (-15%).
* `rapidudf/functions/simd/BUILD` — added `sleef_dispatch-inl.h` to
  `cc_library(name="vector").srcs` so Bazel ships it with the library.

(CMake build is unchanged: `CMakeLists.txt` only enumerates `.cc` sources;
headers resolve via the project-root include path.)

(Line-count reduction is modest because the manual `extern "C"` declarations
were retained for AVX-512 visibility reasons. The structural simplification
— eliminating 11 nearly-identical dispatch blocks — remains.)
