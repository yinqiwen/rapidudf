/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared SLEEF dispatch helpers for files driven by hwy/foreach_target.h.
//
// Design principle: SLEEF is invoked ONLY for math functions that Highway
// does not implement -- specifically `cosh`, `tan`, `pow`, `erf`, `erfc`,
// and `rint`. Everything else (`log`, `log2`, `log10`, `log1p`, `sin`,
// `cos`, `sinh`, `tanh`, `exp`, `exp2`, `expm1`, `hypot`, `atan*`, ...)
// uses Highway's `hwy/contrib/math/math-inl.h` directly.
//
// This header MUST be re-processed once per Highway SIMD target so that
// RUDF_SLEEF_F / RUDF_SLEEF_D resolve to the correct lane-width SLEEF symbols
// for the target currently being compiled. We therefore use Highway's
// "HWY_TARGET_TOGGLE" guard idiom (the same one used by hwy/contrib/math/
// math-inl.h and other -inl.h headers).
//
// IMPORTANT: do NOT replace the manual extern "C" declarations below with
// `#include "sleef.h"`. SLEEF's header gates AVX-512 declarations behind
// `#ifdef __AVX512F__`. Because sleef.h has its own classic include guard,
// it is processed exactly once per TU -- at the first inclusion, before any
// per-target #pragma activates AVX-512. Subsequent re-inclusions are no-ops,
// so the AVX-512 prototypes never become visible. Manually declaring the C
// symbols sidesteps that visibility problem; the linker resolves them
// against the SLEEF library.
//
// Architecture coverage:
//   * x86 (HWY_ARCH_X86): SSE2 / SSE4 / AVX2 / AVX-512 -- f4/f8/f16, d2/d4/d8.
//   * ARM 128-bit NEON (HWY_NEON family): f4 / d2.
//   * Anything else (HWY_SCALAR, HWY_EMU128, HWY_SVE*, ...) is treated as
//     "no SIMD SLEEF available": defines RUDF_SLEEF_NO_SIMD; ops that
//     require SLEEF will static_assert and ops that don't will fall through
//     to Highway. SVE uses a variable lane count and a different SLEEF
//     symbol family (Sleef_*fx_*sve / Sleef_*dx_*sve); not implemented here.

#if defined(RAPIDUDF_FUNCTIONS_SIMD_SLEEF_DISPATCH_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef RAPIDUDF_FUNCTIONS_SIMD_SLEEF_DISPATCH_INL_H_
#undef RAPIDUDF_FUNCTIONS_SIMD_SLEEF_DISPATCH_INL_H_
#else
#define RAPIDUDF_FUNCTIONS_SIMD_SLEEF_DISPATCH_INL_H_
#endif

// ---------------------------------------------------------------------------
// x86 declarations
// ---------------------------------------------------------------------------
#if HWY_ARCH_X86
#include <x86intrin.h>

extern "C" {
// cosh
extern __m128 Sleef_coshf4_u10(__m128);
extern __m256 Sleef_coshf8_u10(__m256);
extern __m512 Sleef_coshf16_u10(__m512);
extern __m128d Sleef_coshd2_u10(__m128d);
extern __m256d Sleef_coshd4_u10(__m256d);
extern __m512d Sleef_coshd8_u10(__m512d);

// tan
extern __m128 Sleef_tanf4_u10(__m128);
extern __m256 Sleef_tanf8_u10(__m256);
extern __m512 Sleef_tanf16_u10(__m512);
extern __m128d Sleef_tand2_u10(__m128d);
extern __m256d Sleef_tand4_u10(__m256d);
extern __m512d Sleef_tand8_u10(__m512d);

// rint (exact, no error suffix)
extern __m128 Sleef_rintf4(__m128);
extern __m256 Sleef_rintf8(__m256);
extern __m512 Sleef_rintf16(__m512);
extern __m128d Sleef_rintd2(__m128d);
extern __m256d Sleef_rintd4(__m256d);
extern __m512d Sleef_rintd8(__m512d);

// erf
extern __m128 Sleef_erff4_u10(__m128);
extern __m256 Sleef_erff8_u10(__m256);
extern __m512 Sleef_erff16_u10(__m512);
extern __m128d Sleef_erfd2_u10(__m128d);
extern __m256d Sleef_erfd4_u10(__m256d);
extern __m512d Sleef_erfd8_u10(__m512d);

// erfc (3.5 ULP)
extern __m128 Sleef_erfcf4_u15(__m128);
extern __m256 Sleef_erfcf8_u15(__m256);
extern __m512 Sleef_erfcf16_u15(__m512);
extern __m128d Sleef_erfcd2_u15(__m128d);
extern __m256d Sleef_erfcd4_u15(__m256d);
extern __m512d Sleef_erfcd8_u15(__m512d);

// pow
extern __m128 Sleef_powf4_u10(__m128, __m128);
extern __m256 Sleef_powf8_u10(__m256, __m256);
extern __m512 Sleef_powf16_u10(__m512, __m512);
extern __m128d Sleef_powd2_u10(__m128d, __m128d);
extern __m256d Sleef_powd4_u10(__m256d, __m256d);
extern __m512d Sleef_powd8_u10(__m512d, __m512d);
}

// ---------------------------------------------------------------------------
// ARM NEON declarations
// ---------------------------------------------------------------------------
// Active when Highway is targeting any 128-bit NEON variant. SLEEF symbol
// names match SSE-base (Sleef_logf4_u10, etc.) but take NEON vector types.
// SLEEF's CMake auto-detects the host ARM variant and builds the advsimd
// backend, so the same Bazel @sleef target gives a NEON-capable libsleef.a
// when built on ARM.
#elif HWY_ARCH_ARM && (HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || \
                       HWY_TARGET == HWY_NEON_WITHOUT_AES)
#include <arm_neon.h>

extern "C" {
extern float32x4_t Sleef_coshf4_u10(float32x4_t);
extern float64x2_t Sleef_coshd2_u10(float64x2_t);
extern float32x4_t Sleef_tanf4_u10(float32x4_t);
extern float64x2_t Sleef_tand2_u10(float64x2_t);
extern float32x4_t Sleef_rintf4(float32x4_t);
extern float64x2_t Sleef_rintd2(float64x2_t);
extern float32x4_t Sleef_erff4_u10(float32x4_t);
extern float64x2_t Sleef_erfd2_u10(float64x2_t);
extern float32x4_t Sleef_erfcf4_u15(float32x4_t);
extern float64x2_t Sleef_erfcd2_u15(float64x2_t);
extern float32x4_t Sleef_powf4_u10(float32x4_t, float32x4_t);
extern float64x2_t Sleef_powd2_u10(float64x2_t, float64x2_t);
}
#endif  // arch dispatch

// ---------------------------------------------------------------------------
// Per-target SLEEF function-name builders.
// ---------------------------------------------------------------------------
// Token-paste the lane-width suffix based on the current HWY_TARGET so
// callers only write:
//   V{RUDF_SLEEF_F(cosh, _u10)(lv.raw)}
//   V{RUDF_SLEEF_D(pow, _u10)(lv.raw, rv.raw)}
//
//   * F = float SIMD width (f4 / f8 / f16)
//   * D = double SIMD width (d2 / d4 / d8)
//   * SUF = trailing accuracy/algorithm suffix (e.g. _u10, _u15, or empty for rint)
//
// foreach_target.h re-includes the host file once per target. The toggle
// guard above ensures this header is re-processed per target, so the macros
// land on the right lane width each time. We `#undef` first to silence
// -Wmacro-redefined.
#undef RUDF_SLEEF_F
#undef RUDF_SLEEF_D
#undef RUDF_SLEEF_NO_SIMD

#if HWY_ARCH_X86 && (HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || \
                     HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR)
#define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f16##SUF
#define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d8##SUF
#elif HWY_ARCH_X86 && HWY_TARGET == HWY_AVX2
#define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f8##SUF
#define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d4##SUF
#elif HWY_ARCH_X86 && (HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2)
#define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f4##SUF
#define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d2##SUF
#elif HWY_ARCH_ARM && (HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || \
                       HWY_TARGET == HWY_NEON_WITHOUT_AES)
#define RUDF_SLEEF_F(NAME, SUF) Sleef_##NAME##f4##SUF
#define RUDF_SLEEF_D(NAME, SUF) Sleef_##NAME##d2##SUF
#else
// SCALAR / EMU128 / SVE / RVV / unknown -- callers gate on
// !defined(RUDF_SLEEF_NO_SIMD) and either fall back to Highway or static_assert.
#define RUDF_SLEEF_NO_SIMD 1
#endif

#endif  // toggle-guarded section
