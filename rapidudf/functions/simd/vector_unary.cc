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

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/vector/vector.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vector_unary.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <x86intrin.h>

extern "C" {

extern __m128 Sleef_coshf4_u10(__m128);
extern __m256 Sleef_coshf8_u10(__m256);
extern __m512 Sleef_coshf16_u10(__m512);

extern __m128d Sleef_coshd2_u10(__m128d);
extern __m256d Sleef_coshd4_u10(__m256d);
extern __m512d Sleef_coshd8_u10(__m512d);

extern __m128 Sleef_tanf4_u10(__m128);
extern __m256 Sleef_tanf8_u10(__m256);
extern __m512 Sleef_tanf16_u10(__m512);

extern __m128d Sleef_tand2_u10(__m128d);
extern __m256d Sleef_tand4_u10(__m256d);
extern __m512d Sleef_tand8_u10(__m512d);

extern __m128 Sleef_rintf4(__m128);
extern __m256 Sleef_rintf8(__m256);
extern __m512 Sleef_rintf16(__m512);

extern __m128d Sleef_rintd2(__m128d);
extern __m256d Sleef_rintd4(__m256d);
extern __m512d Sleef_rintd8(__m512d);

extern __m128 Sleef_erff4_u10(__m128);
extern __m256 Sleef_erff8_u10(__m256);
extern __m512 Sleef_erff16_u10(__m512);
extern __m128d Sleef_erfd2_u10(__m128d);
extern __m256d Sleef_erfd4_u10(__m256d);
extern __m512d Sleef_erfd8_u10(__m512d);

extern __m128 Sleef_erfcf4_u15(__m128);
extern __m256 Sleef_erfcf8_u15(__m256);
extern __m512 Sleef_erfcf16_u15(__m512);
extern __m128d Sleef_erfcd2_u15(__m128d);
extern __m256d Sleef_erfcd4_u15(__m256d);
extern __m512d Sleef_erfcd8_u15(__m512d);
}

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <class D, OpToken op, typename V = hn::VFromD<D>>
static HWY_INLINE auto do_simd_unary_op([[maybe_unused]] D d, V lv) {
  if constexpr (op == OP_SQRT) {
    return hn::Sqrt(lv);
  } else if constexpr (op == OP_FLOOR) {
    return hn::Floor(lv);
  } else if constexpr (op == OP_ROUND) {
    return hn::Round(lv);
  } else if constexpr (op == OP_TRUNC) {
    return hn::Trunc(lv);
  } else if constexpr (op == OP_ABS) {
    return hn::Abs(lv);
  } else if constexpr (op == OP_NOT) {
    return hn::Not(lv);
  } else if constexpr (op == OP_NEGATIVE) {
    return hn::Neg(lv);
  } else if constexpr (op == OP_COS) {
    return hn::Cos(d, lv);
  } else if constexpr (op == OP_SIN) {
    return hn::Sin(d, lv);
  } else if constexpr (op == OP_SINH) {
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_ASIN) {
    return hn::Asin(d, lv);
  } else if constexpr (op == OP_ACOS) {
    return hn::Acos(d, lv);
  } else if constexpr (op == OP_ATAN) {
    return hn::Atan(d, lv);
  } else if constexpr (op == OP_ATANH) {
    return hn::Atanh(d, lv);
  } else if constexpr (op == OP_SINH) {
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_TANH) {
    return hn::Tanh(d, lv);
  } else if constexpr (op == OP_ASINH) {
    return hn::Asinh(d, lv);
  } else if constexpr (op == OP_ACOSH) {
    return hn::Acosh(d, lv);
  } else if constexpr (op == OP_EXP) {
    return hn::Exp(d, lv);
  } else if constexpr (op == OP_EXP2) {
    return hn::Exp2(d, lv);
  } else if constexpr (op == OP_EXPM1) {
    return hn::Expm1(d, lv);
  } else if constexpr (op == OP_LOG) {
    return hn::Log(d, lv);
  } else if constexpr (op == OP_LOG2) {
    return hn::Log2(d, lv);
  } else if constexpr (op == OP_LOG10) {
    return hn::Log10(d, lv);
  } else if constexpr (op == OP_LOG1P) {
    return hn::Log1p(d, lv);
  } else if constexpr (op == OP_CEIL) {
    return hn::Ceil(lv);
  } else if constexpr (op == OP_APPROX_RECIP) {
    return hn::ApproximateReciprocal(lv);
  } else if constexpr (op == OP_APPROX_RECIP_SQRT) {
    return hn::ApproximateReciprocalSqrt(lv);
  } else if constexpr (op == OP_COSH) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_coshf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshd8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshd4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_coshd2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_COSH) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_coshf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshd8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshd4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_coshd2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_TAN) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_tanf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_tanf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_tanf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_tand8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_tand4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_tand2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_ERF) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_erff16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_erff8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_erff4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_erfd8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_erfd4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_erfd2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_ERFC) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_erfcf16_u15(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_erfcf8_u15(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_erfcf4_u15(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_erfcd8_u15(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_erfcd4_u15(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_erfcd2_u15(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_RINT) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_rintf16(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_rintf8(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_rintf4(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_rintd8(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_rintd4(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_rintd2(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <typename T, class Func>
HWY_INLINE void do_unary_transform(const T* input, T* output, const Func& func) {
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  static_assert(simd::kVectorUnitSize % N == 0, "Invalid lanes");
  for (size_t idx = 0; idx < simd::kVectorUnitSize; idx += N) {
    const hn::Vec<D> v = hn::LoadU(d, input + idx);
    hn::StoreU(func(d, v), d, output + idx);
  }
}

template <typename OPT>
HWY_INLINE void simd_vector_unary_op_impl(const typename OPT::operand_t* input, typename OPT::operand_t* output) {
  using D = hn::ScalableTag<typename OPT::operand_t>;
  //   constexpr D d;
  auto transform_func = do_simd_unary_op<D, OPT::op>;
  do_unary_transform(input, output, transform_func);
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
template <typename T, OpToken op>
void simd_vector_unary_op(const T* input, T* output) {
  using OPT = OperandType<T, op>;
  HWY_EXPORT_T(Table, simd_vector_unary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(input, output);
}

#define DEFINE_SIMD_UNARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template void simd_vector_unary_op<TYPE, op>(const TYPE* input, TYPE* output);
#define DEFINE_SIMD_UNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_UNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
// DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
// DEFINE_SIMD_UNARY_OP(OP_NEGATIVE, float, double, int64_t, int32_t, int16_t, int8_t);
DEFINE_SIMD_UNARY_OP(OP_COS, float, double);
DEFINE_SIMD_UNARY_OP(OP_SIN, float, double);
DEFINE_SIMD_UNARY_OP(OP_TAN, float, double);
DEFINE_SIMD_UNARY_OP(OP_TANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ATANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_SINH, float, double);
DEFINE_SIMD_UNARY_OP(OP_COSH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ACOS, float, double);
DEFINE_SIMD_UNARY_OP(OP_ASIN, float, double);
DEFINE_SIMD_UNARY_OP(OP_ATAN, float, double);
DEFINE_SIMD_UNARY_OP(OP_ACOSH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ASINH, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXP, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXP2, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXPM1, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG2, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG10, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG1P, float, double);
DEFINE_SIMD_UNARY_OP(OP_SQRT, float, double);
DEFINE_SIMD_UNARY_OP(OP_FLOOR, float, double);
DEFINE_SIMD_UNARY_OP(OP_CEIL, float, double);
DEFINE_SIMD_UNARY_OP(OP_ROUND, float, double);
DEFINE_SIMD_UNARY_OP(OP_RINT, float, double);
DEFINE_SIMD_UNARY_OP(OP_TRUNC, float, double);
DEFINE_SIMD_UNARY_OP(OP_ERF, float, double);
DEFINE_SIMD_UNARY_OP(OP_ERFC, float, double);
DEFINE_SIMD_UNARY_OP(OP_APPROX_RECIP, float, double);
DEFINE_SIMD_UNARY_OP(OP_APPROX_RECIP_SQRT, float, double);

}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE