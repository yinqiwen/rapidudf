/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <x86intrin.h>
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <functional>
#include <stack>
#include <type_traits>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"

#include "rapidudf/builtin/simd_vector/ops.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/eval_value-inl.h"
#include "rapidudf/types/eval_value.h"
#include "rapidudf/types/scalar.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"
#include "rapidudf/types/string_view.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/eval.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "hwy/cache_control.h"
#include "hwy/contrib/math/math-inl.h"

extern "C" {
extern __m128 Sleef_powf4_u10(__m128, __m128);
extern __m256 Sleef_powf8_u10(__m256, __m256);
extern __m512 Sleef_powf16_u10(__m512, __m512);

extern __m128d Sleef_powd2_u10(__m128d, __m128d);
extern __m256d Sleef_powd4_u10(__m256d, __m256d);
extern __m512d Sleef_powd8_u10(__m512d, __m512d);

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

#define ONLY_FLOATS_BEGIN(op, T) if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
#define ONLY_FLOATS_END(op, T)                                                                      \
  }                                                                                                 \
  else {                                                                                            \
    throw std::logic_error(fmt::format("Unsupported op:{} for non floats:{}", op, get_dtype<T>())); \
  }

HWY_BEFORE_NAMESPACE();

namespace rapidudf {
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static constexpr size_t k_max_stack_var_size = 16;

template <class D, typename V = hn::VFromD<D>>
static HWY_INLINE V do_simd_unary_op([[maybe_unused]] D d, OpToken op, [[maybe_unused]] V lv) {
  using T = hn::TFromV<V>;
  switch (op) {
    case OP_NOT: {
      return hn::Not(lv);
    }
    case OP_NEGATIVE: {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<int64_t, T> ||
                    std::is_same_v<int32_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<int8_t, T>) {
        return hn::Neg(lv);
      } else {
        throw std::logic_error(fmt::format("Unsupported op:{} for non floats:{}", op, get_dtype<T>()));
      }
    }
    case OP_ABS: {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<int64_t, T> ||
                    std::is_same_v<int32_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<int8_t, T>) {
        return hn::Abs(lv);
      } else {
        throw std::logic_error(fmt::format("Unsupported op:{} for non floats:{}", op, get_dtype<T>()));
      }
    }
    case OP_SQRT: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Sqrt(lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_FLOOR: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Floor(lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ROUND: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Round(lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_TRUNC: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Trunc(lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_CEIL: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Ceil(lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_RINT: {
      ONLY_FLOATS_BEGIN(op, T)
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
      ONLY_FLOATS_END(op, T)
    }

    case OP_SIN: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Sin(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_COS: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Cos(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_TAN: {
      ONLY_FLOATS_BEGIN(op, T)
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
      ONLY_FLOATS_END(op, T)
    }
    case OP_SINH: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Sinh(d, lv);
      ONLY_FLOATS_END(op, T)
      break;
    }
    case OP_COSH: {
      ONLY_FLOATS_BEGIN(op, T)
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
      ONLY_FLOATS_END(op, T)
      break;
    }
    case OP_TANH: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Tanh(d, lv);
      ONLY_FLOATS_END(op, T)
      break;
    }
    case OP_ASIN: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Asin(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ACOS: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Acos(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ATAN: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Atan(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ASINH: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Asinh(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ACOSH: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Acosh(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ATANH: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Atanh(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_EXP: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Exp(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_EXP2: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Exp2(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_EXPM1: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Expm1(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_LOG: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Log(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_LOG2: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Log2(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_LOG10: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Log10(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_LOG1P: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Log1p(d, lv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ERF: {
      ONLY_FLOATS_BEGIN(op, T)
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
      ONLY_FLOATS_END(op, T)
    }
    case OP_ERFC: {
      ONLY_FLOATS_BEGIN(op, T)
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
      ONLY_FLOATS_END(op, T)
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}

template <class D, typename V = hn::VFromD<D>>
static HWY_INLINE V do_simd_binary_op([[maybe_unused]] D d, OpToken op, [[maybe_unused]] V lv, [[maybe_unused]] V rv) {
  using T = hn::TFromV<V>;
  switch (op) {
    case OP_PLUS:
    case OP_PLUS_ASSIGN: {
      return hn::Add(lv, rv);
    }
    case OP_MINUS:
    case OP_MINUS_ASSIGN: {
      return hn::Sub(lv, rv);
    }
    case OP_MULTIPLY:
    case OP_MULTIPLY_ASSIGN: {
      return hn::Mul(lv, rv);
    }
    case OP_DIVIDE:
    case OP_DIVIDE_ASSIGN: {
      return hn::Div(lv, rv);
    }
    case OP_MOD:
    case OP_MOD_ASSIGN: {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
        throw std::logic_error(fmt::format("Unsupported op:{} for floats", op));
      } else {
        return hn::Mod(lv, rv);
      }
    }
    case OP_LOGIC_AND: {
      if constexpr (std::is_same_v<uint8_t, T>) {
        return hn::And(lv, rv);
      } else {
        throw std::logic_error(fmt::format("Unsupported op:{} for non bits", op));
      }
    }
    case OP_LOGIC_OR: {
      if constexpr (std::is_same_v<uint8_t, T>) {
        return hn::Or(lv, rv);
      } else {
        throw std::logic_error(fmt::format("Unsupported op:{} for non bits", op));
      }
    }
    case OP_LOGIC_XOR: {
      if constexpr (std::is_same_v<uint8_t, T>) {
        return hn::Xor(lv, rv);
      } else {
        throw std::logic_error(fmt::format("Unsupported op:{} for non bits", op));
      }
    }
    case OP_MAX: {
      return hn::Max(lv, rv);
    }
    case OP_MIN: {
      return hn::Min(lv, rv);
    }
    case OP_HYPOT: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Hypot(d, lv, rv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_ATAN2: {
      ONLY_FLOATS_BEGIN(op, T)
      return hn::Atan2(d, lv, rv);
      ONLY_FLOATS_END(op, T)
    }
    case OP_POW: {
      ONLY_FLOATS_BEGIN(op, T)
      if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
        auto val = Sleef_powf16_u10(lv.raw, rv.raw);
        return V{val};
#elif HWY_TARGET == HWY_AVX2
        auto val = Sleef_powf8_u10(lv.raw, rv.raw);
        return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
        auto val = Sleef_powf4_u10(lv.raw, rv.raw);
        return V{val};
#else
        static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
      } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
        auto val = Sleef_powd8_u10(lv.raw, rv.raw);
        return V{val};
#elif HWY_TARGET == HWY_AVX2
        auto val = Sleef_powd4_u10(lv.raw, rv.raw);
        return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
        auto val = Sleef_powd2_u10(lv.raw, rv.raw);
        return V{val};
#else
        static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
      }
      ONLY_FLOATS_END(op, T)
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}
template <class D, typename V = hn::VFromD<D>>
static HWY_INLINE V do_simd_ternary_op([[maybe_unused]] D d, OpToken op, V a, V b, V c) {
  switch (op) {
    case OP_CLAMP: {
      return hn::Clamp(a, b, c);
    }
    case OP_FMA: {
      return hn::MulAdd(a, b, c);
    }
    case OP_FMS: {
      return hn::MulSub(a, b, c);
    }
    case OP_FNMA: {
      return hn::NegMulAdd(a, b, c);
    }
    case OP_FNMS: {
      return hn::NegMulSub(a, b, c);
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}
template <class D, typename M = hn::Mask<D>>
static HWY_INLINE M do_simd_unary_logic_op([[maybe_unused]] D d, OpToken op, M lv) {
  switch (op) {
    case OP_NOT: {
      return hn::Not(lv);
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}

template <class D, typename M = hn::Mask<D>>
static HWY_INLINE M do_simd_binary_logic_op([[maybe_unused]] D d, OpToken op, M lv, M rv) {
  switch (op) {
    case OP_LOGIC_AND: {
      return hn::And(lv, rv);
    }
    case OP_LOGIC_OR: {
      return hn::Or(lv, rv);
    }
    case OP_LOGIC_XOR: {
      return hn::Xor(lv, rv);
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}
template <class D, typename V = hn::VFromD<D>>
static HWY_INLINE hn::Mask<D> do_simd_cmp_op([[maybe_unused]] D d, OpToken op, V lv, V rv) {
  switch (op) {
    case OP_GREATER: {
      return hn::Gt(lv, rv);
    }
    case OP_GREATER_EQUAL: {
      return hn::Ge(lv, rv);
    }
    case OP_LESS: {
      return hn::Lt(lv, rv);
    }
    case OP_LESS_EQUAL: {
      return hn::Le(lv, rv);
    }
    case OP_EQUAL: {
      return hn::Eq(lv, rv);
    }
    case OP_NOT_EQUAL: {
      return hn::Ne(lv, rv);
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
}

struct Operator {
  uint8_t op = OP_INVALID;
  int8_t operand_count = 0;
  bool is_logic = false;
  bool is_cmp = false;
  Operator(OpToken v = OP_INVALID) {
    op = static_cast<uint8_t>(v);
    if (op != 0) {
      operand_count = static_cast<int8_t>(get_operand_count(v));
      is_logic = is_logic_op(v);
      is_cmp = is_compare_op(v);
    }
  }
};
enum EvalAction : uint8_t {
  ACTION_LOAD_VECTOR = 1,
  ACTION_LOAD_SCALAR,
  ACTION_EVAL_1_LOGIC,
  ACTION_EVAL_1,
  ACTION_EVAL_2_LOGIC,
  ACTION_EVAL_2_CMP,
  ACTION_EVAL_2,
  ACTION_EVAL_3,
};
template <typename T>
struct alignas(sizeof(hn::Vec<hn::ScalableTag<T>>)) EvalNode {
  hn::Vec<hn::ScalableTag<T>> scalar;
  Vector<T> vec;
  Operator op;
  int16_t dup_idx = -1;
  EvalAction action;
  uint8_t reserved[sizeof(uint64_t) - sizeof(Operator) - sizeof(EvalAction) - sizeof(int16_t)];

  explicit EvalNode(OpToken t) : op(t) {}
  explicit EvalNode(hn::Vec<hn::ScalableTag<T>> s) : scalar(s) {}
  explicit EvalNode(Vector<T> v) : vec(v) {}
  inline OpToken GetOp() { return static_cast<OpToken>(op.op); }
};

template <typename T>
using EvalNodeVector =
    std::vector<EvalNode<T>, boost::alignment::aligned_allocator<EvalNode<T>, sizeof(hn::Vec<hn::ScalableTag<T>>)>>;
// template <typename T>
// using RPNEvalNodeVector =
//     absl::InlinedVector<RPNEvalNode<T>, k_max_stack_var_size,
//                         boost::alignment::aligned_allocator<RPNEvalNode<T>, sizeof(hn::Vec<hn::ScalableTag<T>>)>>;

template <bool bit_op, bool with_remaining, typename D, typename T = hn::TFromD<D>>
static HWY_INLINE auto eval(D d, size_t vec_idx, [[maybe_unused]] size_t remaining, EvalNodeVector<T>& inputs) {
  using V = hn::Vec<D>;
  using M = hn::Mask<D>;
  constexpr size_t N = Lanes(d);
  size_t mask_stack_size = 0;
  size_t value_stack_size = 0;
  alignas(sizeof(M)) M mask_stks[k_max_stack_var_size];
  alignas(sizeof(V)) V value_stks[k_max_stack_var_size];
  // for (size_t i = 0; i < inputs.size(); i++) {
  //   auto& operand = inputs[i];
  //   switch (inputs[i].action) {
  //     case ACTION_LOAD_VECTOR: {
  //       value_stks[value_stack_size++] = hn::LoadU(d, operand.vec.Data() + vec_idx);
  //       break;
  //     }
  //     case ACTION_EVAL_2: {
  //       value_stks[value_stack_size - 2] =
  //           do_simd_binary_op(d, operand.GetOp(), value_stks[value_stack_size - 2], value_stks[value_stack_size -
  //           1]);
  //       value_stack_size -= 1;
  //       break;
  //     }
  //     case ACTION_EVAL_1: {
  //       V v = do_simd_unary_op(d, operand.GetOp(), value_stks[value_stack_size - 1]);
  //       value_stks[value_stack_size - 1] = v;
  //       break;
  //     }
  //     case ACTION_EVAL_1_LOGIC: {
  //       M mask = do_simd_unary_logic_op(d, operand.GetOp(), mask_stks[mask_stack_size - 1]);
  //       mask_stks[mask_stack_size - 1] = mask;
  //       break;
  //     }
  //     case ACTION_EVAL_2_CMP: {
  //       M mask = do_simd_cmp_op(d, operand.GetOp(), value_stks[value_stack_size - 2], value_stks[value_stack_size -
  //       1]); value_stack_size -= 2; mask_stks[mask_stack_size++] = mask; break;
  //     }
  //     case ACTION_EVAL_2_LOGIC: {
  //       M mask =
  //           do_simd_binary_logic_op(d, operand.GetOp(), mask_stks[mask_stack_size - 2], mask_stks[mask_stack_size -
  //           1]);
  //       mask_stack_size -= 2;
  //       mask_stks[mask_stack_size++] = mask;
  //       break;
  //     }
  //     case ACTION_EVAL_3: {
  //       V v = do_simd_ternary_op(d, operand.GetOp(), value_stks[value_stack_size - 3], value_stks[value_stack_size -
  //       2],
  //                                value_stks[value_stack_size - 1]);
  //       value_stack_size -= 3;
  //       value_stks[value_stack_size++] = v;
  //       break;
  //     }
  //     case ACTION_LOAD_SCALAR: {
  //       value_stks[value_stack_size++] = operand.scalar;
  //       break;
  //     }
  //     default: {
  //       break;
  //     }
  //   }
  // }

  for (size_t i = 0; i < inputs.size(); i++) {
    auto& operand = inputs[i];
    if (operand.GetOp() > 0) {
      int operand_count = operand.op.operand_count;
      bool is_logic = false;
      if constexpr (bit_op) {
        is_logic = operand.op.is_logic;
      }
      if (ABSL_PREDICT_TRUE(operand_count == 2)) {
        bool is_cmp = false;
        if constexpr (bit_op) {
          is_cmp = operand.op.is_cmp;
        }
        if (ABSL_PREDICT_TRUE(!is_logic && !is_cmp)) {
          value_stks[value_stack_size - 2] =
              do_simd_binary_op(d, operand.GetOp(), value_stks[value_stack_size - 2], value_stks[value_stack_size - 1]);
          value_stack_size -= 1;
        } else if (ABSL_PREDICT_TRUE(is_logic)) {
          M mask = do_simd_binary_logic_op(d, operand.GetOp(), mask_stks[mask_stack_size - 2],
                                           mask_stks[mask_stack_size - 1]);
          mask_stack_size -= 2;
          mask_stks[mask_stack_size++] = mask;
        } else {
          M mask =
              do_simd_cmp_op(d, operand.GetOp(), value_stks[value_stack_size - 2], value_stks[value_stack_size - 1]);
          value_stack_size -= 2;
          mask_stks[mask_stack_size++] = mask;
        }
      } else if (operand_count == 1) {
        if (ABSL_PREDICT_TRUE(!is_logic)) {
          V v = do_simd_unary_op(d, operand.GetOp(), value_stks[value_stack_size - 1]);
          value_stks[value_stack_size - 1] = v;
        } else {
          M mask = do_simd_unary_logic_op(d, operand.GetOp(), mask_stks[mask_stack_size - 1]);
          mask_stks[mask_stack_size - 1] = mask;
        }
      } else if (operand_count == 3) {
        V v = do_simd_ternary_op(d, operand.GetOp(), value_stks[value_stack_size - 3], value_stks[value_stack_size - 2],
                                 value_stks[value_stack_size - 1]);
        value_stack_size -= 3;
        value_stks[value_stack_size++] = v;
      }
    } else if (operand.vec.Size() > 0) {
      hwy::Prefetch(operand.vec.Data() + vec_idx + N);
      if (ABSL_PREDICT_TRUE(operand.dup_idx < 0)) {
        operand.scalar = hn::LoadU(d, operand.vec.Data() + vec_idx);
        value_stks[value_stack_size++] = operand.scalar;
      } else {
        value_stks[value_stack_size++] = inputs[operand.dup_idx].scalar;
      }
      // if constexpr (with_remaining) {
      //   value_stks[value_stack_size++] = hn::LoadN(d, operand.vec.Data() + vec_idx, remaining);
      // } else {
      //   value_stks[value_stack_size++] = hn::LoadU(d, operand.vec.Data() + vec_idx);
      // }

    } else {
      value_stks[value_stack_size++] = operand.scalar;
    }
  }
  if constexpr (bit_op) {
    return mask_stks[mask_stack_size - 1];
  } else {
    return value_stks[value_stack_size - 1];
  }
}
template <typename T>
HWY_INLINE void simd_vector_prepare_eval(Context& ctx, absl::Span<EvalValue> nodes, EvalNodeVector<T>& inputs,
                                         size_t& count) {
  using number_t = typename InternalType<T>::internal_type;
  using D = hn::ScalableTag<number_t>;
  // using V = hn::Vec<D>;
  constexpr D d;
  count = 0;
  DType compute_dtype = get_dtype<T>();
  inputs.reserve(nodes.size());
  absl::flat_hash_map<const void*, int16_t> idx_cache;
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes[i];
    DType dtype(node.dtype);
    if (dtype.IsInvalid()) {
      auto optype = static_cast<OpToken>(node.op);
      EvalNode<T> input(optype);
      int operand_count = get_operand_count(optype);
      bool is_logic = is_logic_op(optype);
      bool is_cmp = is_compare_op(optype);
      if (operand_count == 1) {
        if (is_logic) {
          input.action = ACTION_EVAL_1_LOGIC;
        } else {
          input.action = ACTION_EVAL_1;
        }
      } else if (operand_count == 2) {
        if (is_logic) {
          input.action = ACTION_EVAL_2_LOGIC;
        } else if (is_cmp) {
          input.action = ACTION_EVAL_2_CMP;
        } else {
          input.action = ACTION_EVAL_2;
        }
      } else {
        input.action = ACTION_EVAL_3;
      }
      inputs.emplace_back(std::move(input));
    } else if (dtype.IsSimdVector()) {
      hwy::Prefetch(node.vector.Data());
      Vector<T> vec(node.vector);
      EvalNode<T> input(vec);
      count = vec.ElementSize();
      input.action = ACTION_LOAD_VECTOR;
      auto [found, success] = idx_cache.emplace(node.vector.Data(), i);
      if (!success) {
        input.dup_idx = found->second;
      }
      inputs.emplace_back(input);
    } else {
      T const_v;
      if (dtype != compute_dtype) {
        if (dtype.IsStringView() || compute_dtype.IsStringView()) {
          THROW_DTYPE_MISMATCH_ERR(dtype, compute_dtype);
        }
      }
      switch (dtype.GetFundamentalType()) {
        case DATA_U64: {
          const_v = static_cast<T>(node.scalar_u64);
          break;
        }
        case DATA_U32: {
          const_v = static_cast<T>(node.scalar_u32);
          break;
        }
        case DATA_U16: {
          const_v = static_cast<T>(node.scalar_u16);
          break;
        }
        case DATA_U8: {
          const_v = static_cast<T>(node.scalar_u8);
          break;
        }
        case DATA_I64: {
          const_v = static_cast<T>(node.scalar_i64);
          break;
        }
        case DATA_I32: {
          const_v = static_cast<T>(node.scalar_i32);
          break;
        }
        case DATA_I16: {
          const_v = static_cast<T>(node.scalar_i16);
          break;
        }
        case DATA_I8: {
          const_v = static_cast<T>(node.scalar_i8);
          break;
        }
        case DATA_F32: {
          const_v = static_cast<T>(node.scalar_f32);
          break;
        }
        case DATA_F64: {
          const_v = static_cast<T>(node.scalar_f64);
          break;
        }
        default: {
          THROW_DTYPE_MISMATCH_ERR(dtype, compute_dtype);
          break;
        }
      }

      auto const_val = hn::Set(d, const_v);
      EvalNode<T> input(const_val);
      input.action = ACTION_LOAD_SCALAR;
      inputs.emplace_back(std::move(input));
    }
  }
}

template <typename T>
HWY_INLINE Vector<T> simd_vector_eval_arithmetic_impl(Context& ctx, absl::Span<EvalValue> nodes) {
  using number_t = typename InternalType<T>::internal_type;
  using D = hn::ScalableTag<number_t>;
  using V = hn::Vec<D>;
  constexpr D d;
  constexpr size_t N = Lanes(d);
  size_t count = 0;
  EvalNodeVector<number_t> inputs;
  DType compute_dtype = get_dtype<number_t>();
  simd_vector_prepare_eval<number_t>(ctx, nodes, inputs, count);

  VectorData result_data = ctx.NewSimdVector<number_t>(N, count, true);
  number_t* output = result_data.MutableData<number_t>();
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      V result = eval<false, false, D>(d, idx, 0, inputs);
      hn::StoreU(result, d, output + idx);
    }
  }

  if (HWY_UNLIKELY(idx == count)) {
    return Vector<T>(result_data);
  }
  const size_t remaining = count - idx;
  V result = eval<false, true, D>(d, idx, remaining, inputs);
  hn::StoreN(result, d, output + idx, remaining);
  return Vector<T>(result_data);
}

template <class D, typename M = hn::Mask<D>>
static HWY_INLINE void store_mask(D d, M mask, uint8_t* bits, size_t idx) {
  constexpr size_t N = hn::Lanes(d);
  size_t bits_byte_idx = idx / 8;
  size_t bit_cursor = idx % 8;
  uint8_t cache_byte = 0;
  if constexpr (N < 8) {
    cache_byte = bits[bits_byte_idx];
  }
  hn::StoreMaskBits(d, mask, bits + bits_byte_idx);
  if (bit_cursor > 0) {
    uint8_t current_byte = bits[bits_byte_idx];
    uint8_t restore_value = ((current_byte << bit_cursor) | cache_byte);
    bits[bits_byte_idx] = restore_value;
  }
}

template <typename T>
HWY_INLINE Vector<Bit> simd_vector_eval_bit_impl(Context& ctx, absl::Span<EvalValue> nodes) {
  if constexpr (std::is_same_v<T, Bit>) {
    return simd_vector_eval_arithmetic_impl<T>(ctx, nodes);
  } else {
    using D = hn::ScalableTag<T>;
    using M = hn::Mask<D>;
    const D d;
    constexpr size_t N = hn::Lanes(d);
    VectorData result_data;
    size_t count = 0;
    EvalNodeVector<T> inputs;
    DType compute_dtype = get_dtype<T>();
    simd_vector_prepare_eval<T>(ctx, nodes, inputs, count);

    result_data = ctx.NewSimdVector<Bit>(N, count, true);
    uint8_t* output = result_data.MutableData<uint8_t>();
    size_t idx = 0;
    if (count >= N) {
      for (; idx <= count - N; idx += N) {
        M result = eval<true, false, D>(d, idx, 0, inputs);
        store_mask(d, result, output, idx);
      }
    }

    if (HWY_UNLIKELY(idx == count)) {
      return Vector<Bit>(result_data);
    }
    const size_t remaining = count - idx;
    M result = eval<true, true, D>(d, idx, remaining, inputs);
    store_mask(d, result, output, idx);
    return Vector<Bit>(result_data);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace rapidudf {
namespace simd {
template <typename T>
Vector<T> simd_vector_eval_arithmetic(Context& ctx, absl::Span<EvalValue> operands) {
  HWY_EXPORT_T(Table, simd_vector_eval_arithmetic_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(ctx, operands);
}
template <typename T>
Vector<Bit> simd_vector_eval_bit(Context& ctx, absl::Span<EvalValue> nodes) {
  HWY_EXPORT_T(Table, simd_vector_eval_bit_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(ctx, nodes);
}

#define DEFINE_SIMD_VECTOR_EVAL_ARITH_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_eval_arithmetic<TYPE>(Context & ctx, absl::Span<EvalValue> nodes);
#define DEFINE_SIMD_VECTOR_EVAL_ARITH(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_VECTOR_EVAL_ARITH_TEMPLATE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_VECTOR_EVAL_ARITH(int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double)

#define DEFINE_SIMD_VECTOR_EVAL_BIT_TEMPLATE(r, op, ii, TYPE) \
  template Vector<Bit> simd_vector_eval_bit<TYPE>(Context & ctx, absl::Span<EvalValue> nodes);
#define DEFINE_SIMD_VECTOR_EVAL_BIT(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_VECTOR_EVAL_BIT_TEMPLATE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_VECTOR_EVAL_BIT(int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double)

static inline std::pair<DType, VectorData> extract_column_vector(Column* column) {
  VectorData invalid;
  DType invalid_dtype;
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR("unsupported table type in extract_column_vector");
          return std::pair<DType, VectorData>(invalid_dtype, invalid);
        } else {
          return std::pair<DType, VectorData>(get_dtype<T>(), arg.RawData());
        }
      },
      column->GetInternal());
}

static inline EvalValue simd_vector_string_view_eval_rpn(Context& ctx, absl::Span<EvalValue> nodes) {
  size_t count = 0;
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes[i];
    DType dtype(node.dtype);
    if (dtype.IsSimdVector()) {
      count = node.vector.Size();
      break;
    }
  }
  VectorData result_data = ctx.NewSimdVector<Bit>(8, count, true);
  std::vector<EvalValue> eval_stack;
  uint64_t* bits = result_data.MutableData<uint64_t>();
  for (size_t i = 0; i < count; i++) {
    eval_stack.clear();
    for (auto node : nodes) {
      DType dtype(node.dtype);
      if (dtype.IsSimdVector()) {
        auto strs = node.ToVector<StringView>();
        eval_stack.emplace_back(to_eval_value(strs[i]));
      } else if (dtype.IsStringView()) {
        eval_stack.emplace_back(to_eval_value(node.scalar_sv));
      } else if (dtype.IsInvalid()) {
        OpToken op = static_cast<OpToken>(node.op);
        bool bv = false;
        switch (op) {
          case OP_EQUAL: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv == eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_NOT_EQUAL: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv != eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_GREATER_EQUAL: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv >= eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_GREATER: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv > eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_LESS_EQUAL: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv <= eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_LESS: {
            bv = eval_stack[eval_stack.size() - 2].scalar_sv < eval_stack[eval_stack.size() - 1].scalar_sv;
            break;
          }
          case OP_LOGIC_AND: {
            bv = eval_stack[eval_stack.size() - 2].scalar_bv && eval_stack[eval_stack.size() - 1].scalar_bv;
            break;
          }
          case OP_LOGIC_OR: {
            bv = eval_stack[eval_stack.size() - 2].scalar_bv || eval_stack[eval_stack.size() - 1].scalar_bv;
            break;
          }
          case OP_LOGIC_XOR: {
            bv = eval_stack[eval_stack.size() - 2].scalar_bv ^ eval_stack[eval_stack.size() - 1].scalar_bv;
            break;
          }
          default: {
            THROW_LOGIC_ERR(fmt::format("unsupported op:{} for string_view", op));
          }
        }
        eval_stack.pop_back();
        eval_stack.pop_back();
        eval_stack.emplace_back(to_eval_value(bv));
      } else {
        THROW_LOGIC_ERR(fmt::format("unsupported dtype:{} for string_view rpn expr", dtype));
      }
    }
    if (eval_stack.size() != 1) {
      THROW_LOGIC_ERR(fmt::format("Invalid final stack count:{} at string_view eval", eval_stack.size()));
    }
    size_t bits_byte_idx = i / 64;
    size_t bit_cursor = i % 64;
    if (eval_stack[0].scalar_bv) {
      bits[bits_byte_idx] = bits64_set(bits[bits_byte_idx], bit_cursor);
    } else {
      bits[bits_byte_idx] = bits64_clear(bits[bits_byte_idx], bit_cursor);
    }
  }
  return to_eval_value(Vector<Bit>(result_data));
}

static inline EvalValue simd_vector_eval_rpn(Context& ctx, DType compute_dtype, absl::Span<EvalValue> nodes) {
  OpToken last_op = static_cast<OpToken>(nodes[nodes.size() - 1].op);
  bool is_bit_result = is_logic_op(last_op) || is_compare_op(last_op);
  EvalValue result_value;
  DType result_dtype = compute_dtype.ToSimdVector();
  if (is_bit_result) {
    result_dtype = DType(DATA_BIT).ToSimdVector();
  }
  result_value.dtype = result_dtype.Control();
  switch (compute_dtype.GetFundamentalType()) {
    case DATA_F32: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<float>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<float> result = simd_vector_eval_arithmetic<float>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_F64: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<double>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<double> result = simd_vector_eval_arithmetic<double>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_U64: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<uint64_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<uint64_t> result = simd_vector_eval_arithmetic<uint64_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_I64: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<int64_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<int64_t> result = simd_vector_eval_arithmetic<int64_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_U32: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<uint32_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<uint32_t> result = simd_vector_eval_arithmetic<uint32_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_I32: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<int32_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<int32_t> result = simd_vector_eval_arithmetic<int32_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_U16: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<uint16_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<uint16_t> result = simd_vector_eval_arithmetic<uint16_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_I16: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<int16_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<int16_t> result = simd_vector_eval_arithmetic<int16_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_U8: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<uint8_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<uint8_t> result = simd_vector_eval_arithmetic<uint8_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_I8: {
      if (is_bit_result) {
        Vector<Bit> result = simd_vector_eval_bit<int8_t>(ctx, nodes);
        result_value.vector = result.RawData();
      } else {
        Vector<int8_t> result = simd_vector_eval_arithmetic<int8_t>(ctx, nodes);
        result_value.vector = result.RawData();
      }
      break;
    }
    case DATA_STRING_VIEW: {
      return simd_vector_string_view_eval_rpn(ctx, nodes);
    }
    case DATA_BIT: {
      if (!is_bit_result) {
        THROW_LOGIC_ERR(fmt::format("unsupported compute dtype:{} for arithmetics.", compute_dtype));
      }
      Vector<Bit> result = simd_vector_eval_bit<Bit>(ctx, nodes);
      result_value.vector = result.RawData();
      break;
    }
    default: {
      THROW_LOGIC_ERR(fmt::format("unsupported compute dtype:{} ", compute_dtype));
    }
  }
  return result_value;
}

static inline EvalValue do_simd_vector_eval(Context& ctx, absl::Span<EvalValue> nodes) {
  // size_t vector_size = 0;
  for (auto& node : nodes) {
    DType dtype(node.dtype);
    if (dtype.IsInvalid()) {
      // RUDF_INFO("### op:{}", static_cast<OpToken>(node.op));
      continue;
    }
    // RUDF_INFO("### data:{}", dtype);
    if (dtype.IsSimdColumnPtr()) {
      auto [column_dtype, vdata] = extract_column_vector(node.column);
      node.vector = vdata;
      node.dtype = column_dtype.Control();
      // vector_size = vdata.Size();
    } else if (dtype.IsSimdVector()) {
      // vector_size = node.vector.Size();
    }
  }
  // RUDF_INFO("Eval {} nodes with last op:{}", nodes.size(), nodes[nodes.size() - 1].op);

  DType compute_dtype;
  std::vector<EvalValue> eval_stack;
  eval_stack.reserve(16);
  int first_vector_op_stack_idx = -1;
  // int last_op_idx = -1;
  auto get_compute_dtype = [&](int idx) -> bool {
    if (eval_stack.empty()) {
      THROW_LOGIC_ERR("invalid state in simd_vector get_compute_dtype");
    }
    for (int i = idx - 1; i >= 0; i--) {
      DType dtype(eval_stack[i].dtype);
      if (dtype.IsSimdVector()) {
        compute_dtype = dtype.Elem();
        first_vector_op_stack_idx = idx;
        return true;
      }
    }
    return false;
    // THROW_LOGIC_ERR("Can NOT get compute dtype");
  };
  auto eval_func = [&]() {
    if (eval_stack.empty() || first_vector_op_stack_idx < 0) {
      THROW_LOGIC_ERR("invalid state in simd_vector eval_func");
    }
    // OpToken first_op = static_cast<OpToken>(eval_stack[first_stack_op_idx].op);
    // int operand_count = get_operand_count(first_op);
    // if (operand_count < 0) {
    //   THROW_LOGIC_ERR(fmt::format("invalid op:{} in simd_vector eval_func", first_op));
    // }

    // int start_pos = first_stack_op_idx - operand_count;
    int start_pos = 0;
    int end_pos = eval_stack.size() - 1;
    absl::Span<EvalValue> eval_stack_view(eval_stack);
    auto to_eval_nodes = eval_stack_view.subspan(start_pos, end_pos - start_pos + 1);
    EvalValue eval_val = simd_vector_eval_rpn(ctx, compute_dtype, to_eval_nodes);
    eval_stack[start_pos] = eval_val;
    eval_stack.resize(start_pos + 1);
    first_vector_op_stack_idx = -1;
    compute_dtype.Reset();
  };

  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes[i];
    DType dtype(node.dtype);
    if (dtype.IsInvalid()) {
      eval_stack.emplace_back(node);
      // last_op_idx = i;
      if (first_vector_op_stack_idx == -1) {
        // first_stack_op_idx = eval_stack.size() - 1;
        get_compute_dtype(i);
      }
      continue;
    }
    if (dtype.IsPrimitive()) {
      eval_stack.emplace_back(node);
    } else if (dtype.IsSimdVector()) {
      if (compute_dtype.IsInvalid()) {
        eval_stack.emplace_back(node);
      } else {
        auto ele_dtype = dtype.Elem();
        if (compute_dtype == ele_dtype) {
          eval_stack.emplace_back(node);
        } else {
          while (!eval_stack.empty() && eval_stack[eval_stack.size() - 1].dtype != 0) {
            eval_stack.pop_back();
            i--;
          }
          eval_func();
        }
      }
    } else {
      THROW_LOGIC_ERR(fmt::format("Unexpected dtype:{} in simd_vector simd_vector_eval", dtype));
    }
  }
  if (eval_stack.size() > 1) {
    eval_func();
  }
  if (eval_stack.size() != 1) {
    THROW_LOGIC_ERR(fmt::format("Invalid final stack count:{}", eval_stack.size()));
  }
  return eval_stack[0];
}

template <typename T>
Vector<T> simd_vector_eval(Context& ctx, absl::Span<EvalValue> nodes) {
  return Vector<T>(do_simd_vector_eval(ctx, nodes).vector);
}

#define DEFINE_SIMD_VECTOR_EVAL_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_eval<TYPE>(Context & ctx, absl::Span<EvalValue> nodes);
#define DEFINE_SIMD_VECTOR_EVAL(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_VECTOR_EVAL_TEMPLATE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_VECTOR_EVAL(Bit, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double,
                        StringView)

Column* simd_column_eval(Context& ctx, absl::Span<EvalValue> nodes) {
  auto result = do_simd_vector_eval(ctx, nodes);
  DType dtype(result.dtype);
  switch (dtype.GetFundamentalType()) {
    case DATA_BIT: {
      return ctx.New<Column>(ctx, Vector<Bit>(result.vector));
    }
    case DATA_I8: {
      return ctx.New<Column>(ctx, Vector<int8_t>(result.vector));
    }
    case DATA_U8: {
      return ctx.New<Column>(ctx, Vector<uint8_t>(result.vector));
    }
    case DATA_I16: {
      return ctx.New<Column>(ctx, Vector<int16_t>(result.vector));
    }
    case DATA_U16: {
      return ctx.New<Column>(ctx, Vector<uint16_t>(result.vector));
    }
    case DATA_U32: {
      return ctx.New<Column>(ctx, Vector<uint32_t>(result.vector));
    }
    case DATA_I32: {
      return ctx.New<Column>(ctx, Vector<int32_t>(result.vector));
    }
    case DATA_U64: {
      return ctx.New<Column>(ctx, Vector<uint64_t>(result.vector));
    }
    case DATA_I64: {
      return ctx.New<Column>(ctx, Vector<int64_t>(result.vector));
    }
    case DATA_F32: {
      return ctx.New<Column>(ctx, Vector<float>(result.vector));
    }
    case DATA_F64: {
      return ctx.New<Column>(ctx, Vector<double>(result.vector));
    }
    case DATA_STRING_VIEW: {
      return ctx.New<Column>(ctx, Vector<StringView>(result.vector));
    }
    default: {
      THROW_LOGIC_ERR(fmt::format("Unexpected dtype:{} in simd_column_eval", dtype));
    }
  }

  return nullptr;
}
}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE