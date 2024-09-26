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
#include "rapidudf/builtin/simd_vector/ops.h"

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cstring>
#include <type_traits>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/unary_ops.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"

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
#include "sleef.h"
HWY_BEFORE_NAMESPACE();

namespace rapidudf {
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
static constexpr size_t get_lanes() {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  return hn::Lanes(d);
}

template <class D, typename T, class Func>
void do_unary_transform(D d, const T* in1, size_t count, T* out, const Func& func) {
  const size_t N = hn::Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      const hn::Vec<D> v1 = hn::LoadU(d, in1 + idx);
      hn::StoreU(func(d, v1), d, out + idx);
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;
  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v1 = hn::LoadN(d, in1 + idx, remaining);
  hn::StoreN(func(d, v1), d, out + idx, remaining);
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static inline auto do_simd_unary_op([[maybe_unused]] D d, V lv) {
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

template <typename T>
static auto get_constant(T v) {
  if constexpr (std::is_same_v<Bit, T>) {
    uint8_t t = v ? 1 : 0;
    return t;
  } else {
    return v;
  }
}

template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_unary_op_impl(Context& ctx, Vector<typename OPT::operand_t> left) {
  using number_t = typename InternalType<typename OPT::operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<number_t>(lanes, left.Size(), true);
  }
  auto transform_func = do_simd_unary_op<decltype(d), OPT::op>;
  do_unary_transform(d, left.Data(), left.ElementSize(), result_data.MutableData<number_t>(), transform_func);
  return Vector<typename OPT::operand_t>(result_data);
}

template <typename T>
T simd_vector_dot_impl(Vector<T> left, Vector<T> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left.Size(), right.Size()));
  }
  using D = hn::ScalableTag<T>;
  const D d;
  constexpr auto lanes = hn::Lanes(d);
  T val;
  if (left.Size() >= lanes) {
    constexpr auto assumptions = hn::Dot::Assumptions::kAtLeastOneVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left.Data(), right.Data(), left.Size());
  } else {
    constexpr auto assumptions = hn::Dot::Assumptions::kPaddedToVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left.Data(), right.Data(), left.Size());
  }
  return val;
}

template <typename T>
T simd_vector_sum_impl(Vector<T> left) {
  using number_t = typename InternalType<T>::internal_type;
  T sum = {};
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  for (; (i + lanes) < left.Size(); i += lanes) {
    auto lv = hn::LoadU(d, left.Data() + i);
    auto sum_v = hn::ReduceSum(d, lv);
    sum += sum_v;
  }
  if (i < left.Size()) {
    for (; i < left.Size(); i++) {
      sum += left[i];
    }
  }
  return sum;
}

template <typename T>
Vector<T> simd_vector_iota_impl(Context& ctx, T start, uint32_t n) {
  // auto result_data = arena_new_vector<T>(n);
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  auto result_data = ctx.NewSimdVector<T>(lanes, n, true);
  uint8_t* arena_data = result_data.template MutableData<uint8_t>();
  size_t i = 0;
  for (; i < n; i += lanes) {
    auto v = hn::Iota(d, start + i);
    hn::StoreU(v, d, reinterpret_cast<T*>(arena_data + i * sizeof(T)));
  }
  return Vector<T>(result_data);
}

template <typename T>
Vector<T> simd_vector_clone_impl(Context& ctx, Vector<T> data) {
  auto result_data = ctx.NewSimdVector<T>(get_lanes<T>(), data.Size(), true);
  memcpy(result_data.template MutableData<uint8_t>(), data.Data(), result_data.BytesCapacity());
  return Vector<T>(result_data);
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace simd {

template <typename T>
Vector<T> simd_vector_clone(Context& ctx, Vector<T> data) {
  auto* p = ctx.ArenaAllocate(data.BytesCapacity());
  memcpy(p, data.Data(), data.BytesCapacity());
  VectorData result_data(p, data.Size(), data.BytesCapacity());
  return Vector<T>(data);
}
template <typename T>
Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n) {
  HWY_EXPORT_T(Table1, simd_vector_iota_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, start, n);
}
template <typename T>
T simd_vector_sum(Vector<T> left) {
  HWY_EXPORT_T(Table1, simd_vector_sum_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left);
}

template <typename T>
T simd_vector_dot(Vector<T> left, Vector<T> right) {
  HWY_EXPORT_T(Table1, simd_vector_dot_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left, right);
}

template <typename T, OpToken op>
Vector<T> simd_vector_unary_op(Context& ctx, Vector<T> left) {
  using OPT = OpTypes<T, op>;
  HWY_EXPORT_T(Table1, simd_vector_unary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left);
}

#define DEFINE_SIMD_UNARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_unary_op<TYPE, op>(Context & ctx, Vector<TYPE> left);
#define DEFINE_SIMD_UNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_UNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
DEFINE_SIMD_UNARY_OP(OP_NEGATIVE, float, double, int64_t, int32_t, int16_t, int8_t);
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
DEFINE_SIMD_UNARY_OP(OP_ABS, float, double, int64_t, int32_t, int16_t, int8_t);

#define DEFINE_SIMD_DOT_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_dot(Vector<TYPE> left, Vector<TYPE> right);
#define DEFINE_SIMD_DOT_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_DOT_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_DOT_OP(float, double);

#define DEFINE_SIMD_IOTA_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_iota(Context&, TYPE start, uint32_t n);
#define DEFINE_SIMD_IOTA_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_IOTA_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_IOTA_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_CLONE_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_clone(Context& ctx, Vector<TYPE> data);
#define DEFINE_SIMD_CLONE_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_CLONE_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_CLONE_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t, Bit,
                     StringView);

#define DEFINE_SIMD_SUM_OP_TEMPLATE(r, op, ii, TYPE) template TYPE simd_vector_sum(Vector<TYPE> vec);
#define DEFINE_SIMD_SUM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_SUM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_SUM_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE
