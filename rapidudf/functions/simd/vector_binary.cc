/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd/vector.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vector_binary.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <x86intrin.h>

extern "C" {
extern __m128 Sleef_powf4_u10(__m128, __m128);
extern __m256 Sleef_powf8_u10(__m256, __m256);
extern __m512 Sleef_powf16_u10(__m512, __m512);

extern __m128d Sleef_powd2_u10(__m128d, __m128d);
extern __m256d Sleef_powd4_u10(__m256d, __m256d);
extern __m512d Sleef_powd8_u10(__m512d, __m512d);
}
#include "sleef.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <class D, OpToken op, typename V = hn::VFromD<D>>
HWY_INLINE auto do_simd_binary_op([[maybe_unused]] D d, V lv, V rv) {
  if constexpr (op == OP_HYPOT) {
    return hn::Hypot(d, lv, rv);
  } else if constexpr (op == OP_ATAN2) {
    return hn::Atan2(d, lv, rv);
  } else if constexpr (op == OP_MULTIPLY) {
    return hn::Mul(lv, rv);
  } else if constexpr (op == OP_DIVIDE) {
    return hn::Div(lv, rv);
  } else if constexpr (op == OP_POW) {
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
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <class T, class Func>
HWY_INLINE void do_binary_transform(const T* left, const T* right, T* output, const Func& func) {
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  static_assert(simd::kVectorUnitSize % N == 0, "Invalid lanes");
  for (size_t idx = 0; idx < simd::kVectorUnitSize; idx += N) {
    const hn::Vec<D> v1 = hn::LoadU(d, left + idx);
    const hn::Vec<D> v2 = hn::LoadU(d, right + idx);
    hn::StoreU(func(d, v1, v2), d, output + idx);
  }
}

template <typename OPT>
HWY_INLINE void simd_vector_binary_op_impl(const typename OPT::operand_t* left, const typename OPT::operand_t* right,
                                           typename OPT::operand_t* output) {
  using D = hn::ScalableTag<typename OPT::operand_t>;
  constexpr D d;
  auto transform_func = do_simd_binary_op<D, OPT::op>;
  do_binary_transform(left, right, output, transform_func);
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
template <typename T, OpToken op>
void simd_vector_binary_op(const T* left, const T* right, T* output) {
  using OPT = OperandType<T, op>;
  HWY_EXPORT_T(Table, simd_vector_binary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left, right, output);
}

#define DEFINE_SIMD_BINARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template void simd_vector_binary_op<TYPE, op>(const TYPE*, const TYPE*, TYPE* output);
#define DEFINE_SIMD_BINARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
// DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
// DEFINE_SIMD_UNARY_OP(OP_NEGATIVE, float, double, int64_t, int32_t, int16_t, int8_t);
DEFINE_SIMD_BINARY_OP(OP_HYPOT, float, double);
DEFINE_SIMD_BINARY_OP(OP_ATAN2, float, double);
DEFINE_SIMD_BINARY_OP(OP_POW, float, double);
// DEFINE_SIMD_BINARY_OP(OP_MULTIPLY, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
//                       int8_t);
// DEFINE_SIMD_BINARY_OP(OP_DIVIDE, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
//                       int8_t);
}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE