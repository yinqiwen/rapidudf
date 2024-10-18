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
#include "rapidudf/types/simd/vector.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vector_ternary.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <class D, OpToken op, typename V = hn::VFromD<D>>
static HWY_INLINE auto do_simd_ternary_op(D d, V a, V b, V c) {
  if constexpr (op == OP_CLAMP) {
    return hn::Clamp(a, b, c);
  } else if constexpr (op == OP_FMA) {
    return hn::MulAdd(a, b, c);
    // return hn::Add(hn::Mul(a, b), c);
  } else if constexpr (op == OP_FMS) {
    return hn::MulSub(a, b, c);
  } else if constexpr (op == OP_FNMA) {
    return hn::NegMulAdd(a, b, c);
  } else if constexpr (op == OP_FNMS) {
    return hn::NegMulSub(a, b, c);
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <class T, class Func>
HWY_INLINE void do_ternary_transform(const T* a, const T* b, const T* c, T* output, const Func& func) {
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  static_assert(simd::kVectorUnitSize % N == 0, "Invalid lanes");
  for (size_t idx = 0; idx < simd::kVectorUnitSize; idx += N) {
    const hn::Vec<D> v1 = hn::LoadU(d, a + idx);
    const hn::Vec<D> v2 = hn::LoadU(d, b + idx);
    const hn::Vec<D> v3 = hn::LoadU(d, c + idx);
    hn::StoreU(func(d, v1, v2, v3), d, output + idx);
  }
}

template <typename OPT>
HWY_INLINE void simd_vector_ternary_op_impl(const typename OPT::operand_t* a, const typename OPT::operand_t* b,
                                            const typename OPT::operand_t* c, typename OPT::operand_t* output) {
  using D = hn::ScalableTag<typename OPT::operand_t>;
  constexpr D d;
  auto transform_func = do_simd_ternary_op<D, OPT::op>;
  do_ternary_transform(a, b, c, output, transform_func);
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
template <typename T, OpToken op>
void simd_vector_ternary_op(const T* a, const T* b, const T* c, T* output) {
  using OPT = OperandType<T, op>;
  HWY_EXPORT_T(Table, simd_vector_ternary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(a, b, c, output);
}

#define DEFINE_SIMD_TERNARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template void simd_vector_ternary_op<TYPE, op>(const TYPE*, const TYPE*, const TYPE*, TYPE* output);
#define DEFINE_SIMD_TERNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
// DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
// DEFINE_SIMD_UNARY_OP(OP_NEGATIVE, float, double, int64_t, int32_t, int16_t, int8_t);
DEFINE_SIMD_TERNARY_OP(OP_CLAMP, float, double, uint64_t, int64_t, uint32_t, int32_t);
DEFINE_SIMD_TERNARY_OP(OP_FMA, float, double, uint64_t, int64_t, uint32_t, int32_t);
DEFINE_SIMD_TERNARY_OP(OP_FMS, float, double, uint64_t, int64_t, uint32_t, int32_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMA, float, double, uint64_t, int64_t, uint32_t, int32_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMS, float, double, uint64_t, int64_t, uint32_t, int32_t);
}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE