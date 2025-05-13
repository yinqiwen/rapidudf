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

#include "rapidudf/functions/simd/bits.h"
#include <cstring>
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/vector.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/bits.cc"  // this file

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
static HWY_INLINE auto do_simd_unary_op([[maybe_unused]] D d, V lv) {
  if constexpr (op == OP_NOT) {
    return hn::Not(lv);
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
HWY_INLINE auto do_simd_binary_op([[maybe_unused]] D d, V lv, V rv) {
  if constexpr (op == OP_LOGIC_AND) {
    return hn::And(lv, rv);
  } else if constexpr (op == OP_LOGIC_OR) {
    return hn::Or(lv, rv);
  } else if constexpr (op == OP_LOGIC_XOR) {
    return hn::Xor(lv, rv);
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

HWY_INLINE void simd_vector_bits_not_impl(const uint8_t* src, uint8_t* dst, size_t bit_length) {
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  size_t idx = 0;
  size_t count = bit_length;
  size_t byte_count = count / 8;
  if (count % 8 > 0) {
    byte_count++;
  }
  const uint8_t* in = src;
  uint8_t* out = dst;
  if (byte_count >= N) {
    for (; idx <= byte_count - N; idx += N) {
      const hn::Vec<D> v1 = hn::LoadU(d, in + idx);
      hn::StoreU(do_simd_unary_op<D, OP_NOT>(d, v1), d, out + idx);
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == byte_count)) return;
  const size_t remaining = byte_count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v1 = hn::LoadN(d, in + idx, remaining);
  hn::StoreN(do_simd_unary_op<D, OP_NOT>(d, v1), d, out + idx, remaining);
}

template <OpToken op>
HWY_INLINE void simd_vector_bits_binary_impl(const uint8_t* left, const uint8_t* right, uint8_t* dst,
                                             size_t bit_length) {
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  size_t idx = 0;
  size_t count = bit_length;
  size_t byte_count = count / 8;
  if (count % 8 > 0) {
    byte_count++;
  }
  const uint8_t* left_in = left;
  const uint8_t* right_in = right;
  uint8_t* out = dst;
  if (byte_count >= N) {
    for (; idx <= byte_count - N; idx += N) {
      const hn::Vec<D> v1 = hn::LoadU(d, left_in + idx);
      const hn::Vec<D> v2 = hn::LoadU(d, right_in + idx);
      hn::StoreU(do_simd_binary_op<D, op>(d, v1, v2), d, out + idx);
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == byte_count)) return;
  const size_t remaining = byte_count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v1 = hn::LoadN(d, left_in + idx, remaining);
  const hn::Vec<D> v2 = hn::LoadN(d, right_in + idx, remaining);
  hn::StoreN(do_simd_binary_op<D, op>(d, v1, v2), d, out + idx, remaining);
}

// template <OpToken op>
// HWY_INLINE void simd_vector_bits_binary_impl(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst) {
//   using D = hn::ScalableTag<uint8_t>;
//   constexpr D d;
//   constexpr size_t N = hn::Lanes(d);
//   size_t idx = 0;
//   size_t count = left.Size();
//   size_t byte_count = count / 8;
//   if (count % 8 > 0) {
//     byte_count++;
//   }
//   const uint8_t* left_in = left.GetVectorBuf().ReadableData<uint8_t>();
//   const uint8_t* right_in = right.GetVectorBuf().ReadableData<uint8_t>();
//   uint8_t* out = dst.GetVectorBuf().MutableData<uint8_t>();
//   if (byte_count >= N) {
//     for (; idx <= byte_count - N; idx += N) {
//       const hn::Vec<D> v1 = hn::LoadU(d, left_in + idx);
//       const hn::Vec<D> v2 = hn::LoadU(d, right_in + idx);
//       hn::StoreU(do_simd_binary_op<D, op>(d, v1, v2), d, out + idx);
//     }
//   }
//   // `count` was a multiple of the vector length `N`: already done.
//   if (HWY_UNLIKELY(idx == byte_count)) return;
//   const size_t remaining = byte_count - idx;
//   HWY_DASSERT(0 != remaining && remaining < N);
//   const hn::Vec<D> v1 = hn::LoadN(d, left_in + idx, remaining);
//   const hn::Vec<D> v2 = hn::LoadN(d, right_in + idx, remaining);
//   hn::StoreN(do_simd_binary_op<D, op>(d, v1, v2), d, out + idx, remaining);
// }

HWY_INLINE size_t simd_vector_count_true_impl(const uint8_t* data, size_t bit_length) {
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  size_t idx = 0;
  size_t count = bit_length;
  size_t byte_count = count / 8;
  if (count % 8 > 0) {
    byte_count++;
  }

  const uint8_t* src_in = data;
  size_t n = 0;
  if (byte_count >= N) {
    for (; idx <= byte_count - N; idx += N) {
      const hn::Vec<D> v = hn::LoadU(d, src_in + idx);
      auto results = hn::PopulationCount(v);
      n += hn::ReduceSum(d, results);
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == byte_count)) return n;
  const size_t remaining = byte_count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v = hn::LoadN(d, src_in + idx, remaining);
  auto results = hn::PopulationCount(v);
  n += hn::ReduceSum(d, results);
  return n;
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
void simd_vector_bits_not(Vector<Bit> src, Vector<Bit> dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_not_impl);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (src.GetVectorBuf().ReadableData<uint8_t>(), dst.GetVectorBuf().MutableData<uint8_t>(), src.Size());
}
void simd_vector_bits_and(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_AND>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left.GetVectorBuf().ReadableData<uint8_t>(), right.GetVectorBuf().ReadableData<uint8_t>(),
   dst.GetVectorBuf().MutableData<uint8_t>(), left.Size());
}
void simd_vector_bits_or(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_OR>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left.GetVectorBuf().ReadableData<uint8_t>(), right.GetVectorBuf().ReadableData<uint8_t>(),
   dst.GetVectorBuf().MutableData<uint8_t>(), left.Size());
}
void simd_vector_bits_xor(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_XOR>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left.GetVectorBuf().ReadableData<uint8_t>(), right.GetVectorBuf().ReadableData<uint8_t>(),
   dst.GetVectorBuf().MutableData<uint8_t>(), left.Size());
}
size_t simd_vector_bits_count_true(Vector<Bit> src) {
  HWY_EXPORT_T(Table, simd_vector_count_true_impl);
  return HWY_DYNAMIC_DISPATCH_T(Table)(src.GetVectorBuf().ReadableData<uint8_t>(), src.Size());
}

void simd_vector_bits_not(const ArrowVector<Bit>* src, ArrowVector<Bit>* dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_not_impl);
  return HWY_DYNAMIC_DISPATCH_T(Table)(src->Data(), dst->MutableData(), src->Size());
}

void simd_vector_bits_and(const ArrowVector<Bit>* left, const ArrowVector<Bit>* right, ArrowVector<Bit>* dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_AND>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left->Data(), right->Data(), dst->MutableData(), left->Size());
}
void simd_vector_bits_or(const ArrowVector<Bit>* left, const ArrowVector<Bit>* right, ArrowVector<Bit>* dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_OR>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left->Data(), right->Data(), dst->MutableData(), left->Size());
}
void simd_vector_bits_xor(const ArrowVector<Bit>* left, const ArrowVector<Bit>* right, ArrowVector<Bit>* dst) {
  HWY_EXPORT_T(Table, simd_vector_bits_binary_impl<OP_LOGIC_XOR>);
  HWY_DYNAMIC_DISPATCH_T(Table)
  (left->Data(), right->Data(), dst->MutableData(), left->Size());
}

size_t simd_vector_bits_count_true(const ArrowVector<Bit>* src) {
  HWY_EXPORT_T(Table, simd_vector_count_true_impl);
  return HWY_DYNAMIC_DISPATCH_T(Table)(src->Data(), src->Size());
}

}  // namespace functions

}  // namespace rapidudf
#endif  // HWY_ONCE