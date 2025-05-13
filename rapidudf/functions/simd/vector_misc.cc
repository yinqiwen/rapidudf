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
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vector_misc.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE T simd_vector_dot_impl(const T* left, size_t left_len, const T* right, size_t right_len) {
  if (left_len != right_len) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left_len, right_len));
  }
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr auto lanes = hn::Lanes(d);
  T val;
  if (left_len >= lanes) {
    constexpr auto assumptions = hn::Dot::Assumptions::kAtLeastOneVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left, right, left_len);
  } else {
    constexpr auto assumptions = hn::Dot::Assumptions::kPaddedToVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left, right, left_len);
  }
  return val;
}

template <typename T>
HWY_INLINE T simd_vector_cos_distance_impl(const T* left, size_t left_len, const T* right, size_t right_len) {
  if (left_len != right_len) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left_len, right_len));
  }
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr auto N = hn::Lanes(d);
  size_t idx = 0;
  size_t count = left_len;
  hn::Vec<D> dot_v = hn::Zero(d);
  hn::Vec<D> norm_left_v = hn::Zero(d);
  hn::Vec<D> norm_right_v = hn::Zero(d);

  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      auto lv = hn::LoadU(d, left + idx);
      auto rv = hn::LoadU(d, right + idx);
      dot_v = hn::MulAdd(lv, rv, dot_v);
      norm_left_v = hn::MulAdd(lv, lv, norm_left_v);
      norm_right_v = hn::MulAdd(rv, rv, norm_right_v);
    }
  }
  if (HWY_LIKELY(idx != left_len)) {
    const size_t remaining = left_len - idx;
    const hn::Vec<D> lv = hn::LoadN(d, left + idx, remaining);
    const hn::Vec<D> rv = hn::LoadN(d, right + idx, remaining);
    dot_v = hn::MulAdd(lv, rv, dot_v);
    norm_left_v = hn::MulAdd(lv, lv, norm_left_v);
    norm_right_v = hn::MulAdd(rv, rv, norm_right_v);
  }
  T norm_left = std::sqrt(hn::ReduceSum(d, norm_left_v));
  T norm_right = std::sqrt(hn::ReduceSum(d, norm_right_v));
  T dot = hn::ReduceSum(d, dot_v);
  T cosine_similarity = dot / (norm_left * norm_right);

  return 1.0f - cosine_similarity;
}

template <typename T>
HWY_INLINE T simd_vector_l2_distance_impl(const T* left, size_t left_len, const T* right, size_t right_len) {
  if (left_len != right_len) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left_len, right_len));
  }
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr auto N = hn::Lanes(d);
  size_t idx = 0;
  size_t count = left_len;
  hn::Vec<D> distance_v = hn::Zero(d);
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      auto lv = hn::LoadU(d, left + idx);
      auto rv = hn::LoadU(d, right + idx);
      auto sub_v = hn::Abs(hn::Sub(lv, rv));
      distance_v = hn::MulAdd(sub_v, sub_v, distance_v);
    }
  }
  if (HWY_LIKELY(idx != count)) {
    const size_t remaining = count - idx;
    const hn::Vec<D> lv = hn::LoadN(d, left + idx, remaining);
    const hn::Vec<D> rv = hn::LoadN(d, right + idx, remaining);
    auto sub_v = hn::Abs(hn::Sub(lv, rv));
    distance_v = hn::MulAdd(sub_v, sub_v, distance_v);
  }
  return std::sqrt(hn::ReduceSum(d, distance_v));
}

template <typename T>
HWY_INLINE T simd_vector_sum_impl(const T* left, size_t left_len) {
  T sum = {};
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  for (; (i + lanes) < left_len; i += lanes) {
    auto lv = hn::LoadU(d, left + i);
    auto sum_v = hn::ReduceSum(d, lv);
    sum += sum_v;
  }
  if (i < left_len) {
    for (; i < left_len; i++) {
      sum += left[i];
    }
  }
  return sum;
}

template <typename T>
HWY_INLINE T simd_vector_reduce_max_impl(const T* left, size_t left_len) {
  T max_val = std::numeric_limits<T>::min();
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  for (; (i + lanes) < left_len; i += lanes) {
    auto lv = hn::LoadU(d, left + i);
    auto max_v = hn::ReduceMax(d, lv);
    if (max_v > max_val) {
      max_val = max_v;
    }
  }
  if (i < left_len) {
    for (; i < left_len; i++) {
      if (left[i] > max_val) {
        max_val = left[i];
      }
    }
  }
  return max_val;
}

template <typename T>
HWY_INLINE T simd_vector_reduce_min_impl(const T* left, size_t left_len) {
  T min_val = std::numeric_limits<T>::max();
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  for (; (i + lanes) < left_len; i += lanes) {
    auto lv = hn::LoadU(d, left + i);
    auto min_v = hn::ReduceMin(d, lv);
    if (min_v < min_val) {
      min_val = min_v;
    }
  }
  if (i < left_len) {
    for (; i < left_len; i++) {
      if (left[i] < min_val) {
        min_val = left[i];
      }
    }
  }
  return min_val;
}

template <typename T>
HWY_INLINE Vector<T> simd_vector_iota_impl(Context& ctx, T start, uint32_t n) {
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  auto result_data = ctx.NewVectorBuf<T>(n, lanes * sizeof(T));
  uint8_t* arena_data = result_data.template MutableData<uint8_t>();
  size_t i = 0;
  for (; i < n; i += lanes) {
    auto v = hn::Iota(d, start + i);
    hn::StoreU(v, d, reinterpret_cast<T*>(arena_data + i * sizeof(T)));
  }
  result_data.SetReadonly(false);
  return Vector<T>(result_data);
}

template <typename T>
HWY_INLINE Vector<T> simd_vector_gather_impl(Context& ctx, Vector<T> data, Vector<int32_t> indices) {
  using number_t = T;
  using D = hn::ScalableTag<number_t>;
  const D d;
  const T* base = data.Data();
  constexpr size_t N = hn::Lanes(d);
  constexpr hn::CappedTag<int32_t, N> indice_d;
  size_t dst_size = indices.Size();
  VectorBuf result_data = ctx.NewVectorBuf<T>(dst_size, N * sizeof(T));
  result_data.SetReadonly(false);
  T* dst = result_data.MutableData<T>();
  const int32_t* indice_data = indices.Data();
  size_t idx = 0;
  size_t count = indices.Size();
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      auto indice = hn::LoadU(indice_d, indice_data + idx);
      hn::VFromD<D> gathered_val;
      if constexpr (sizeof(uint32_t) == sizeof(T)) {
        gathered_val = hn::GatherIndex(d, base, indice);
      } else {
        using ToIndiceD = hn::RebindToSigned<D>;
        ToIndiceD to_d;
        auto promoted_indice = hn::PromoteTo(to_d, indice);
        gathered_val = hn::GatherIndex(d, base, promoted_indice);
      }
      hn::StoreU(gathered_val, d, dst + idx);
    }
  }

  if (HWY_UNLIKELY(idx == count)) {
    return Vector<T>(result_data);
  }
  for (size_t i = idx; i < count; i++) {
    dst[i] = base[indices[i]];
  }
  return Vector<T>(result_data);
}

template <typename OPT>
HWY_INLINE size_t simd_vector_find_impl(Vector<typename OPT::operand_t> data, typename OPT::operand_t v) {
  using D = hn::ScalableTag<typename OPT::operand_t>;
  constexpr D d;
  if constexpr (OPT::op == OP_EQUAL) {
    return hn::Find(d, v, data.Data(), data.Size());
  } else {
    if constexpr (OPT::op == OP_GREATER_EQUAL) {
      return hn::FindIf(d, data.Data(), data.Size(),
                        [v](const auto d, const auto vec) HWY_ATTR { return hn::Ge(vec, hn::Set(d, v)); });
    } else if constexpr (OPT::op == OP_GREATER) {
      return hn::FindIf(d, data.Data(), data.Size(),
                        [v](const auto d, const auto vec) HWY_ATTR { return hn::Gt(vec, hn::Set(d, v)); });
    } else if constexpr (OPT::op == OP_NOT_EQUAL) {
      return hn::FindIf(d, data.Data(), data.Size(),
                        [v](const auto d, const auto vec) HWY_ATTR { return hn::Ne(vec, hn::Set(d, v)); });
    } else if constexpr (OPT::op == OP_LESS_EQUAL) {
      return hn::FindIf(d, data.Data(), data.Size(),
                        [v](const auto d, const auto vec) HWY_ATTR { return hn::Le(vec, hn::Set(d, v)); });
    } else if constexpr (OPT::op == OP_LESS) {
      return hn::FindIf(d, data.Data(), data.Size(),
                        [v](const auto d, const auto vec) HWY_ATTR { return hn::Lt(vec, hn::Set(d, v)); });
    }
  }
}

template <typename T>
HWY_INLINE void simd_vector_random_impl(Context& ctx, uint64_t seed, T* output) {
  hn::VectorXoshiro* rand = ctx.GetPtr<hn::VectorXoshiro>({}, seed);
  if constexpr (std::is_same_v<double, T>) {
    auto result = rand->Uniform(kVectorUnitSize);
    memcpy(output, result.data(), result.size() * sizeof(T));
  } else if constexpr (std::is_same_v<uint64_t, T>) {
    auto result = (*rand)(kVectorUnitSize);
    memcpy(output, result.data(), result.size() * sizeof(T));
  } else {
    static_assert(sizeof(T) == -1, "Invalid random");
  }
}

template <typename T>
HWY_INLINE T random_impl(uint64_t seed) {
  hn::internal::Xoshiro rand(seed);
  if constexpr (std::is_same_v<double, T>) {
    return rand.Uniform();
  } else if constexpr (std::is_same_v<uint64_t, T>) {
    return rand();
  } else {
    static_assert(sizeof(T) == -1, "Invalid random");
  }
}

HWY_INLINE const uint8_t* get_mask_bits(const uint8_t* bits, size_t idx, uint64_t& tmp) {
  size_t bits_offset = idx / 8;
  size_t bits_cursor = idx % 8;
  uint8_t shift_bits = (bits[bits_offset] >> bits_cursor);
  tmp = shift_bits;
  return reinterpret_cast<uint8_t*>(&tmp);
}

template <typename D>
HWY_INLINE void store_mask_bits(hn::Mask<D> mask, uint8_t* bits, size_t idx) {
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  size_t bits_offset = idx / 8;
  size_t bits_cursor = idx % 8;

  if constexpr (N < 8) {
    uint8_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    hn::StoreMaskBits(d, mask, tmp);
    bits[bits_offset] = (bits[bits_offset] | (tmp[0] << bits_cursor));
  } else {
    hn::StoreMaskBits(d, mask, bits + bits_offset);
  }
}
template <typename D, OpToken op>
HWY_INLINE hn::Mask<D> compare_value(hn::Vec<D> left, hn::Vec<D> right) {
  if constexpr (op == OP_EQUAL) {
    return hn::Eq(left, right);
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return hn::Ge(left, right);
  } else if constexpr (op == OP_GREATER) {
    return hn::Gt(left, right);
  } else if constexpr (op == OP_NOT_EQUAL) {
    return hn::Ne(left, right);
  } else if constexpr (op == OP_LESS_EQUAL) {
    return hn::Le(left, right);
  } else if constexpr (op == OP_LESS) {
    return hn::Lt(left, right);
  }
}

template <typename T>
HWY_INLINE size_t simd_vector_filter_impl(const T* in, const uint8_t* bits, T* out, size_t count) {
  using D = hn::ScalableTag<T>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  if constexpr (N % 8 > 0) {
    size_t idx = 0;
    size_t out_offset = 0;
    uint64_t tmp_bits = 0;
    if (count >= N) {
      for (; idx <= count - N; idx += N) {
        auto v = hn::LoadU(d, in + idx);
        auto mask = hn::LoadMaskBits(d, get_mask_bits(bits, idx, tmp_bits));
        size_t n = hn::CompressBlendedStore(v, mask, d, out + out_offset);
        out_offset += n;
      }
    }
    // `count` was a multiple of the vector length `N`: already done.
    if (HWY_UNLIKELY(idx == count)) return out_offset;
    const size_t remaining = count - idx;
    HWY_DASSERT(0 != remaining && remaining < N);
    const hn::Vec<D> v = hn::LoadN(d, in + idx, remaining);
    auto mask = hn::And(hn::FirstN(d, remaining), hn::LoadMaskBits(d, get_mask_bits(bits, idx, tmp_bits)));
    out_offset += hn::CompressBlendedStore(v, mask, d, out + out_offset);
    return out_offset;
  } else {
    size_t idx = 0;
    size_t out_offset = 0;
    if (count >= N) {
      for (; idx <= count - N; idx += N) {
        auto v = hn::LoadU(d, in + idx);
        size_t bits_offset = idx / 8;
        out_offset += hn::CompressBitsStore(v, bits + bits_offset, d, out + out_offset);
      }
    }
    // `count` was a multiple of the vector length `N`: already done.
    if (HWY_UNLIKELY(idx == count)) return out_offset;
    const size_t remaining = count - idx;
    HWY_DASSERT(0 != remaining && remaining < N);
    const hn::Vec<D> v = hn::LoadN(d, in + idx, remaining);
    size_t bits_offset = idx / 8;
    auto mask = hn::And(hn::FirstN(d, remaining), hn::LoadMaskBits(d, bits + bits_offset));
    out_offset += hn::CompressBlendedStore(v, mask, d, out + out_offset);
    return out_offset;
  }
}

template <typename OPT>
HWY_INLINE size_t simd_vector_match_impl(const typename OPT::operand_t* data, size_t len, typename OPT::operand_t cmp,
                                         uint64_t& mask_bits) {
  size_t max_match_len = len;
  if (max_match_len > sizeof(uint64_t) * 8) {
    max_match_len = sizeof(uint64_t) * 8;
  }
  uint64_t mask_bits_tmp[2] = {0, 0};
  uint8_t* mask_bits_p = reinterpret_cast<uint8_t*>(mask_bits_tmp);
  using D = hn::ScalableTag<typename OPT::operand_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  const hn::Vec<D> cmp_v = hn::Set(d, cmp);
  size_t idx = 0;
  if (len >= N) {
    for (; idx <= max_match_len - N; idx += N) {
      const hn::Vec<D> v = hn::LoadU(d, data + idx);
      hn::Mask<D> mask = compare_value<D, OPT::op>(v, cmp_v);
      store_mask_bits<D>(mask, mask_bits_p, idx);
    }
  }
  if (HWY_UNLIKELY(idx == max_match_len)) {
    mask_bits = mask_bits_tmp[0];
    return max_match_len;
  }
  const size_t remaining = max_match_len - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v = hn::LoadN(d, data + idx, remaining);
  hn::Mask<D> mask = compare_value<D, OPT::op>(v, cmp_v);
  store_mask_bits<D>(mask, mask_bits_p, idx);
  mask_bits = mask_bits_tmp[0];
  return max_match_len;
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
template <typename T>
Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n) {
  HWY_EXPORT_T(Table1, simd_vector_iota_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, start, n);
}

template <typename T>
T simd_vector_sum(Vector<T> left) {
  HWY_EXPORT_T(Table1, simd_vector_sum_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left.Data(), left.Size());
}

template <typename T>
T simd_vector_dot_distance(Vector<T> left, Vector<T> right) {
  HWY_EXPORT_T(Table, simd_vector_dot_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left.Data(), left.Size(), right.Data(), right.Size());
}
template <typename T>
T simd_vector_cosine_distance(Vector<T> left, Vector<T> right) {
  HWY_EXPORT_T(Table, simd_vector_cos_distance_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left.Data(), left.Size(), right.Data(), right.Size());
}

template <typename T>
T simd_vector_l2_distance(Vector<T> left, Vector<T> right) {
  HWY_EXPORT_T(Table, simd_vector_l2_distance_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left.Data(), left.Size(), right.Data(), right.Size());
}

template <typename T>
T simd_vector_avg(Vector<T> left) {
  T sum = simd_vector_sum(left);
  return sum / left.Size();
}

template <typename T>
Vector<T> simd_vector_filter(Context& ctx, Vector<T> data, Vector<Bit> bits) {
  T* raw = reinterpret_cast<T*>(ctx.ArenaAllocate(data.BytesCapacity()));
  size_t filter_cursor = 0;
  if constexpr (std::is_same_v<T, Bit>) {
    uint64_t* bits64 = reinterpret_cast<uint64_t*>(raw);
    for (size_t idx = 0; idx < data.Size(); idx++) {
      if (bits[idx]) {
        Bit bit = data[filter_cursor++];
        size_t bits_idx = filter_cursor / 64;
        size_t bits_cursor = filter_cursor % 64;
        if (bit) {
          bits64[bits_idx] = bits64_set(bits64[bits_idx], bits_cursor);
        } else {
          bits64[bits_idx] = bits64_clear(bits64[bits_idx], bits_cursor);
        }
        filter_cursor++;
      }
    }
    // } else {
    //   for (size_t i = 0; i < data.Size(); i++) {
    //     if (bits[i]) {
    //       raw[filter_cursor++] = data[i];
    //     }
    //   }
    // }
  } else if constexpr (std::is_same_v<T, StringView>) {
    for (size_t i = 0; i < data.Size(); i++) {
      if (bits[i]) {
        raw[filter_cursor++] = data[i];
      }
    }
  } else if constexpr (std::is_same_v<T, Pointer>) {
    HWY_EXPORT_T(Table, simd_vector_filter_impl<uint64_t>);
    const uint64_t* in = reinterpret_cast<const uint64_t*>(data.Data());
    uint64_t* out = reinterpret_cast<uint64_t*>(raw);
    filter_cursor = HWY_DYNAMIC_DISPATCH_T(Table)(in, bits.Data(), out, data.Size());
  } else {
    HWY_EXPORT_T(Table, simd_vector_filter_impl<T>);
    filter_cursor = HWY_DYNAMIC_DISPATCH_T(Table)(data.Data(), bits.Data(), raw, data.Size());
  }

  VectorBuf vdata(raw, filter_cursor, data.BytesCapacity());
  vdata.SetReadonly(false);
  return Vector<T>(vdata);
}

template <typename T>
Vector<T> simd_vector_gather(Context& ctx, Vector<T> data, Vector<int32_t> indices) {
  if constexpr (std::is_same_v<StringView, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint16_t, T> ||
                std::is_same_v<int8_t, T> || std::is_same_v<uint8_t, T>) {
    T* raw = reinterpret_cast<T*>(ctx.ArenaAllocate(sizeof(T) * indices.Size()));
    for (size_t i = 0; i < indices.Size(); i++) {
      raw[i] = data[indices[i]];
    }
    VectorBuf vdata(raw, indices.Size());
    vdata.SetTemporary(true);
    return Vector<T>(vdata);
  } else if constexpr (std::is_same_v<T, Bit>) {
    size_t n = (indices.Size() + 7) / 8;
    uint64_t* raw = reinterpret_cast<uint64_t*>(ctx.ArenaAllocate(sizeof(uint64_t) * n));
    for (size_t i = 0; i < indices.Size(); i++) {
      size_t bits_idx = i / 64;
      size_t bits_cursor = i % 64;
      if (data[indices[i]]) {
        raw[bits_idx] = bits64_set(raw[bits_idx], bits_cursor);
      } else {
        raw[bits_idx] = bits64_clear(raw[bits_idx], bits_cursor);
      }
    }
    VectorBuf vdata(raw, indices.Size());
    return Vector<T>(vdata);
  } else if constexpr (std::is_same_v<T, Pointer>) {
    Vector<uint64_t> pointers(data.GetVectorBuf());
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<uint64_t>);
    auto result = HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, pointers, indices);
    return Vector<T>(result.GetVectorBuf());
  } else {
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<T>);
    return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, data, indices);
  }
}

template <typename T, OpToken op>
int simd_vector_find(Vector<T> data, T v) {
  if constexpr (std::is_same_v<StringView, T>) {
    for (size_t i = 0; i < data.Size(); i++) {
      if (compare_string_view(op, data[i], v)) {
        return i;
      }
    }
    return -1;
  } else {
    using OPT = OperandType<T, op>;
    HWY_EXPORT_T(Table, simd_vector_find_impl<OPT>);
    size_t n = HWY_DYNAMIC_DISPATCH_T(Table)(data, v);
    if (n == data.Size()) {
      return -1;
    }
    return static_cast<int>(n);
  }
}

template <typename T, OpToken op>
size_t simd_vector_match(const T* data, size_t len, T v, uint64_t& mask) {
  using OPT = OperandType<T, op>;
  HWY_EXPORT_T(Table, simd_vector_match_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(data, len, v, mask);
}

template <typename T>
T simd_vector_reduce_max(Vector<T> left) {
  HWY_EXPORT_T(Table, simd_vector_reduce_max_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left.Data(), left.Size());
}
template <typename T>
T simd_vector_reduce_min(Vector<T> left) {
  HWY_EXPORT_T(Table, simd_vector_reduce_min_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(left.Data(), left.Size());
}

template <typename T>
void simd_vector_random(Context& ctx, uint64_t seed, T* output) {
  HWY_EXPORT_T(Table, simd_vector_random_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(ctx, seed, output);
}

template <typename T>
T random(uint64_t seed) {
  HWY_EXPORT_T(Table, random_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table)(seed);
}

#define DEFINE_SIMD_DOT_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_dot_distance(Vector<TYPE> left, Vector<TYPE> right);
#define DEFINE_SIMD_DOT_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_DOT_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_DOT_OP(float, double);

#define DEFINE_SIMD_COS_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_cosine_distance(Vector<TYPE> left, Vector<TYPE> right);
#define DEFINE_SIMD_COS_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_COS_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_COS_OP(float, double);

#define DEFINE_SIMD_L2_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_l2_distance(Vector<TYPE> left, Vector<TYPE> right);
#define DEFINE_SIMD_L2_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_L2_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_L2_OP(float, double);

#define DEFINE_SIMD_IOTA_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_iota(Context&, TYPE start, uint32_t n);
#define DEFINE_SIMD_IOTA_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_IOTA_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_IOTA_OP(float, double, uint64_t, int64_t, uint32_t, int32_t);

#define DEFINE_SIMD_SUM_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_sum(Vector<TYPE> vec);   \
  template TYPE simd_vector_avg(Vector<TYPE> vec);
#define DEFINE_SIMD_SUM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_SUM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

DEFINE_SIMD_SUM_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_REDUCE_MAX_MIN_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_reduce_max(Vector<TYPE> vec);       \
  template TYPE simd_vector_reduce_min(Vector<TYPE> vec);
#define DEFINE_SIMD_REDUCE_MAX_MIN_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_REDUCE_MAX_MIN_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_REDUCE_MAX_MIN_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_GATHER_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_gather(Context& ctx, Vector<TYPE> data, Vector<int32_t> indices);
#define DEFINE_SIMD_GATHER_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_GATHER_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_GATHER_OP(float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                      StringView, Bit, Pointer);

#define DEFINE_SIMD_FILTER_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_filter(Context& ctx, Vector<TYPE> data, Vector<Bit> bits);
#define DEFINE_SIMD_FILTER_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_FILTER_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_FILTER_OP(float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                      StringView, Bit, Pointer);

#define DEFINE_SIMD_FIND_OP_TEMPLATE(r, op, ii, TYPE) \
  template int simd_vector_find<TYPE, op>(Vector<TYPE> data, TYPE val);
#define DEFINE_SIMD_FIND_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_FIND_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_FIND_OP(OP_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                    StringView);
DEFINE_SIMD_FIND_OP(OP_GREATER_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                    int64_t, StringView);
DEFINE_SIMD_FIND_OP(OP_GREATER, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                    StringView);
DEFINE_SIMD_FIND_OP(OP_LESS, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                    StringView);
DEFINE_SIMD_FIND_OP(OP_LESS_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                    int64_t, StringView);
DEFINE_SIMD_FIND_OP(OP_NOT_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                    int64_t, StringView);

#define DEFINE_SIMD_MATCH_OP_TEMPLATE(r, op, ii, TYPE) \
  template size_t simd_vector_match<TYPE, op>(const TYPE* data, size_t len, TYPE v, uint64_t& mask);
#define DEFINE_SIMD_MATCH_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_MATCH_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_MATCH_OP(OP_EQUAL, uint8_t);
// DEFINE_SIMD_MATCH_OP(OP_GREATER_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
// uint64_t,
//                      int64_t);
// DEFINE_SIMD_MATCH_OP(OP_GREATER, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
//                      int64_t);
// DEFINE_SIMD_MATCH_OP(OP_LESS, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
// int64_t); DEFINE_SIMD_MATCH_OP(OP_LESS_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
// uint64_t,
//                      int64_t);
// DEFINE_SIMD_MATCH_OP(OP_NOT_EQUAL, float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
//                      int64_t);

#define DEFINE_SIMD_RANDOM_OP_TEMPLATE(r, op, ii, TYPE) \
  template void simd_vector_random(Context& ctx, uint64_t seed, TYPE* output);
#define DEFINE_SIMD_RANDOM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_RANDOM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_RANDOM_OP(uint64_t, double);

#define DEFINE_RANDOM_OP_TEMPLATE(r, op, ii, TYPE) template TYPE random(uint64_t seed);
#define DEFINE_RANDOM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_RANDOM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_RANDOM_OP(uint64_t, double);

}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE