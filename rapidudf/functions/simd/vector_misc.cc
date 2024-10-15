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

#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd/vector.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vector_misc.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE T simd_vector_dot_impl(simd::Vector<T> left, simd::Vector<T> right) {
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
T simd_vector_sum_impl(simd::Vector<T> left) {
  T sum = {};
  const hn::ScalableTag<T> d;
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
HWY_INLINE simd::Vector<T> simd_vector_iota_impl(Context& ctx, T start, uint32_t n) {
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
  result_data.SetReadonly(false);
  return simd::Vector<T>(result_data);
}

template <typename T>
simd::Vector<T> simd_vector_gather_impl(Context& ctx, simd::Vector<T> data, simd::Vector<int32_t> indices) {
  using number_t = T;
  using D = hn::ScalableTag<number_t>;
  const D d;
  const T* base = data.Data();
  constexpr size_t N = hn::Lanes(d);
  constexpr hn::CappedTag<int32_t, N> indice_d;
  size_t dst_size = indices.Size();
  simd::VectorData result_data = ctx.NewSimdVector<T>(N, dst_size, true);
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
    return simd::Vector<T>(result_data);
  }
  for (size_t i = idx; i < count; i++) {
    dst[i] = base[indices[i]];
  }
  return simd::Vector<T>(result_data);
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
template <typename T>
simd::Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n) {
  HWY_EXPORT_T(Table1, simd_vector_iota_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, start, n);
}

template <typename T>
T simd_vector_sum(simd::Vector<T> left) {
  HWY_EXPORT_T(Table1, simd_vector_sum_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left);
}

template <typename T>
T simd_vector_dot(simd::Vector<T> left, simd::Vector<T> right) {
  HWY_EXPORT_T(Table1, simd_vector_dot_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left, right);
}

template <typename T>
T simd_vector_avg(simd::Vector<T> left) {
  T sum = simd_vector_sum(left);
  return sum / left.Size();
}

template <typename T>
simd::Vector<T> simd_vector_filter(Context& ctx, simd::Vector<T> data, simd::Vector<Bit> bits) {
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
  } else {
    for (size_t i = 0; i < data.Size(); i++) {
      if (bits[i]) {
        raw[filter_cursor++] = data[i];
      }
    }
  }

  simd::VectorData vdata(raw, filter_cursor, data.BytesCapacity());
  vdata.SetTemporary(true);
  vdata.SetReadonly(false);
  return simd::Vector<T>(vdata);
}

template <typename T>
simd::Vector<T> simd_vector_gather(Context& ctx, simd::Vector<T> data, simd::Vector<int32_t> indices) {
  if constexpr (std::is_same_v<StringView, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint16_t, T> ||
                std::is_same_v<int8_t, T> || std::is_same_v<uint8_t, T>) {
    T* raw = reinterpret_cast<T*>(ctx.ArenaAllocate(sizeof(T) * indices.Size()));
    for (size_t i = 0; i < indices.Size(); i++) {
      raw[i] = data[indices[i]];
    }
    simd::VectorData vdata(raw, indices.Size());
    vdata.SetTemporary(true);
    return simd::Vector<T>(vdata);
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
    simd::VectorData vdata(raw, indices.Size());
    vdata.SetTemporary(true);
    return simd::Vector<T>(vdata);
  } else if constexpr (std::is_same_v<T, Pointer>) {
    simd::Vector<uint64_t> pointers(data.GetVectorData());
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<uint64_t>);
    auto result = HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, pointers, indices);
    return simd::Vector<T>(result.GetVectorData());
  } else {
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<T>);
    return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, data, indices);
  }
}

#define DEFINE_SIMD_DOT_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_dot(simd::Vector<TYPE> left, simd::Vector<TYPE> right);
#define DEFINE_SIMD_DOT_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_DOT_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_DOT_OP(float, double);

#define DEFINE_SIMD_IOTA_OP_TEMPLATE(r, op, ii, TYPE) \
  template simd::Vector<TYPE> simd_vector_iota(Context&, TYPE start, uint32_t n);
#define DEFINE_SIMD_IOTA_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_IOTA_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_IOTA_OP(float, double, uint64_t, int64_t, uint32_t, int32_t);

#define DEFINE_SIMD_SUM_OP_TEMPLATE(r, op, ii, TYPE)     \
  template TYPE simd_vector_sum(simd::Vector<TYPE> vec); \
  template TYPE simd_vector_avg(simd::Vector<TYPE> vec);
#define DEFINE_SIMD_SUM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_SUM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

DEFINE_SIMD_SUM_OP(float, double, uint64_t, int64_t, uint32_t, int32_t);

#define DEFINE_SIMD_GATHER_OP_TEMPLATE(r, op, ii, TYPE) \
  template simd::Vector<TYPE> simd_vector_gather(Context& ctx, simd::Vector<TYPE> data, simd::Vector<int32_t> indices);
#define DEFINE_SIMD_GATHER_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_GATHER_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_GATHER_OP(float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                      StringView, Bit, Pointer);

#define DEFINE_SIMD_FILTER_OP_TEMPLATE(r, op, ii, TYPE) \
  template simd::Vector<TYPE> simd_vector_filter(Context& ctx, simd::Vector<TYPE> data, simd::Vector<Bit> bits);
#define DEFINE_SIMD_FILTER_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_FILTER_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_FILTER_OP(float, double, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                      StringView, Bit, Pointer);
}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE