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

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <type_traits>
#include "rapidudf/log/log.h"
#include "rapidudf/types/string_view.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/misc_ops.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "rapidudf/builtin/simd_vector/ops.h"
#include "rapidudf/types/simd_vector.h"

HWY_BEFORE_NAMESPACE();

namespace rapidudf {
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
Vector<T> simd_vector_gather_impl(Context& ctx, Vector<T> data, Vector<int32_t> indices) {
  using number_t = T;
  using D = hn::ScalableTag<number_t>;
  const D d;
  const T* base = data.Data();
  constexpr size_t N = hn::Lanes(d);
  constexpr hn::CappedTag<int32_t, N> indice_d;
  size_t dst_size = indices.Size();
  VectorData result_data = ctx.NewSimdVector<T>(N, dst_size, true);
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

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace rapidudf {
namespace simd {
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
  } else {
    for (size_t i = 0; i < data.Size(); i++) {
      if (bits[i]) {
        raw[filter_cursor++] = data[i];
      }
    }
  }

  VectorData vdata(raw, filter_cursor, data.BytesCapacity());
  vdata.SetTemporary(true);
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
    VectorData vdata(raw, indices.Size());
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
    VectorData vdata(raw, indices.Size());
    vdata.SetTemporary(true);
    return Vector<T>(vdata);
  } else if constexpr (std::is_same_v<T, Pointer>) {
    Vector<uint64_t> pointers(data.GetVectorData());
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<uint64_t>);
    auto result = HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, pointers, indices);
    return Vector<T>(result.GetVectorData());
  } else {
    HWY_EXPORT_T(Table1, simd_vector_gather_impl<T>);
    return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, data, indices);
  }
}

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

}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE