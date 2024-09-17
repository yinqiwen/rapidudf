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
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/binary_ops.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "hwy/bit_set.h"
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
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
static constexpr size_t get_lanes() {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  return hn::Lanes(d);
}

template <class D, class Func, typename T1, typename T2, typename OUT>
void do_binary_transform(D d, T1 in1, T2 in2, size_t count, OUT* out, const Func& func) {
  constexpr size_t N = Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      hn::Vec<D> v1;
      if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
        v1 = in1;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
        v1 = hn::LoadU(d, in1 + idx);
      } else {
        static_assert(sizeof(T1) == -1, "invalid T1");
      }
      hn::Vec<D> v2;
      if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
        v2 = in2;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
        v2 = hn::LoadU(d, in2 + idx);
      } else {
        static_assert(sizeof(T2) == -1, "invalid T2");
      }
      auto result = func(d, v1, v2);
      if constexpr (std::is_same_v<decltype(result), hn::VFromD<D>>) {
        if constexpr (std::is_same_v<hn::TFromD<D>, OUT>) {
          hn::StoreU(result, d, out + idx);
        } else {
          static_assert(sizeof(OUT) == -1, "invalid out type");
        }
      } else if constexpr (std::is_same_v<decltype(result), hn::Mask<D>>) {
        if constexpr (std::is_same_v<uint8_t, OUT>) {
          size_t bits_byte_idx = idx / 8;
          size_t bit_cursor = idx % 8;
          uint8_t cache_byte = 0;
          if constexpr (N < 8) {
            cache_byte = out[bits_byte_idx];
          }
          hn::StoreMaskBits(d, result, out + bits_byte_idx);
          if (bit_cursor > 0) {
            uint8_t current_byte = out[bits_byte_idx];
            uint8_t restore_value = ((current_byte << bit_cursor) | cache_byte);
            out[bits_byte_idx] = restore_value;
          }
        } else {
          static_assert(sizeof(OUT) == -1, "invalid out type");
        }
      } else {
        static_assert(sizeof(OUT) == -1, "invalid result type");
      }
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  hn::Vec<D> v1;
  if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
    v1 = in1;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
    v1 = hn::LoadN(d, in1 + idx, remaining);
  } else {
    static_assert(sizeof(T1) == -1, "invalid T1");
  }
  hn::Vec<D> v2;
  if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
    v2 = in2;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
    v2 = hn::LoadN(d, in2 + idx, remaining);
  } else {
    static_assert(sizeof(T2) == -1, "invalid T2");
  }
  auto result = func(d, v1, v2);
  if constexpr (std::is_same_v<decltype(result), hn::VFromD<D>>) {
    if constexpr (std::is_same_v<hn::TFromD<D>, OUT>) {
      hn::StoreN(result, d, out + idx, remaining);
    } else {
      static_assert(sizeof(OUT) == -1, "invalid out type");
    }
  } else if constexpr (std::is_same_v<decltype(result), hn::Mask<D>>) {
    if constexpr (std::is_same_v<uint8_t, OUT>) {
      size_t bits_byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t cache_byte = out[bits_byte_idx];
      hn::StoreMaskBits(d, result, out + bits_byte_idx);
      if (bit_cursor > 0) {
        uint8_t current_byte = out[bits_byte_idx];
        uint8_t restore_value = ((cache_byte << bit_cursor) | current_byte);
        out[bits_byte_idx] = restore_value;
      }
    } else {
      static_assert(sizeof(OUT) == -1, "invalid out type");
    }
  } else {
    static_assert(sizeof(result) == -1, "invalid result type");
  }
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static inline auto do_simd_binary_op(D d, V lv, V rv) {
  if constexpr (op == OP_PLUS || op == OP_PLUS_ASSIGN) {
    // if constexpr (is_bit) {
    //   return hn::Or(lv, rv);
    // } else {
    //   return hn::Add(lv, rv);
    // }
    return hn::Add(lv, rv);
  } else if constexpr (op == OP_MINUS || op == OP_MINUS_ASSIGN) {
    // if constexpr (is_bit) {
    //   return hn::And(lv, hn::Not(rv));
    // } else {
    //   return hn::Sub(lv, rv);
    // }
    return hn::Sub(lv, rv);
  } else if constexpr (op == OP_MULTIPLY || op == OP_MULTIPLY_ASSIGN) {
    return hn::Mul(lv, rv);
  } else if constexpr (op == OP_DIVIDE || op == OP_DIVIDE_ASSIGN) {
    return hn::Div(lv, rv);
  } else if constexpr (op == OP_MOD || op == OP_MOD_ASSIGN) {
    return hn::Mod(lv, rv);
  } else if constexpr (op == OP_LOGIC_OR) {
    return hn::Or(lv, rv);
  } else if constexpr (op == OP_LOGIC_AND) {
    return hn::And(lv, rv);
  } else if constexpr (op == OP_GREATER) {
    return hn::Gt(lv, rv);
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return hn::Ge(lv, rv);
  } else if constexpr (op == OP_LESS) {
    return hn::Lt(lv, rv);
  } else if constexpr (op == OP_LESS_EQUAL) {
    return hn::Le(lv, rv);
  } else if constexpr (op == OP_EQUAL) {
    return hn::Eq(lv, rv);
  } else if constexpr (op == OP_NOT_EQUAL) {
    return hn::Ne(lv, rv);
  } else if constexpr (op == OP_MAX) {
    return hn::Max(lv, rv);
  } else if constexpr (op == OP_MIN) {
    return hn::Min(lv, rv);
  } else if constexpr (op == OP_HYPOT) {
    return hn::Hypot(d, lv, rv);
  } else if constexpr (op == OP_ATAN2) {
    return hn::Atan2(d, lv, rv);
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

template <typename T>
static auto get_constant(T v) {
  if constexpr (std::is_same_v<Bit, T>) {
    uint8_t t = v ? 1 : 0;
    return t;
  } else {
    return v;
  }
}
template <OpToken op, typename T = void>
static inline bool do_string_view_cmp(StringView lv, StringView rv) {
  if constexpr (op == OP_GREATER) {
    return lv > rv;
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return lv >= rv;
  } else if constexpr (op == OP_LESS) {
    return lv < rv;
  } else if constexpr (op == OP_LESS_EQUAL) {
    return lv <= rv;
  } else if constexpr (op == OP_EQUAL) {
    return lv == rv;
  } else if constexpr (op == OP_NOT_EQUAL) {
    return lv != rv;
  } else {
    static_assert(sizeof(T) == -1, "unsupported cmp op");
  }
}

template <typename T, typename R, OpToken op>
static Vector<R> simd_vector_binary_scalar_op(Context& ctx, Vector<T> left, T right, bool reverse) {
  using number_t = typename InternalType<T>::internal_type;
  // constexpr bool is_bit = std::is_same_v<T, Bit>;
  const hn::ScalableTag<number_t> d;
  // constexpr auto lanes = hn::Lanes(d);
  using output_t = typename InternalType<R>::internal_type;

  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<R>(get_lanes<R>(), left.Size(), true);
  }

  auto rv = hn::Set(d, get_constant(right));
  if (reverse) {
    do_binary_transform(d, rv, left.Data(), left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), op>);
  } else {
    do_binary_transform(d, left.Data(), rv, left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), op>);
  }

  return Vector<R>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp(Context& ctx, Vector<StringView> left, Vector<StringView> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector string_view size mismatch {}:{}", left.Size(), right.Size()));
  }
  auto lanes = get_lanes<Bit>();
  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else if (ctx.IsTemporary(right)) {
    result_data = right.RawData();
  } else {
    result_data = ctx.NewSimdVector<Bit>(lanes, left.Size(), true);
  }
  size_t bitset_n = left.Size() / 64;
  if (left.Size() % 64 > 0) {
    bitset_n++;
  }
  std::vector<hwy::BitSet64> bitsets(bitset_n);
  for (size_t i = 0; i < left.Size(); i++) {
    bool v = do_string_view_cmp<op>(left[i], right[i]);
    if (v) {
      bitsets[i / 64].Set(i % 64);
    } else {
      bitsets[i / 64].Clear(i % 64);
    }
  }
  memcpy(result_data.MutableData<uint8_t*>(), bitsets.data(), sizeof(uint64_t) * bitsets.size());
  return Vector<Bit>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp_scalar(Context& ctx, Vector<StringView> left, StringView right, bool reverse) {
  VectorData result_data;
  auto lanes = get_lanes<Bit>();
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<Bit>(lanes, left.Size(), true);
  }
  size_t bitset_n = left.Size() / 64;
  if (left.Size() % 64 > 0) {
    bitset_n++;
  }
  std::vector<hwy::BitSet64> bitsets(bitset_n);
  for (size_t i = 0; i < left.Size(); i++) {
    bool v = reverse ? do_string_view_cmp<op>(right, left[i]) : do_string_view_cmp<op>(left[i], right);
    if (v) {
      bitsets[i / 64].Set(i % 64);
    } else {
      bitsets[i / 64].Clear(i % 64);
    }
  }
  memcpy(result_data.MutableData<uint8_t*>(), bitsets.data(), sizeof(uint64_t) * bitsets.size());
  return Vector<Bit>(result_data);
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_vector_scalar_op_impl(Context& ctx,
                                                                           Vector<typename OPT::operand_t> left,
                                                                           typename OPT::operand_t right) {
  if constexpr (std::is_same_v<StringView, typename OPT::operand_t>) {
    return simd_vector_string_cmp_scalar<OPT::op>(ctx, left, right, false);
  } else {
    return simd_vector_binary_scalar_op<typename OPT::operand_t, typename OPT::operand_t_1, OPT::op>(ctx, left, right,
                                                                                                     false);
  }
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_scalar_vector_op_impl(Context& ctx, typename OPT::operand_t left,
                                                                           Vector<typename OPT::operand_t> right) {
  if constexpr (std::is_same_v<StringView, typename OPT::operand_t>) {
    return simd_vector_string_cmp_scalar<OPT::op>(ctx, right, left, true);
  } else {
    return simd_vector_binary_scalar_op<typename OPT::operand_t, typename OPT::operand_t_1, OPT::op>(ctx, right, left,
                                                                                                     true);
  }
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_op_impl(Context& ctx, Vector<typename OPT::operand_t> left,
                                                             Vector<typename OPT::operand_t> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(
        fmt::format("vector binary op:{} arg vector's size mismatch {}:{}", OPT::op, left.Size(), right.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using operand_t_1 = typename OPT::operand_t_1;
  if constexpr (std::is_same_v<StringView, operand_t>) {
    return simd_vector_string_cmp<OPT::op>(ctx, left, right);
  } else {
    using number_t = typename InternalType<operand_t>::internal_type;
    using output_t = typename InternalType<operand_t_1>::internal_type;
    // constexpr bool is_bit = std::is_same_v<T, Bit>;
    const hn::ScalableTag<number_t> d;
    // constexpr auto lanes = hn::Lanes(d);
    VectorData result_data;
    if (ctx.IsTemporary(left)) {
      result_data = left.RawData();
    } else if (ctx.IsTemporary(right)) {
      result_data = right.RawData();
    } else {
      result_data = ctx.NewSimdVector<operand_t_1>(get_lanes<operand_t_1>(), left.Size(), true);
    }
    do_binary_transform(d, left.Data(), right.Data(), left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), OPT::op>);

    return Vector<operand_t_1>(result_data);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace simd {

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_op(Context& ctx, Vector<T> left, Vector<T> right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_vector_scalar_op(Context& ctx, Vector<T> left, T right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_scalar_vector_op(Context& ctx, T left, Vector<T> right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

#define DEFINE_SIMD_BINARY_BOOL_OP_TEMPLATE(r, op, ii, TYPE)                                                       \
  template Vector<Bit> simd_vector_binary_op<TYPE, Bit, op>(Context & ctx, Vector<TYPE> left, Vector<TYPE> right); \
  template Vector<Bit> simd_vector_binary_vector_scalar_op<TYPE, Bit, op>(Context & ctx, Vector<TYPE> left,        \
                                                                          TYPE right);                             \
  template Vector<Bit> simd_vector_binary_scalar_vector_op<TYPE, Bit, op>(Context & ctx, TYPE left, Vector<TYPE> right);

#define DEFINE_SIMD_BINARY_BOOL_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_BOOL_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_BINARY_BOOL_OP(OP_GREATER, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_GREATER_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LESS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LESS_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_NOT_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LOGIC_AND, Bit);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LOGIC_OR, Bit);

#define DEFINE_SIMD_BINARY_MATH_OP_TEMPLATE(r, op, ii, TYPE)                                                         \
  template Vector<TYPE> simd_vector_binary_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> left, Vector<TYPE> right); \
  template Vector<TYPE> simd_vector_binary_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> left,        \
                                                                            TYPE right);                             \
  template Vector<TYPE> simd_vector_binary_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE left,                \
                                                                            Vector<TYPE> right);

#define DEFINE_SIMD_BINARY_MATH_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_MATH_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

DEFINE_SIMD_BINARY_MATH_OP(OP_PLUS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MINUS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MULTIPLY, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_DIVIDE, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MOD, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_PLUS_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MINUS_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MULTIPLY_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_DIVIDE_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MOD_ASSIGN, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MAX, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MIN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_HYPOT, float, double);
DEFINE_SIMD_BINARY_MATH_OP(OP_ATAN2, float, double);
DEFINE_SIMD_BINARY_MATH_OP(OP_POW, float, double);
}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE
