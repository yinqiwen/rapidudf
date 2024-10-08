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
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/ternary_ops.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

// #include "hwy/bit_set.h"
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

template <class D, typename T1, typename T2, typename T3, typename OUT, class Func>
HWY_INLINE void do_ternary_transform(D d, T1 in1, T2 in2, T3 in3, size_t count, OUT* out, const Func& func) {
  const size_t N = hn::Lanes(d);
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
      hn::Vec<D> v3;
      if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
        v3 = in3;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
        v3 = hn::LoadU(d, in3 + idx);
      } else {
        static_assert(sizeof(T3) == -1, "invalid T2");
      }
      hn::StoreU(func(d, v1, v2, v3), d, out + idx);
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
  hn::Vec<D> v3;
  if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
    v3 = in3;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
    v3 = hn::LoadN(d, in3 + idx, remaining);
  } else {
    static_assert(sizeof(T3) == -1, "invalid T3");
  }
  hn::StoreN(func(d, v1, v2, v3), d, out + idx, remaining);
}

template <class D, typename T2, typename T3, typename OUT = hn::TFromD<D>>
HWY_INLINE void do_select(D d, Vector<Bit> cond, T2 in2, T3 in3, size_t count, OUT* out) {
  const size_t N = hn::Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      hn::Vec<D> v2;
      if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
        v2 = in2;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
        v2 = hn::LoadU(d, in2 + idx);
      } else {
        static_assert(sizeof(T2) == -1, "invalid T2");
      }
      hn::Vec<D> v3;
      if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
        v3 = in3;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
        v3 = hn::LoadU(d, in3 + idx);
      } else {
        static_assert(sizeof(T3) == -1, "invalid T2");
      }
      size_t bits_byte_cursor = idx / 8;
      size_t bits_cursor = idx % 8;
      hn::Vec<D> result;
      if (bits_cursor > 0) {
        uint8_t tmp_bits[8];
        tmp_bits[0] = cond.Data()[bits_byte_cursor];
        tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
        auto mask = hn::LoadMaskBits(d, tmp_bits);
        result = hn::IfThenElse(mask, v2, v3);
      } else {
        auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
        result = hn::IfThenElse(mask, v2, v3);
      }
      hn::StoreU(result, d, out + idx);
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);

  hn::Vec<D> v2;
  if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
    v2 = in2;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
    v2 = LoadN(d, in2 + idx, remaining);
  } else {
    static_assert(sizeof(T2) == -1, "invalid T2");
  }
  hn::Vec<D> v3;
  if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
    v3 = in3;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
    v3 = LoadN(d, in3 + idx, remaining);
  } else {
    static_assert(sizeof(T3) == -1, "invalid T2");
  }
  size_t bits_byte_cursor = idx / 8;
  size_t bits_cursor = idx % 8;
  hn::Vec<D> result;
  if (bits_cursor > 0) {
    uint8_t tmp_bits[8];
    tmp_bits[0] = cond.Data()[bits_byte_cursor];
    tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
    auto mask = hn::LoadMaskBits(d, tmp_bits);
    result = hn::IfThenElse(mask, v2, v3);
  } else {
    auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
    result = hn::IfThenElse(mask, v2, v3);
  }
  hn::StoreN(result, d, out + idx, remaining);
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static HWY_INLINE auto do_simd_ternary_op(D d, V a, V b, V c) {
  if constexpr (op == OP_CLAMP) {
    return hn::Clamp(a, b, c);
  } else if constexpr (op == OP_CONDITIONAL) {
    return hn::IfThenElse(a, b, c);
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

template <typename T>
static auto get_constant(T v) {
  if constexpr (std::is_same_v<Bit, T>) {
    uint8_t t = v ? 1 : 0;
    return t;
  } else {
    return v;
  }
}

template <typename T, typename D>
static HWY_INLINE auto select_ternary_value(Vector<Bit> cond, hn::VFromD<D> true_val, hn::VFromD<D> false_val,
                                            size_t i) {
  const D d;
  constexpr auto lanes = hn::Lanes(d);
  size_t bits_byte_cursor = i / 8;
  size_t bits_cursor = i % 8;
  if constexpr (lanes < 8) {
    uint8_t tmp_bits[8];
    tmp_bits[0] = cond.Data()[bits_byte_cursor];
    tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
    auto mask = hn::LoadMaskBits(d, tmp_bits);
    auto v = hn::IfThenElse(mask, true_val, false_val);
    return v;
  } else {
    auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
    auto v = hn::IfThenElse(mask, true_val, false_val);
    return v;
  }
}

template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_op_impl(Context& ctx, Vector<typename OPT::operand_t_1> a,
                                                            Vector<typename OPT::operand_t> b,
                                                            Vector<typename OPT::operand_t> c) {
  if (a.Size() != b.Size() || a.Size() != c.Size() || c.Size() != b.Size()) {
    RUDF_ERROR("op:{}, a size:{}, b size:{}, c size:{}", OPT::op, a.Size(), b.Size(), c.Size());
    THROW_SIZE_MISMATCH_ERR(a.Size(), c.Size());
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size(), true);
  }
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>());
  } else {
    auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
    do_ternary_transform(d, a.Data(), b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>(),
                         transform_func);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_vector_scalar_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 typename OPT::operand_t c) {
  if (a.Size() != b.Size()) {
    THROW_SIZE_MISMATCH_ERR(a.Size(), b.Size());
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size(), true);
  }
  auto cv = hn::Set(d, get_constant(c));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, b.Data(), cv, b.Size(), result_data.MutableData<number_t>());
  } else {
    auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
    do_ternary_transform(d, a.Data(), b.Data(), cv, b.Size(), result_data.MutableData<number_t>(), transform_func);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_scalar_vector_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 typename OPT::operand_t b,
                                                                                 Vector<typename OPT::operand_t> c) {
  if (a.Size() != c.Size()) {
    THROW_SIZE_MISMATCH_ERR(a.Size(), c.Size());
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);

  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size());
  }
  auto bv = hn::Set(d, get_constant(b));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, bv, c.Data(), a.Size(), result_data.MutableData<number_t>());
  } else {
    auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
    do_ternary_transform(d, a.Data(), bv, c.Data(), a.Size(), result_data.MutableData<number_t>(), transform_func);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_scalar_scalar_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 typename OPT::operand_t b,
                                                                                 typename OPT::operand_t c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);

  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size(), true);
  }
  auto bv = hn::Set(d, get_constant(b));
  auto cv = hn::Set(d, get_constant(c));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, bv, cv, a.Size(), result_data.MutableData<number_t>());
  } else {
    auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
    do_ternary_transform(d, a.Data(), bv, cv, a.Size(), result_data.MutableData<number_t>(), transform_func);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_vector_vector_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 Vector<typename OPT::operand_t> c) {
  if (b.Size() != c.Size()) {
    THROW_SIZE_MISMATCH_ERR(b.Size(), c.Size());
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(b.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, b.Size(), true);
  }
  auto av = hn::Set(d, get_constant(a));
  auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
  do_ternary_transform(d, av, b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>(), transform_func);

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_scalar_vector_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 typename OPT::operand_t b,
                                                                                 Vector<typename OPT::operand_t> c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(c.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, c.Size(), true);
  }
  auto av = hn::Set(d, get_constant(a));
  auto bv = hn::Set(d, get_constant(b));
  auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
  do_ternary_transform(d, av, bv, c.Data(), c.Size(), result_data.MutableData<number_t>(), transform_func);

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_vector_scalar_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 typename OPT::operand_t c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else {
    // result_data = arena_new_vector<T>(b.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, b.Size(), true);
  }
  auto av = hn::Set(d, get_constant(a));
  auto cv = hn::Set(d, get_constant(c));
  auto transform_func = do_simd_ternary_op<decltype(d), OPT::op>;
  do_ternary_transform(d, av, b.Data(), cv, b.Size(), result_data.MutableData<number_t>(), transform_func);

  return Vector<operand_t>(result_data);
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace simd {

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_scalar_op(Context& ctx, R a, Vector<T> b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_scalar_vector_op(Context& ctx, R a, T b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_vector_op(Context& ctx, R a, Vector<T> b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_vector_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_scalar_op(Context& ctx, Vector<R> a, T b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_scalar_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_vector_op(Context& ctx, Vector<R> a, T b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_vector_scalar_op(Context& ctx, Vector<R> a, Vector<T> b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_op(Context& ctx, Vector<R> a, Vector<T> b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

#define DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE(r, op, ii, TYPE)                                              \
  template Vector<TYPE> simd_vector_ternary_op<Bit, TYPE, OP_CONDITIONAL>(Context & ctx, Vector<Bit> a,    \
                                                                          Vector<TYPE> b, Vector<TYPE> c); \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, Vector<TYPE> b, TYPE c);                                               \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, TYPE b, Vector<TYPE> c);                                               \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, TYPE b, TYPE c);

#define DEFINE_SIMD_TERNARY_COND_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_COND_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_TERNARY_OP_TEMPLATE(r, op, ii, TYPE)                                                             \
  template Vector<TYPE> simd_vector_ternary_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a, Vector<TYPE> b,        \
                                                               Vector<TYPE> c);                                      \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    Vector<TYPE> b, TYPE c);         \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    TYPE b, Vector<TYPE> c);         \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    TYPE b, TYPE c);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE a,           \
                                                                                    Vector<TYPE> b, Vector<TYPE> c); \
  template Vector<TYPE> simd_vector_ternary_scalar_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE a, TYPE b,   \
                                                                                    Vector<TYPE> c);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, TYPE a,           \
                                                                                    Vector<TYPE> b, TYPE c);

#define DEFINE_SIMD_TERNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_OP(OP_CLAMP, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FMA, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FMS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMA, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);

}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE
