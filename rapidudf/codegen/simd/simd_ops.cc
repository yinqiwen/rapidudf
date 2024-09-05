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
#include "rapidudf/codegen/simd/simd_ops.h"

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cstring>
#include <type_traits>
#include <vector>

#include "hwy/bit_set.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/sort/sorting_networks-inl.h"
#include "hwy/contrib/sort/traits-inl.h"
#include "hwy/contrib/sort/traits128-inl.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"

#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/simd.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace simd {
namespace hn = hwy::HWY_NAMESPACE;

template <typename V, OpToken op>
static inline auto do_simd_unary_op(V lv) {
  if constexpr (op == OP_SQRT) {
    return hn::Sqrt(lv);
  } else if constexpr (op == OP_FLOOR) {
    return hn::Floor(lv);
  } else if constexpr (op == OP_ABS) {
    return hn::Abs(lv);
  } else if constexpr (op == OP_NOT) {
    return hn::Not(lv);
  } else if constexpr (op == OP_COS) {
    constexpr hn::DFromV<V> d;
    return hn::Cos(d, lv);
  } else if constexpr (op == OP_SIN) {
    constexpr hn::DFromV<V> d;
    return hn::Sin(d, lv);
  } else if constexpr (op == OP_SINH) {
    constexpr hn::DFromV<V> d;
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_ASIN) {
    constexpr hn::DFromV<V> d;
    return hn::Asin(d, lv);
  } else if constexpr (op == OP_ACOS) {
    constexpr hn::DFromV<V> d;
    return hn::Acos(d, lv);
  } else if constexpr (op == OP_ATANH) {
    constexpr hn::DFromV<V> d;
    return hn::Atanh(d, lv);
  } else if constexpr (op == OP_SINH) {
    constexpr hn::DFromV<V> d;
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_TANH) {
    constexpr hn::DFromV<V> d;
    return hn::Tanh(d, lv);
  } else if constexpr (op == OP_ASINH) {
    constexpr hn::DFromV<V> d;
    return hn::Asinh(d, lv);
  } else if constexpr (op == OP_ACOSH) {
    constexpr hn::DFromV<V> d;
    return hn::Acosh(d, lv);
  } else if constexpr (op == OP_EXP) {
    constexpr hn::DFromV<V> d;
    return hn::Exp(d, lv);
  } else if constexpr (op == OP_EXP2) {
    constexpr hn::DFromV<V> d;
    return hn::Exp2(d, lv);
  } else if constexpr (op == OP_EXPM1) {
    constexpr hn::DFromV<V> d;
    return hn::Expm1(d, lv);
  } else if constexpr (op == OP_LOG) {
    constexpr hn::DFromV<V> d;
    return hn::Log(d, lv);
  } else if constexpr (op == OP_LOG2) {
    constexpr hn::DFromV<V> d;
    return hn::Log2(d, lv);
  } else if constexpr (op == OP_LOG10) {
    constexpr hn::DFromV<V> d;
    return hn::Log10(d, lv);
  } else if constexpr (op == OP_LOG1P) {
    constexpr hn::DFromV<V> d;
    return hn::Log1p(d, lv);
  } else if constexpr (op == OP_CEIL) {
    return hn::Ceil(lv);
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <typename T, OpToken op, bool is_bit>
static inline auto do_simd_op(T lv, T rv) {
  if constexpr (op == OP_PLUS) {
    if constexpr (is_bit) {
      return hn::Or(lv, rv);
    } else {
      return hn::Add(lv, rv);
    }
  } else if constexpr (op == OP_MINUS) {
    if constexpr (is_bit) {
      return hn::And(lv, hn::Not(rv));
    } else {
      return hn::Sub(lv, rv);
    }
  } else if constexpr (op == OP_MULTIPLY) {
    return hn::Mul(lv, rv);
  } else if constexpr (op == OP_DIVIDE) {
    return hn::Div(lv, rv);
  } else if constexpr (op == OP_MOD) {
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
    constexpr hn::DFromV<T> d;
    return hn::Hypot(d, lv, rv);
  } else if constexpr (op == OP_ATAN2) {
    constexpr hn::DFromV<T> d;
    return hn::Atan2(d, lv, rv);
  } else {
    static_assert(sizeof(T) == -1, "unsupported op");
  }
}

template <typename R, typename T, OpToken op>
static inline auto do_simd_op(R a, T b, T c) {
  if constexpr (op == OP_CLAMP) {
    return hn::Clamp(a, b, c);
  } else if constexpr (op == OP_CONDITIONAL) {
    return hn::IfThenElse(a, b, c);
  } else if constexpr (op == OP_MULADD) {
    return hn::MulAdd(a, b, c);
  } else if constexpr (op == OP_MULSUB) {
    return hn::MulSub(a, b, c);
  } else if constexpr (op == OP_MULADDSUB) {
    return hn::MulAddSub(a, b, c);
  } else if constexpr (op == OP_NEG_MULADD) {
    return hn::NegMulAdd(a, b, c);
  } else if constexpr (op == OP_NEG_MULSUB) {
    return hn::NegMulSub(a, b, c);
  } else {
    static_assert(sizeof(T) == -1, "unsupported op");
  }
}

template <typename T, OpToken op>
static inline auto do_residue_op(T lv, T rv) {
  if constexpr (op == OP_PLUS) {
    return lv + rv;
  } else if constexpr (op == OP_MINUS) {
    return lv - rv;
  } else if constexpr (op == OP_MULTIPLY) {
    return lv * rv;
  } else if constexpr (op == OP_DIVIDE) {
    return lv / rv;
  } else if constexpr (op == OP_MOD) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      static_assert(sizeof(T) == -1, "can not do mod on floats");
    } else {
      return lv % rv;
    }
  } else if constexpr (op == OP_LOGIC_OR) {
    return Bit(lv || rv);
  } else if constexpr (op == OP_LOGIC_AND) {
    return Bit(lv && rv);
  } else if constexpr (op == OP_GREATER) {
    return Bit(lv > rv);
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return Bit(lv >= rv);
  } else if constexpr (op == OP_LESS) {
    return Bit(lv < rv);
  } else if constexpr (op == OP_LESS_EQUAL) {
    return Bit(lv <= rv);
  } else if constexpr (op == OP_EQUAL) {
    return Bit(lv == rv);
  } else if constexpr (op == OP_NOT_EQUAL) {
    return Bit(lv != rv);
  } else {
    static_assert(sizeof(T) == -1, "unsupported op");
  }
}
inline uint8_t bit_set(uint8_t number, uint8_t n) { return number | ((uint8_t)1 << n); }
inline uint8_t bit_clear(uint8_t number, uint8_t n) { return number & ~((uint8_t)1 << n); }
template <typename T>
class VectorDataHelper {
 public:
  using D = hn::ScalableTag<T>;
  VectorDataHelper(void* data) { data_ = reinterpret_cast<uint8_t*>(data); }

  void AddMask(hn::Mask<D> mask) {
    constexpr uint32_t lanes = hn::Lanes(D());
    if constexpr (lanes < 8) {
      if (bit_cursor_ > 0) {
        uint8_t tmp[8];
        hn::StoreMaskBits(d, mask, tmp);
        tmp[0] = (tmp[0] << bit_cursor_);
        data_[cursor_] = (tmp[0] | data_[cursor_]);
      } else {
        hn::StoreMaskBits(d, mask, data_ + cursor_);
      }
      bit_cursor_ += lanes;
      if (bit_cursor_ == 8) {
        bit_cursor_ = 0;
        cursor_++;
      }
    } else {
      size_t n = hn::StoreMaskBits(d, mask, data_ + cursor_);
      cursor_ += n;
    }
  }
  void Add(typename hn::VFromD<D> temp) {
    hn::StoreU(temp, d, reinterpret_cast<T*>(data_ + cursor_));
    cursor_ += (hn::Lanes(d) * sizeof(T));
  }

  void AddResidue(T v) {
    RUDF_INFO("AddResidue:{}", v);
    *(reinterpret_cast<T*>(data_ + cursor_)) = v;
    cursor_ += sizeof(T);
  }

  void AddResidue(Bit v) {
    if (v) {
      data_[cursor_] = bit_set(data_[cursor_], bit_cursor_);
    } else {
      data_[cursor_] = bit_clear(data_[cursor_], bit_cursor_);
    }
    bit_cursor_++;
    if (bit_cursor_ % 8 == 0) {
      cursor_++;
      bit_cursor_ = 0;
    }
  }

 private:
  uint8_t* data_ = nullptr;
  size_t cursor_ = 0;
  uint32_t bit_cursor_ = 0;
  hn::ScalableTag<T> d;
};

static size_t get_bits_byte_size(size_t n) { return (n + 7) / 8 + 8; }
static size_t get_arena_element_size(size_t n, size_t lanes) {
  size_t rest = n % lanes;
  if (rest == 0) {
    return n;
  }
  return n + lanes - rest;
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
Vector<R> simd_vector_binary_scalar_op(Vector<T> left, T right, bool reverse, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  constexpr bool is_bit = std::is_same_v<T, Bit>;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  using MaskType = hn::Mask<decltype(d)>;
  size_t i = 0;
  DType result_dtype = get_dtype<T>();
  size_t result_size = left.Size();
  size_t element_size = get_arena_element_size(left.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  if (op >= OP_NOT && op <= OP_LOGIC_OR) {
    // use bits array
    byte_size = get_bits_byte_size(left.Size());
    result_dtype = DType(DATA_BIT);
  }
  uint8_t* arena_data = nullptr;
  RUDF_DEBUG("op:{},reuse:{},size:{}", static_cast<int>(op), reuse, left.Size());
  if (reuse > 0) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(left.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  // uint8_t* arena_data = Arena::Get().Allocate(byte_size);
  VectorDataHelper<number_t> helper(arena_data);

  auto rv = hn::Set(d, get_constant(right));
  for (; (i) < left.Size(); i += lanes) {
    auto lv = hn::LoadU(d, left.Data() + i);
    auto temp = do_simd_op<decltype(lv), op, is_bit>(reverse ? rv : lv, reverse ? lv : rv);
    if constexpr (std::is_same_v<MaskType, decltype(temp)>) {
      helper.AddMask(temp);
    } else {
      helper.Add(temp);
    }
  }
  // if (i < left.Size()) {
  //   for (; i < left.Size(); i++) {
  //     DO_RESIDUE_OP(op, left[i], right[i], [&]() { helper.AddResidue(temp); });
  //   }
  // }
  VectorData result_data(arena_data, result_size);
  return Vector<R>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp(Vector<StringView> left, Vector<StringView> right, uint32_t reuse) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector string_view size mismatch {}:{}", left.Size(), right.Size()));
  }
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_LEFT) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(left.Data()));
  } else if (reuse == REUSE_RIGHT) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(right.Data()));
  } else {
    uint32_t byte_size = get_bits_byte_size(left.Size());
    arena_data = GetArena().Allocate(byte_size);
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
    }
  }
  memcpy(arena_data, bitsets.data(), sizeof(uint64_t) * bitsets.size());
  VectorData result_data(arena_data, left.Size());
  return Vector<Bit>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp_scalar(Vector<StringView> left, StringView right, bool reverse, uint32_t reuse) {
  uint8_t* arena_data = nullptr;
  if (reuse > 0) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(left.Data()));
  } else {
    uint32_t byte_size = get_bits_byte_size(left.Size());
    arena_data = GetArena().Allocate(byte_size);
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
    }
  }
  memcpy(arena_data, bitsets.data(), sizeof(uint64_t) * bitsets.size());
  VectorData result_data(arena_data, left.Size());
  return Vector<Bit>(result_data);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_vector_scalar_op(Vector<T> left, T right, uint32_t reuse) {
  if constexpr (std::is_same_v<StringView, T>) {
    return simd_vector_string_cmp_scalar<op>(left, right, false, reuse);
  } else {
    return simd_vector_binary_scalar_op<T, R, op>(left, right, false, reuse);
  }
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_scalar_vector_op(T left, Vector<T> right, uint32_t reuse) {
  if constexpr (std::is_same_v<StringView, T>) {
    return simd_vector_string_cmp_scalar<op>(right, left, true, reuse);
  } else {
    return simd_vector_binary_scalar_op<T, R, op>(right, left, true, reuse);
  }
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_op(Vector<T> left, Vector<T> right, uint32_t reuse) {
  if constexpr (std::is_same_v<StringView, T>) {
    return simd_vector_string_cmp<op>(left, right, reuse);
  } else {
    using number_t = typename InternalType<T>::internal_type;
    constexpr bool is_bit = std::is_same_v<T, Bit>;
    const hn::ScalableTag<number_t> d;
    constexpr auto lanes = hn::Lanes(d);
    using MaskType = hn::Mask<decltype(d)>;
    size_t i = 0;
    DType result_dtype = get_dtype<T>();
    size_t result_size = left.Size();
    size_t element_size = get_arena_element_size(left.Size(), lanes);
    uint32_t byte_size = sizeof(number_t) * element_size;
    if (op >= OP_NOT && op <= OP_LOGIC_OR) {
      // use bits array
      byte_size = get_bits_byte_size(left.Size());
      result_dtype = DType(DATA_BIT);
    }
    uint8_t* arena_data = nullptr;
    if (reuse == REUSE_LEFT) {
      arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(left.Data()));
    } else if (reuse == REUSE_RIGHT) {
      arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(right.Data()));
    } else {
      arena_data = GetArena().Allocate(byte_size);
    }

    VectorDataHelper<number_t> helper(arena_data);
    for (; (i) < left.Size(); i += lanes) {
      auto lv = hn::LoadU(d, left.Data() + i);
      auto rv = hn::LoadU(d, right.Data() + i);
      auto temp = do_simd_op<decltype(lv), op, is_bit>(lv, rv);
      if constexpr (std::is_same_v<MaskType, decltype(temp)>) {
        helper.AddMask(temp);
      } else {
        helper.Add(temp);
      }
    }
    VectorData result_data(arena_data, result_size);
    return Vector<R>(result_data);
  }
}

template <typename T, typename D>
static inline auto select_ternary_value(Vector<Bit> cond, hn::VFromD<D> true_val, hn::VFromD<D> false_val, size_t i) {
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

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_op(Vector<R> a, Vector<T> b, Vector<T> c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = a.Size();
  size_t element_size = get_arena_element_size(a.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_FIRST) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a.Data()));
  } else if (reuse == REUSE_SECOND) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(b.Data()));
  } else if (reuse == REUSE_THIRD) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(c.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < a.Size(); i += lanes) {
    if constexpr (op == OP_CONDITIONAL) {
      auto true_v = hn::LoadU(d, b.Data() + i);
      auto false_v = hn::LoadU(d, c.Data() + i);
      helper.Add(select_ternary_value<T, decltype(d)>(a, true_v, false_v, i));
    } else {
      auto av = hn::LoadU(d, a.Data() + i);
      auto bv = hn::LoadU(d, b.Data() + i);
      auto cv = hn::LoadU(d, c.Data() + i);
      auto temp = do_simd_op<decltype(av), decltype(bv), op>(av, bv, cv);
      helper.Add(temp);
    }
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_vector_scalar_op(Vector<R> a, Vector<T> b, T c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = a.Size();
  size_t element_size = get_arena_element_size(a.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_FIRST) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a.Data()));
  } else if (reuse == REUSE_SECOND) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(b.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto cv = hn::Set(d, get_constant(c));
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < a.Size(); i += lanes) {
    if constexpr (op == OP_CONDITIONAL) {
      auto true_v = hn::LoadU(d, b.Data() + i);
      helper.Add(select_ternary_value<T, decltype(d)>(a, true_v, cv, i));
    } else {
      auto av = hn::LoadU(d, a.Data() + i);
      auto bv = hn::LoadU(d, b.Data() + i);
      auto temp = do_simd_op<decltype(av), decltype(bv), op>(av, bv, cv);
      helper.Add(temp);
    }
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_vector_op(Vector<R> a, T b, Vector<T> c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = a.Size();
  size_t element_size = get_arena_element_size(a.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_FIRST) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a.Data()));
  } else if (reuse == REUSE_SECOND || reuse == REUSE_THIRD) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(c.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto bv = hn::Set(d, get_constant(b));

  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < a.Size(); i += lanes) {
    auto cv = hn::LoadU(d, c.Data() + i);
    if constexpr (op == OP_CONDITIONAL) {
      helper.Add(select_ternary_value<T, decltype(d)>(a, bv, cv, i));
    } else {
      auto av = hn::LoadU(d, a.Data() + i);
      auto temp = do_simd_op<decltype(av), decltype(av), op>(av, bv, cv);
      helper.Add(temp);
    }
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_scalar_op(Vector<R> a, T b, T c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = a.Size();
  size_t element_size = get_arena_element_size(a.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_FIRST) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto bv = hn::Set(d, get_constant(b));
  auto cv = hn::Set(d, get_constant(c));
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < a.Size(); i += lanes) {
    if constexpr (op == OP_CONDITIONAL) {
      helper.Add(select_ternary_value<T, decltype(d)>(a, bv, cv, i));
    } else {
      auto av = hn::LoadU(d, a.Data() + i);
      auto temp = do_simd_op<decltype(av), decltype(bv), op>(av, bv, cv);
      helper.Add(temp);
    }
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_vector_op(R a, Vector<T> b, Vector<T> c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = b.Size();
  size_t element_size = get_arena_element_size(b.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_FIRST) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(b.Data()));
  } else if (reuse == REUSE_SECOND || reuse == REUSE_THIRD) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(c.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto av = hn::Set(d, get_constant(a));
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < b.Size(); i += lanes) {
    auto cv = hn::LoadU(d, c.Data() + i);
    auto bv = hn::LoadU(d, b.Data() + i);
    auto temp = do_simd_op<decltype(av), decltype(av), op>(av, bv, cv);
    helper.Add(temp);
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_scalar_vector_op(R a, T b, Vector<T> c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = c.Size();
  size_t element_size = get_arena_element_size(c.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse > 0) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(c.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto av = hn::Set(d, get_constant(a));
  auto bv = hn::Set(d, get_constant(b));
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < c.Size(); i += lanes) {
    auto cv = hn::LoadU(d, c.Data() + i);
    auto temp = do_simd_op<decltype(av), decltype(av), op>(av, bv, cv);
    helper.Add(temp);
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_scalar_op(R a, Vector<T> b, T c, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  size_t result_size = b.Size();
  size_t element_size = get_arena_element_size(b.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  uint8_t* arena_data = nullptr;
  if (reuse > 0) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(b.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  auto av = hn::Set(d, get_constant(a));
  auto cv = hn::Set(d, get_constant(c));
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < b.Size(); i += lanes) {
    auto bv = hn::LoadU(d, b.Data() + i);
    auto temp = do_simd_op<decltype(av), decltype(av), op>(av, bv, cv);
    helper.Add(temp);
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}

template <typename T, OpToken op>
Vector<T> simd_vector_unary_op(Vector<T> left, uint32_t reuse) {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  DType result_dtype = get_dtype<T>();
  size_t result_size = left.Size();
  size_t element_size = get_arena_element_size(left.Size(), lanes);
  uint32_t byte_size = sizeof(number_t) * element_size;
  if (op >= OP_NOT && op <= OP_LOGIC_OR) {
    // use bits array
    byte_size = get_bits_byte_size(left.Size());
    result_dtype = DType(DATA_BIT);
  }
  uint8_t* arena_data = nullptr;
  if (reuse == REUSE_LEFT) {
    arena_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(left.Data()));
  } else {
    arena_data = GetArena().Allocate(byte_size);
  }
  VectorDataHelper<number_t> helper(arena_data);
  for (; (i) < left.Size(); i += lanes) {
    auto lv = hn::LoadU(d, left.Data() + i);
    auto temp = do_simd_unary_op<decltype(lv), op>(lv);
    helper.Add(temp);
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}

template <typename T>
T simd_vector_dot(Vector<T> left, Vector<T> right, uint32_t reuse) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left.Size(), right.Size()));
  }
  using D = hn::ScalableTag<T>;
  const D d;
  constexpr auto assumptions = hn::Dot::Assumptions::kAtLeastOneVector;
  T val = hn::Dot::Compute<assumptions, D, T>(d, left.Data(), right.Data(), left.Size());
  return val;
}

template <typename T>
Vector<T> simd_vector_iota(T start, uint32_t n, uint32_t reuse) {
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t element_size = get_arena_element_size(n, lanes);
  uint32_t byte_size = sizeof(T) * element_size;
  uint8_t* arena_data = GetArena().Allocate(byte_size);
  size_t result_size = n;
  size_t i = 0;
  for (; i < result_size; i += lanes) {
    auto v = hn::Iota(d, start + i);
    hn::StoreU(v, d, reinterpret_cast<T*>(arena_data + i * sizeof(T)));
  }
  VectorData result_data(arena_data, result_size);
  return Vector<T>(result_data);
}

#define DEFINE_SIMD_BINARY_MATH_OP_TEMPLATE(r, op, ii, TYPE)                                                          \
  template Vector<TYPE> simd_vector_binary_op<TYPE, TYPE, op>(Vector<TYPE> left, Vector<TYPE> right, uint32_t reuse); \
  template Vector<TYPE> simd_vector_binary_vector_scalar_op<TYPE, TYPE, op>(Vector<TYPE> left, TYPE right,            \
                                                                            uint32_t reuse);                          \
  template Vector<TYPE> simd_vector_binary_scalar_vector_op<TYPE, TYPE, op>(TYPE left, Vector<TYPE> right,            \
                                                                            uint32_t reuse);

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
DEFINE_SIMD_BINARY_MATH_OP(OP_MAX, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MIN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_HYPOT, float, double);
DEFINE_SIMD_BINARY_MATH_OP(OP_ATAN2, float, double);

#define DEFINE_SIMD_BINARY_BOOL_OP_TEMPLATE(r, op, ii, TYPE)                                                        \
  template Vector<Bit> simd_vector_binary_op<TYPE, Bit, op>(Vector<TYPE> left, Vector<TYPE> right, uint32_t reuse); \
  template Vector<Bit> simd_vector_binary_vector_scalar_op<TYPE, Bit, op>(Vector<TYPE> left, TYPE right,            \
                                                                          uint32_t reuse);                          \
  template Vector<Bit> simd_vector_binary_scalar_vector_op<TYPE, Bit, op>(TYPE left, Vector<TYPE> right,            \
                                                                          uint32_t reuse);

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

#define DEFINE_SIMD_UNARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_unary_op<TYPE, op>(Vector<TYPE> left, uint32_t reuse);
#define DEFINE_SIMD_UNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_UNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
DEFINE_SIMD_UNARY_OP(OP_COS, float, double);
DEFINE_SIMD_UNARY_OP(OP_SIN, float, double);
DEFINE_SIMD_UNARY_OP(OP_TANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ATANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_SINH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ACOS, float, double);
DEFINE_SIMD_UNARY_OP(OP_ASIN, float, double);
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
DEFINE_SIMD_UNARY_OP(OP_ABS, float, double, int64_t, int32_t, int16_t, int8_t);

#define DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE(r, op, ii, TYPE)                                              \
  template Vector<TYPE> simd_vector_ternary_op<Bit, TYPE, OP_CONDITIONAL>(Vector<Bit> a, Vector<TYPE> b,   \
                                                                          Vector<TYPE> c, uint32_t reuse); \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Vector<Bit> a, Vector<TYPE> b, TYPE c, uint32_t reuse);                                              \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Vector<Bit> a, TYPE b, Vector<TYPE> c, uint32_t reuse);                                              \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Vector<Bit> a, TYPE b, TYPE c, uint32_t reuse);

#define DEFINE_SIMD_TERNARY_COND_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_COND_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_TERNARY_OP_TEMPLATE(r, op, ii, TYPE)                                                             \
  template Vector<TYPE> simd_vector_ternary_op<TYPE, TYPE, op>(Vector<TYPE> a, Vector<TYPE> b, Vector<TYPE> c,       \
                                                               uint32_t reuse);                                      \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<TYPE, TYPE, op>(Vector<TYPE> a, Vector<TYPE> b,  \
                                                                                    TYPE c, uint32_t reuse);         \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<TYPE, TYPE, op>(Vector<TYPE> a, TYPE b,          \
                                                                                    Vector<TYPE> c, uint32_t reuse); \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<TYPE, TYPE, op>(Vector<TYPE> a, TYPE b, TYPE c,  \
                                                                                    uint32_t reuse);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_vector_op<TYPE, TYPE, op>(TYPE a, Vector<TYPE> b,          \
                                                                                    Vector<TYPE> c, uint32_t reuse); \
  template Vector<TYPE> simd_vector_ternary_scalar_scalar_vector_op<TYPE, TYPE, op>(TYPE a, TYPE b, Vector<TYPE> c,  \
                                                                                    uint32_t reuse);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_scalar_op<TYPE, TYPE, op>(TYPE a, Vector<TYPE> b, TYPE c,  \
                                                                                    uint32_t reuse);

#define DEFINE_SIMD_TERNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_OP(OP_CLAMP, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_MULADD, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_MULSUB, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_MULADDSUB, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_NEG_MULADD, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_NEG_MULSUB, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);

#define DEFINE_SIMD_DOT_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_dot(Vector<TYPE> left, Vector<TYPE> right, uint32_t reuse);
#define DEFINE_SIMD_DOT_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_DOT_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_DOT_OP(float, double);

#define DEFINE_SIMD_IOTA_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_iota(TYPE start, uint32_t n, uint32_t reuse);
#define DEFINE_SIMD_IOTA_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_IOTA_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_IOTA_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

}  // namespace simd
}  // namespace rapidudf