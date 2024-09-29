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

#include <array>
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <stack>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/string_view.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/fused_ops.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "hwy/cache_control.h"
#include "hwy/contrib/math/math-inl.h"

#include "rapidudf/builtin/simd_vector/ops.h"
#include "rapidudf/types/simd_vector.h"

HWY_BEFORE_NAMESPACE();

namespace rapidudf {
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename V, std::size_t N = 16>
struct __attribute__((aligned(sizeof(V)))) InlineOperands : public std::array<V, N> {
  inline void pop_back(size_t n = 1) { cursor_ -= n; }
  inline void emplace_back(V v) {
    this->at(cursor_) = v;
    cursor_++;
  }
  inline void clear() { cursor_ = 0; }
  inline void reserve(size_t n) {}
  inline size_t size() const { return cursor_; }
  // V& top() { return this->at(cursor_ - 1); }
  // V& next() {
  //   cursor_++;
  //   return top();
  // }
  uint32_t cursor_ = 0;
};

template <typename V, std::size_t N = 8>
// using FusedOperands = boost::container::small_vector<V, N, boost::alignment::aligned_allocator<V, sizeof(V) * 8>>;
// using FusedOperands = absl::InlinedVector<V, N, boost::alignment::aligned_allocator<V, sizeof(V) * 8>>;
// using FusedOperands = std::vector<V, boost::alignment::aligned_allocator<V, sizeof(V) * 8>>;
using FusedOperands = InlineOperands<V, N>;

template <class D, typename V = hn::VFromD<D>>
static inline V do_simd_fused_op([[maybe_unused]] D d, OpToken op, FusedOperands<V>& vals) {
  using T = hn::TFromV<V>;
  size_t operand_size = vals.size();
  V result;
  bool is_op_supported = true;
  switch (op) {
    case OP_PLUS:
    case OP_PLUS_ASSIGN: {
      result = hn::Add(vals[operand_size - 1], vals[operand_size - 2]);
      break;
    }
    case OP_MINUS:
    case OP_MINUS_ASSIGN: {
      result = hn::Sub(vals[operand_size - 2], vals[operand_size - 1]);
      break;
    }
    case OP_MULTIPLY:
    case OP_MULTIPLY_ASSIGN: {
      result = hn::Mul(vals[operand_size - 2], vals[operand_size - 1]);
      break;
    }
    case OP_DIVIDE:
    case OP_DIVIDE_ASSIGN: {
      result = hn::Div(vals[operand_size - 2], vals[operand_size - 1]);
      break;
    }
    case OP_MOD:
    case OP_MOD_ASSIGN: {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
        is_op_supported = false;
        break;
      } else {
        result = hn::Mod(vals[operand_size - 2], vals[operand_size - 1]);
        break;
      }
    }
    case OP_SQRT: {
      vals.pop_back();
      return hn::Sqrt(vals[operand_size - 1]);
    }
    case OP_SIN: {
      vals.pop_back();
      return hn::Sin(d, vals[operand_size - 1]);
    }
    default: {
      throw std::logic_error(fmt::format("Unsupported op:{}", op));
    }
  }
  if (!is_op_supported) {
    throw std::logic_error(fmt::format("Unsupported op:{}", op));
  }
  vals.pop_back();
  vals.pop_back();
  return result;
}

template <typename T>
struct VectorOperand {
  hn::Vec<hn::ScalableTag<T>> scalar;
  Vector<T> vec;
  OpToken op = OP_INVALID;
  explicit VectorOperand(OpToken t) : op(t) {}
  explicit VectorOperand(hn::Vec<hn::ScalableTag<T>> s) : scalar(s) {}
  explicit VectorOperand(Vector<T> v) : vec(v) {}
};

template <typename T, typename R>
static inline OpToken get_operands(size_t vec_idx, size_t& stack_cursor, size_t remaining,
                                   std::vector<VectorOperand<T>>& inputs, R& results) {
  using D = hn::ScalableTag<T>;
  using V = hn::Vec<D>;
  constexpr D d;
  using V = hn::Vec<hn::ScalableTag<T>>;
  OpToken optoken = OP_INVALID;
  while (stack_cursor < inputs.size()) {
    auto& operand = inputs[stack_cursor];
    stack_cursor++;
    if (operand.op > 0) {
      return operand.op;
    } else if (operand.vec.Size() > 0) {
      if (remaining > 0) {
        results.emplace_back(hn::LoadN(d, operand.vec.Data() + vec_idx, remaining));
      } else {
        results.emplace_back(hn::LoadU(d, operand.vec.Data() + vec_idx));
      }
    } else {
      results.emplace_back(operand.scalar);
    }
  }
  return optoken;
}

template <typename T>
Vector<T> simd_vector_fused_op_impl(Context& ctx, std::vector<Operand<T>>& operands) {
  using number_t = typename InternalType<T>::internal_type;
  using D = hn::ScalableTag<number_t>;
  using V = hn::Vec<D>;
  const D d;
  constexpr size_t N = Lanes(d);

  VectorData result_data;
  size_t count = 0;
  std::vector<VectorOperand<T>> inputs;
  for (auto& operand : operands) {
    std::visit(
        [&](auto&& arg) {
          using ARG = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<OpToken, ARG>) {
            inputs.emplace_back(arg);
          } else if constexpr (std::is_same_v<T, ARG>) {
            V new_v = hn::Set(d, arg);
            inputs.emplace_back(new_v);
          } else {
            if (count == 0) {
              count = arg.Size();
            } else {
              if (count != arg.Size()) {
                THROW_SIZE_MISMATCH_ERR(arg.Size(), count);
              }
            }
            if (result_data.Size() == 0 && ctx.IsTemporary(arg)) {
              result_data = arg.RawData();
            }
            inputs.emplace_back(arg);
          }
        },
        operand);
  }
  if (result_data.Size() == 0) {
    result_data = ctx.NewSimdVector<T>(N, count, true);
  }

  T* output = result_data.MutableData<T>();
  size_t idx = 0;
  alignas(32) FusedOperands<V> fused_operands;
  fused_operands.reserve(16);
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      fused_operands.clear();
      size_t stack_pop_cursor = 0;
      V result;
      while (stack_pop_cursor < inputs.size()) {
        OpToken op = get_operands(idx, stack_pop_cursor, 0, inputs, fused_operands);
        result = do_simd_fused_op(d, op, fused_operands);
        fused_operands.emplace_back(result);
      }
      hn::StoreU(result, d, output + idx);
    }
  }

  if (HWY_UNLIKELY(idx == count)) {
    return Vector<T>(result_data);
  }
  const size_t remaining = count - idx;
  size_t stack_pop_cursor = 0;
  V result;
  fused_operands.clear();
  while (stack_pop_cursor < operands.size()) {
    OpToken op = get_operands(idx, stack_pop_cursor, remaining, inputs, fused_operands);
    result = do_simd_fused_op(d, op, fused_operands);
    fused_operands.emplace_back(result);
  }
  hn::StoreN(result, d, output + idx, remaining);
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
Vector<T> simd_vector_fused_op(Context& ctx, std::vector<Operand<T>>& operands) {
  HWY_EXPORT_T(Table1, simd_vector_fused_op_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, operands);
}
template Vector<float> simd_vector_fused_op<float>(Context& ctx, std::vector<Operand<float>>& operands);
template Vector<double> simd_vector_fused_op<double>(Context& ctx, std::vector<Operand<double>>& operands);
}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE