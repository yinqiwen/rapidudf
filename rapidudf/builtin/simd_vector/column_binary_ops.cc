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

#include <type_traits>
#include <variant>

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/simd_vector.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"
namespace rapidudf {
namespace simd {

template <OpToken op>
Column* simd_column_binary_op(Column* left, Column* right) {
  if (!left->TypeEquals(*right)) {
    THROW_LOGIC_ERR("left/right have different internal dtype");
  }
  auto& ctx = left->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(op)) {
            if constexpr (op >= OP_EQUAL && op <= OP_LOGIC_OR) {
              auto right_vec = right->ToVector<value_type>().value();
              auto result = simd_vector_binary_op<value_type, Bit, op>(ctx, arg, right_vec);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              auto right_vec = right->ToVector<value_type>().value();
              auto result = simd_vector_binary_op<value_type, value_type, op>(ctx, arg, right_vec);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            }
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      left->GetInternal());
}

template <OpToken op>
Column* simd_column_binary_column_scalar_op(Column* left, Scalar* right) {
  auto& ctx = left->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto right_result = right->To<value_type>();
          if (!right_result.ok()) {
            THROW_LOGIC_ERR(right_result.status().ToString());
          }
          if constexpr (is_valid_operand<value_type>(op)) {
            if constexpr (op >= OP_EQUAL && op <= OP_LOGIC_OR) {
              value_type right_val = right_result.value();
              auto result = simd_vector_binary_vector_scalar_op<value_type, Bit, op>(ctx, arg, right_val);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              value_type right_val = right_result.value();
              auto result = simd_vector_binary_vector_scalar_op<value_type, value_type, op>(ctx, arg, right_val);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            }
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      left->GetInternal());
}
template <OpToken op>
Column* simd_column_binary_scalar_column_op(Scalar* left, Column* right) {
  auto& ctx = right->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto left_result = left->To<value_type>();
          if (!left_result.ok()) {
            THROW_LOGIC_ERR(left_result.status().ToString());
          }
          if constexpr (is_valid_operand<value_type>(op)) {
            value_type left_val = left_result.value();
            if constexpr (op >= OP_EQUAL && op <= OP_LOGIC_OR) {
              auto result = simd_vector_binary_scalar_vector_op<value_type, Bit, op>(ctx, left_val, arg);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              auto result = simd_vector_binary_scalar_vector_op<value_type, value_type, op>(ctx, left_val, arg);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            }
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      right->GetInternal());
}

#define DEFINE_SIMD_BINARY_OP_TEMPLATE(r, _, ii, op)                                       \
  template Column* simd_column_binary_op<op>(Column * left, Column * right);               \
  template Column* simd_column_binary_column_scalar_op<op>(Column * left, Scalar * right); \
  template Column* simd_column_binary_scalar_column_op<op>(Scalar * left, Column * right);

#define DEFINE_BINARY_UNARY_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_BINARY_UNARY_OP(OP_PLUS, OP_MINUS, OP_MULTIPLY, OP_DIVIDE, OP_MOD, OP_PLUS_ASSIGN, OP_MINUS_ASSIGN,
                       OP_MULTIPLY_ASSIGN, OP_DIVIDE_ASSIGN, OP_MOD_ASSIGN, OP_MAX, OP_MIN);
DEFINE_BINARY_UNARY_OP(OP_HYPOT, OP_ATAN2, OP_POW, OP_GREATER, OP_GREATER_EQUAL, OP_LESS, OP_LESS_EQUAL, OP_EQUAL,
                       OP_NOT_EQUAL, OP_LOGIC_AND, OP_LOGIC_OR);
}  // namespace simd
}  // namespace rapidudf