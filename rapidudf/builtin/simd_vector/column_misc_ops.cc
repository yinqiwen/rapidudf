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

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/simd_vector.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"
namespace rapidudf {
namespace simd {
Scalar* simd_column_sum(Column* a) {
  auto& ctx = a->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_SUM)) {
            auto val = simd_vector_sum(arg);
            return to_scalar(ctx, val);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
        return (Scalar*)0;
      },
      a->GetInternal());
}
Scalar* simd_column_dot(Column* left, Column* right) {
  if (!left->TypeEquals(*right)) {
    THROW_LOGIC_ERR("left/right have different internal dtype");
  }
  auto& ctx = left->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_DOT));
        } else {
          using value_type = typename T::value_type;
          auto right_vec = right->ToVector<value_type>().value();
          if constexpr (is_valid_operand<value_type>(OP_DOT)) {
            auto val = simd_vector_dot(arg, right_vec);
            return to_scalar(ctx, val);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_DOT, get_dtype<value_type>()));
          }
        }
        return (Scalar*)0;
      },
      left->GetInternal());
}

Column* simd_column_clone(Column* data) {
  auto& ctx = data->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_CLONE));
        } else {
          auto result = simd_vector_clone(ctx, arg);
          auto* c = ctx.New<Column>(ctx, result);
          return c;
        }
        return (Column*)0;
      },
      data->GetInternal());
}

}  // namespace simd
}  // namespace rapidudf