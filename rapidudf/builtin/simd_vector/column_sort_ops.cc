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
void column_sort(Column* data, bool descending) {
  auto& ctx = data->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_SORT)) {
            sort(ctx, arg, descending);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
      },
      data->GetInternal());
}
void column_select(Column* data, size_t k, bool descending) {
  auto& ctx = data->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_SORT)) {
            select(ctx, arg, k, descending);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
      },
      data->GetInternal());
}
void column_topk(Column* data, size_t k, bool descending) {
  auto& ctx = data->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_SORT)) {
            topk(ctx, arg, k, descending);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
      },
      data->GetInternal());
}
Column* column_argsort(Column* data, bool descending) {
  auto& ctx = data->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_ARG_SORT)) {
            auto v = argsort(ctx, arg, descending);
            return ctx.New<Column>(ctx, v);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      data->GetInternal());
}
Column* column_argselect(Column* data, size_t k, bool descending) {
  auto& ctx = data->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SUM));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(OP_ARG_SORT)) {
            auto v = argselect(ctx, arg, k, descending);
            return ctx.New<Column>(ctx, v);
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SUM, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      data->GetInternal());
}
void column_sort_key_value(Column* key, Column* value, bool descending) {
  auto& ctx = key->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SORT_KV));
        } else {
          using key_type = typename T::value_type;
          if constexpr (is_valid_operand<key_type>(OP_SORT_KV)) {
            std::visit(
                [&](auto&& val_arg) {
                  using R = std::decay_t<decltype(val_arg)>;
                  if constexpr (std::is_same_v<R, TablePtr>) {
                    THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SORT_KV));
                  } else {
                    using value_type = typename R::value_type;
                    if constexpr (is_valid_operand<value_type>(OP_SORT_KV)) {
                      sort_key_value(ctx, arg, val_arg, descending);
                    } else {
                      THROW_LOGIC_ERR(
                          fmt::format("Unsupported op:{} with column dtype:{}", OP_SORT_KV, get_dtype<value_type>()));
                    }
                  }
                },
                value->GetInternal());
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SORT_KV, get_dtype<key_type>()));
          }
        }
      },
      key->GetInternal());
}
void column_select_key_value(Column* key, Column* value, size_t k, bool descending) {
  auto& ctx = key->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SELECT_KV));
        } else {
          using key_type = typename T::value_type;
          if constexpr (is_valid_operand<key_type>(OP_SELECT_KV)) {
            std::visit(
                [&](auto&& val_arg) {
                  using R = std::decay_t<decltype(val_arg)>;
                  if constexpr (std::is_same_v<R, TablePtr>) {
                    THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_SELECT_KV));
                  } else {
                    using value_type = typename R::value_type;
                    if constexpr (is_valid_operand<value_type>(OP_SELECT_KV)) {
                      select_key_value(ctx, arg, val_arg, k, descending);
                    } else {
                      THROW_LOGIC_ERR(
                          fmt::format("Unsupported op:{} with column dtype:{}", OP_SELECT_KV, get_dtype<value_type>()));
                    }
                  }
                },
                value->GetInternal());
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_SORT_KV, get_dtype<key_type>()));
          }
        }
      },
      key->GetInternal());
}
void column_topk_key_value(Column* key, Column* value, size_t k, bool descending) {
  auto& ctx = key->GetContext();
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_TOPK_KV));
        } else {
          using key_type = typename T::value_type;
          if constexpr (is_valid_operand<key_type>(OP_TOPK_KV)) {
            std::visit(
                [&](auto&& val_arg) {
                  using R = std::decay_t<decltype(val_arg)>;
                  if constexpr (std::is_same_v<R, TablePtr>) {
                    THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", OP_TOPK_KV));
                  } else {
                    using value_type = typename R::value_type;
                    if constexpr (is_valid_operand<value_type>(OP_TOPK_KV)) {
                      topk_key_value(ctx, arg, val_arg, k, descending);
                    } else {
                      THROW_LOGIC_ERR(
                          fmt::format("Unsupported op:{} with column dtype:{}", OP_TOPK_KV, get_dtype<value_type>()));
                    }
                  }
                },
                value->GetInternal());
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", OP_TOPK_KV, get_dtype<key_type>()));
          }
        }
      },
      key->GetInternal());
}

}  // namespace simd
}  // namespace rapidudf