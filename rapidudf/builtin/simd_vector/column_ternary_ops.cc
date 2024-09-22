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
Column* simd_column_ternary_op(Column* a, Column* b, Column* c) {
  auto& ctx = a->GetContext();
  if (!b->TypeEquals(*b)) {
    THROW_LOGIC_ERR("ternary ops have different internal dtype");
  }
  if constexpr (op != OP_CONDITIONAL) {
    if (!a->TypeEquals(*b)) {
      THROW_LOGIC_ERR("ternary ops have different internal dtype");
    }
  } else {
    if (!a->IsBit()) {
      THROW_LOGIC_ERR("can NOT do conditional op while column is not bit/bool");
    }
  }
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto c_vec = c->ToVector<value_type>().value();
          if constexpr (op == OP_CONDITIONAL) {
            if constexpr (is_valid_operand<value_type>(op)) {
              auto a_vec = a->ToVector<Bit>().value();
              auto result = simd_vector_ternary_op<Bit, value_type, op>(ctx, a_vec, arg, c_vec);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          } else {
            auto a_vec = a->ToVector<value_type>().value();
            if constexpr (is_valid_operand<value_type>(op)) {
              auto result = simd_vector_ternary_op<value_type, value_type, op>(ctx, a_vec, arg, c_vec);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          }
        }
        return (Column*)0;
      },
      b->GetInternal());
}

template <OpToken op>
Column* simd_column_ternary_column_column_scalar_op(Column* a, Column* b, Scalar* c) {
  if constexpr (op != OP_CONDITIONAL) {
    if (!a->TypeEquals(*b)) {
      THROW_LOGIC_ERR("ternary ops have different internal dtype");
    }
  } else {
    if (!a->IsBit()) {
      THROW_LOGIC_ERR("can NOT do conditional op while column is not bit/bool");
    }
  }
  auto& ctx = a->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto c_result = c->To<value_type>();
          if (!c_result.ok()) {
            THROW_LOGIC_ERR(c_result.status().ToString());
          }
          auto c_val = c_result.value();
          if constexpr (op == OP_CONDITIONAL) {
            if constexpr (is_valid_operand<value_type>(op)) {
              auto a_vec = a->ToVector<Bit>().value();
              auto result = simd_vector_ternary_vector_vector_scalar_op<Bit, value_type, op>(ctx, a_vec, arg, c_val);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          } else {
            auto a_vec = a->ToVector<value_type>().value();
            if constexpr (is_valid_operand<value_type>(op)) {
              auto result =
                  simd_vector_ternary_vector_vector_scalar_op<value_type, value_type, op>(ctx, a_vec, arg, c_val);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          }
        }
        return (Column*)0;
      },
      b->GetInternal());
}

template <OpToken op>
Column* simd_column_ternary_column_scalar_column_op(Column* a, Scalar* b, Column* c) {
  if constexpr (op != OP_CONDITIONAL) {
    if (!a->TypeEquals(*c)) {
      THROW_LOGIC_ERR("ternary ops have different internal dtype");
    }
  } else {
    if (!a->IsBit()) {
      THROW_LOGIC_ERR("can NOT do conditional op while column is not bit/bool");
    }
  }
  auto& ctx = a->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto b_result = b->To<value_type>();
          if (!b_result.ok()) {
            THROW_LOGIC_ERR(b_result.status().ToString());
          }
          auto b_val = b_result.value();
          if constexpr (op == OP_CONDITIONAL) {
            if constexpr (is_valid_operand<value_type>(op)) {
              auto a_vec = a->ToVector<Bit>().value();
              auto result = simd_vector_ternary_vector_scalar_vector_op<Bit, value_type, op>(ctx, a_vec, b_val, arg);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          } else {
            auto a_vec = a->ToVector<value_type>().value();
            if constexpr (is_valid_operand<value_type>(op)) {
              auto result =
                  simd_vector_ternary_vector_scalar_vector_op<value_type, value_type, op>(ctx, a_vec, b_val, arg);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          }
        }
        return (Column*)0;
      },
      c->GetInternal());
}

template <OpToken op>
Column* simd_column_ternary_column_scalar_scalar_op(Column* a, Scalar* b, Scalar* c) {
  if constexpr (op == OP_CONDITIONAL) {
    if (!a->IsBit()) {
      THROW_LOGIC_ERR("can NOT do conditional op while column is not bit/bool");
    }
  }
  auto& ctx = a->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          if constexpr (op == OP_CONDITIONAL) {
            auto a_vec = a->ToVector<Bit>().value();
            return std::visit(
                [&](auto&& b_arg) {
                  using value_type = std::decay_t<decltype(b_arg)>;
                  if constexpr (!is_valid_operand<value_type>(op)) {
                    THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
                  }
                  auto c_result = c->To<value_type>();
                  if (!c_result.ok()) {
                    THROW_LOGIC_ERR(c_result.status().ToString());
                  }
                  auto c_val = c_result.value();
                  auto result =
                      simd_vector_ternary_vector_scalar_scalar_op<Bit, value_type, op>(ctx, a_vec, b_arg, c_val);
                  auto* c = ctx.New<Column>(ctx, result);
                  return c;
                },
                b->GetInternal());
          } else {
            using value_type = typename T::value_type;
            auto b_result = c->To<value_type>();
            if (!b_result.ok()) {
              THROW_LOGIC_ERR(b_result.status().ToString());
            }
            auto b_val = b_result.value();
            auto c_result = c->To<value_type>();
            if (!c_result.ok()) {
              THROW_LOGIC_ERR(c_result.status().ToString());
            }
            auto c_val = c_result.value();
            if constexpr (is_valid_operand<value_type>(op)) {
              auto result =
                  simd_vector_ternary_vector_scalar_scalar_op<value_type, value_type, op>(ctx, arg, b_val, c_val);
              auto* c = ctx.New<Column>(ctx, result);
              return c;
            } else {
              THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
            }
          }
        }
        return (Column*)0;
      },
      a->GetInternal());
}

template <OpToken op>
Column* simd_column_ternary_scalar_column_column_op(Scalar* a, Column* b, Column* c) {
  if (!b->TypeEquals(*c)) {
    THROW_LOGIC_ERR("ternary ops have different internal dtype");
  }
  auto& ctx = b->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto a_result = a->To<value_type>();
          if (!a_result.ok()) {
            THROW_LOGIC_ERR(a_result.status().ToString());
          }
          auto a_val = a_result.value();
          auto c_vec = c->ToVector<value_type>().value();
          if constexpr (is_valid_operand<value_type>(op)) {
            auto result =
                simd_vector_ternary_scalar_vector_vector_op<value_type, value_type, op>(ctx, a_val, arg, c_vec);
            auto* c = ctx.New<Column>(ctx, result);
            return c;
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      b->GetInternal());
}
template <OpToken op>
Column* simd_column_ternary_scalar_scalar_column_op(Scalar* a, Scalar* b, Column* c) {
  auto& ctx = c->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto a_result = a->To<value_type>();
          if (!a_result.ok()) {
            THROW_LOGIC_ERR(a_result.status().ToString());
          }
          auto a_val = a_result.value();
          auto b_result = b->To<value_type>();
          if (!b_result.ok()) {
            THROW_LOGIC_ERR(b_result.status().ToString());
          }
          auto b_val = b_result.value();
          if constexpr (is_valid_operand<value_type>(op)) {
            auto result =
                simd_vector_ternary_scalar_scalar_vector_op<value_type, value_type, op>(ctx, a_val, b_val, arg);
            auto* c = ctx.New<Column>(ctx, result);
            return c;
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      c->GetInternal());
}
template <OpToken op>
Column* simd_column_ternary_scalar_column_scalar_op(Scalar* a, Column* b, Scalar* c) {
  auto& ctx = b->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          auto a_result = a->To<value_type>();
          if (!a_result.ok()) {
            THROW_LOGIC_ERR(a_result.status().ToString());
          }
          auto a_val = a_result.value();
          auto c_result = c->To<value_type>();
          if (!c_result.ok()) {
            THROW_LOGIC_ERR(c_result.status().ToString());
          }
          auto c_val = c_result.value();
          if constexpr (is_valid_operand<value_type>(op)) {
            auto result =
                simd_vector_ternary_scalar_vector_scalar_op<value_type, value_type, op>(ctx, a_val, arg, c_val);
            auto* c = ctx.New<Column>(ctx, result);
            return c;
          } else {
            THROW_LOGIC_ERR(fmt::format("Unsupported op:{} with column dtype:{}", op, get_dtype<value_type>()));
          }
        }
        return (Column*)0;
      },
      b->GetInternal());
}

#define DEFINE_SIMD_TERNARY_OP_TEMPLATE(r, _, ii, op)                                                   \
  template Column* simd_column_ternary_op<op>(Column * a, Column * b, Column * c);                      \
  template Column* simd_column_ternary_column_column_scalar_op<op>(Column * a, Column * b, Scalar * c); \
  template Column* simd_column_ternary_column_scalar_column_op<op>(Column * a, Scalar * b, Column * c); \
  template Column* simd_column_ternary_column_scalar_scalar_op<op>(Column * a, Scalar * b, Scalar * c); \
  template Column* simd_column_ternary_scalar_column_column_op<op>(Scalar * a, Column * b, Column * c); \
  template Column* simd_column_ternary_scalar_scalar_column_op<op>(Scalar * a, Scalar * b, Column * c); \
  template Column* simd_column_ternary_scalar_column_scalar_op<op>(Scalar * a, Column * b, Scalar * c);

#define DEFINE_SIMD_TERNARY_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_OP_TEMPLATE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_OP(OP_CLAMP, OP_FMA, OP_FMS, OP_FNMA, OP_FNMS);

template Column* simd_column_ternary_op<OP_CONDITIONAL>(Column* a, Column* b, Column* c);
template Column* simd_column_ternary_column_column_scalar_op<OP_CONDITIONAL>(Column* a, Column* b, Scalar* c);
template Column* simd_column_ternary_column_scalar_column_op<OP_CONDITIONAL>(Column* a, Scalar* b, Column* c);
template Column* simd_column_ternary_column_scalar_scalar_op<OP_CONDITIONAL>(Column* a, Scalar* b, Scalar* c);

}  // namespace simd
}  // namespace rapidudf