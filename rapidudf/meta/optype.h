/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <string>
#include <string_view>

#include "fmt/format.h"
namespace rapidudf {
enum OpToken {
  OP_INVALID = 0,
  OP_UNARY_BEGIN = 1,
  OP_POSITIVE,
  OP_NEGATIVE,
  OP_NOT,
  OP_SQRT,
  OP_CBRT,
  OP_FLOOR,
  OP_CEIL,
  OP_ROUND,
  OP_RINT,
  OP_TRUNC,
  OP_ERF,
  OP_ERFC,
  OP_ABS,
  OP_SIN,
  OP_COS,
  OP_TAN,
  OP_ASIN,
  OP_ACOS,
  OP_ATAN,
  OP_ATANH,
  OP_SINH,
  OP_COSH,
  OP_TANH,
  OP_ASINH,
  OP_ACOSH,
  OP_EXP,
  OP_EXP2,
  OP_EXPM1,
  OP_LOG,
  OP_LOG2,
  OP_LOG10,
  OP_LOG1P,
  OP_APPROX_RECIP,
  OP_APPROX_RECIP_SQRT,
  OP_UNARY_END,
  OP_BINARY_BEGIN,
  OP_PLUS,
  OP_MINUS,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_MOD,
  OP_ASSIGN,
  OP_PLUS_ASSIGN,
  OP_MINUS_ASSIGN,
  OP_MULTIPLY_ASSIGN,
  OP_DIVIDE_ASSIGN,
  OP_MOD_ASSIGN,
  OP_EQUAL,
  OP_NOT_EQUAL,
  OP_LESS,
  OP_LESS_EQUAL,
  OP_GREATER,
  OP_GREATER_EQUAL,
  OP_LOGIC_AND,
  OP_LOGIC_OR,
  OP_LOGIC_XOR,
  OP_ATAN2,
  OP_MAX,
  OP_MIN,
  OP_POW,
  OP_HYPOT,
  OP_ABS_DIFF,
  OP_BINARY_END,
  OP_TERNARY_BEGIN,
  OP_CONDITIONAL,
  OP_CLAMP,
  OP_FMA,
  OP_FMS,
  OP_FNMA,
  OP_FNMS,
  OP_TERNARY_END,
  OP_MISC_BEGIN,
  OP_DOT,
  OP_COS_DISTANCE,
  OP_L2_DISTANCE,
  OP_IOTA,
  OP_SORT,
  OP_SELECT,
  OP_TOPK,
  OP_SORT_KV,
  OP_SELECT_KV,
  OP_TOPK_KV,
  OP_ARG_SORT,
  OP_ARG_SELECT,
  OP_SUM,
  OP_AVG,
  OP_CLONE,
  OP_FILTER,
  OP_GATHER,
  OP_MISC_END,
  OP_END,
};
constexpr std::array<std::string_view, OP_END> kOpTokenStrs = {"invalid",
                                                               "unary_begin",
                                                               "positive",
                                                               "negative",
                                                               "not",
                                                               "sqrt",
                                                               "cbrt",
                                                               "floor",
                                                               "ceil",
                                                               "round",
                                                               "rint",
                                                               "trunc",
                                                               "erf",
                                                               "erfc",
                                                               "abs",
                                                               "sin",
                                                               "cos",
                                                               "tan",
                                                               "asin",
                                                               "acos",
                                                               "atan",
                                                               "atanh",
                                                               "sinh",
                                                               "cosh",
                                                               "tanh",
                                                               "asinh",
                                                               "acosh",
                                                               "exp",
                                                               "exp2",
                                                               "expm1",
                                                               "log",
                                                               "log2",
                                                               "log10",
                                                               "log1p",
                                                               "approx_recip",
                                                               "approx_recip_sqrt",
                                                               "unary_end",
                                                               "binary_begin",
                                                               "add",
                                                               "sub",
                                                               "multiply",
                                                               "divide",
                                                               "mod",
                                                               "assign",
                                                               "add_assign",
                                                               "sub_assign",
                                                               "multiply_assign",
                                                               "divide_assign",
                                                               "mod_assign",
                                                               "eq",
                                                               "ne",
                                                               "lt",
                                                               "le",
                                                               "gt",
                                                               "ge",
                                                               "and",
                                                               "or",
                                                               "xor",
                                                               "atan2",
                                                               "max",
                                                               "min",
                                                               "pow",
                                                               "hypot",
                                                               "abs_diff",
                                                               "binary_end",
                                                               "ternary_begin",
                                                               "conditional",
                                                               "clamp",
                                                               "fma",
                                                               "fms",
                                                               "fnma",
                                                               "fnms",
                                                               "ternary_end",
                                                               "misc_begin",
                                                               "dot",
                                                               "cos_distance",
                                                               "l2_distance",
                                                               "iota",
                                                               "sort",
                                                               "select",
                                                               "topk",
                                                               "sort_kv",
                                                               "select_kv",
                                                               "topk_kv",
                                                               "argsort",
                                                               "argselect",
                                                               "sum",
                                                               "avg",
                                                               "clone",
                                                               "filter",
                                                               "gather",
                                                               "misc_end"};
}  // namespace rapidudf

template <>
struct fmt::formatter<rapidudf::OpToken> : formatter<std::string> {
  // parse is inherited from formatter<string_view>.
  auto format(rapidudf::OpToken c, format_context& ctx) const -> format_context::iterator {
    std::string view = "object";
    if (c <= rapidudf::OP_END) {
      view = std::string(rapidudf::kOpTokenStrs[c]);
    } else {
      view = fmt::format("op/{}", static_cast<int>(c));
    }
    return formatter<std::string>::format(view, ctx);
  }
};