/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once
#include <fmt/format.h>
#include <array>
#include <string>
#include <string_view>
namespace rapidudf {
enum OpToken {
  OP_INVALID = 0,
  OP_ASSIGN,
  OP_PLUS,
  OP_MINUS,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_MOD,
  OP_POSITIVE,
  OP_NEGATIVE,
  OP_NOT,
  OP_EQUAL,
  OP_NOT_EQUAL,
  OP_LESS,
  OP_LESS_EQUAL,
  OP_GREATER,
  OP_GREATER_EQUAL,
  OP_LOGIC_AND,
  OP_LOGIC_OR,
  OP_CONDITIONAL,
  OP_MAX,
  OP_MIN,
  OP_POW,
  OP_SQRT,
  OP_FLOOR,
  OP_CEIL,
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
  OP_ATAN2,
  OP_SINH,
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
  OP_HYPOT,
  OP_DOT,
  OP_IOTA,
  OP_CLAMP,
  OP_MULADD,
  OP_MULSUB,
  OP_MULADDSUB,
  OP_NEG_MULADD,
  OP_NEG_MULSUB,
  OP_END,
};
constexpr std::array<std::string_view, OP_END> kOpTokenStrs = {
    "invalid",   "assign",   "add",         "sub",       "multiply", "divide",     "mod",        "positive",
    "negative",  "not",      "equal",       "not_equal", "less",     "less_equal", "greater",    "greater_equal",
    "logic_and", "logic_or", "conditional", "max",       "min",      "pow",        "sqrt",       "floor",
    "ceil",      "erf",      "erfc",        "abs",       "sin",      "cos",        "tan",        "asin",
    "acos",      "atan",     "atanh",       "atan2",     "sinh",     "tanh",       "asinh",      "acosh",
    "exp",       "exp2",     "expm1",       "log",       "log2",     "log10",      "log1p",      "hypot",
    "dot",       "iota",     "clamp",       "muladd",    "mulsub",   "muladdsub",  "neg_muladd", "neg_mulsub"};
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