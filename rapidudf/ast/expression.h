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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#include "rapidudf/ast/context.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/optype.h"

namespace rapidudf {
namespace ast {

struct VarDefine {
  // DType dtype;
  std::string name;
  uint32_t position = 0;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};

struct VarRef {
  std::string name;
  uint32_t position = 0;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};

struct BinaryExpr;
struct UnaryExpr;
struct FuncInvoke;
struct VarAccessor;
using BinaryExprPtr = std::shared_ptr<BinaryExpr>;
using UnaryExprPtr = std::shared_ptr<UnaryExpr>;
using Operand = std::variant<double, int64_t, bool, std::string, VarAccessor, BinaryExprPtr, UnaryExprPtr, VarDefine>;
using Expression = BinaryExprPtr;

struct FuncInvokeArgs {
  std::optional<std::vector<BinaryExprPtr>> args;
  absl::StatusOr<std::vector<VarTag>> Validate(ParseContext& ctx);
};

struct FieldAccess {
  std::string field;
  std::optional<FuncInvokeArgs> func_args;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx, DType src_dtype);
};
using DynamicParamAccess = std::variant<std::string, uint32_t, VarRef>;
using MemberAccess = std::variant<FieldAccess, DynamicParamAccess>;

struct VarAccessor {
  std::string name;
  std::optional<std::vector<MemberAccess>> access_args;
  std::optional<FuncInvokeArgs> func_args;
  uint32_t position = 0;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};

struct BinaryExpr {
  Operand left;
  // std::optional<std::tuple<OpToken, Operand>> right;
  std::vector<std::tuple<OpToken, Operand>> right;
  uint32_t position = 0;
  void SetRight(const std::vector<std::tuple<OpToken, UnaryExprPtr>>& ops) {
    for (const auto& [op, expr] : ops) {
      Operand operand = expr;
      right.emplace_back(std::make_tuple(op, operand));
    }
  }
  void SetRight(const std::vector<std::tuple<OpToken, BinaryExprPtr>>& ops) {
    for (const auto& [op, expr] : ops) {
      Operand operand = expr;
      right.emplace_back(std::make_tuple(op, operand));
    }
  }
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};
struct UnaryExpr {
  std::optional<OpToken> op;
  Operand operand;
  uint32_t position = 0;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};

}  // namespace ast
}  // namespace rapidudf