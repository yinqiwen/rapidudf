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
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/reflect.h"

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
struct ConstantNumber {
  double dv = 0;
  std::optional<DType> dtype;
  std::string ToString() const;
};

struct BinaryExpr;
struct UnaryExpr;
struct SelectExpr;
struct FuncInvoke;
struct VarAccessor;
struct Array;
struct SelectRPNNode;
using BinaryExprPtr = std::shared_ptr<BinaryExpr>;
using UnaryExprPtr = std::shared_ptr<UnaryExpr>;
using SelectExprPtr = std::shared_ptr<SelectExpr>;
using SelectRPNNodePtr = std::shared_ptr<SelectRPNNode>;

using RPNNode =
    std::variant<OpToken, bool, ConstantNumber, std::string, SelectRPNNodePtr, VarDefine, Array, VarAccessor>;

struct RPN {
  std::vector<RPNNode> nodes;
  DType dtype;
  void SetDType(ParseContext& ctx, DType dtype);
  void Print();
};
struct Array {
  std::vector<BinaryExprPtr> elements;
  DType dtype;
  uint32_t position = 0;

  std::vector<RPN> rpns;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx);
};

using Operand = std::variant<bool, ConstantNumber, std::string, VarAccessor, SelectExprPtr, BinaryExprPtr, UnaryExprPtr,
                             VarDefine, Array>;

struct Expression {
  BinaryExprPtr expr;
  RPN rpn_expr;
};

struct FuncInvokeArgs {
  std::optional<std::vector<BinaryExprPtr>> args;
  std::vector<RPN> rpns;
  absl::StatusOr<std::vector<VarTag>> Validate(ParseContext& ctx, RPN* rpn = nullptr);
};

struct FieldAccess {
  std::string field;
  std::optional<FuncInvokeArgs> func_args;
  uint32_t position = 0;

  StructMember struct_member;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx, DType src_dtype);
};
using DynamicParamAccess = std::variant<std::string, uint32_t, VarRef>;
using MemberAccess = std::variant<FieldAccess, DynamicParamAccess>;

struct VarAccessor {
  std::string name;
  std::optional<std::vector<MemberAccess>> access_args;
  std::optional<FuncInvokeArgs> func_args;

  OpToken builtin_op = OP_INVALID;
  std::vector<std::string> access_func_names;
  uint32_t position = 0;

  absl::StatusOr<VarTag> Validate(ParseContext& ctx, RPN& rpn, bool& as_builtin_op);
};

struct SelectRPNNode {
  RPN cond_rpn;
  RPN true_rpn;
  RPN false_rpn;
};

struct SelectExpr {
  Operand cond;
  std::optional<std::tuple<Operand, Operand>> true_false_operands;
  uint32_t position = 0;
  DType ternary_result_dtype;
  SelectRPNNodePtr select_rpn;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx, RPN& rpn);
};

struct BinaryExpr {
  Operand left;
  // std::optional<std::tuple<OpToken, Operand>> right;
  std::vector<std::tuple<OpToken, Operand>> right;
  uint32_t position = 0;
  static BinaryExprPtr New(Operand operand, uint32_t pos) {
    auto p = std::make_shared<BinaryExpr>();
    p->left = operand;
    p->position = pos;
    return p;
  }
  void SetRight(std::optional<std::tuple<rapidudf::OpToken, BinaryExprPtr>> operand) {
    if (operand.has_value()) {
      auto [op, right_op] = *operand;
      Operand operand = right_op;
      right.emplace_back(std::make_tuple(op, operand));
    }
  }

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
  absl::StatusOr<VarTag> Validate(ParseContext& ctx, RPN& rpn);
};
struct UnaryExpr {
  std::optional<OpToken> op;
  Operand operand;
  uint32_t position = 0;
  absl::StatusOr<VarTag> Validate(ParseContext& ctx, RPN& rpn);
};

}  // namespace ast
}  // namespace rapidudf