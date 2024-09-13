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
#include <variant>
#include <vector>
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/expression.h"

namespace rapidudf {
namespace ast {

struct ReturnStatement;
struct ExpressionStatement;
struct WhileStatement;
struct IfElseStatement;
struct ContinueStatement;
struct BreakStatement;
using Statement = std::variant<ReturnStatement, ExpressionStatement, WhileStatement, IfElseStatement, ContinueStatement,
                               BreakStatement>;

struct ContinueStatement {
  uint32_t position = 0;
  absl::Status Validate(ParseContext& ctx);
};
struct BreakStatement {
  uint32_t position = 0;
  absl::Status Validate(ParseContext& ctx);
};

struct ChoiceStatement {
  BinaryExprPtr expr;
  std::vector<Statement> statements;

  absl::Status Validate(ParseContext& ctx);
};
struct IfElseStatement {
  ChoiceStatement if_statement;
  std::vector<ChoiceStatement> elif_statements;
  std::optional<std::vector<Statement>> else_statements;
  absl::Status Validate(ParseContext& ctx);
};
struct ExpressionStatement {
  BinaryExprPtr expr;
  absl::Status Validate(ParseContext& ctx) {
    auto result = expr->Validate(ctx);
    if (!result.ok()) {
      return result.status();
    }
    return absl::OkStatus();
  }
};

struct WhileStatement {
  ChoiceStatement body;
  absl::Status Validate(ParseContext& ctx);
};

struct ReturnStatement {
  std::optional<BinaryExprPtr> expr;
  // uint32_t position = 0;
  absl::Status Validate(ParseContext& ctx);
};

}  // namespace ast
}  // namespace rapidudf