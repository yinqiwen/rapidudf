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
#include <optional>
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
  RPN rpn;
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
  RPN rpn;
  bool is_vector_expr = false;
  absl::Status Validate(ParseContext& ctx);
};

struct WhileStatement {
  ChoiceStatement body;
  absl::Status Validate(ParseContext& ctx);
};

struct ReturnStatement {
  RPN rpn;
  std::optional<BinaryExprPtr> expr;
  bool is_vector_expr = false;
  absl::Status Validate(ParseContext& ctx);
};

}  // namespace ast
}  // namespace rapidudf