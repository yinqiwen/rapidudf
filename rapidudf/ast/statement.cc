/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include "rapidudf/ast/statement.h"
#include "fmt/core.h"
// #include "rapidudf/builtin/builtin_symbols.h"

namespace rapidudf {
namespace ast {
static absl::Status check_statements(ParseContext& ctx, std::vector<Statement>& statements) {
  for (auto& statement : statements) {
    auto status = std::visit([&](auto&& arg) { return arg.Validate(ctx); }, statement);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status ExpressionStatement::Validate(ParseContext& ctx) {
  is_vector_expr = false;
  ctx.SetVectorExressionFlag(false);
  auto result = expr->Validate(ctx, rpn);
  if (!result.ok()) {
    return result.status();
  }
  is_vector_expr = ctx.IsVectorExpression();
  return absl::OkStatus();
}

absl::Status ChoiceStatement::Validate(ParseContext& ctx) {
  auto result = expr->Validate(ctx, rpn);
  if (!result.ok()) {
    return result.status();
  }
  if (!result->dtype.IsBit()) {
    return ctx.GetErrorStatus(fmt::format("Can NOT do if/elif/while on non bool expression:{}", result.value().dtype));
  }
  return check_statements(ctx, statements);
}

absl::Status IfElseStatement::Validate(ParseContext& ctx) {
  if (!if_statement.expr && elif_statements.size() > 0) {
    if_statement = elif_statements[0];
    elif_statements.erase(elif_statements.begin());
  }
  absl::Status rc = if_statement.Validate(ctx);
  if (!rc.ok()) {
    return rc;
  }
  for (auto& st : elif_statements) {
    rc = st.Validate(ctx);
    if (!rc.ok()) {
      return rc;
    }
  }
  if (else_statements) {
    return check_statements(ctx, *else_statements);
  }
  return absl::OkStatus();
}

absl::Status ReturnStatement::Validate(ParseContext& ctx) {
  absl::Status status = absl::OkStatus();
  if (expr) {
    ctx.SetVectorExressionFlag(false);
    auto expr_result = (*expr)->Validate(ctx, rpn);
    if (!expr_result.ok()) {
      return expr_result.status();
    }
    if (!ctx.CanCastTo(expr_result->dtype, ctx.GetFuncReturnDType())) {
      return ctx.GetErrorStatus(fmt::format("Can NOT return invalid dtype:{}, while func return dtype:{}",
                                            expr_result->dtype, ctx.GetFuncReturnDType()));
    }
    is_vector_expr = ctx.IsVectorExpression();
    return absl::OkStatus();
  }
  return status;
}

absl::Status WhileStatement::Validate(ParseContext& ctx) {
  ctx.EnterLoop();
  auto status = body.Validate(ctx);
  ctx.ExitLoop();
  return status;
}

absl::Status ContinueStatement::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  if (ctx.IsInLoop()) {
    return absl::OkStatus();
  } else {
    return ctx.GetErrorStatus(fmt::format("Can NOT continue in non loop block"));
  }
}
absl::Status BreakStatement::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  if (ctx.IsInLoop()) {
    return absl::OkStatus();
  } else {
    return ctx.GetErrorStatus(fmt::format("Can NOT break in non loop block"));
  }
}
}  // namespace ast
}  // namespace rapidudf