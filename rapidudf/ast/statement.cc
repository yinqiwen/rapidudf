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
#include "rapidudf/ast/statement.h"
#include "fmt/core.h"
#include "rapidudf/builtin/builtin_symbols.h"
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
absl::Status ChoiceStatement::Validate(ParseContext& ctx) {
  auto result = expr->Validate(ctx);
  if (!result.ok()) {
    return result.status();
  }
  if (!result->dtype.IsBit()) {
    return ctx.GetErrorStatus(fmt::format("Can NOT do if/elif/while on non bool expression:{}", result.value().dtype));
  }
  return check_statements(ctx, statements);
}

absl::Status IfElseStatement::Validate(ParseContext& ctx) {
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
    auto expr_result = (*expr)->Validate(ctx);
    if (!expr_result.ok()) {
      return expr_result.status();
    }
    if (!ctx.CanCastTo(expr_result->dtype, ctx.GetFuncReturnDType())) {
      return ctx.GetErrorStatus(fmt::format("Can NOT return invalid dtype:{}, while func return dtype:{}",
                                            expr_result->dtype, ctx.GetFuncReturnDType()));
    }
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