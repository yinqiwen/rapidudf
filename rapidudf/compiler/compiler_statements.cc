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

#include "rapidudf/compiler/codegen.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {
absl::Status JitCompiler::BuildIR(const std::vector<ast::Statement>& statements) {
  for (auto& statement : statements) {
    auto rc = std::visit([&](auto&& arg) { return BuildIR(arg); }, statement);
    if (!rc.ok()) {
      return rc;
    }
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(const ast::ReturnStatement& statement) {
  //   auto* ir_builder = GetSession()->GetIRBuilder();
  ValuePtr return_val;
  if (statement.expr.has_value()) {
    auto val_result = BuildIR(statement.rpn);
    if (!val_result.ok()) {
      return val_result.status();
    }
    return_val = val_result.value();
  }
  return codegen_->Return(return_val);
}
absl::Status JitCompiler::BuildIR(const ast::IfElseStatement& statement) {
  auto if_cond_val_result = BuildIR(statement.if_statement.rpn);
  if (!if_cond_val_result.ok()) {
    return if_cond_val_result.status();
  }
  auto if_cond_val = if_cond_val_result.value();
  auto condition = codegen_->NewCondition(statement.elif_statements.size(), statement.else_statements.has_value());
  codegen_->BeginIf(condition, if_cond_val);
  auto status = BuildIR(statement.if_statement.statements);
  if (!status.ok()) {
    return status;
  }
  codegen_->EndIf(condition);
  for (size_t i = 0; i < statement.elif_statements.size(); i++) {
    codegen_->BeginElif(condition, i);
    auto elif_cond_val_result = BuildIR(statement.elif_statements[i].rpn);
    if (!elif_cond_val_result.ok()) {
      return elif_cond_val_result.status();
    }
    auto elif_cond_val = elif_cond_val_result.value();
    codegen_->EndElifCond(condition, i, elif_cond_val);
    status = BuildIR(statement.elif_statements[i].statements);
    if (!status.ok()) {
      return status;
    }
    codegen_->EndElif(condition, i);
  }
  if (statement.else_statements.has_value()) {
    codegen_->BeginElse(condition);
    status = BuildIR(*statement.else_statements);
    if (!status.ok()) {
      return status;
    }
    codegen_->EndElse(condition);
  }
  codegen_->FinishCondition(condition);
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(const ast::WhileStatement& statement) {
  auto loop = codegen_->NewLoop();
  auto cond_result = BuildIR(statement.body.rpn);
  if (!cond_result.ok()) {
    return cond_result.status();
  }
  auto cond_val = cond_result.value();
  codegen_->AddLoopCond(loop, cond_val);
  auto status = BuildIR(statement.body.statements);
  if (!status.ok()) {
    return status;
  }
  codegen_->FinishLoop(loop);
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(const ast::ExpressionStatement& statement) {
  auto val = BuildIR(statement.rpn);
  if (!val.ok()) {
    return val.status();
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(const ast::ContinueStatement& statement) {
  ast_ctx_.SetPosition(statement.position);
  return codegen_->ContinueLoop();
}
absl::Status JitCompiler::BuildIR(const ast::BreakStatement& statement) {
  ast_ctx_.SetPosition(statement.position);
  return codegen_->BreakLoop();
}
}  // namespace compiler
}  // namespace rapidudf