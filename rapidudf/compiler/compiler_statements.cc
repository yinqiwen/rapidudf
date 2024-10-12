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