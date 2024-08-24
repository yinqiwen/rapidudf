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

#include <fmt/core.h>
#include <variant>
#include <vector>
#include "rapidudf/codegen/optype.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
namespace rapidudf {
absl::Status JitCompiler::CompileStatement(const ast::ReturnStatement& statement) {
  if (statement.expr.has_value()) {
    auto val_result = CompileExpression(*statement.expr);
    if (!val_result.ok()) {
      return val_result.status();
    }
    auto val = val_result.value();
    auto ret_val = val->CastTo(func_desc_.return_type);
    if (!ret_val) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT cast to return dtype:{} from dtype:{}", func_desc_.return_type, val->GetDType())));
    }
    GetCodeGenerator().ReturnValue(ret_val);
    GetCodeGenerator().DropTmpValue(val);
    GetCodeGenerator().DropTmpValue(ret_val);
  }
  GetCodeGenerator().Jump(std::string(kFuncExistLabel));
  return absl::OkStatus();
}
absl::Status JitCompiler::CompileStatement(const ast::IfElseStatement& statement) {
  RUDF_DEBUG("Start compile ifelse statement.");
  auto expr_val_result = CompileExpression(statement.if_statement.expr);
  if (!expr_val_result.ok()) {
    RUDF_LOG_ERROR_STATUS(expr_val_result.status());
  }
  auto expr_val = expr_val_result.value();
  if (!expr_val->GetDType().IsBool()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("if expr MUST return bool value, while got:{}", expr_val->GetDType())));
  }
  RUDF_DEBUG("if expr compiled success.");
  uint32_t ifelse_cursor = label_cursor_++;

  std::string if_final_label = fmt::format("if_final_{}", ifelse_cursor);
  std::vector<std::string> elif_labels;
  for (size_t i = 0; i < statement.elif_statements.size(); i++) {
    elif_labels.emplace_back(fmt::format("elif_{}_{}", ifelse_cursor, i));
  }
  std::string else_label;
  if (statement.else_statements) {
    else_label = fmt::format("else_{}", ifelse_cursor);
  }
  std::string if_fail_label;
  if (!elif_labels.empty()) {
    if_fail_label = elif_labels[0];
  } else if (!else_label.empty()) {
    if_fail_label = else_label;
  } else {
    if_fail_label = if_final_label;
  }
  auto false_bin = Value::New(&GetCodeGenerator(), DATA_U8, static_cast<uint64_t>(0));
  int rc = expr_val->Cmp(OP_EQUAL, *false_bin, nullptr);
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("cmp if expr failed.")));
  }
  RUDF_DEBUG("if expr cmp compiled success.");
  GetCodeGenerator().Jump(if_fail_label, OP_EQUAL);
  for (auto& st : statement.if_statement.statements) {
    auto status = std::visit([&](auto&& arg) { return CompileStatement(arg); }, st);
    if (!status.ok()) {
      return status;
    }
  }
  RUDF_DEBUG("if body compiled success.");
  for (size_t i = 0; i < statement.elif_statements.size(); i++) {
    GetCodeGenerator().Label(elif_labels[i]);
    auto expr_val_result = CompileExpression(statement.elif_statements[i].expr);
    if (!expr_val_result.ok()) {
      RUDF_LOG_ERROR_STATUS(expr_val_result.status());
    }
    auto expr_val = expr_val_result.value();
    if (!expr_val->GetDType().IsBool()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("elif expr MUST return bool value, while got:{}", expr_val->GetDType())));
    }
    std::string elif_fail_label;
    if (i != statement.elif_statements.size() - 1) {
      elif_fail_label = elif_labels[i + 1];
    } else if (!else_label.empty()) {
      elif_fail_label = else_label;
    } else {
      elif_fail_label = if_final_label;
    }
    rc = expr_val->Cmp(OP_EQUAL, *false_bin, nullptr);
    if (0 != rc) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("cmp elif expr failed.")));
    }
    GetCodeGenerator().Jump(elif_fail_label, OP_EQUAL);
    for (auto& st : statement.elif_statements[i].statements) {
      auto status = std::visit([&](auto&& arg) { return CompileStatement(arg); }, st);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (statement.else_statements) {
    GetCodeGenerator().Label(else_label);
    for (auto& st : *statement.else_statements) {
      auto status = std::visit([&](auto&& arg) { return CompileStatement(arg); }, st);
      if (!status.ok()) {
        return status;
      }
    }
  }
  GetCodeGenerator().Label(if_final_label);
  GetCodeGenerator().GetCodeGen().nop();
  return absl::OkStatus();
}
absl::Status JitCompiler::CompileStatement(const ast::WhileStatement& statement) {
  uint32_t while_cursor = label_cursor_++;
  std::string while_start_label = fmt::format("while_start_{}", while_cursor);
  std::string while_end_label = fmt::format("while_end_{}", while_cursor);
  GetCodeGenerator().Label(while_start_label);
  auto expr_val_result = CompileExpression(statement.body.expr);
  if (!expr_val_result.ok()) {
    RUDF_LOG_ERROR_STATUS(expr_val_result.status());
  }
  auto expr_val = expr_val_result.value();
  if (!expr_val->GetDType().IsBool()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("while expr MUST return bool value, while got:{}", expr_val->GetDType())));
  }
  RUDF_DEBUG("while expr compiled success.");
  auto false_bin = Value::New(&GetCodeGenerator(), DATA_U8, static_cast<uint64_t>(0));

  int rc = expr_val->Cmp(OP_EQUAL, *false_bin, nullptr);
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("cmp while expr failed.")));
  }
  RUDF_DEBUG("while expr cmp compiled success.");
  GetCodeGenerator().Jump(while_end_label, OP_EQUAL);
  for (auto& st : statement.body.statements) {
    auto rc = std::visit([&](auto&& arg) { return CompileStatement(arg); }, st);
    if (!rc.ok()) {
      return rc;
    }
  }
  RUDF_DEBUG("while body compiled success.");
  GetCodeGenerator().Jump(while_start_label);
  GetCodeGenerator().Label(while_end_label);
  GetCodeGenerator().GetCodeGen().nop();
  return absl::OkStatus();
}
absl::Status JitCompiler::CompileStatement(const ast::ExpressionStatement& statement) {
  auto val = CompileExpression(statement.expr);
  if (!val.ok()) {
    return val.status();
  }
  GetCodeGenerator().DropTmpValue(val.value());
  return absl::OkStatus();
}
}  // namespace rapidudf