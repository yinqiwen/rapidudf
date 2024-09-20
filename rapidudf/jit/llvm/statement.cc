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
#include <utility>
#include <variant>
#include <vector>
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/jit_session.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
namespace rapidudf {
namespace llvm {
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const std::vector<ast::Statement>& statements) {
  for (auto& statement : statements) {
    auto rc = std::visit([&](auto&& arg) { return BuildIR(ctx, arg); }, statement);
    if (!rc.ok()) {
      return rc;
    }
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::ReturnStatement& statement) {
  if (statement.expr.has_value()) {
    auto val_result = BuildIR(ctx, *statement.expr);
    if (!val_result.ok()) {
      return val_result.status();
    }
    ValuePtr val = val_result.value();

    auto ret_val = val->CastTo(ctx->desc.return_type);
    if (!ret_val) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT cast to return dtype:{} from dtype:{}", ctx->desc.return_type, val->GetDType())));
    }
    if (GetCompileContext()->return_value != nullptr) {
      // auto status = GetCompileContext()->return_value->CopyFrom(ret_val);
      // if (!status.ok()) {
      //   return status;
      // }
      // GetCompileContext()->return_value->CopyFrom(ret_val);
      GetSession()->GetIRBuilder()->CreateStore(ret_val->GetValue(), GetCompileContext()->return_value->GetRawValue());
    }
    GetSession()->GetIRBuilder()->CreateBr(GetCompileContext()->exit_block);
  } else {
    // GetSession()->GetIRBuilder()->CreateRetVoid();
    GetSession()->GetIRBuilder()->CreateBr(GetCompileContext()->exit_block);
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::IfElseStatement& statement) {
  RUDF_DEBUG("Start compile ifelse statement.");
  auto if_cond_val_result = BuildIR(ctx, statement.if_statement.expr);
  if (!if_cond_val_result.ok()) {
    return if_cond_val_result.status();
  }
  auto if_cond_val = if_cond_val_result.value();

  auto* current_func = GetSession()->GetIRBuilder()->GetInsertBlock()->getParent();

  uint32_t label_cursor = GetLabelCursor();
  std::string if_block_label = fmt::format("if_block_{}", label_cursor);
  std::string continue_label = fmt::format("continue_{}", label_cursor);
  ::llvm::BasicBlock* if_block =
      ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), if_block_label, current_func);

  std::vector<::llvm::BasicBlock*> elif_blocks;
  std::vector<::llvm::BasicBlock*> elif_cond_blocks;
  ::llvm::BasicBlock* else_block = nullptr;
  for (size_t i = 0; i < statement.elif_statements.size(); i++) {
    std::string elif_cond_label = fmt::format("elif_cond_{}_{}", label_cursor, i);
    ::llvm::BasicBlock* elif_cond_block =
        ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), elif_cond_label, current_func);
    elif_cond_blocks.emplace_back(elif_cond_block);
    std::string elif_label = fmt::format("elif_{}_{}", label_cursor, i);
    ::llvm::BasicBlock* elif_block =
        ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), elif_label, current_func);
    elif_blocks.emplace_back(elif_block);
  }
  if (statement.else_statements.has_value()) {
    std::string else_label = fmt::format("else_{}", label_cursor);
    else_block = ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), else_label, current_func);
  }
  ::llvm::BasicBlock* continue_block =
      ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), continue_label, current_func);
  ::llvm::BasicBlock* if_next_block = continue_block;
  if (elif_cond_blocks.size() > 0) {
    if_next_block = elif_cond_blocks[0];
  } else if (else_block != nullptr) {
    if_next_block = else_block;
  }

  GetSession()->GetIRBuilder()->CreateCondBr(if_cond_val->GetValue(), if_block, if_next_block);
  GetSession()->GetIRBuilder()->SetInsertPoint(if_block);
  auto status = BuildIR(ctx, statement.if_statement.statements);
  if (!status.ok()) {
    return status;
  }
  if (if_block->getTerminator() == nullptr) {
    GetSession()->GetIRBuilder()->CreateBr(continue_block);  // end if
  }
  for (size_t i = 0; i < elif_cond_blocks.size(); i++) {
    GetSession()->GetIRBuilder()->SetInsertPoint(elif_cond_blocks[i]);
    auto elif_cond_val_result = BuildIR(ctx, statement.elif_statements[i].expr);
    if (!elif_cond_val_result.ok()) {
      return elif_cond_val_result.status();
    }
    auto elif_cond_val = elif_cond_val_result.value();
    ::llvm::BasicBlock* next_block = continue_block;
    if (i < elif_blocks.size() - 1) {
      next_block = elif_cond_blocks[i + 1];
    } else if (else_block != nullptr) {
      next_block = else_block;
    }
    GetSession()->GetIRBuilder()->CreateCondBr(elif_cond_val->GetValue(), elif_blocks[i], next_block);
    GetSession()->GetIRBuilder()->SetInsertPoint(elif_blocks[i]);
    auto status = BuildIR(ctx, statement.elif_statements[i].statements);
    if (!status.ok()) {
      return status;
    }
    if (elif_blocks[i]->getTerminator() == nullptr) {
      GetSession()->GetIRBuilder()->CreateBr(continue_block);  // end elif
    }
  }
  if (else_block != nullptr) {
    GetSession()->GetIRBuilder()->SetInsertPoint(else_block);
    auto status = BuildIR(ctx, *statement.else_statements);
    if (!status.ok()) {
      return status;
    }
    if (else_block->getTerminator() == nullptr) {
      GetSession()->GetIRBuilder()->CreateBr(continue_block);  // end else
    }
  }
  // Continue
  if (!continue_block->hasNPredecessorsOrMore(1)) {
    continue_block->removeFromParent();
  } else {
    GetSession()->GetIRBuilder()->SetInsertPoint(continue_block);
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::WhileStatement& statement) {
  uint32_t label_cursor = GetLabelCursor();
  std::string while_cond_label = fmt::format("while_cond_{}", label_cursor);
  std::string while_body_label = fmt::format("while_body_{}", label_cursor);
  std::string while_end_label = fmt::format("while_end_{}", label_cursor);
  auto* current_func = GetSession()->GetIRBuilder()->GetInsertBlock()->getParent();
  ::llvm::BasicBlock* while_cond_block =
      ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), while_cond_label, current_func);
  ::llvm::BasicBlock* while_end_block = ::llvm::BasicBlock::Create(*GetLLVMContext(), while_end_label);
  ::llvm::BasicBlock* while_body_block =
      ::llvm::BasicBlock::Create(GetSession()->GetIRBuilder()->getContext(), while_body_label);
  ctx->loop_blocks.emplace_back(std::make_pair(while_cond_block, while_end_block));
  if (GetSession()->GetIRBuilder()->GetInsertBlock()->getTerminator() == nullptr) {
    GetSession()->GetIRBuilder()->CreateBr(while_cond_block);
  }
  GetSession()->GetIRBuilder()->SetInsertPoint(while_cond_block);
  auto cond_result = BuildIR(ctx, statement.body.expr);
  if (!cond_result.ok()) {
    return cond_result.status();
  }
  auto cond_val = cond_result.value();
  GetSession()->GetIRBuilder()->CreateCondBr(cond_val->GetValue(), while_body_block, while_end_block);
  while_body_block->insertInto(current_func);
  GetSession()->GetIRBuilder()->SetInsertPoint(while_body_block);
  auto status = BuildIR(ctx, statement.body.statements);
  if (!status.ok()) {
    return status;
  }
  if (GetSession()->GetIRBuilder()->GetInsertBlock()->getTerminator() == nullptr) {
    GetSession()->GetIRBuilder()->CreateBr(while_cond_block);  // end while body
  }
  while_end_block->insertInto(current_func);
  GetSession()->GetIRBuilder()->SetInsertPoint(while_end_block);
  ctx->loop_blocks.pop_back();
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::ExpressionStatement& statement) {
  auto val = BuildIR(ctx, statement.expr);
  if (!val.ok()) {
    return val.status();
  }
  return absl::OkStatus();
}

absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::ContinueStatement& statement) {
  ast_ctx_.SetPosition(statement.position);
  if (ctx->loop_blocks.empty()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("continue in non loop block"));
  }
  GetSession()->GetIRBuilder()->CreateBr(ctx->loop_blocks.back().first);  // continue to loop cond
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::BreakStatement& statement) {
  ast_ctx_.SetPosition(statement.position);
  if (ctx->loop_blocks.empty()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("break in non loop block"));
  }
  GetSession()->GetIRBuilder()->CreateBr(ctx->loop_blocks.back().second);  // continue to loop end
  return absl::OkStatus();
}
}  // namespace llvm
}  // namespace rapidudf