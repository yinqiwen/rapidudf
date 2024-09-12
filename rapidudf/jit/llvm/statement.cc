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
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::ReturnStatement& statement) {
  if (statement.expr.has_value()) {
    auto val_result = BuildIR(ctx, *statement.expr);
    if (!val_result.ok()) {
      return val_result.status();
    }
    ValuePtr val = val_result.value();
    auto ret_val = val->CastTo(current_compile_functon_ctx_->desc.return_type);
    if (!ret_val) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT cast to return dtype:{} from dtype:{}",
                                              current_compile_functon_ctx_->desc.return_type, val->GetDType())));
    }
    ir_builder_->CreateRet(ret_val->GetValue());
  } else {
    ir_builder_->CreateRetVoid();
  }
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::IfElseStatement& statement) {
  RUDF_DEBUG("Start compile ifelse statement.");

  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::WhileStatement& statement) {
  return absl::OkStatus();
}
absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::ExpressionStatement& statement) {
  auto val = BuildIR(ctx, statement.expr);
  if (!val.ok()) {
    return val.status();
  }
  return absl::OkStatus();
}
}  // namespace llvm
}  // namespace rapidudf