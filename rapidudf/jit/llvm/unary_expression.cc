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
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
namespace rapidudf {
namespace llvm {
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::Operand& expr) {
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::string>) {
          return BuildIR(ctx, arg);
        } else if constexpr (std::is_same_v<T, ast::VarAccessor> || std::is_same_v<T, ast::VarDefine>) {
          return BuildIR(ctx, arg);
        } else if constexpr (std::is_same_v<T, ast::BinaryExprPtr> || std::is_same_v<T, ast::UnaryExprPtr> ||
                             std::is_same_v<T, ast::TernaryExprPtr>) {
          return BuildIR(ctx, arg);
        } else if constexpr (std::is_same_v<T, ast::ConstantNumber>) {
          if (arg.dtype.has_value()) {
            return BuildIR(ctx, arg.dv, *arg.dtype);
          } else {
            return BuildIR(ctx, arg.dv);
          }
        } else if constexpr (std::is_same_v<T, ast::Array>) {
          return BuildIR(ctx, arg);
        } else {
          static_assert(sizeof(arg) == -1, "non-exhaustive visitor!");
          ValuePtr empty;
          return empty;
        }
      },
      expr);
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ast::UnaryExprPtr expr) {
  ast_ctx_.SetPosition(expr->position);
  auto val_result = BuildIR(ctx, expr->operand);
  if (!val_result.ok()) {
    return val_result.status();
  }
  auto val = val_result.value();
  if (expr->op.has_value()) {
    if (val->GetDType().IsVoid()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with void operands", *(expr->op))));
    }
    auto result = val->UnaryOp(*(expr->op));
    if (!result) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with  operands:{}", *(expr->op), val->GetDType())));
    }
    return result;
  }
  return val;
}

}  // namespace llvm
}  // namespace rapidudf