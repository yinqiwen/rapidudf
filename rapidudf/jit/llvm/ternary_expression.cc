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
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ast::TernaryExprPtr expr) {
  ast_ctx_.SetPosition(expr->position);
  auto cond_result = BuildIR(ctx, expr->cond);
  if (expr->true_false_operands.has_value()) {
    if (!cond_result.ok()) {
      return cond_result.status();
    }
    auto cond_val = cond_result.value();
    auto [true_expr, false_expr] = *(expr->true_false_operands);
    if (cond_val->GetDType().IsBool()) {
      auto true_val_result = BuildIR(ctx, true_expr);
      if (!true_val_result.ok()) {
        return true_val_result.status();
      }
      auto false_val_result = BuildIR(ctx, false_expr);
      if (!false_val_result.ok()) {
        return false_val_result.status();
      }
      auto result = cond_val->Select(true_val_result.value(), false_val_result.value());
      if (!result) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT do select true:{}, false:{}",
                                                                  true_val_result.value()->GetDType(),
                                                                  false_val_result.value()->GetDType())));
      }
      return result;
    } else if (cond_val->GetDType().IsSimdVectorBit() || cond_val->GetDType().IsSimdColumnPtr()) {
      auto true_expr_result = BuildIR(ctx, true_expr);
      if (!true_expr_result.ok()) {
        return true_expr_result.status();
      }
      auto true_expr_val = true_expr_result.value();
      auto false_expr_result = BuildIR(ctx, false_expr);
      if (!false_expr_result.ok()) {
        return false_expr_result.status();
      }
      auto false_expr_val = false_expr_result.value();
      auto func_name =
          GetFunctionName(OP_CONDITIONAL, cond_val->GetDType(), true_expr_val->GetDType(), false_expr_val->GetDType());
      std::vector<ValuePtr> args{cond_val, true_expr_val, false_expr_val};
      if (cond_val->GetDType().IsSimdColumnPtr()) {
        for (size_t i = 0; i < args.size(); i++) {
          if (!args[i]->GetDType().IsSimdColumnPtr()) {
            auto cast_func = GetFunctionName(OP_SCALAR_CAST, args[i]->GetDType());
            auto cast_result = CallFunction(cast_func, {args[i]});
            if (!cast_result.ok()) {
              return cast_result.status();
            }
            args[i] = cast_result.value();
          }
        }
      }

      auto result = CallFunction(func_name, args, false);
      if (!result.ok()) {
        return result.status();
      }
      return result.value();
    }
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("ternary op with cond:{}", cond_val->GetDType())));
  } else {
    return cond_result;
  }
}

}  // namespace llvm
}  // namespace rapidudf