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

#include "fmt/core.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, OpToken op, ValuePtr cond_val,
                                              ValuePtr true_expr_val, ValuePtr false_expr_val) {
  if (op == OP_CONDITIONAL) {
    if (cond_val->GetDType().IsBool()) {
      auto result = cond_val->Select(true_expr_val, false_expr_val);
      if (!result) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
            fmt::format("Can NOT do select true:{}, false:{}", true_expr_val->GetDType(), false_expr_val->GetDType())));
      }
      return result;
    } else if (cond_val->GetDType().IsSimdVectorBit() || cond_val->GetDType().IsSimdColumnPtr()) {
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
      auto result = CallFunction(func_name, args);
      if (!result.ok()) {
        return result.status();
      }
      return result.value();
    }
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("ternary op with cond:{}", cond_val->GetDType())));
  } else {
    auto a = cond_val;
    auto b = true_expr_val;
    auto c = false_expr_val;
    ValuePtr result;
    if (a->GetDType().IsSimdVector() || a->GetDType().IsSimdColumnPtr() || b->GetDType().IsSimdVector() ||
        b->GetDType().IsSimdColumnPtr() || c->GetDType().IsSimdVector() || c->GetDType().IsSimdColumnPtr()) {
      auto func_name = GetFunctionName(op, a->GetDType(), b->GetDType(), c->GetDType());
      std::vector<ValuePtr> args{a, b, c};
      auto call_result = CallFunction(func_name, args);
      if (!call_result.ok()) {
        return call_result.status();
      }
      result = call_result.value();
    } else {
      if (HasIntrinsic(op)) {
        result = a->TernaryOp(op, b, c);
      } else {
        auto func_name = GetFunctionName(op, a->GetDType());
        std::vector<ValuePtr> args{a, b, c};
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      }
    }
    return result;
  }
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ast::SelectRPNNodePtr select) {
  auto cond_result = BuildIR(ctx, select->cond_rpn);
  if (!cond_result.ok()) {
    return cond_result.status();
  }
  auto cond_value = cond_result.value();
  // if (!select->true_false_operands.has_value()) {
  //   return cond_value;
  // }
  auto true_value_result = BuildIR(ctx, select->true_rpn);
  auto false_value_result = BuildIR(ctx, select->false_rpn);
  if (!true_value_result.ok()) {
    return true_value_result.status();
  }
  if (!false_value_result.ok()) {
    return false_value_result.status();
  }
  auto true_value = true_value_result.value();
  auto false_value = false_value_result.value();
  return BuildIR(ctx, OP_CONDITIONAL, cond_value, true_value, false_value);
}

}  // namespace llvm
}  // namespace rapidudf