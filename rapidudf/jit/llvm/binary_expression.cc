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

#include <vector>
#include "fmt/core.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, OpToken op, ValuePtr left,
                                              ValuePtr right) {
  if (op != OP_ASSIGN) {
    if (left->GetDType().IsVoid() || right->GetDType().IsVoid()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format(
          "Can NOT do op:{} with void operands, while left:{}, right:{}", op, left->GetDType(), right->GetDType())));
    }
  }
  ValuePtr result;
  switch (op) {
    case OP_PLUS:
    case OP_MINUS:
    case OP_MULTIPLY:
    case OP_DIVIDE:
    case OP_MOD:
    case OP_PLUS_ASSIGN:
    case OP_MINUS_ASSIGN:
    case OP_MULTIPLY_ASSIGN:
    case OP_DIVIDE_ASSIGN:
    case OP_MOD_ASSIGN:
    case OP_EQUAL:
    case OP_NOT_EQUAL:
    case OP_LESS:
    case OP_LESS_EQUAL:
    case OP_GREATER:
    case OP_GREATER_EQUAL:
    case OP_LOGIC_AND:
    case OP_LOGIC_OR:
    case OP_POW:
    case OP_MAX:
    case OP_MIN:
    case OP_ATAN2:
    case OP_HYPOT: {
      if (left->GetDType().IsSimdVector() || right->GetDType().IsSimdVector()) {
        auto func_name = GetFunctionName(op, left->GetDType(), right->GetDType());
        std::vector<ValuePtr> args{left, right};
        if (left->GetDType().IsSimdVector() && (op >= OP_PLUS_ASSIGN && op <= OP_MOD_ASSIGN)) {
          auto status = left->SetSimdVectorTemporary(true);
          if (!status.ok()) {
            return status;
          }
        }
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      } else if (left->GetDType().IsSimdColumnPtr() || right->GetDType().IsSimdColumnPtr()) {
        auto func_name = GetFunctionName(op, left->GetDType(), right->GetDType());
        std::vector<ValuePtr> args{left, right};
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
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      } else {
        if (HasIntrinsic(op)) {
          result = left->BinaryOp(op, right);
        } else {
          auto func_name = GetFunctionName(op, left->GetDType());
          std::vector<ValuePtr> args{left, right};
          auto call_result = CallFunction(func_name, args);
          if (!call_result.ok()) {
            return call_result.status();
          }
          result = call_result.value();
        }
        if (!result) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
      }
      if (op >= OP_PLUS_ASSIGN && op <= OP_MOD_ASSIGN) {
        if (!left->IsWritable()) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with non writable value:{}", op, left->GetDType())));
        }
        auto status = left->CopyFrom(result);
        if (!status.ok()) {
          return status;
        }
        if (left->GetDType().IsSimdVector()) {
          auto status = left->SetSimdVectorTemporary(false);
          if (!status.ok()) {
            return status;
          }
        }
      }
      break;
    }
    case OP_ASSIGN: {
      if (!left->IsWritable()) {
        RUDF_LOG_ERROR_STATUS(
            ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with non writable value:{}", op, left->GetDType())));
      }
      auto status = left->CopyFrom(right);
      if (!status.ok()) {
        return status;
      }
      if (left->GetDType().IsSimdVector()) {
        auto status = left->SetSimdVectorTemporary(false);
        if (!status.ok()) {
          return status;
        }
      }
      result = left;
      break;
    }
    default: {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
    }
  }
  return result;
}

}  // namespace llvm
}  // namespace rapidudf