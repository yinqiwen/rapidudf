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

#include <variant>
#include "fmt/core.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, OpToken op, ValuePtr val) {
  if (val->GetDType().IsVoid()) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with void operands", op)));
  }
  ValuePtr result;
  switch (op) {
    case OP_NOT: {
      if (val->GetDType().IsSimdVector() || val->GetDType().IsSimdColumnPtr()) {
        auto func_name = GetFunctionName(op, val->GetDType());
        std::vector<ValuePtr> args{val};
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      } else {
        result = val->UnaryOp(op);
      }
      break;
    }
    case OP_NEGATIVE: {
      if (val->GetDType().IsSimdVector() || val->GetDType().IsSimdColumnPtr()) {
        auto func_name = GetFunctionName(op, val->GetDType());
        std::vector<ValuePtr> args{val};
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      } else {
        result = val->UnaryOp(op);
      }
      break;
    }
    default: {
      if (val->GetDType().IsSimdVector() || val->GetDType().IsSimdColumnPtr()) {
        auto func_name = GetFunctionName(op, val->GetDType());
        std::vector<ValuePtr> args{val};
        auto call_result = CallFunction(func_name, args);
        if (!call_result.ok()) {
          return call_result.status();
        }
        result = call_result.value();
      } else {
        if (HasIntrinsic(op)) {
          result = val->UnaryOp(op);
        } else {
          auto func_name = GetFunctionName(op, val->GetDType());
          std::vector<ValuePtr> args{val};
          auto call_result = CallFunction(func_name, args);
          if (!call_result.ok()) {
            return call_result.status();
          }
          result = call_result.value();
        }
      }
      break;
    }
  }
  if (!result) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with  operands:{}", op, val->GetDType())));
  }
  return result;
}

}  // namespace llvm
}  // namespace rapidudf