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

#include <string_view>
#include <vector>
#include "absl/cleanup/cleanup.h"
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/ops/cmp.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/value.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/reflect.h"
namespace rapidudf {
using namespace Xbyak::util;
absl::StatusOr<ValuePtr> JitCompiler::CompileFieldAccess(ValuePtr var, const ast::FieldAccess& field) {
  if (!var->GetDType().IsSimdVector() && !var->GetDType().IsPtr()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, var->GetDType())));
  }
  auto accessor = ReflectFactory::GetStructMember(var->GetDType().PtrTo(), field.field);
  if (!accessor) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
        fmt::format("Can NOT get reflect accessor with dtype:{} & member:{}", var->GetDType().PtrTo(), field.field)));
  }
  if (field.func_args.has_value()) {
    if (!accessor->HasMemberFunc()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & member func:{}", var->GetDType(), field.field)));
    }
    std::vector<ValuePtr> arg_values;
    if (field.func_args->args.has_value()) {
      for (auto func_arg_expr : *(field.func_args->args)) {
        auto arg_val = CompileExpression(func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }

    auto result = accessor->BuildFuncCall(GetCodeGenerator(), *var, arg_values);
    if (!result.ok()) {
      return result.status();
    }
    auto result_value = result.value();
    RUDF_DEBUG("func:{} call result dtype:{} {}/{}/{}", field.field, result_value->GetDType(),
               result_value->IsRegister(), result_value->IsStack(), result_value->IsConst());
    for (auto p : arg_values) {
      GetCodeGenerator().DropTmpValue(p);
    }
    return result_value;
  } else {
    if (!accessor->HasField()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & field:{}", var->GetDType(), field.field)));
    }
    auto rcx_val = Value::New(&GetCodeGenerator(), var->GetDType(), &rcx, false);
    rcx_val->Copy(*var);
    DType field_read_dtype = *accessor->member_field_dtype;
    RUDF_DEBUG("Access dtype:{} field:{} with dtype:{}", var->GetDType(), field.field, field_read_dtype);
    auto result = accessor->BuildFieldAccess(GetCodeGenerator());
    if (!result.ok()) {
      return result.status();
    }
    auto result_value = result.value();
    RUDF_DEBUG("Access result dtype:{}", result_value->GetDType());
    auto ret_value = GetCodeGenerator().NewValue(result_value->GetDType());
    ret_value->Copy(*(result.value()));
    return ret_value;
  }
}
absl::StatusOr<ValuePtr> JitCompiler::CompileJsonAccess(ValuePtr var, uint32_t idx) {
  if (!var->GetDType().IsJsonPtr()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
  }

  auto key_arg = GetCodeGenerator().NewConstValue(DATA_U64, idx);
  std::vector<ValuePtr> args{var, key_arg};
  auto result = GetCodeGenerator().CallFunction(std::string(kBuiltinJsonArrayGet), args);
  if (!result) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Call func:{} failed", kBuiltinJsonMemberGet)));
  }
  auto json_result = GetCodeGenerator().NewValue(var->GetDType());
  int rc = json_result->Copy(*result);
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Failed to copy json array get result")));
  }
  return json_result;
}
absl::StatusOr<ValuePtr> JitCompiler::CompileJsonAccess(ValuePtr var, const std::string& key) {
  if (!var->GetDType().IsJsonPtr()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
  }
  std::unique_ptr<std::string> str = std::make_unique<std::string>(key);
  std::string_view key_view = *str;
  GetCompileContext().const_strings.emplace_back(std::move(str));

  auto key_arg = GetCodeGenerator().NewConstValue(DATA_STRING_VIEW);
  key_arg->Set(key_view);
  std::vector<ValuePtr> args{var, key_arg};
  auto result = GetCodeGenerator().CallFunction(std::string(kBuiltinJsonMemberGet), args);
  if (!result) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Call func:{} failed", kBuiltinJsonMemberGet)));
  }

  auto json_result = GetCodeGenerator().NewValue(var->GetDType());
  int rc = json_result->Copy(*result);
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Failed to copy json member get result")));
  }
  return json_result;
}
absl::StatusOr<ValuePtr> JitCompiler::CompileJsonAccess(ValuePtr var, const ast::VarRef& key) {
  if (!var->GetDType().IsJsonPtr()) {
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
  }
  auto ref_result = GetLocalVar(key.name);
  if (!ref_result.ok()) {
    return ref_result.status();
  }
  auto key_val = ref_result.value();
  std::vector<ValuePtr> args{var, key_val};

  if (key_val->GetDType().IsStringPtr()) {
    key_val = key_val->CastTo(DATA_STRING_VIEW);
  }
  ValuePtr result;
  if (key_val->GetDType().IsStringView()) {
    result = GetCodeGenerator().CallFunction(std::string(kBuiltinJsonMemberGet), args);
  } else if (key_val->GetDType().IsInteger()) {
    result = GetCodeGenerator().CallFunction(std::string(kBuiltinJsonArrayGet), args);
  } else {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
        fmt::format("Can NOT do json member/array access  with key dtype", key_val->GetDType())));
  }
  auto json_result = GetCodeGenerator().NewValue(var->GetDType());
  int rc = json_result->Copy(*result);
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Failed to copy json get result")));
  }
  return json_result;
}
}  // namespace rapidudf