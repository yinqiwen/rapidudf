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
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::GetLocalVar(const std::string& name) {
  auto found = current_compile_functon_ctx_->named_values.find(name);
  if (found != current_compile_functon_ctx_->named_values.end()) {
    return found->second;
  }
  return absl::NotFoundError(fmt::format("No var '{}' found", name));
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var,
                                              const ast::FieldAccess& field) {
  auto accessor = field.struct_member;
  if (field.func_args.has_value()) {
    if (!var->GetDType().IsSimdVector() && !var->GetDType().IsPtr() && !var->GetDType().IsStringView()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, var->GetDType())));
    }
    if (!accessor.HasMemberFunc()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & member func:{}", var->GetDType(), field.field)));
    }
    std::vector<ValuePtr> arg_values;
    if (field.func_args->args.has_value()) {
      for (auto func_arg_expr : *(field.func_args->args)) {
        auto arg_val = BuildIR(ctx, func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }

    // auto result = BuildStructFuncCall(GetCodeGenerator(), accessor, *var, arg_values);
    // if (!result.ok()) {
    //   return result.status();
    // }
    // auto result_value = result.value();
    // RUDF_DEBUG("func:{} call result dtype:{} {}/{}/{}", field.field, result_value->GetDType(),
    //            result_value->IsRegister(), result_value->IsStack(), result_value->IsConst());

    // return result_value;
    return absl::UnimplementedError("VarAccessor json access func");
  } else {
    if (!var->GetDType().IsPtr()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, var->GetDType())));
    }
    if (!accessor.HasField()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & field:{}", var->GetDType(), field.field)));
    }
    ::llvm::Value* offset =
        ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ir_builder_->getContext()), accessor.member_field_offset);
    auto field_ptr = ir_builder_->CreateInBoundsGEP(::llvm::Type::getInt8Ty(ir_builder_->getContext()), var->GetValue(),
                                                    std::vector<::llvm::Value*>{offset});

    if (accessor.member_field_dtype->IsNumber()) {
      auto dst_type_result = GetType(*accessor.member_field_dtype);
      if (!dst_type_result.ok()) {
        return dst_type_result.status();
      }

      auto field_val = ir_builder_->CreateAlignedLoad(dst_type_result.value(), field_ptr,
                                                      ::llvm::MaybeAlign(accessor.member_field_dtype->ByteSize()));
      return NewValue(*accessor.member_field_dtype, field_val);
    } else if (accessor.member_field_dtype->IsStringView() || accessor.member_field_dtype->IsStdStringView() ||
               accessor.member_field_dtype->IsSimdVector()) {
    } else {
      return NewValue(accessor.member_field_dtype->ToPtr(), field_ptr);
    }
    return absl::UnimplementedError("Field access func");
  }
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, uint32_t idx) {
  return absl::UnimplementedError("VarAccessor json access func");
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const std::string& key) {
  return absl::UnimplementedError("VarAccessor json access func");
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const ast::VarRef& key) {
  auto result = GetLocalVar(key.name);
  if (!result.ok()) {
    return result.status();
  }
  auto val = result.value();
  if (val->GetDType().IsInteger()) {
  } else if (val->GetDType().IsStringView()) {
  }
  return absl::UnimplementedError("VarAccessor json access func");
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::VarDefine& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto var = NewValue(DATA_VOID, nullptr);
  GetCompileContext().named_values.emplace(expr.name, var);
  return var;
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::VarAccessor& expr) {
  ast_ctx_.SetPosition(expr.position);
  ValuePtr var;
  if (expr.func_args.has_value()) {
    // name is func name
    std::vector<ValuePtr> arg_values;
    if (expr.func_args->args.has_value()) {
      for (auto func_arg_expr : *(expr.func_args->args)) {
        auto arg_val = BuildIR(ctx, func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }
    return CallFunction(expr.name, arg_values);
  } else if (expr.access_args.has_value()) {
    // name is var name
    auto var_result = GetLocalVar(expr.name);
    if (!var_result.ok()) {
      return var_result.status();
    }

    var = var_result.value();
    for (auto access_arg : *expr.access_args) {
      auto next_result = std::visit(
          [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ast::FieldAccess>) {
              return BuildIR(ctx, var, arg);
            } else if constexpr (std::is_same_v<T, ast::DynamicParamAccess>) {
              return std::visit([&](auto&& json_arg) { return BuildIR(ctx, var, json_arg); }, arg);
            } else {
              static_assert(sizeof(arg) == -1, "non-exhaustive visitor!");
              return absl::StatusOr<ValuePtr>(absl::OkStatus());
            }
          },
          access_arg);
      if (!next_result.ok()) {
        return next_result.status();
      }
      var = next_result.value();
    }
    return var;
  } else {
    // name is var name
    return GetLocalVar(expr.name);
  }
}

}  // namespace llvm
}  // namespace rapidudf