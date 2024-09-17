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
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/jit_session.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/constants.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::GetLocalVar(const std::string& name) {
  auto found = GetCompileContext()->named_values.find(name);
  if (found != GetCompileContext()->named_values.end()) {
    return found->second;
  }
  for (size_t i = 0; i < kConstantCount; i++) {
    if (name == kConstantNames[i]) {
      ::llvm::APFloat fv(kConstantValues[i]);
      auto val = ::llvm::ConstantFP::get(GetSession()->GetIRBuilder()->getContext(), fv);
      return NewValue(DATA_F64, val);
    }
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
    arg_values.emplace_back(var);

    if (field.func_args->args.has_value()) {
      for (auto func_arg_expr : *(field.func_args->args)) {
        auto arg_val = BuildIR(ctx, func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }
    std::string member_func_name = GetMemberFuncName(var->GetDType().PtrTo(), field.field);
    return CallFunction(member_func_name, arg_values);
  } else {
    if (!var->GetDType().IsPtr()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, var->GetDType())));
    }
    if (!accessor.HasField()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & field:{}", var->GetDType(), field.field)));
    }
    ::llvm::Value* offset = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(GetSession()->GetIRBuilder()->getContext()), accessor.member_field_offset);
    auto field_ptr = GetSession()->GetIRBuilder()->CreateInBoundsGEP(
        ::llvm::Type::getInt8Ty(GetSession()->GetIRBuilder()->getContext()), var->GetValue(),
        std::vector<::llvm::Value*>{offset});

    auto member_field_dtype = *accessor.member_field_dtype;
    if (member_field_dtype.IsNumber() || member_field_dtype.IsStringView() || member_field_dtype.IsStdStringView() ||
        member_field_dtype.IsSimdVector() || member_field_dtype.IsAbslSpan() || member_field_dtype.IsPtr()) {
      auto dst_type_result = GetType(member_field_dtype);
      if (!dst_type_result.ok()) {
        return dst_type_result.status();
      }
      auto field_val = GetSession()->GetIRBuilder()->CreateAlignedLoad(
          dst_type_result.value(), field_ptr, ::llvm::MaybeAlign(member_field_dtype.ByteSize()));
      return NewValue(member_field_dtype, field_val);
    } else {
      return NewValue(accessor.member_field_dtype->ToPtr(), field_ptr);
    }
    return absl::UnimplementedError("Field access func");
  }
}

// absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, uint32_t idx) {
//   if (var->GetDType().IsJsonPtr()) {
//     auto key_arg = NewValue(DATA_U64, GetSession()->GetIRBuilder()->getInt64(idx));
//     std::vector<ValuePtr> args{var, key_arg};
//     return CallFunction(std::string(kBuiltinJsonArrayGet), args);
//   } else if (var->GetDType().IsVectorPtr()) {
//     RUDF_LOG_ERROR_STATUS(
//         ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
//   } else {
//     RUDF_LOG_ERROR_STATUS(
//         ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
//   }
// }
// absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const std::string& key) {
//   if (!var->GetDType().IsJsonPtr()) {
//     RUDF_LOG_ERROR_STATUS(
//         ast_ctx_.GetErrorStatus(fmt::format("Can NOT do member access on dtype:{}", var->GetDType())));
//   }
//   auto member_result = BuildIR(ctx, key);
//   if (!member_result.ok()) {
//     return member_result.status();
//   }
//   std::vector<ValuePtr> args{var, member_result.value()};
//   return CallFunction(std::string(kBuiltinJsonMemberGet), args);
// }
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::VarRef& key) {
  auto result = GetLocalVar(key.name);
  if (!result.ok()) {
    return result.status();
  }
  return result.value();
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::VarDefine& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto var = NewValue(DATA_VOID, nullptr);
  GetCompileContext()->named_values.emplace(expr.name, var);
  return var;
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::VarAccessor& expr) {
  ast_ctx_.SetPosition(expr.position);

  if (expr.func_args.has_value()) {
    // name is func name
    OpToken builtin_op = get_buitin_func_op(expr.name);

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
    if (builtin_op != OP_INVALID) {
      if (arg_values.size() == 1) {
        auto result = arg_values[0]->UnaryOp(builtin_op);
        if (!result) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do builtin op:{} with dtype:{}", builtin_op, arg_values[0]->GetDType())));
        }
        return result;
      } else if (arg_values.size() == 2) {
        auto result = arg_values[0]->BinaryOp(builtin_op, arg_values[1]);
        if (!result) {
          RUDF_LOG_ERROR_STATUS(
              ast_ctx_.GetErrorStatus(fmt::format("Can NOT do builtin op:{} with left dtype:{}, right dtype:{}",
                                                  builtin_op, arg_values[0]->GetDType(), arg_values[1]->GetDType())));
        }
        return result;
      } else if (arg_values.size() == 3) {
        auto result = arg_values[0]->TernaryOp(builtin_op, arg_values[1], arg_values[2]);
        if (!result) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do builtin op:{} with 1st dtype:{}, 2nd dtype:{},3rd dtype:{}", builtin_op,
                          arg_values[0]->GetDType(), arg_values[1]->GetDType(), arg_values[2]->GetDType())));
        }
        return result;
      }
    }
    return CallFunction(expr.name, arg_values);
  } else if (expr.access_args.has_value()) {
    // name is var name
    ValuePtr var;
    auto var_result = GetLocalVar(expr.name);
    if (!var_result.ok()) {
      return var_result.status();
    }

    var = var_result.value();
    size_t access_idx = 0;
    for (auto access_arg : *expr.access_args) {
      auto next_result = std::visit(
          [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ast::FieldAccess>) {
              return BuildIR(ctx, var, arg);
            } else if constexpr (std::is_same_v<T, ast::DynamicParamAccess>) {
              auto param_result = std::visit([&](auto&& json_arg) { return BuildIR(ctx, json_arg); }, arg);
              if (!param_result.ok()) {
                return absl::StatusOr<ValuePtr>(param_result.status());
              }
              if (expr.access_func_names.size() <= access_idx) {
                return absl::StatusOr<ValuePtr>(ast_ctx_.GetErrorStatus("Empty access func."));
              }
              std::vector<ValuePtr> arg_values{var, param_result.value()};
              return CallFunction(expr.access_func_names[access_idx], arg_values);
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
      access_idx++;
    }
    return var;
  } else {
    // name is var name
    return GetLocalVar(expr.name);
  }
}

}  // namespace llvm
}  // namespace rapidudf