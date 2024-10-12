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

#include "rapidudf/compiler/codegen.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::Array& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto element_dtype = expr.dtype.Elem();

  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < expr.elements.size(); i++) {
    auto result = BuildIR(expr.rpns[i]);
    if (!result.ok()) {
      return result.status();
    }
    elements.emplace_back(result.value());
  }
  return codegen_->NewArray(element_dtype, elements);
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::VarDefine& expr) {
  ast_ctx_.SetPosition(expr.position);
  return codegen_->NewVoid(expr.name);
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::VarRef& key) {
  ast_ctx_.SetPosition(key.position);
  return codegen_->GetLocalVar(key.name);
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(ValuePtr obj, const ast::FieldAccess& field) {
  auto accessor = field.struct_member;
  if (field.func_args.has_value()) {
    if (!obj->GetDType().IsSimdVector() && !obj->GetDType().IsPtr() && !obj->GetDType().IsStringView() &&
        !obj->GetDType().IsStdStringView()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, obj->GetDType())));
    }
    if (!accessor.HasMemberFunc()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & member func:{}", obj->GetDType(), field.field)));
    }
    std::vector<ValuePtr> arg_values;
    arg_values.emplace_back(obj);
    if (field.func_args->args.has_value()) {
      for (auto func_arg_expr : (field.func_args->rpns)) {
        auto arg_val = BuildIR(func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }
    std::string member_func_name = GetMemberFuncName(obj->GetDType().PtrTo(), field.field);
    return codegen_->CallFunction(member_func_name, arg_values);
  } else {
    if (!obj->GetDType().IsPtr()) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT access field:{} with dtype:{}", field.field, obj->GetDType())));
    }
    if (!accessor.HasField()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
          fmt::format("Can NOT get reflect accessor with dtype:{} & field:{}", obj->GetDType(), field.field)));
    }
    uint32_t field_offset = accessor.member_field_offset;
    auto field_dtype = *accessor.member_field_dtype;
    return codegen_->GetStructField(obj, field_dtype, field_offset);
  }
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::VarAccessor& expr) {
  ast_ctx_.SetPosition(expr.position);
  if (expr.func_args.has_value()) {
    // // name is func name
    std::string func_name = expr.name;
    std::vector<ValuePtr> arg_values;
    // bool is_simd_func = false;
    if (expr.func_args->args.has_value()) {
      for (auto func_arg_expr : (expr.func_args->rpns)) {
        auto arg_val = BuildIR(func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }
    return codegen_->CallFunction(func_name, arg_values);
  } else if (expr.access_args.has_value()) {
    // name is var name
    auto var_result = codegen_->GetLocalVar(expr.name);
    if (!var_result.ok()) {
      return var_result.status();
    }
    ValuePtr var = var_result.value();
    size_t access_idx = 0;
    for (auto access_arg : *expr.access_args) {
      auto next_result = std::visit(
          [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ast::FieldAccess>) {
              return BuildIR(var, arg);
            } else if constexpr (std::is_same_v<T, ast::DynamicParamAccess>) {
              auto param_result = std::visit([&](auto&& json_arg) { return BuildIR(json_arg); }, arg);
              if (!param_result.ok()) {
                return absl::StatusOr<ValuePtr>(param_result.status());
              }
              if (expr.access_func_names.size() <= access_idx) {
                return absl::StatusOr<ValuePtr>(ast_ctx_.GetErrorStatus("Empty access func."));
              }
              std::vector<ValuePtr> arg_values{var, param_result.value()};
              return codegen_->CallFunction(expr.access_func_names[access_idx], arg_values);
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
    return codegen_->GetLocalVar(expr.name);
  }
}
}  // namespace compiler
}  // namespace rapidudf