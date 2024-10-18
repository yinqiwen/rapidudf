/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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