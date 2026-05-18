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
#include "rapidudf/ast/context.h"
#include <array>
#include <vector>
#include "absl/strings/str_split.h"
#include "fmt/format.h"

#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/constants.h"

namespace rapidudf {
namespace ast {

void ParseContext::Clear() {
  ast_pool_.Reset();
  source_.clear();
  source_lines_.clear();
  function_parse_ctxs_.clear();
  ast_err_.clear();
  validate_posistion_ = 0;
}

void ParseContext::SetFuncDesc(const FunctionDesc& d, uint32_t idx) {
  FunctionDesc desc = d;
  desc.Init();
  GetFunctionParseContext(idx).desc = desc;
}

std::vector<FunctionDesc> ParseContext::GetAllFunctionDescs() const {
  std::vector<FunctionDesc> fs;
  for (auto& ctx : function_parse_ctxs_) {
    fs.emplace_back(ctx.desc);
  }
  return fs;
}

void ParseContext::EnterLoop() { GetFunctionParseContext(current_function_cursor_).in_loop++; }
bool ParseContext::IsInLoop() { return GetFunctionParseContext(current_function_cursor_).in_loop > 0; }
void ParseContext::ExitLoop() { GetFunctionParseContext(current_function_cursor_).in_loop--; }

void ParseContext::SetSource(const std::string& src, bool clear_vars) {
  ast_pool_.Reset();
  source_ = src;
  source_lines_.clear();
  source_lines_computed_ = false;
  if (clear_vars) {
    function_parse_ctxs_.clear();
  } else {
    for (auto& ctx : function_parse_ctxs_) {
      // ctx.builtin_func_calls.clear();
      ctx.func_calls.clear();
      ctx.implicit_func_calls.clear();
    }
  }
  ast_err_.clear();
  validate_posistion_ = 0;
}

absl::StatusOr<VarTag> ParseContext::IsVarExist(std::string_view name, bool error_on_exist) {
  auto& local_vars = GetFunctionParseContext(current_function_cursor_).local_vars;
  auto found = local_vars.find(name);
  if (found != local_vars.end()) {
    if (error_on_exist) {
      return absl::AlreadyExistsError(fmt::format("var:{} already exist at {}", name, GetErrorLine()));
    }
    return found->second;
  }
  for (size_t i = 0; i < kConstantCount; i++) {
    if (kConstantNames[i] == name) {
      return DType(DATA_F64);
    }
  }
  if (!error_on_exist) {
    return absl::NotFoundError(fmt::format("var:{} is not exist at {}", name, GetErrorLine()));
  }
  return DType(DATA_VOID);
}
void ParseContext::EnsureSourceLines() const {
  if (!source_lines_computed_) {
    const_cast<ParseContext*>(this)->source_lines_ = absl::StrSplit(absl::string_view(source_), '\n');
    const_cast<ParseContext*>(this)->source_lines_computed_ = true;
  }
}

std::string ParseContext::GetSourceLine(int line) const {
  EnsureSourceLines();
  int idx = line - 1;
  if (idx >= 0 && idx < source_lines_.size()) {
    return std::string(source_lines_[idx]);
  }
  return "";
}
int ParseContext::GetLineNo() const {
  EnsureSourceLines();
  uint32_t cursor = 0;
  uint32_t lineno = 1;
  for (auto line : source_lines_) {
    if (validate_posistion_ <= (cursor + line.size())) {
      break;
    }
    cursor += line.size();
    lineno++;
  }
  return lineno;
}
std::string ParseContext::GetErrorLine() const {
  EnsureSourceLines();
  uint32_t cursor = 0;
  uint32_t lineno = 1;
  for (auto line : source_lines_) {
    if (validate_posistion_ <= (cursor + line.size())) {
      return fmt::format("line:{}, error line: {}", lineno, line);
    }
    cursor += line.size();
    lineno++;
  }
  return fmt::format("cursor:{}, source lines:'{}'", validate_posistion_, source_lines_.size());
}

bool ParseContext::AddLocalVar(std::string_view name, DType dtype, const DynObjectSchema* schema) {
  auto [iter, success] =
      GetFunctionParseContext(current_function_cursor_).local_vars.emplace(name, VarTag{dtype, name, schema});
  if (!success && iter->second.dtype.IsVoid()) {
    iter->second.dtype = dtype;
    iter->second.schema = schema;
    return true;
  }
  return success;
}

bool ParseContext::CanCastTo(DType from_dtype, DType to_dtype) {
  bool v = from_dtype.CanCastTo(to_dtype);
  if (v) {
    std::string implicit_func_call;
    if (to_dtype.IsStringView()) {
      if (from_dtype.IsStringPtr()) {
        implicit_func_call = functions::kBuiltinCastStdStrToStringView;
      } else if (from_dtype.IsFlatbuffersStringPtr()) {
        implicit_func_call = functions::kBuiltinCastFbsStrToStringView;
      } else if (from_dtype.IsStdStringView()) {
        implicit_func_call = functions::kBuiltinCastStdStrViewToStringView;
      }
    }
    if (!implicit_func_call.empty()) {
      auto _ = CheckFuncExist(implicit_func_call, true);
    }
  }

  return v;
}

absl::StatusOr<const FunctionDesc*> ParseContext::CheckFuncExist(std::string_view name, bool implicit) {
  const FunctionDesc* desc = nullptr;
  bool local_func = false;
  for (uint32_t i = 0; i <= current_function_cursor_; i++) {
    auto& func_ctx = GetFunctionParseContext(i);
    if (func_ctx.desc.name == name) {
      desc = &func_ctx.desc;
      local_func = true;
      break;
    }
  }
  if (nullptr == desc) {
    desc = FunctionFactory::GetFunction(name);
  }
  if (desc == nullptr) {
    return absl::NotFoundError(fmt::format("func:{} not exist at `{}`", name, GetErrorLine()));
  }
  if (desc->context_arg_idx >= 0 && GetFuncContextArgIdx() < 0) {
    return absl::InvalidArgumentError(fmt::format(
        "Function:{} need `rapidudf::Context` arg, missing in expression/udf args, at `{}`", name, GetErrorLine()));
  }
  if (!local_func) {
    std::string name_str(name);
    if (implicit) {
      GetFunctionParseContext(current_function_cursor_).implicit_func_calls.emplace(std::move(name_str), desc);
    } else {
      GetFunctionParseContext(current_function_cursor_).func_calls.emplace(std::move(name_str), desc);
    }
  }
  return desc;
}

void ParseContext::AddMemberFuncCall(DType dtype, const std::string& name, FunctionDesc desc) {
  GetFunctionParseContext(current_function_cursor_).member_func_calls[dtype][name] = desc;
}

}  // namespace ast
}  // namespace rapidudf