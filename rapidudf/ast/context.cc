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
#include "rapidudf/ast/context.h"
#include <fmt/format.h>
#include <vector>
#include "rapidudf/builtin/builtin_symbols.h"

namespace rapidudf {
namespace ast {
void ParseContext::Clear() {
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

void ParseContext::EnterLoop() { GetFunctionParseContext(current_function_cursor_).in_loop++; }
bool ParseContext::IsInLoop() { return GetFunctionParseContext(current_function_cursor_).in_loop > 0; }
void ParseContext::ExitLoop() { GetFunctionParseContext(current_function_cursor_).in_loop--; }

void ParseContext::SetSource(const std::string& src, bool clear_vars) {
  source_ = src;
  source_lines_.clear();
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
  source_lines_ = absl::StrSplit(absl::string_view(source_), '\n');
}

absl::StatusOr<DType> ParseContext::IsVarExist(const std::string& name, bool error_on_exist) {
  auto found = GetFunctionParseContext(current_function_cursor_).local_vars.find(name);
  if (found != GetFunctionParseContext(current_function_cursor_).local_vars.end()) {
    if (error_on_exist) {
      return absl::AlreadyExistsError(fmt::format("var:{} already exist at {}", name, GetErrorLine()));
    }
    return found->second;
  }
  if (!error_on_exist) {
    return absl::NotFoundError(fmt::format("var:{} is not exist at {}", name, GetErrorLine()));
  }
  return DType(DATA_VOID);
}
std::string ParseContext::GetErrorLine() const {
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

bool ParseContext::AddLocalVar(const std::string& name, DType dtype) {
  auto [iter, success] = GetFunctionParseContext(current_function_cursor_).local_vars.emplace(name, dtype);
  if (!success && iter->second.IsVoid()) {
    iter->second = dtype;
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
        implicit_func_call = kBuiltinCastStdStrToStringView;
      } else if (from_dtype.IsFlatbuffersStringPtr()) {
        implicit_func_call = kBuiltinCastFbsStrToStringView;
      } else if (from_dtype.IsStdStringView()) {
        implicit_func_call = kBuiltinCastStdStrViewToStringView;
      }
    }
    if (!implicit_func_call.empty()) {
      auto _ = CheckFuncExist(implicit_func_call, true);
    }
  }

  return v;
}

absl::StatusOr<const FunctionDesc*> ParseContext::CheckFuncExist(const std::string& name, bool implicit) {
  const FunctionDesc* desc = nullptr;
  bool local_func = false;
  for (uint32_t i = 0; i <= current_function_cursor_; i++) {
    if (GetFunctionParseContext(i).desc.name == name) {
      desc = &(GetFunctionParseContext(i).desc);
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
    if (implicit) {
      GetFunctionParseContext(current_function_cursor_).implicit_func_calls.emplace(name, desc);
    } else {
      GetFunctionParseContext(current_function_cursor_).func_calls.emplace(name, desc);
    }
  }
  return desc;
}

void ParseContext::AddMemberFuncCall(DType dtype, const std::string& name, FunctionDesc desc) {
  GetFunctionParseContext(current_function_cursor_).member_func_calls[dtype][name] = desc;
}

}  // namespace ast
}  // namespace rapidudf