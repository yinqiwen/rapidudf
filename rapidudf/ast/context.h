/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once
#include <fmt/core.h>
#include <fmt/format.h>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
namespace rapidudf {
namespace ast {

enum ParseResult { kParseOK = 0, kParseFuncNotExist, kParseResultEnd };
constexpr std::array<std::string_view, kParseResultEnd> kParseResultStrs = {"OK", "Function Not Exist"};

struct VarTag {
  std::string name;
  DType dtype;
  VarTag(DType d, const std::string& n = "") {
    dtype = d;
    name = n;
  }
};

class ParseContext {
 public:
  void Clear() {
    source_.clear();
    source_lines_.clear();
    function_parse_ctxs_.clear();
    ast_err_.clear();
    validate_posistion_ = 0;
  }

  const std::string& GetAstErr() const { return ast_err_; }
  void SetAstErr(const std::string& err) { ast_err_ = err; }
  void SetPosition(uint32_t pos) { validate_posistion_ = pos; }
  void SetSource(const std::string& src, bool clear_vars = true) {
    source_ = src;
    source_lines_.clear();
    if (clear_vars) {
      function_parse_ctxs_.clear();
    } else {
      for (auto& ctx : function_parse_ctxs_) {
        ctx.builtin_func_calls.clear();
        ctx.func_calls.clear();
      }
    }
    ast_err_.clear();
    validate_posistion_ = 0;
    source_lines_ = absl::StrSplit(absl::string_view(source_), '\n');
  }

  absl::Status GetErrorStatus(const std::string& err) {
    return absl::InvalidArgumentError(fmt::format("{} at {}", err, GetErrorLine()));
  }

  absl::StatusOr<DType> IsVarExist(const std::string& name, bool error_on_exist) {
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

  std::string GetErrorLine() const {
    uint32_t cursor = 0;
    uint32_t lineno = 1;
    for (auto line : source_lines_) {
      if (validate_posistion_ <= (cursor + line.size())) {
        // return {lineno, fmt::format("{}", line)};
        return fmt::format("line:{}, error line: {}", lineno, line);
      }
      cursor += line.size();
      lineno++;
    }
    return fmt::format("cursor:{}, source lines:'{}'", validate_posistion_, source_lines_.size());
  }

  bool AddLocalVar(const std::string& name, DType dtype) {
    auto [iter, success] = GetFunctionParseContext(current_function_cursor_).local_vars.emplace(name, dtype);
    if (!success && iter->second.IsVoid()) {
      iter->second = dtype;
      return true;
    }
    return success;
  }

  absl::StatusOr<const FuncDesc*> CheckFuncExist(const std::string& name) {
    const FuncDesc* desc = nullptr;
    for (uint32_t i = 0; i < current_function_cursor_; i++) {
      if (GetFunctionParseContext(i).desc.name == name) {
        desc = &(GetFunctionParseContext(i).desc);
        break;
      }
    }
    if (nullptr == desc) {
      desc = FuncFactory::GetFunc(name);
    }
    if (desc == nullptr) {
      return absl::NotFoundError(fmt::format("func:{} not exist at {}'", name, GetErrorLine()));
    }
    GetFunctionParseContext(current_function_cursor_).func_calls.emplace(name, desc);
    return desc;
  }

  void AddBuiltinFuncCall(std::string_view name) {
    GetFunctionParseContext(current_function_cursor_).builtin_func_calls.insert(std::string(name));
  }
  const std::unordered_set<std::string>& GetAllBuiltinFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).builtin_func_calls;
  }

  DType GetFuncReturnDType(uint32_t idx = 0) { return GetFunctionParseContext(idx).desc.return_type; }
  void SetFuncDesc(const FuncDesc& d, uint32_t idx = 0) { GetFunctionParseContext(idx).desc = d; }

  const std::unordered_map<std::string, const FuncDesc*>& GetAllFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).func_calls;
  }

  void SetFunctionCursor(uint32_t idx) { current_function_cursor_ = idx; }

 private:
  using LocalVarMap = std::unordered_map<std::string, DType>;
  using FunctionCallMap = std::unordered_map<std::string, const FuncDesc*>;
  using BuiltinFuncationCallSet = std::unordered_set<std::string>;

  struct FunctionParseContext {
    LocalVarMap local_vars;
    FunctionCallMap func_calls;
    BuiltinFuncationCallSet builtin_func_calls;
    FuncDesc desc;
  };
  const FunctionParseContext& GetFunctionParseContext(uint32_t idx) const { return function_parse_ctxs_[idx]; }
  FunctionParseContext& GetFunctionParseContext(uint32_t idx) {
    if (idx >= function_parse_ctxs_.size()) {
      function_parse_ctxs_.resize(idx + 1);
    }
    return function_parse_ctxs_[idx];
  }

  std::string source_;
  std::vector<std::string_view> source_lines_;
  std::vector<FunctionParseContext> function_parse_ctxs_;
  uint32_t current_function_cursor_ = 0;

  // std::unordered_map<std::string, DType> local_vars_;
  // std::unordered_map<std::string, const FuncDesc*> func_calls_;
  // std::unordered_set<std::string> builtin_func_calls_;
  std::string ast_err_;
  uint32_t validate_posistion_ = 0;
  // std::vector<DType> func_ret_dtypes_;
};

}  // namespace ast
}  // namespace rapidudf