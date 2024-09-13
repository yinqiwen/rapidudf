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

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
namespace rapidudf {
namespace ast {

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
  using FunctionCallMap = std::unordered_map<std::string, const FunctionDesc*>;
  using MemberFuncCallMap = std::unordered_map<DType, std::unordered_map<std::string, FunctionDesc>>;
  void Clear();

  const std::string& GetAstErr() const { return ast_err_; }
  void SetAstErr(const std::string& err) { ast_err_ = err; }
  void SetPosition(uint32_t pos) { validate_posistion_ = pos; }
  void SetSource(const std::string& src, bool clear_vars = true);

  absl::Status GetErrorStatus(const std::string& err) {
    return absl::InvalidArgumentError(fmt::format("{} at {}", err, GetErrorLine()));
  }

  absl::StatusOr<DType> IsVarExist(const std::string& name, bool error_on_exist);

  std::string GetErrorLine() const;

  bool AddLocalVar(const std::string& name, DType dtype);

  absl::StatusOr<const FunctionDesc*> CheckFuncExist(const std::string& name, bool implicit = false);
  absl::StatusOr<const FunctionDesc*> CheckFuncExist(std::string_view name, bool implicit = false) {
    return CheckFuncExist(std::string(name), implicit);
  }
  void AddMemberFuncCall(DType dtype, const std::string& name, FunctionDesc desc);

  DType GetFuncReturnDType(uint32_t idx = 0) { return GetFunctionParseContext(idx).desc.return_type; }
  void SetFuncDesc(const FunctionDesc& d, uint32_t idx = 0) { GetFunctionParseContext(idx).desc = d; }

  const FunctionCallMap& GetAllFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).func_calls;
  }
  const FunctionCallMap& GetAllImplicitFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).implicit_func_calls;
  }
  const MemberFuncCallMap& GetAllMemberFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).member_func_calls;
  }

  void SetFunctionCursor(uint32_t idx) { current_function_cursor_ = idx; }

  void ReserveFunctionParseContext(uint32_t n) { GetFunctionParseContext(n - 1); }

  bool CanCastTo(DType from_dtype, DType to_dtype);

  void EnterLoop();
  bool IsInLoop();
  void ExitLoop();

 private:
  using LocalVarMap = std::unordered_map<std::string, DType>;

  using BuiltinFuncationCallSet = std::unordered_set<std::string>;

  struct FunctionParseContext {
    LocalVarMap local_vars;
    FunctionCallMap func_calls;
    FunctionCallMap implicit_func_calls;
    MemberFuncCallMap member_func_calls;
    FunctionDesc desc;
    uint32_t in_loop = 0;
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

  std::string ast_err_;
  uint32_t validate_posistion_ = 0;
};

}  // namespace ast
}  // namespace rapidudf