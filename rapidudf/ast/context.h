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

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "fmt/format.h"

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/dyn_object_schema.h"
namespace rapidudf {
namespace ast {

struct VarTag {
  std::string name;
  DType dtype;
  const DynObjectSchema* schema = nullptr;
  VarTag(DType d, const std::string& n = "", const DynObjectSchema* s = nullptr) {
    dtype = d;
    name = n;
    schema = s;
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

  absl::StatusOr<VarTag> IsVarExist(const std::string& name, bool error_on_exist);

  std::string GetErrorLine() const;

  int GetLineNo() const;
  std::string GetSourceLine(int line) const;

  bool AddLocalVar(const std::string& name, DType dtype, const DynObjectSchema* schema);

  absl::StatusOr<const FunctionDesc*> CheckFuncExist(const std::string& name, bool implicit = false);
  absl::StatusOr<const FunctionDesc*> CheckFuncExist(std::string_view name, bool implicit = false) {
    return CheckFuncExist(std::string(name), implicit);
  }
  void AddMemberFuncCall(DType dtype, const std::string& name, FunctionDesc desc);

  DType GetFuncReturnDType(uint32_t idx = 0) { return GetFunctionParseContext(idx).desc.return_type; }
  int GetFuncContextArgIdx() { return GetFunctionParseContext(current_function_cursor_).desc.context_arg_idx; }
  void SetFuncDesc(const FunctionDesc& d, uint32_t idx = 0);

  const FunctionCallMap& GetAllFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).func_calls;
  }
  const FunctionCallMap& GetAllImplicitFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).implicit_func_calls;
  }
  const MemberFuncCallMap& GetAllMemberFuncCalls(uint32_t funcion_idx) const {
    return GetFunctionParseContext(funcion_idx).member_func_calls;
  }

  std::vector<FunctionDesc> GetAllFunctionDescs() const;

  void SetFunctionCursor(uint32_t idx) { current_function_cursor_ = idx; }

  void ReserveFunctionParseContext(uint32_t n) { GetFunctionParseContext(n - 1); }

  bool CanCastTo(DType from_dtype, DType to_dtype);

  void EnterLoop();
  bool IsInLoop();
  void ExitLoop();
  std::chrono::microseconds GetParseCost() { return parse_cost_; }
  std::chrono::microseconds GetParseValidateCost() { return parse_validate_cost_; }
  void SetParseCost(std::chrono::microseconds cost) { parse_cost_ = cost; }
  void SetParseValidateCost(std::chrono::microseconds cost) { parse_validate_cost_ = cost; }

  bool IsVectorExpression() const { return vector_expr_flag_; }
  void SetVectorExressionFlag(bool v = true) { vector_expr_flag_ = v; }

 private:
  using LocalVarMap = std::unordered_map<std::string, VarTag>;

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
  bool vector_expr_flag_ = false;

  std::string ast_err_;
  uint32_t validate_posistion_ = 0;

  std::chrono::microseconds parse_cost_;
  std::chrono::microseconds parse_validate_cost_;
};

}  // namespace ast
}  // namespace rapidudf