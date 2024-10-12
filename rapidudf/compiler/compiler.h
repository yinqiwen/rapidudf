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
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "fmt/format.h"

#include "rapidudf/ast/block.h"
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/expression.h"
#include "rapidudf/ast/function.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/compiler/function.h"
#include "rapidudf/compiler/options.h"
#include "rapidudf/compiler/value.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace compiler {

class CodeGen;
class JitCompiler {
 public:
  static constexpr std::string_view kExpressionFuncName = "rapidudf_expresion";
  JitCompiler(const Options& opts = Options{});

  absl::StatusOr<std::vector<std::string>> CompileSource(const std::string& source, bool dump_asm = false);

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> LoadFunction(const std::string& name) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    if (!codegen_) {
      return absl::InvalidArgumentError("null compiled session to load function");
    }
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);

    auto verify_result = VerifyFunctionSignature(name, return_type, arg_types);
    if (!verify_result.ok()) {
      return verify_result.status();
    }
    auto func_ptr_result = GetFunctionPtr(name);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();
    return JitFunction<RET, Args...>(name, func_ptr, codegen_, stat_);
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileFunction(const std::string& source, bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewCodegen(dump_asm);
    auto status = CompileFunction(source);
    if (!status.ok()) {
      return status;
    }

    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);

    auto verify_result = VerifyFunctionSignature(return_type, arg_types);
    if (!verify_result.ok()) {
      return verify_result.status();
    }
    std::string fname = verify_result.value();
    auto func_ptr_result = GetFunctionPtr(fname);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();

    return JitFunction<RET, Args...>(fname, func_ptr, codegen_, stat_);
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileExpression(const std::string& source,
                                                              const std::vector<std::string>& arg_names,
                                                              bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewCodegen(dump_asm);
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    if (arg_names.size() != arg_types.size()) {
      return absl::InvalidArgumentError(
          fmt::format("need {} arg names, while only {} provided", arg_types.size(), arg_names.size()));
    }
    ast::Function gen_func_ast;
    gen_func_ast.return_type = return_type;
    if (!arg_names.empty()) {
      gen_func_ast.args = std::vector<ast::FunctionArg>{};
    }
    gen_func_ast.name = std::string(kExpressionFuncName);
    ast_ctx_.ReserveFunctionParseContext(1);
    for (size_t i = 0; i < arg_names.size(); i++) {
      if (!ast_ctx_.AddLocalVar(arg_names[i], arg_types[i])) {
        return absl::InvalidArgumentError(fmt::format("Duplicate arg name:{}", arg_names[i]));
      }
      ast::FunctionArg ast_arg;
      ast_arg.dtype = arg_types[i];
      ast_arg.name = arg_names[i];
      gen_func_ast.args->emplace_back(ast_arg);
    }

    auto status = CompileExpression(source, gen_func_ast);
    if (!status.ok()) {
      return status;
    }

    auto func_ptr_result = GetFunctionPtr(gen_func_ast.name);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();

    return JitFunction<RET, Args...>(gen_func_ast.name, func_ptr, codegen_, stat_);
  }

 private:
  struct RPNEvalNode {
    ValuePtr val;
    OpToken op = OP_INVALID;
    ValuePtr op_temp_val;
    explicit RPNEvalNode(OpToken v) : op(v) {}
    explicit RPNEvalNode(ValuePtr v) : val(v) {}
  };

  void NewCodegen(bool print_asm);
  absl::Status Compile();
  absl::Status CompileFunction(const std::string& source);
  absl::Status CompileFunction(const ast::Function& function);
  absl::Status CompileFunctions(const std::vector<ast::Function>& functions);
  absl::Status CompileExpression(const std::string& expr, ast::Function& function);

  absl::StatusOr<std::string> VerifyFunctionSignature(DType rtype, const std::vector<DType>& args_types);
  absl::StatusOr<std::string> VerifyFunctionSignature(const std::string& name, DType rtype,
                                                      const std::vector<DType>& args_types);

  absl::StatusOr<void*> GetFunctionPtr(const std::string& name);

  absl::Status ThrowVectorExprError(const std::string& msg);

  absl::Status BuildIR(const ast::Function& function);
  absl::Status BuildIR(const ast::Block& block);

  absl::Status BuildIR(const std::vector<ast::Statement>& statements);
  absl::Status BuildIR(const ast::ReturnStatement& statement);
  absl::Status BuildIR(const ast::IfElseStatement& statement);
  absl::Status BuildIR(const ast::WhileStatement& statement);
  absl::Status BuildIR(const ast::ExpressionStatement& statement);
  absl::Status BuildIR(const ast::ContinueStatement& statement);
  absl::Status BuildIR(const ast::BreakStatement& statement);

  absl::StatusOr<ValuePtr> BuildIR(const ast::RPN& rpn);
  absl::StatusOr<ValuePtr> BuildIR(DType dtype, const std::vector<RPNEvalNode>& nodes);
  absl::StatusOr<ValuePtr> BuildVectorIR(DType dtype, std::vector<RPNEvalNode>& nodes);
  absl::Status BuildVectorEvalIR(DType dtype, std::vector<RPNEvalNode>& nodes, ValuePtr curosr, ValuePtr remaining,
                                 ::llvm::Value* output);

  absl::StatusOr<ValuePtr> BuildIR(const ast::ConstantNumber& expr);
  absl::StatusOr<ValuePtr> BuildIR(double v, DType dtype);
  absl::StatusOr<ValuePtr> BuildIR(double v);
  absl::StatusOr<ValuePtr> BuildIR(bool v);
  absl::StatusOr<ValuePtr> BuildIR(uint32_t v);
  absl::StatusOr<ValuePtr> BuildIR(const std::string& v);

  absl::StatusOr<ValuePtr> BuildIR(const ast::VarRef& key);
  absl::StatusOr<ValuePtr> BuildIR(const ast::VarAccessor& expr);
  absl::StatusOr<ValuePtr> BuildIR(const ast::VarDefine& expr);
  absl::StatusOr<ValuePtr> BuildIR(const ast::Array& expr);
  absl::StatusOr<ValuePtr> BuildIR(ValuePtr obj, const ast::FieldAccess& field);

  Options opts_;

  ast::ParseContext ast_ctx_;
  std::vector<ast::Function> parsed_ast_funcs_;

  std::shared_ptr<CodeGen> codegen_;
  std::mutex jit_mutex_;
  JitFunctionStat stat_;

  friend class JitCompilerCache;
};

}  // namespace compiler
}  // namespace rapidudf