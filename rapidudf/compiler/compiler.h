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
#include <memory>
#include <mutex>
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
#include "rapidudf/types/dyn_object_schema.h"

namespace rapidudf {

namespace compiler {

class CodeGen;
class JitCompiler {
 public:
  struct Arg {
    std::string name;
    std::string schema;
  };

  static constexpr std::string_view kExpressionFuncName = "rapidudf_expresion";
  JitCompiler(Options opts = Options{});

  absl::StatusOr<std::vector<std::string>> CompileSource(const std::string& source);

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
  absl::StatusOr<JitFunction<RET, Args...>> CompileFunction(const std::string& source) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewCodegen();
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
  absl::StatusOr<JitFunction<RET, Args...>> CompileDynObjExpression(const std::string& source,
                                                                    const std::vector<Arg>& args) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewCodegen();
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    if (args.size() != arg_types.size()) {
      return absl::InvalidArgumentError(
          fmt::format("need {} arg names, while only {} provided", arg_types.size(), args.size()));
    }
    ast::Function gen_func_ast;
    gen_func_ast.return_type = return_type;
    if (!args.empty()) {
      gen_func_ast.args = std::vector<ast::FunctionArg>{};
    }
    gen_func_ast.name = std::string(kExpressionFuncName);
    ast_ctx_.ReserveFunctionParseContext(1);
    for (size_t i = 0; i < args.size(); i++) {
      ast::FunctionArg ast_arg;
      ast_arg.dtype = arg_types[i];
      ast_arg.name = args[i].name;

      if (arg_types[i].IsDynObjectPtr()) {
        if (args[i].schema.empty()) {
          return absl::InvalidArgumentError(fmt::format("Missing schema for arg:{}", args[i].name));
        }
        const DynObjectSchema* schema = DynObjectSchema::Get(args[i].schema);
        if (schema == nullptr) {
          return absl::InvalidArgumentError(fmt::format("Invalid schema:{} for arg:{}", args[i].schema, args[i].name));
        }
        ast_arg.schema = schema;
      }
      if (!ast_ctx_.AddLocalVar(args[i].name, arg_types[i], ast_arg.schema)) {
        return absl::InvalidArgumentError(fmt::format("Duplicate arg name:{}", args[i].name));
      }
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
  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileExpression(const std::string& source,
                                                              const std::vector<std::string>& arg_names) {
    std::vector<Arg> args;
    for (auto& name : arg_names) {
      args.emplace_back(Arg{.name = name});
    }
    return CompileDynObjExpression<RET, Args...>(source, args);
  }

  absl::StatusOr<void*> GetFunctionPtr(const std::string& name);
  std::shared_ptr<CodeGen> GetCodeGen() { return codegen_; }
  const JitFunctionStat& GetStat() const { return stat_; }
  const std::vector<ast::Function>& GetParsedAST() const { return parsed_ast_funcs_; }

 private:
  struct RPNEvalNode {
    ValuePtr val;
    OpToken op = OP_INVALID;
    DType op_compute_dtype;
    ast::FuncInvocation func_invocation;
    ::llvm::Value* op_temp_val = nullptr;
    ::llvm::Value* constant_vector_val = nullptr;
    ::llvm::Value* constant_vector_val_ptr = nullptr;
    explicit RPNEvalNode(OpToken v) : op(v) {}
    explicit RPNEvalNode(ValuePtr v) : val(v) {}
    explicit RPNEvalNode(const ast::FuncInvocation& f) : func_invocation(f) {}
  };

  void NewCodegen();
  absl::Status Compile();
  absl::Status CompileFunction(const std::string& source);
  absl::Status CompileFunction(const ast::Function& function);
  absl::Status CompileFunctions(const std::vector<ast::Function>& functions);
  absl::Status CompileExpression(const std::string& expr, ast::Function& function);

  absl::StatusOr<std::string> VerifyFunctionSignature(DType rtype, const std::vector<DType>& args_types);
  absl::StatusOr<std::string> VerifyFunctionSignature(const std::string& name, DType rtype,
                                                      const std::vector<DType>& args_types);

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
};

}  // namespace compiler
}  // namespace rapidudf