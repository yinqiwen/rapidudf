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

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/StandardInstrumentations.h"

#include "rapidudf/ast/block.h"
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/expression.h"
#include "rapidudf/ast/function.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/jit/function.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
namespace rapidudf {
namespace llvm {
class JitCompiler {
 public:
  JitCompiler();

  absl::StatusOr<std::vector<std::string>> CompileSource(const std::string& source, bool dump_asm = false);

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> LoadFunction(const std::string& name) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    if (!session_) {
      return absl::InvalidArgumentError("null compiled session");
    }
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    auto found = GetSession().compile_functon_ctxs.find(name);
    if (found == GetSession().compile_functon_ctxs.end()) {
      return absl::NotFoundError(fmt::format("No function:{} found in compiled functions.", name));
    }
    auto ctx = found->second;
    std::string err;
    if (!ctx->func_ast.CompareSignature(return_type, arg_types, err)) {
      RUDF_ERROR("{}", err);
      return absl::InvalidArgumentError(err);
    }
    auto func_ptr_result = GetFunctionPtr(name);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();
    return JitFunction<RET, Args...>(name, func_ptr, session_, false);
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileFunction(const std::string& source, bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewSession(dump_asm);
    auto status = CompileFunction(source);
    if (!status.ok()) {
      return status;
    }

    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    std::string err;
    if (!GetCompileContext().func_ast.CompareSignature(return_type, arg_types, err)) {
      RUDF_ERROR("{}", err);
      return absl::InvalidArgumentError(err);
    }
    std::string fname = GetCompileContext().func_ast.name;
    auto func_ptr_result = GetFunctionPtr(fname);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();
    auto resource = std::move(session_);
    return JitFunction<RET, Args...>(fname, func_ptr, resource, false);
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileExpression(const std::string& source,
                                                              const std::vector<std::string>& arg_names,
                                                              bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    NewSession(dump_asm);
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
    gen_func_ast.name = "rapidudf_expresion";
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
    auto resource = std::move(session_);
    return JitFunction<RET, Args...>(gen_func_ast.name, func_ptr, resource, false);
  }

 private:
  struct ExternFunction {
    FunctionDesc desc;
    ::llvm::FunctionType* func_type = nullptr;
    ::llvm::Function* func = nullptr;
  };
  using ExternFunctionPtr = std::shared_ptr<ExternFunction>;
  using LoopBlocks = std::pair<::llvm::BasicBlock*, ::llvm::BasicBlock*>;
  struct FunctionCompileContext {
    FunctionDesc desc;
    ast::Function func_ast;
    ::llvm::Function* func = nullptr;
    ::llvm::Type* return_type = nullptr;
    ::llvm::Value* return_value = nullptr;
    ::llvm::BasicBlock* exit_block = nullptr;
    std::unordered_map<std::string, ValuePtr> named_values;
    std::vector<LoopBlocks> loop_blocks;
  };
  using FunctionCompileContextPtr = std::shared_ptr<FunctionCompileContext>;

  struct JitSession {
    std::unique_ptr<::llvm::orc::LLJIT> jit;
    std::vector<std::unique_ptr<std::string>> const_strings;
    std::unordered_map<std::string, ExternFunctionPtr> extern_funcs;
    std::unordered_map<std::string, FunctionCompileContextPtr> compile_functon_ctxs;
    FunctionCompileContextPtr current_compile_functon_ctx;
    uint32_t compile_function_idx = 0;
    uint32_t label_cursor = 0;
    bool print_asm = false;
  };

  using JitTypeArray = std::vector<::llvm::Type*>;
  using JitTypeArrayPtr = std::unique_ptr<JitTypeArray>;
  using JitValueArray = std::vector<::llvm::Value*>;
  using JitValueArrayPtr = std::unique_ptr<JitValueArray>;

  // void Init();
  void NewSession(bool print_asm);
  absl::Status Compile();
  absl::StatusOr<void*> GetFunctionPtr(const std::string& name);

  ValuePtr NewValue(DType dtype, ::llvm::Value* val, ::llvm::Type* type = nullptr);

  absl::Status CompileFunction(const std::string& source);
  absl::Status CompileFunction(const ast::Function& function);
  absl::Status CompileFunctions(const std::vector<ast::Function>& functions);
  absl::Status CompileExpression(const std::string& expr, ast::Function& function);

  absl::StatusOr<::llvm::Type*> GetType(DType dtype);
  absl::StatusOr<::llvm::FunctionType*> GetFunctionType(const FunctionDesc& desc);

  absl::Status BuildIR(const ast::Function& function);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::Block& block);

  absl::Status BuildIR(FunctionCompileContextPtr ctx, const std::vector<ast::Statement>& statements);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::ReturnStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::IfElseStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::WhileStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::ExpressionStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::ContinueStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::BreakStatement& statement);

  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ast::UnaryExprPtr expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ast::BinaryExprPtr expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ast::TernaryExprPtr expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, const ast::VarAccessor& expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, const ast::VarDefine& expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, const ast::Array& expr);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, const ast::Operand& expr);

  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, double v, DType dtype);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, double v);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, bool v);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, const std::string& v);

  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const ast::FieldAccess& field);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, uint32_t idx);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const std::string& key);
  absl::StatusOr<ValuePtr> BuildIR(FunctionCompileContextPtr ctx, ValuePtr var, const ast::VarRef& key);

  absl::StatusOr<ValuePtr> GetLocalVar(const std::string& name);
  ExternFunctionPtr GetFunction(const std::string& name);
  std::string GetMemberFuncName(DType dtype, const std::string& member);
  absl::StatusOr<ValuePtr> CallFunction(const std::string& name, const std::vector<ValuePtr>& arg_values);
  absl::StatusOr<ValuePtr> CallFunction(std::string_view name, const std::vector<ValuePtr>& arg_values) {
    return CallFunction(std::string(name), arg_values);
  }

  ::llvm::IRBuilder<>* GetIRBuilder() { return ir_builder_.get(); }

  FunctionCompileContext& GetCompileContext() { return *GetSession().current_compile_functon_ctx; }
  JitSession& GetSession() { return *session_; }

  uint32_t GetLabelCursor() { return session_->label_cursor++; }

  ast::ParseContext ast_ctx_;

  std::unique_ptr<::llvm::LLVMContext> context_;
  std::unique_ptr<::llvm::Module> module_;
  std::unique_ptr<::llvm::IRBuilder<>> ir_builder_;

  std::unique_ptr<::llvm::FunctionPassManager> func_pass_manager_;
  std::unique_ptr<::llvm::LoopAnalysisManager> loop_analysis_manager_;
  std::unique_ptr<::llvm::FunctionAnalysisManager> func_analysis_manager_;
  std::unique_ptr<::llvm::CGSCCAnalysisManager> cgscc_analysis_manager_;
  std::unique_ptr<::llvm::ModuleAnalysisManager> module_analysis_manager_;
  std::unique_ptr<::llvm::PassInstrumentationCallbacks> pass_inst_callbacks_;
  std::unique_ptr<::llvm::StandardInstrumentations> std_insts_;

  std::mutex jit_mutex_;

  std::shared_ptr<JitSession> session_;

  friend class Value;
};
}  // namespace llvm
}  // namespace rapidudf