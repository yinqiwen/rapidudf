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

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

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

  void Dump();

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileFunction(const std::string& source, bool dump_asm = false) {
    auto result = CompileFunction(source);
    if (!result.ok()) {
      return result.status();
    }
    auto compiled_ctx = result.value();

    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    std::string err;
    if (!compiled_ctx->func_ast.CompareSignature(return_type, arg_types, err)) {
      RUDF_ERROR("{}", err);
      return absl::InvalidArgumentError(err);
    }
    if (dump_asm) {
      Dump();
    }
    auto status = Compile();
    if (!status.ok()) {
      return status;
    }

    auto func_addr_result = jit_->lookup(compiled_ctx->func_ast.name);
    if (!func_addr_result) {
      auto err = func_addr_result.takeError();
      std::string err_str;
      ::llvm::raw_string_ostream err_stream(err_str);
      err_stream << err;
      RUDF_ERROR("{} for func:{}", err_str, compiled_ctx->func_ast.name);
      return absl::InvalidArgumentError(err_str);
    }
    auto func_addr = std::move(*func_addr_result);

    auto func_ptr = reinterpret_cast<void*>(func_addr.toPtr<RET(Args...)>());
    return JitFunction<RET, Args...>(compiled_ctx->desc.name, func_ptr, std::make_shared<int>(), false);
  }

 private:
  struct ExternFunction {
    FunctionDesc desc;
    ::llvm::FunctionType* func_type = nullptr;
    ::llvm::Function* func = nullptr;
  };
  using ExternFunctionPtr = std::shared_ptr<ExternFunction>;
  struct FunctionCompileContext {
    FunctionDesc desc;
    ast::Function func_ast;
    ::llvm::Function* func = nullptr;
    std::vector<::llvm::Type*> func_arg_types;
    std::unordered_map<std::string, ValuePtr> named_values;
    std::vector<std::unique_ptr<std::string>> const_strings;
  };
  using FunctionCompileContextPtr = std::shared_ptr<FunctionCompileContext>;

  using JitTypeArray = std::vector<::llvm::Type*>;
  using JitTypeArrayPtr = std::unique_ptr<JitTypeArray>;
  using JitValueArray = std::vector<::llvm::Value*>;
  using JitValueArrayPtr = std::unique_ptr<JitValueArray>;

  void Init();

  absl::Status Compile();

  ValuePtr NewValue(DType dtype, ::llvm::Value* val, const std::string& name = "");

  absl::StatusOr<FunctionCompileContextPtr> CompileFunction(const std::string& source);
  absl::StatusOr<FunctionCompileContextPtr> CompileFunction(const ast::Function& function);

  absl::StatusOr<::llvm::Type*> GetType(DType dtype);
  absl::StatusOr<::llvm::FunctionType*> GetFunctionType(const FunctionDesc& desc);

  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::Block& block);

  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::ReturnStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::IfElseStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::WhileStatement& statement);
  absl::Status BuildIR(FunctionCompileContextPtr ctx, const ast::ExpressionStatement& statement);

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
  absl::StatusOr<ValuePtr> CallFunction(const std::string& name, const std::vector<ValuePtr>& arg_values);
  absl::StatusOr<ValuePtr> CallFunction(std::string_view name, const std::vector<ValuePtr>& arg_values) {
    return CallFunction(std::string(name), arg_values);
  }

  JitTypeArray& AllocateJitTypes();
  JitValueArray& AllocateJitValues();

  ::llvm::IRBuilder<>* GetIRBuilder() { return ir_builder_.get(); }

  FunctionCompileContext& GetCompileContext() { return *current_compile_functon_ctx_; }

  ast::ParseContext ast_ctx_;

  std::unique_ptr<::llvm::orc::LLJIT> jit_;
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

  std::unordered_map<std::string, ExternFunctionPtr> extern_funcs_;

  std::unordered_map<std::string, FunctionCompileContextPtr> compile_functon_ctxs_;
  FunctionCompileContextPtr current_compile_functon_ctx_;
  uint32_t compile_function_idx_ = 0;

  std::vector<JitTypeArrayPtr> jit_types_;
  std::vector<JitValueArrayPtr> jit_values_;

  friend class Value;
};
}  // namespace llvm
}  // namespace rapidudf