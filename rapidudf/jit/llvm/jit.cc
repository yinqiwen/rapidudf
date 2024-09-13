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
#include "rapidudf/jit/llvm/jit.h"
#include <fmt/core.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Use.h>
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#include <memory>
#include <vector>
#include "rapidudf/ast/grammar.h"
#include "rapidudf/ast/symbols.h"
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/jit/llvm/macros.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
namespace rapidudf {
namespace llvm {
JitCompiler::JitCompiler() {
  init_builtin();
  ast::Symbols::Init();
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();
  ::llvm::InitializeNativeTargetAsmParser();
  auto result = ::llvm::orc::LLJITBuilder().create();
  jit_ = std::move(*result);
  Init();
}
void JitCompiler::Init() {
  // Open a new context and module.
  context_ = std::make_unique<::llvm::LLVMContext>();
  module_ = std::make_unique<::llvm::Module>("RapidUDF", *context_);
  ir_builder_ = std::make_unique<::llvm::IRBuilder<>>(*context_);
  module_->setDataLayout(jit_->getDataLayout());

  init_buitin_types(*context_);

  // Create new pass and analysis managers.
  func_pass_manager_ = std::make_unique<::llvm::FunctionPassManager>();
  loop_analysis_manager_ = std::make_unique<::llvm::LoopAnalysisManager>();
  func_analysis_manager_ = std::make_unique<::llvm::FunctionAnalysisManager>();
  cgscc_analysis_manager_ = std::make_unique<::llvm::CGSCCAnalysisManager>();
  module_analysis_manager_ = std::make_unique<::llvm::ModuleAnalysisManager>();
  pass_inst_callbacks_ = std::make_unique<::llvm::PassInstrumentationCallbacks>();
  std_insts_ = std::make_unique<::llvm::StandardInstrumentations>(*context_,
                                                                  /*DebugLogging*/ true);
  std_insts_->registerCallbacks(*pass_inst_callbacks_, module_analysis_manager_.get());

  // Add transform passes.

  func_pass_manager_->addPass(::llvm::InstCombinePass());
  func_pass_manager_->addPass(::llvm::ReassociatePass());
  // func_pass_manager_->addPass(::llvm::InlinerPass());
  func_pass_manager_->addPass(::llvm::GVNPass());
  func_pass_manager_->addPass(::llvm::SimplifyCFGPass());

  // Register analysis passes used in these transform passes.
  ::llvm::PassBuilder pass_builder;
  pass_builder.registerModuleAnalyses(*module_analysis_manager_);
  pass_builder.registerFunctionAnalyses(*func_analysis_manager_);
  pass_builder.crossRegisterProxies(*loop_analysis_manager_, *func_analysis_manager_, *cgscc_analysis_manager_,
                                    *module_analysis_manager_);
}

ValuePtr JitCompiler::NewValue(DType dtype, ::llvm::Value* val, ::llvm::Type* type) {
  return Value::New(dtype, this, val, type);
}

typename JitCompiler::JitTypeArray& JitCompiler::AllocateJitTypes() {
  auto p = std::make_unique<JitTypeArray>();
  jit_types_.emplace_back(std::move(p));
  return *jit_types_[jit_types_.size() - 1];
}
typename JitCompiler::JitValueArray& JitCompiler::AllocateJitValues() {
  auto p = std::make_unique<JitValueArray>();
  jit_values_.emplace_back(std::move(p));
  return *jit_values_[jit_values_.size() - 1];
}

absl::StatusOr<::llvm::FunctionType*> JitCompiler::GetFunctionType(const FunctionDesc& desc) {
  auto return_type_result = GetType(desc.return_type);
  if (!return_type_result.ok()) {
    return return_type_result.status();
  }
  ::llvm::Type* return_type = return_type_result.value();
  auto& func_arg_types = AllocateJitTypes();
  for (size_t i = 0; i < desc.arg_types.size(); i++) {
    auto arg_type = GetType(desc.arg_types[i]);
    if (!arg_type.ok()) {
      return arg_type.status();
    }

    func_arg_types.emplace_back(arg_type.value());
  }
  auto func_type = ::llvm::FunctionType::get(return_type, ::llvm::ArrayRef<::llvm::Type*>(func_arg_types), false);
  return func_type;
}

absl::StatusOr<typename JitCompiler::FunctionCompileContextPtr> JitCompiler::CompileFunction(
    const std::string& source) {
  auto f = ast::parse_function_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }

  return CompileFunction(f.value());
}

absl::StatusOr<typename JitCompiler::FunctionCompileContextPtr> JitCompiler::CompileFunction(
    const ast::Function& function) {
  auto& dylib = jit_->getMainJITDylib();
  ::llvm::orc::SymbolMap extern_func_map;
  ::llvm::orc::MangleAndInterner mangle(jit_->getExecutionSession(), jit_->getDataLayout());
  auto all_func_calls = ast_ctx_.GetAllFuncCalls(compile_function_idx_);
  auto implicit_func_calls = ast_ctx_.GetAllImplicitFuncCalls(compile_function_idx_);
  all_func_calls.merge(implicit_func_calls);

  for (auto [_, desc] : all_func_calls) {
    auto exec_addr = ::llvm::orc::ExecutorAddr::fromPtr(desc->func);
    extern_func_map.insert({mangle(desc->name), {exec_addr, ::llvm::JITSymbolFlags::Callable}});
    RUDF_DEBUG("Inject extern func:{}", desc->name);
    auto func_type_result = GetFunctionType(*desc);
    if (!func_type_result.ok()) {
      return func_type_result.status();
    }
    ExternFunctionPtr extern_func = std::make_shared<ExternFunction>();
    extern_func->desc = *desc;
    auto func_type = func_type_result.value();
    extern_func->func =
        ::llvm::Function::Create(func_type, ::llvm::Function::ExternalLinkage, extern_func->desc.name, *module_);
    extern_func->func_type = func_type;
    extern_funcs_[desc->name] = extern_func;
  }
  if (!extern_func_map.empty()) {
    auto err = dylib.define(::llvm::orc::absoluteSymbols(extern_func_map));
    RUDF_LOG_RETURN_LLVM_ERROR(err);
  }
  std::shared_ptr<FunctionCompileContext> func_compile_ctx = std::make_shared<FunctionCompileContext>();
  func_compile_ctx->func_ast = function;
  func_compile_ctx->desc = function.ToFuncDesc();
  auto func_type_result = GetFunctionType(function.ToFuncDesc());
  if (!func_type_result.ok()) {
    return func_type_result.status();
  }
  ::llvm::FunctionType* func_type = func_type_result.value();
  ::llvm::Function* f =
      ::llvm::Function::Create(func_type, ::llvm::Function::ExternalLinkage, func_compile_ctx->func_ast.name, *module_);

  // Add a basic block to the function. As before, it automatically inserts
  // because of the last argument.
  ::llvm::BasicBlock* entry_block = ::llvm::BasicBlock::Create(*context_, "entry", f);
  func_compile_ctx->exit_block = ::llvm::BasicBlock::Create(*context_, "exit");
  ir_builder_->SetInsertPoint(entry_block);
  if (!function.return_type.IsVoid()) {
    auto return_type_result = GetType(function.return_type);
    if (!return_type_result.ok()) {
      return return_type_result.status();
    }
    func_compile_ctx->return_type = return_type_result.value();
    func_compile_ctx->return_value = ir_builder_->CreateAlloca(return_type_result.value());
  }

  // Create a basic block builder with default parameters.  The builder will
  // automatically append instructions to the basic block `BB'.

  current_compile_functon_ctx_ = func_compile_ctx;

  if (!f->arg_empty()) {
    for (size_t i = 0; i < f->arg_size(); i++) {
      ::llvm::Argument* arg = f->getArg(i);
      std::string name = (*function.args)[i].name;
      DType dtype = (*function.args)[i].dtype;
      arg->setName(name);
      // func_compile_ctx->named_values[name] = NewValue(dtype, arg, arg->getType());
      auto* arg_val = ir_builder_->CreateAlloca(arg->getType());
      ir_builder_->CreateStore(arg, arg_val);
      func_compile_ctx->named_values[name] = NewValue(dtype, arg_val, arg->getType());
    }
  }
  func_compile_ctx->func = f;
  auto status = BuildIR(func_compile_ctx, function.body);
  if (!status.ok()) {
    return status;
  }
  if (ir_builder_->GetInsertBlock()->getTerminator() == nullptr) {
    ir_builder_->CreateBr(func_compile_ctx->exit_block);
  }
  func_compile_ctx->exit_block->insertInto(f);
  ir_builder_->SetInsertPoint(func_compile_ctx->exit_block);
  if (nullptr != func_compile_ctx->return_value) {
    ir_builder_->CreateRet(ir_builder_->CreateLoad(func_compile_ctx->return_type, func_compile_ctx->return_value));
  }

  // Validate the generated code, checking for consistency.
  std::string err_str;
  ::llvm::raw_string_ostream err_stream(err_str);
  bool r = ::llvm::verifyFunction(*f, &err_stream);
  if (r) {
    RUDF_ERROR("verify failed:{}", err_str);
    Dump();
    return absl::InvalidArgumentError(err_str);
  }

  // Run the optimizer on the function.
  // func_pass_manager_->run(*f, *func_analysis_manager_);
  return func_compile_ctx;
}

absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::Block& block) {
  return BuildIR(ctx, block.statements);
}

absl::Status JitCompiler::Compile() {
  ::llvm::orc::ThreadSafeModule module(std::move(module_), std::move(context_));

  auto err = jit_->addIRModule(std::move(module));
  RUDF_LOG_RETURN_LLVM_ERROR(err);
  return absl::OkStatus();
}

absl::StatusOr<::llvm::Type*> JitCompiler::GetType(DType dtype) {
  auto type = get_type(*context_, dtype);
  if (nullptr == type) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("get type failed for:{}", dtype)));
  }
  return type;
}

typename JitCompiler::ExternFunctionPtr JitCompiler::GetFunction(const std::string& name) {
  auto found = extern_funcs_.find(name);
  if (found == extern_funcs_.end()) {
    return nullptr;
  }
  return found->second;
}

absl::StatusOr<ValuePtr> JitCompiler::CallFunction(const std::string& name, const std::vector<ValuePtr>& arg_values) {
  ExternFunctionPtr func = GetFunction(name);
  if (!func) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("No func:{} found", name)));
  }
  std::vector<::llvm::Value*> arg_vals;
  for (auto& arg : arg_values) {
    arg_vals.emplace_back(arg->GetValue());
  }
  ::llvm::Value* result = ir_builder_->CreateCall(func->func, arg_vals);
  return NewValue(func->desc.return_type, result);
}

void JitCompiler::Dump() { module_->print(::llvm::errs(), nullptr); }
}  // namespace llvm
}  // namespace rapidudf