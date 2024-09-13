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
#include "llvm/ADT/APFloat.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#include <memory>
#include <vector>
#include "rapidudf/ast/context.h"
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
}

absl::StatusOr<void*> JitCompiler::GetFunctionPtr(const std::string& name) {
  if (!session_) {
    return absl::InvalidArgumentError("null compiled session");
  }
  auto func_addr_result = GetSession().jit->lookup(name);
  if (!func_addr_result) {
    RUDF_LOG_RETURN_LLVM_ERROR(func_addr_result.takeError());
  }
  auto func_addr = std::move(*func_addr_result);
  auto func_ptr = reinterpret_cast<void*>(func_addr.toPtr<void()>());
  if (nullptr == func_ptr) {
    RUDF_INFO("####Null func ptr:{}", name);
  }
  return func_ptr;
}

void JitCompiler::NewSession(bool print_asm) {
  ast_ctx_.Clear();
  session_ = std::make_shared<JitSession>();
  session_->print_asm = print_asm;
  ::llvm::orc::LLJITBuilder jit_builder;
  auto result = jit_builder.create();
  session_->jit = std::move(*result);
  context_ = std::make_unique<::llvm::LLVMContext>();
  module_ = std::make_unique<::llvm::Module>("RapidUDF", *context_);
  ir_builder_ = std::make_unique<::llvm::IRBuilder<>>(*context_);
  module_->setDataLayout(session_->jit->getDataLayout());

  init_buitin_types(*context_);

  // Create new pass and analysis managers.

  loop_analysis_manager_ = std::make_unique<::llvm::LoopAnalysisManager>();
  func_analysis_manager_ = std::make_unique<::llvm::FunctionAnalysisManager>();
  cgscc_analysis_manager_ = std::make_unique<::llvm::CGSCCAnalysisManager>();
  module_analysis_manager_ = std::make_unique<::llvm::ModuleAnalysisManager>();
  pass_inst_callbacks_ = std::make_unique<::llvm::PassInstrumentationCallbacks>();
  std_insts_ = std::make_unique<::llvm::StandardInstrumentations>(*context_,
                                                                  /*DebugLogging*/ true);
  std_insts_->registerCallbacks(*pass_inst_callbacks_, module_analysis_manager_.get());

  // Add transform passes.

  // Register analysis passes used in these transform passes.
  ::llvm::PassBuilder pass_builder;

  pass_builder.registerModuleAnalyses(*module_analysis_manager_);
  pass_builder.registerFunctionAnalyses(*func_analysis_manager_);
  pass_builder.registerCGSCCAnalyses(*cgscc_analysis_manager_);
  pass_builder.registerLoopAnalyses(*loop_analysis_manager_);
  pass_builder.crossRegisterProxies(*loop_analysis_manager_, *func_analysis_manager_, *cgscc_analysis_manager_,
                                    *module_analysis_manager_);

  // pass_builder.buildModuleOptimizationPipeline(::llvm::OptimizationLevel::O2, ::llvm::ThinOrFullLTOPhase::None).
  auto func_pass_manager =
      pass_builder.buildFunctionSimplificationPipeline(::llvm::OptimizationLevel::O2, ::llvm::ThinOrFullLTOPhase::None);
  func_pass_manager_ = std::make_unique<::llvm::FunctionPassManager>(std::move(func_pass_manager));
  func_pass_manager_->addPass(::llvm::InstCombinePass());
  func_pass_manager_->addPass(::llvm::ReassociatePass());
  func_pass_manager_->addPass(::llvm::GVNPass());
  func_pass_manager_->addPass(::llvm::SimplifyCFGPass());
}

ValuePtr JitCompiler::NewValue(DType dtype, ::llvm::Value* val, ::llvm::Type* type) {
  return Value::New(dtype, this, val, type);
}

absl::StatusOr<::llvm::FunctionType*> JitCompiler::GetFunctionType(const FunctionDesc& desc) {
  auto return_type_result = GetType(desc.return_type);
  if (!return_type_result.ok()) {
    return return_type_result.status();
  }
  ::llvm::Type* return_type = return_type_result.value();
  std::vector<::llvm::Type*> func_arg_types;
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

absl::Status JitCompiler::CompileFunction(const std::string& source) {
  auto f = ast::parse_function_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }

  return CompileFunction(f.value());
}

std::string JitCompiler::GetMemberFuncName(DType dtype, const std::string& member) {
  std::string fname = fmt::format("{}_{}", dtype.GetTypeString(), member);
  return fname;
}

absl::Status JitCompiler::CompileFunctions(const std::vector<ast::Function>& functions) {
  ast::ParseContext::FunctionCallMap all_func_calls;
  ast::ParseContext::MemberFuncCallMap all_member_func_calls;
  for (size_t i = 0; i < functions.size(); i++) {
    auto func_calls = ast_ctx_.GetAllFuncCalls(i);
    auto implicit_func_calls = ast_ctx_.GetAllImplicitFuncCalls(i);
    all_func_calls.merge(func_calls);
    all_func_calls.merge(implicit_func_calls);
    const auto& member_func_calls = ast_ctx_.GetAllMemberFuncCalls(i);
    for (const auto& [dtype, func_calls] : member_func_calls) {
      for (const auto& [name, desc] : func_calls) {
        all_member_func_calls[dtype].emplace(name, desc);
      }
    }
  }

  auto& dylib = GetSession().jit->getMainJITDylib();
  ::llvm::orc::SymbolMap extern_func_map;
  ::llvm::orc::MangleAndInterner mangle(GetSession().jit->getExecutionSession(), GetSession().jit->getDataLayout());
  for (auto [_, desc] : all_func_calls) {
    auto exec_addr = ::llvm::orc::ExecutorAddr::fromPtr(desc->func);
    extern_func_map.insert({mangle(desc->name), {exec_addr, ::llvm::JITSymbolFlags::Callable}});
    RUDF_DEBUG("Inject extern func {}", desc->name);
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
    GetSession().extern_funcs[desc->name] = extern_func;
  }
  const auto& member_func_calls = ast_ctx_.GetAllMemberFuncCalls(GetSession().compile_function_idx);
  for (const auto& [dtype, func_calls] : member_func_calls) {
    for (const auto& [name, desc] : func_calls) {
      auto exec_addr = ::llvm::orc::ExecutorAddr::fromPtr(desc.func);
      std::string fname = GetMemberFuncName(dtype, name);
      extern_func_map.insert({mangle(fname), {exec_addr, ::llvm::JITSymbolFlags::Callable}});
      RUDF_DEBUG("Inject member func {}", fname);
      auto func_type_result = GetFunctionType(desc);
      if (!func_type_result.ok()) {
        return func_type_result.status();
      }
      ExternFunctionPtr extern_func = std::make_shared<ExternFunction>();
      extern_func->desc = desc;
      auto func_type = func_type_result.value();
      extern_func->func = ::llvm::Function::Create(func_type, ::llvm::Function::ExternalLinkage, fname, *module_);
      extern_func->func_type = func_type;
      GetSession().extern_funcs[fname] = extern_func;
    }
  }

  if (!extern_func_map.empty()) {
    auto err = dylib.define(::llvm::orc::absoluteSymbols(extern_func_map));
    RUDF_LOG_RETURN_LLVM_ERROR(err);
  }

  for (auto& func : functions) {
    auto status = BuildIR(func);
    RUDF_LOG_RETURN_ERROR_STATUS(status);
  }
  return Compile();
}

absl::Status JitCompiler::CompileExpression(const std::string& expr, ast::Function& function) {
  auto f = ast::parse_expression_ast(ast_ctx_, expr);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }
  // RUDF_INFO("after ast parse, {}", ast_ctx_.)
  ast::ReturnStatement return_statement;
  return_statement.expr = *f;
  function.body.statements.emplace_back(return_statement);

  return CompileFunctions(std::vector<ast::Function>{function});
}

absl::Status JitCompiler::CompileFunction(const ast::Function& function) {
  return CompileFunctions(std::vector<ast::Function>{function});
}

absl::StatusOr<std::vector<std::string>> JitCompiler::CompileSource(const std::string& source, bool dump_asm) {
  auto funcs = ast::parse_functions_ast(ast_ctx_, source);
  if (!funcs.ok()) {
    RUDF_LOG_ERROR_STATUS(funcs.status());
  }
  std::vector<std::string> fnames;
  for (auto& func : funcs.value()) {
    fnames.emplace_back(func.name);
  }

  std::lock_guard<std::mutex> guard(jit_mutex_);
  NewSession(dump_asm);
  auto status = CompileFunctions(funcs.value());
  if (!status.ok()) {
    return status;
  }
  return fnames;
}

absl::Status JitCompiler::BuildIR(const ast::Function& function) {
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
  RUDF_DEBUG("create func:{}", func_compile_ctx->func_ast.name);
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

  GetSession().current_compile_functon_ctx = func_compile_ctx;
  GetSession().compile_functon_ctxs.emplace(function.name, func_compile_ctx);

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
  } else {
    ir_builder_->CreateRetVoid();
  }

  // Validate the generated code, checking for consistency.
  std::string err_str;
  ::llvm::raw_string_ostream err_stream(err_str);
  bool r = ::llvm::verifyFunction(*f, &err_stream);
  if (r) {
    RUDF_ERROR("verify failed:{}", err_str);
    module_->print(::llvm::errs(), nullptr);
    return absl::InvalidArgumentError(err_str);
  }

  // Run the optimizer on the function.
  func_pass_manager_->run(*f, *func_analysis_manager_);
  return absl::OkStatus();
}

absl::Status JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::Block& block) {
  return BuildIR(ctx, block.statements);
}

absl::Status JitCompiler::Compile() {
  if (GetSession().print_asm) {
    module_->print(::llvm::errs(), nullptr);
  }
  ::llvm::orc::ThreadSafeModule module(std::move(module_), std::move(context_));
  auto err = GetSession().jit->addIRModule(std::move(module));
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
  auto local_found = GetSession().compile_functon_ctxs.find(name);
  if (local_found != GetSession().compile_functon_ctxs.end()) {
    //
  }
  auto found = GetSession().extern_funcs.find(name);
  if (found == GetSession().extern_funcs.end()) {
    return nullptr;
  }
  return found->second;
}

absl::StatusOr<ValuePtr> JitCompiler::CallFunction(const std::string& name, const std::vector<ValuePtr>& arg_values) {
  ::llvm::Function* found_func = nullptr;
  FunctionDesc found_func_desc;
  ExternFunctionPtr func = GetFunction(name);
  if (func) {
    found_func_desc = func->desc;
    found_func = func->func;
  } else {
    auto found = GetSession().compile_functon_ctxs.find(name);
    if (found != GetSession().compile_functon_ctxs.end()) {
      found_func_desc = found->second->desc;
      found_func = found->second->func;
    }
  }
  if (!found_func) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("No func:{} found", name)));
  }
  if (arg_values.size() != found_func_desc.arg_types.size()) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
        fmt::format("Expect {} args, while {} given", found_func_desc.arg_types.size(), arg_values.size())));
  }

  std::vector<::llvm::Value*> arg_vals(arg_values.size());
  for (size_t i = 0; i < arg_values.size(); i++) {
    ValuePtr arg_val = arg_values[i];
    if (arg_val->GetDType() != found_func_desc.arg_types[i]) {
      DType src_dtype = arg_val->GetDType();
      RUDF_INFO("[{}]name:{}, var dtype:{} {}", i, name, src_dtype, found_func_desc.arg_types[i]);
      arg_val = arg_val->CastTo(found_func_desc.arg_types[i]);
      if (!arg_val) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Func:{} cast arg:{} from {} to {} failed.", name, i,
                                                                  src_dtype, found_func_desc.arg_types[i])));
      }
    }
    arg_vals[i] = arg_val->GetValue();
  }

  ::llvm::Value* result = ir_builder_->CreateCall(found_func, arg_vals);
  return NewValue(found_func_desc.return_type, result);
}

}  // namespace llvm
}  // namespace rapidudf