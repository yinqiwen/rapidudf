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
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"

namespace rapidudf {
namespace llvm {
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
  ValuePtr return_value = nullptr;
  ::llvm::BasicBlock* exit_block = nullptr;
  ::llvm::BasicBlock* exception_block = nullptr;
  ValuePtr context_arg_value;
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

  std::unique_ptr<::llvm::LLVMContext> context;
  std::unique_ptr<::llvm::Module> module;
  std::unique_ptr<::llvm::IRBuilder<>> ir_builder;

  std::unique_ptr<::llvm::FunctionPassManager> func_pass_manager;
  std::unique_ptr<::llvm::LoopAnalysisManager> loop_analysis_manager;
  std::unique_ptr<::llvm::FunctionAnalysisManager> func_analysis_manager;
  std::unique_ptr<::llvm::CGSCCAnalysisManager> cgscc_analysis_manager;
  std::unique_ptr<::llvm::ModuleAnalysisManager> module_analysis_manager;
  std::unique_ptr<::llvm::PassInstrumentationCallbacks> pass_inst_callbacks;
  std::unique_ptr<::llvm::StandardInstrumentations> std_insts;

  JitFunctionStat stat;
  ::llvm::IRBuilder<>* GetIRBuilder() { return ir_builder.get(); }
};

}  // namespace llvm
}  // namespace rapidudf