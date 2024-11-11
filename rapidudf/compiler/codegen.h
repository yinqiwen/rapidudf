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

#include <string>
#include <string_view>
#include <utility>

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/StandardInstrumentations.h"

#include "absl/status/statusor.h"
#include "rapidudf/compiler/macros.h"
#include "rapidudf/compiler/options.h"
#include "rapidudf/compiler/value.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace compiler {
struct ExternFunction {
  FunctionDesc desc;
  ::llvm::FunctionType* func_type = nullptr;
  ::llvm::Function* func = nullptr;
};
using ExternFunctionPtr = std::shared_ptr<ExternFunction>;
using LoopBlocks = std::pair<::llvm::BasicBlock*, ::llvm::BasicBlock*>;
struct FunctionValue {
  FunctionDesc desc;
  ::llvm::Function* func = nullptr;
  ::llvm::Type* return_type = nullptr;
  ValuePtr return_value = nullptr;
  ::llvm::BasicBlock* exit_block = nullptr;
  ValuePtr context_arg_value;
  std::unordered_map<std::string, ValuePtr> named_values;
  std::vector<LoopBlocks> loop_blocks;
};
using FunctionValuePtr = std::shared_ptr<FunctionValue>;

class CodeGen {
 public:
  class Loop {
   private:
    ::llvm::Function* func_ = nullptr;
    ::llvm::BasicBlock* cond_ = nullptr;
    ::llvm::BasicBlock* body_ = nullptr;
    ::llvm::BasicBlock* end_ = nullptr;
    friend class CodeGen;
  };

  class Condition {
   private:
    ::llvm::BasicBlock* if_body = nullptr;
    std::vector<::llvm::BasicBlock*> elif_bodies;
    std::vector<::llvm::BasicBlock*> elif_conds;
    ::llvm::BasicBlock* else_body = nullptr;
    ::llvm::BasicBlock* end = nullptr;
    friend class CodeGen;
  };

  explicit CodeGen(const Options& opts);

  absl::Status DeclareExternFunctions(
      std::unordered_map<std::string, const FunctionDesc*>& func_calls,
      std::unordered_map<DType, std::unordered_map<std::string, FunctionDesc>>& member_func_calls);
  absl::Status DefineFunction(const FunctionDesc& desc, const std::vector<std::string>& arg_names);

  Loop NewLoop();
  void AddLoopCond(Loop loop, ValuePtr cond);
  void AddLoopCond(Loop loop, ::llvm::Value* cond);
  void FinishLoop(Loop loop);

  Condition NewCondition(size_t elif_count, bool has_else);
  void BeginIf(Condition condition, ValuePtr cond);
  void EndIf(Condition condition);
  void BeginElif(Condition condition, size_t i);
  void EndElifCond(Condition condition, size_t i, ValuePtr cond);
  void EndElif(Condition condition, size_t i);
  void BeginElse(Condition condition);
  void EndElse(Condition condition);
  void FinishCondition(Condition condition);

  absl::Status Return(ValuePtr val);
  absl::Status BreakLoop();
  absl::Status ContinueLoop();
  absl::Status FinishFunction();

  absl::StatusOr<::llvm::Value*> NewConstantVectorValue(DType dtype, ::llvm::Value* val);
  absl::StatusOr<::llvm::Value*> NewConstantVectorValue(ValuePtr val);
  absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> NewStackConstantVector(ValuePtr constant);
  ::llvm::Value* NewVectorVar(DType dtype);

  absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> LoadVector(DType dtype, ::llvm::Value* ptr,
                                                                       ::llvm::Value* idx);
  absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> LoadNVector(DType dtype, ::llvm::Value* ptr,
                                                                        ::llvm::Value* idx, ::llvm::Value* n);
  absl::Status StoreVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx);
  absl::Status StoreNVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx, ::llvm::Value* n);
  void Store(::llvm::Value* val, ::llvm::Value* ptr);
  ::llvm::Value* Load(::llvm::Type* typ, ::llvm::Value* ptr);

  absl::StatusOr<ValuePtr> CastTo(ValuePtr val, DType dst_dtype);
  absl::StatusOr<::llvm::Value*> CastTo(::llvm::Value* val, DType src_dtype, DType dst_dtype);
  absl::StatusOr<::llvm::Value*> UnaryOp(OpToken op, DType dtype, ::llvm::Value* val);
  absl::StatusOr<::llvm::Value*> BinaryOp(OpToken op, DType dtype, ::llvm::Value* left, ::llvm::Value* right);
  absl::StatusOr<::llvm::Value*> TernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                           ::llvm::Value* c);
  absl::StatusOr<ValuePtr> UnaryOp(OpToken op, ValuePtr val);
  absl::StatusOr<ValuePtr> BinaryOp(OpToken op, ValuePtr left, ValuePtr right);
  absl::StatusOr<ValuePtr> TernaryOp(OpToken op, ValuePtr a, ValuePtr b, ValuePtr c);

  absl::StatusOr<::llvm::Value*> VectorUnaryOp(OpToken op, DType dtype, ::llvm::Value* input, ::llvm::Value* output);
  absl::StatusOr<::llvm::Value*> VectorBinaryOp(OpToken op, DType dtype, ::llvm::Value* left, ::llvm::Value* right,
                                                ::llvm::Value* output);
  absl::StatusOr<::llvm::Value*> VectorTernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                                 ::llvm::Value* c, ::llvm::Value* output);

  absl::StatusOr<ValuePtr> CallFunction(const std::string& name, const std::vector<ValuePtr>& const_arg_values);
  absl::StatusOr<ValuePtr> CallFunction(std::string_view name, const std::vector<ValuePtr>& arg_values) {
    return CallFunction(std::string(name), arg_values);
  }

  absl::StatusOr<ValuePtr> GetLocalVar(const std::string& name);

  ValuePtr NewValue(DType dtype, ::llvm::Value* val, ::llvm::Type* ptr_element_type = nullptr);
  ValuePtr NewU32(uint32_t);
  ValuePtr NewI32(uint32_t);
  ValuePtr NewU32Var(uint32_t init = 0);
  ValuePtr NewI32Var(uint32_t init = 0);
  ValuePtr NewBool(bool);
  ValuePtr NewF32(float);
  ValuePtr NewF64(double);
  ValuePtr NewVoid(const std::string& name);
  ValuePtr NewStringView(const std::string& str);
  ValuePtr NewVar(DType dtype);
  absl::StatusOr<ValuePtr> NewArray(DType dtype, const std::vector<ValuePtr>& vals);
  absl::StatusOr<ValuePtr> GetStructField(ValuePtr obj, DType field_dtype, uint32_t offset);

  absl::StatusOr<::llvm::Type*> GetType(DType dtype);
  absl::StatusOr<::llvm::FunctionType*> GetFunctionType(const FunctionDesc& desc);

  absl::StatusOr<void*> GetFunctionPtr(const std::string& name);

  absl::Status Finish();

  ::llvm::LLVMContext& GetContext() { return *context_; }

  bool IsExternFunctionExist(const std::string& name);

  void PrintAsm();

 private:
  uint32_t GetLabelCursor() { return label_cursor_++; }
  ::llvm::Type* GetElementType(::llvm::Type* t);

  absl::StatusOr<::llvm::Value*> CallFunction(const std::string& name, const std::vector<::llvm::Value*>& arg_values);

  absl::StatusOr<DType> NormalizeDType(const std::vector<DType>& dtypes);

  ExternFunctionPtr GetFunction(const std::string& name);

  Options opts_;

  std::vector<std::unique_ptr<std::string>> const_strings_;
  std::unordered_map<std::string, ExternFunctionPtr> extern_funcs_;
  std::unordered_map<std::string, FunctionValuePtr> funcs_;
  FunctionValuePtr current_func_;

  std::unique_ptr<::llvm::orc::LLJIT> jit_;
  std::unique_ptr<::llvm::LLVMContext> context_;
  std::unique_ptr<::llvm::Module> module_;
  std::unique_ptr<::llvm::IRBuilder<>> builder_;

  std::unique_ptr<::llvm::FunctionPassManager> func_pass_manager_;
  std::unique_ptr<::llvm::LoopAnalysisManager> loop_analysis_manager_;
  std::unique_ptr<::llvm::FunctionAnalysisManager> func_analysis_manager_;
  std::unique_ptr<::llvm::CGSCCAnalysisManager> cgscc_analysis_manager_;
  std::unique_ptr<::llvm::ModuleAnalysisManager> module_analysis_manager_;
  std::unique_ptr<::llvm::PassInstrumentationCallbacks> pass_inst_callbacks_;
  std::unique_ptr<::llvm::StandardInstrumentations> std_insts_;

  uint32_t label_cursor_;
};
}  // namespace compiler
}  // namespace rapidudf