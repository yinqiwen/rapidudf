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
#include "rapidudf/compiler/codegen.h"

#include "fmt/format.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/Transforms/Vectorize/VectorCombine.h"

#include "rapidudf/compiler/type.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/constants.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/vector.h"
namespace rapidudf {
namespace compiler {

CodeGen::CodeGen(const Options& opts) : opts_(opts), label_cursor_(0) {
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();
  ::llvm::InitializeNativeTargetAsmParser();

  auto JTMB = ::llvm::orc::JITTargetMachineBuilder::detectHost();
  // RUDF_INFO("features:{}", JTMB->getFeatures().getString());
  // RUDF_INFO("cpu:{}", JTMB->getCPU());
  ::llvm::orc::LLJITBuilder jit_builder;
  jit_builder.setJITTargetMachineBuilder(*JTMB);
  // jit_builder.getJITTargetMachineBuilder()->setCPU("haswell");
  auto result = jit_builder.create();
  jit_ = std::move(*result);
  context_ = std::make_unique<::llvm::LLVMContext>();
  module_ = std::make_unique<::llvm::Module>("RapidUDF", *context_);
  builder_ = std::make_unique<::llvm::IRBuilder<>>(*context_);
  module_->setDataLayout(jit_->getDataLayout());
  if (opts_.fast_math) {
    builder_->setFastMathFlags(::llvm::FastMathFlags::getFast());
  }

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

  ::llvm::OptimizationLevel opt_level = ::llvm::OptimizationLevel::O2;
  switch (opts_.optimize_level) {
    case 0: {
      opt_level = ::llvm::OptimizationLevel::O0;
      break;
    }
    case 1: {
      opt_level = ::llvm::OptimizationLevel::O1;
      break;
    }
    case 3: {
      opt_level = ::llvm::OptimizationLevel::O3;
      break;
    }
    default: {
      break;
    }
  }

  auto func_pass_manager =
      pass_builder.buildFunctionSimplificationPipeline(opt_level, ::llvm::ThinOrFullLTOPhase::ThinLTOPostLink);

  func_pass_manager_ = std::make_unique<::llvm::FunctionPassManager>(std::move(func_pass_manager));
  func_pass_manager_->addPass(::llvm::InstCombinePass());
  func_pass_manager_->addPass(::llvm::ReassociatePass());
  // func_pass_manager_->addPass(::llvm::GVNPass());
  func_pass_manager_->addPass(llvm::NewGVNPass());
  func_pass_manager_->addPass(::llvm::SimplifyCFGPass());
  func_pass_manager_->addPass(::llvm::PartiallyInlineLibCallsPass());
  func_pass_manager_->addPass(::llvm::MergedLoadStoreMotionPass());
  func_pass_manager_->addPass(::llvm::TailCallElimPass());
  func_pass_manager_->addPass(::llvm::LoadStoreVectorizerPass());
  func_pass_manager_->addPass(::llvm::SLPVectorizerPass());
  func_pass_manager_->addPass(::llvm::VectorCombinePass());
  func_pass_manager_->addPass(::llvm::LoopVectorizePass());
}

absl::Status CodeGen::Finish() {
  if (opts_.print_asm) {
    module_->print(::llvm::errs(), nullptr);
  }
  ::llvm::orc::ThreadSafeModule module(std::move(module_), std::move(context_));
  auto err = jit_->addIRModule(std::move(module));
  RUDF_LOG_RETURN_LLVM_ERROR(err);
  return absl::OkStatus();
}

absl::StatusOr<void*> CodeGen::GetFunctionPtr(const std::string& name) {
  auto func_addr_result = jit_->lookup(name);
  if (!func_addr_result) {
    RUDF_LOG_RETURN_LLVM_ERROR(func_addr_result.takeError());
  }
  auto func_addr = std::move(*func_addr_result);
  auto func_ptr = reinterpret_cast<void*>(func_addr.toPtr<void()>());
  return func_ptr;
}

::llvm::Type* CodeGen::GetElementType(::llvm::Type* t) {
  if (t->isVectorTy()) {
    ::llvm::VectorType* vtype = reinterpret_cast<::llvm::VectorType*>(t);
    return vtype->getElementType();
  } else {
    return t;
  }
}
absl::StatusOr<DType> CodeGen::NormalizeDType(const std::vector<DType>& dtypes) {
  DType normalize_dtype;
  for (auto dtype : dtypes) {
    if (dtype.IsNumber()) {
      if (dtype > normalize_dtype) {
        normalize_dtype = dtype;
      }
    } else if (dtype.IsStringPtr() || dtype.IsStdStringView() || dtype.IsFlatbuffersStringPtr() ||
               dtype.IsStringView()) {
      normalize_dtype = DType(DATA_STRING_VIEW);
      break;
    }
  }
  for (auto dtype : dtypes) {
    if (!dtype.CanCastTo(normalize_dtype)) {
      RUDF_LOG_RETURN_FMT_ERROR("Can NOT cast from {} to {}", dtype, normalize_dtype);
    }
  }
  return normalize_dtype;
}

absl::StatusOr<::llvm::Type*> CodeGen::GetType(DType dtype) {
  auto type = get_type(*context_, dtype);
  if (nullptr == type) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("get type failed for:{}", dtype)));
  }
  return type;
}

absl::StatusOr<::llvm::FunctionType*> CodeGen::GetFunctionType(const FunctionDesc& desc) {
  auto return_type_result = GetType(desc.return_type);
  if (!return_type_result.ok()) {
    abort();
    return return_type_result.status();
  }
  ::llvm::Type* return_type = return_type_result.value();
  std::vector<::llvm::Type*> func_arg_types;

  for (size_t i = 0; i < desc.arg_types.size(); i++) {
    auto arg_type_result = GetType(desc.arg_types[i]);
    if (!arg_type_result.ok()) {
      return arg_type_result.status();
    }
    auto* arg_type = arg_type_result.value();
    bool byval = desc.PassArgByValue(i);
    if (byval) {
      arg_type = ::llvm::PointerType::get(arg_type, 0);
    }
    func_arg_types.emplace_back(arg_type);
  }
  auto func_type = ::llvm::FunctionType::get(return_type, ::llvm::ArrayRef<::llvm::Type*>(func_arg_types), false);
  return func_type;
}

absl::StatusOr<ValuePtr> CodeGen::GetLocalVar(const std::string& name) {
  auto found = current_func_->named_values.find(name);
  if (found != current_func_->named_values.end()) {
    return found->second;
  }
  for (size_t i = 0; i < kConstantCount; i++) {
    if (name == kConstantNames[i]) {
      ::llvm::APFloat fv(kConstantValues[i]);
      auto val = ::llvm::ConstantFP::get(builder_->getContext(), fv);
      return NewValue(DATA_F64, val);
    }
  }
  return absl::NotFoundError(fmt::format("No var '{}' found", name));
}

bool CodeGen::IsExternFunctionExist(const std::string& name) { return extern_funcs_.find(name) != extern_funcs_.end(); }

ExternFunctionPtr CodeGen::GetFunction(const std::string& name) {
  auto local_found = funcs_.find(name);
  if (local_found != funcs_.end()) {
    //
  }
  auto found = extern_funcs_.find(name);
  if (found == extern_funcs_.end()) {
    return nullptr;
  }
  return found->second;
}

void CodeGen::PrintAsm() {
  if (opts_.print_asm) {
    module_->print(::llvm::errs(), nullptr);
  }
}

absl::Status CodeGen::DeclareExternFunctions(
    std::unordered_map<std::string, const FunctionDesc*>& func_calls,
    std::unordered_map<DType, std::unordered_map<std::string, FunctionDesc>>& member_func_calls) {
  auto& dylib = jit_->getMainJITDylib();
  ::llvm::orc::SymbolMap extern_func_map;
  ::llvm::orc::MangleAndInterner mangle(jit_->getExecutionSession(), jit_->getDataLayout());

  FunctionDesc vector_size_func_desc;
  vector_size_func_desc.name = std::string(kVectorGetSizeFuncName);
  vector_size_func_desc.return_type = DATA_I32;
  vector_size_func_desc.arg_types.emplace_back(get_dtype<void*>());
  vector_size_func_desc.func = reinterpret_cast<void*>(VectorBase::GetSize);
  func_calls.emplace(vector_size_func_desc.name, &vector_size_func_desc);

  FunctionDesc vector_data_func_desc;
  vector_data_func_desc.name = std::string(kVectorGetDataFuncName);
  vector_data_func_desc.return_type = get_dtype<const uint8_t*>();
  vector_data_func_desc.arg_types.emplace_back(get_dtype<void*>());
  vector_data_func_desc.func = reinterpret_cast<void*>(VectorBase::GetData);
  func_calls.emplace(vector_data_func_desc.name, &vector_data_func_desc);

  for (auto [_, desc] : func_calls) {
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

    if (!extern_func->func->arg_empty()) {
      for (size_t i = 0; i < extern_func->func->arg_size(); i++) {
        if (desc->PassArgByValue(i)) {
          auto* arg_type = get_type(*context_, desc->arg_types[i]);
          extern_func->func->addParamAttr(i, ::llvm::Attribute::getWithByValType(*context_, arg_type));
          extern_func->func->addParamAttr(i, ::llvm::Attribute::getWithAlignment(*context_, ::llvm::Align(8)));
          extern_func->func->addParamAttr(i, ::llvm::Attribute::AttrKind::NoUndef);
        }
      }
    }
    extern_funcs_[desc->name] = extern_func;
  }

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
      extern_funcs_[fname] = extern_func;
    }
  }

  if (!extern_func_map.empty()) {
    auto err = dylib.define(::llvm::orc::absoluteSymbols(extern_func_map));
    RUDF_LOG_RETURN_LLVM_ERROR(err);
  }

  return absl::OkStatus();
}

absl::Status CodeGen::DefineFunction(const FunctionDesc& desc, const std::vector<std::string>& arg_names) {
  auto func_type_result = GetFunctionType(desc);
  if (!func_type_result.ok()) {
    return func_type_result.status();
  }
  FunctionValuePtr func_value = std::make_shared<FunctionValue>();
  func_value->desc = desc;
  ::llvm::FunctionType* func_type = func_type_result.value();
  ::llvm::Function* f = ::llvm::Function::Create(func_type, ::llvm::Function::ExternalLinkage, desc.name, *module_);
  func_value->func = f;
  ::llvm::BasicBlock* entry_block = ::llvm::BasicBlock::Create(*context_, "entry", f);
  func_value->exit_block = ::llvm::BasicBlock::Create(*context_, "exit");
  builder_->SetInsertPoint(entry_block);
  if (!desc.return_type.IsVoid()) {
    auto return_type_result = GetType(desc.return_type);
    if (!return_type_result.ok()) {
      return return_type_result.status();
    }
    auto* return_type = return_type_result.value();
    func_value->return_type = return_type;
    ValuePtr return_value = NewValue(desc.return_type, builder_->CreateAlloca(return_type), return_type);
    func_value->return_value = return_value;
  }
  // RUDF_DEBUG("create func:{}", desc.name);
  if (!f->arg_empty()) {
    for (size_t i = 0; i < f->arg_size(); i++) {
      ::llvm::Argument* arg = f->getArg(i);
      std::string name = arg_names[i];
      DType dtype = desc.arg_types[i];
      arg->setName(name);
      // RUDF_INFO("{}:{}", name, f->getParamByValType(i)->getScalarSizeInBits());
      if (dtype.IsContextPtr()) {
        auto val = NewValue(dtype, arg);
        func_value->context_arg_value = val;
        func_value->named_values[name] = val;
      } else {
        auto* arg_type = get_type(*context_, dtype);
        ValuePtr val;
        if (arg_type != arg->getType()) {
          auto* arg_val = builder_->CreateAlloca(arg_type);
          auto load_val = builder_->CreateLoad(arg_type, arg);
          builder_->CreateStore(load_val, arg_val);
          val = NewValue(dtype, arg_val, arg_type);
          f->addParamAttr(i, ::llvm::Attribute::getWithByValType(*context_, arg_type));
        } else {
          auto* arg_val = builder_->CreateAlloca(arg->getType());
          builder_->CreateStore(arg, arg_val);
          val = NewValue(dtype, arg_val, arg->getType());
        }
        func_value->named_values[name] = val;
      }
    }
  }

  funcs_[desc.name] = func_value;
  current_func_ = func_value;
  return absl::OkStatus();
}

absl::Status CodeGen::FinishFunction() {
  if (builder_->GetInsertBlock()->getTerminator() == nullptr) {
    builder_->CreateBr(current_func_->exit_block);
  }
  current_func_->exit_block->insertInto(current_func_->func);
  builder_->SetInsertPoint(current_func_->exit_block);
  if (nullptr != current_func_->return_value) {
    builder_->CreateRet(current_func_->return_value->LoadValue());
  } else {
    builder_->CreateRetVoid();
  }
  // func_compile_ctx->exception_block->insertInto(f);
  // ir_builder->SetInsertPoint(func_compile_ctx->exception_block);

  // if (nullptr != func_compile_ctx->return_value) {
  //   ir_builder->CreateRet(func_compile_ctx->return_value->GetValue());
  // } else {
  //   ir_builder->CreateRetVoid();
  // }

  // Validate the generated code, checking for consistency.
  std::string err_str;
  ::llvm::raw_string_ostream err_stream(err_str);
  bool r = ::llvm::verifyFunction(*current_func_->func, &err_stream);
  if (r) {
    RUDF_ERROR("verify failed:{}", err_str);
    module_->print(::llvm::errs(), nullptr);
    return absl::InvalidArgumentError(err_str);
  }

  // Run the optimizer on the function.
  if (opts_.optimize_level > 0) {
    func_pass_manager_->run(*current_func_->func, *func_analysis_manager_);
  }
  return absl::OkStatus();
}

absl::Status CodeGen::Return(ValuePtr val) {
  if (!val) {
    builder_->CreateBr(current_func_->exit_block);
  } else {
    auto ret_val_result = CastTo(val, current_func_->desc.return_type);
    if (!ret_val_result.ok()) {
      return ret_val_result.status();
    }
    auto ret_val = ret_val_result.value();
    if (current_func_->return_value != nullptr) {
      builder_->CreateStore(ret_val->LoadValue(), current_func_->return_value->GetPtrValue());
    }
    builder_->CreateBr(current_func_->exit_block);
  }
  return absl::OkStatus();
}
absl::Status CodeGen::BreakLoop() {
  if (current_func_->loop_blocks.empty()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("break in non loop block"));
  }
  builder_->CreateBr(current_func_->loop_blocks.back().second);  // continue to loop end
  return absl::OkStatus();
}

absl::Status CodeGen::ContinueLoop() {
  if (current_func_->loop_blocks.empty()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("continue in non loop block"));
  }
  builder_->CreateBr(current_func_->loop_blocks.back().first);  // continue to loop cond
  return absl::OkStatus();
}

typename CodeGen::Loop CodeGen::NewLoop() {
  Loop loop;
  uint32_t label_cursor = GetLabelCursor();
  std::string while_cond_label = fmt::format("while_cond_{}", label_cursor);
  std::string while_body_label = fmt::format("while_body_{}", label_cursor);
  std::string while_end_label = fmt::format("while_end_{}", label_cursor);
  loop.cond_ =
      ::llvm::BasicBlock::Create(builder_->getContext(), while_cond_label, builder_->GetInsertBlock()->getParent());
  loop.end_ = ::llvm::BasicBlock::Create(*context_, while_end_label);
  loop.body_ = ::llvm::BasicBlock::Create(*context_, while_body_label);
  loop.func_ = builder_->GetInsertBlock()->getParent();
  current_func_->loop_blocks.emplace_back(std::make_pair(loop.cond_, loop.end_));
  if (builder_->GetInsertBlock()->getTerminator() == nullptr) {
    builder_->CreateBr(loop.cond_);
  }
  builder_->SetInsertPoint(loop.cond_);
  return loop;
}
void CodeGen::AddLoopCond(Loop loop, ::llvm::Value* cond) {
  builder_->CreateCondBr(cond, loop.body_, loop.end_);
  loop.body_->insertInto(loop.func_);
  builder_->SetInsertPoint(loop.body_);
}
void CodeGen::AddLoopCond(Loop loop, ValuePtr cond) { AddLoopCond(loop, cond->LoadValue()); }
void CodeGen::FinishLoop(Loop loop) {
  if (builder_->GetInsertBlock()->getTerminator() == nullptr) {
    builder_->CreateBr(loop.cond_);  // end while body
  }
  loop.end_->insertInto(loop.func_);
  builder_->SetInsertPoint(loop.end_);
  current_func_->loop_blocks.pop_back();
}

typename CodeGen::Condition CodeGen::NewCondition(size_t elif_count, bool has_else) {
  Condition condition;
  uint32_t label_cursor = GetLabelCursor();
  std::string if_block_label = fmt::format("if_block_{}", label_cursor);
  std::string continue_label = fmt::format("continue_{}", label_cursor);
  auto* current_func = builder_->GetInsertBlock()->getParent();
  condition.if_body = ::llvm::BasicBlock::Create(builder_->getContext(), if_block_label, current_func);

  for (size_t i = 0; i < elif_count; i++) {
    std::string elif_cond_label = fmt::format("elif_cond_{}_{}", label_cursor, i);
    ::llvm::BasicBlock* elif_cond_block =
        ::llvm::BasicBlock::Create(builder_->getContext(), elif_cond_label, current_func);
    condition.elif_conds.emplace_back(elif_cond_block);
    std::string elif_label = fmt::format("elif_{}_{}", label_cursor, i);
    ::llvm::BasicBlock* elif_block = ::llvm::BasicBlock::Create(builder_->getContext(), elif_label, current_func);
    condition.elif_bodies.emplace_back(elif_block);
  }
  if (has_else) {
    std::string else_label = fmt::format("else_{}", label_cursor);
    condition.else_body = ::llvm::BasicBlock::Create(builder_->getContext(), else_label, current_func);
  }
  condition.end = ::llvm::BasicBlock::Create(builder_->getContext(), continue_label, current_func);
  return condition;
}

void CodeGen::BeginIf(Condition condition, ValuePtr cond) {
  ::llvm::BasicBlock* if_next_block = condition.end;
  if (condition.elif_conds.size() > 0) {
    if_next_block = condition.elif_conds[0];
  } else if (condition.else_body != nullptr) {
    if_next_block = condition.else_body;
  }

  builder_->CreateCondBr(cond->LoadValue(), condition.if_body, if_next_block);
  builder_->SetInsertPoint(condition.if_body);
}
void CodeGen::EndIf(Condition condition) {
  // if (condition.if_body->getTerminator() == nullptr) {
  //   builder_->CreateBr(condition.end);
  // }

  if (builder_->GetInsertBlock()->getTerminator() == nullptr) {
    builder_->CreateBr(condition.end);
  }
}
void CodeGen::BeginElif(Condition condition, size_t i) { builder_->SetInsertPoint(condition.elif_conds[i]); }
void CodeGen::EndElifCond(Condition condition, size_t i, ValuePtr cond) {
  ::llvm::BasicBlock* next_block = condition.end;
  if (i < condition.elif_conds.size() - 1) {
    next_block = condition.elif_conds[i + 1];
  } else if (condition.else_body != nullptr) {
    next_block = condition.else_body;
  }
  builder_->CreateCondBr(cond->LoadValue(), condition.elif_bodies[i], next_block);
  builder_->SetInsertPoint(condition.elif_bodies[i]);
}
void CodeGen::EndElif(Condition condition, size_t i) {
  if (condition.elif_bodies[i]->getTerminator() == nullptr) {
    builder_->CreateBr(condition.end);  // end elif
  }
}
void CodeGen::BeginElse(Condition condition) { builder_->SetInsertPoint(condition.else_body); }
void CodeGen::EndElse(Condition condition) {
  if (condition.else_body->getTerminator() == nullptr) {
    builder_->CreateBr(condition.end);  // end else
  }
}
void CodeGen::FinishCondition(Condition condition) {
  if (!condition.end->hasNPredecessorsOrMore(1)) {
    condition.end->removeFromParent();
  } else {
    builder_->SetInsertPoint(condition.end);
  }
}

absl::StatusOr<::llvm::Value*> CodeGen::CallFunction(const std::string& name,
                                                     const std::vector<::llvm::Value*>& arg_values) {
  ::llvm::Function* found_func = nullptr;
  FunctionDesc found_func_desc;
  ExternFunctionPtr func = GetFunction(name);
  if (func) {
    found_func_desc = func->desc;
    found_func = func->func;
  } else {
    auto found = funcs_.find(name);
    if (found != funcs_.end()) {
      found_func_desc = found->second->desc;
      found_func = found->second->func;
    }
  }
  if (!found_func) {
    RUDF_LOG_RETURN_FMT_ERROR("CallFunction:No func:{} found", name);
  }
  ::llvm::FunctionCallee callee(found_func);
  ::llvm::Value* result = builder_->CreateCall(callee, arg_values);
  return result;
}

absl::StatusOr<ValuePtr> CodeGen::CallFunction(const std::string& name, const std::vector<ValuePtr>& const_arg_values) {
  ::llvm::Function* found_func = nullptr;
  FunctionDesc found_func_desc;
  ExternFunctionPtr func = GetFunction(name);
  if (func) {
    found_func_desc = func->desc;
    found_func = func->func;
  } else {
    auto found = funcs_.find(name);
    if (found != funcs_.end()) {
      found_func_desc = found->second->desc;
      found_func = found->second->func;
    }
  }
  if (!found_func) {
    RUDF_LOG_RETURN_FMT_ERROR("CallFunction:No func:{} found", name);
  }
  std::vector<ValuePtr> arg_values = const_arg_values;
  if (found_func_desc.context_arg_idx >= 0 && arg_values.size() == found_func_desc.arg_types.size() - 1) {
    if (current_func_->context_arg_value) {
      arg_values.insert(arg_values.begin() + found_func_desc.context_arg_idx, current_func_->context_arg_value);
    }
  }

  if (arg_values.size() != found_func_desc.arg_types.size()) {
    RUDF_LOG_RETURN_FMT_ERROR("Func:{} expect {} args, while {} given", name, found_func_desc.arg_types.size(),
                              arg_values.size());
  }

  std::vector<::llvm::Value*> arg_vals(arg_values.size());
  std::vector<::llvm::Value*> byval_args;

  for (size_t i = 0; i < arg_values.size(); i++) {
    ValuePtr arg_val = arg_values[i];
    if (arg_val->GetDType() != found_func_desc.arg_types[i]) {
      DType src_dtype = arg_val->GetDType();
      auto cast_result = CastTo(arg_val, found_func_desc.arg_types[i]);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      arg_val = cast_result.value();
    }
    if (found_func_desc.PassArgByValue(i)) {
      arg_vals[i] = arg_val->GetPtrValue();
      if (arg_vals[i] == nullptr) {
        auto* arg_type = get_type(*context_, arg_val->GetDType());
        auto* arg_tmp_ptr_val = builder_->CreateAlloca(arg_type);
        builder_->CreateStore(arg_val->LoadValue(), arg_tmp_ptr_val);
        arg_vals[i] = arg_tmp_ptr_val;
      }
      byval_args.emplace_back(arg_vals[i]);
    } else {
      arg_vals[i] = arg_val->LoadValue();
    }
  }
  // std::vector<::llvm::OperandBundleDef> operand_bundles{::llvm::OperandBundleDef("", arg_vals)};
  std::vector<::llvm::OperandBundleDef> operand_bundles;
  if (!byval_args.empty()) {
    operand_bundles.emplace_back(::llvm::OperandBundleDef("byval", byval_args));
  }
  ::llvm::FunctionCallee callee(found_func);
  ::llvm::Value* result = builder_->CreateCall(callee, arg_vals, operand_bundles);
  ::llvm::Type* return_val_type = nullptr;
  DType return_type = found_func_desc.return_type;

  if (return_type.IsPtr()) {
    auto ret_type_ptr_to = return_type.PtrTo();
    if (ret_type_ptr_to.IsInteger() || ret_type_ptr_to.IsFloat()) {
      return_val_type = GetType(ret_type_ptr_to).value();
      return_type = return_type.PtrTo();
    }
  } else if (return_type.IsSimdVector()) {
    return_val_type = GetType(return_type).value();
    auto* vector_ptr = builder_->CreateAlloca(return_val_type);
    builder_->CreateStore(result, vector_ptr);
    result = vector_ptr;
  }
  return NewValue(return_type, result, return_val_type);
}

void CodeGen::Store(::llvm::Value* val, ::llvm::Value* ptr) { builder_->CreateStore(val, ptr); }

::llvm::Value* CodeGen::Load(::llvm::Type* typ, ::llvm::Value* ptr) { return builder_->CreateLoad(typ, ptr); }

}  // namespace compiler
}  // namespace rapidudf