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
#include "rapidudf/compiler/compiler.h"

#include <string_view>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/ast/symbols.h"
#include "rapidudf/compiler/codegen.h"
#include "rapidudf/functions/functions.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {
JitCompiler::JitCompiler(Options opts) : opts_(opts) {
  functions::init_builtin();
  ast::Symbols::Init();
}

absl::StatusOr<std::vector<std::string>> JitCompiler::CompileSource(const std::string& source) {
  std::lock_guard<std::mutex> guard(jit_mutex_);
  NewCodegen();
  auto funcs = ast::parse_functions_ast(ast_ctx_, source);
  if (!funcs.ok()) {
    RUDF_LOG_ERROR_STATUS(funcs.status());
  }
  stat_.parse_cost = ast_ctx_.GetParseCost();
  stat_.parse_validate_cost = ast_ctx_.GetParseValidateCost();

  std::vector<std::string> fnames;
  for (auto& func : funcs.value()) {
    fnames.emplace_back(func.name);
  }
  auto status = CompileFunctions(funcs.value());
  if (!status.ok()) {
    return status;
  }
  return fnames;
}

absl::Status JitCompiler::CompileFunction(const std::string& source) {
  auto f = ast::parse_function_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }

  stat_.parse_cost = ast_ctx_.GetParseCost();
  stat_.parse_validate_cost = ast_ctx_.GetParseValidateCost();
  auto start_time = std::chrono::high_resolution_clock::now();
  auto status = CompileFunction(f.value());
  auto compile_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  stat_.parse_cost = compile_duration;
  return status;
}

absl::Status JitCompiler::CompileFunctions(const std::vector<ast::Function>& functions) {
  parsed_ast_funcs_ = functions;
  auto start_time = std::chrono::high_resolution_clock::now();
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
  auto status = codegen_->DeclareExternFunctions(all_func_calls, all_member_func_calls);
  if (!status.ok()) {
    return status;
  }
  //   auto throw_func = FunctionFactory::GetFunction(std::string(k_throw_size_exception_func));
  //   if (nullptr != throw_func) {
  //     all_func_calls[std::string(k_throw_size_exception_func)] = throw_func;
  //   }

  for (auto& func : functions) {
    auto status = BuildIR(func);
    RUDF_LOG_RETURN_ERROR_STATUS(status);
  }
  stat_.ir_build_cost =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  start_time = std::chrono::high_resolution_clock::now();
  status = Compile();
  stat_.compile_cost =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  return status;
}

absl::Status JitCompiler::CompileExpression(const std::string& expr, ast::Function& function) {
  auto f = ast::parse_expression_ast(ast_ctx_, expr, function.ToFuncDesc());
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }
  stat_.parse_cost = ast_ctx_.GetParseCost();
  stat_.parse_validate_cost = ast_ctx_.GetParseValidateCost();
  ast::ReturnStatement return_statement;
  return_statement.expr = f->expr;
  return_statement.rpn = f->rpn_expr;
  function.body.statements.emplace_back(return_statement);
  auto status = CompileFunctions(std::vector<ast::Function>{function});
  return status;
}

absl::Status JitCompiler::CompileFunction(const ast::Function& function) {
  return CompileFunctions(std::vector<ast::Function>{function});
}

void JitCompiler::NewCodegen() {
  ast_ctx_.Clear();
  codegen_ = std::make_shared<CodeGen>(opts_);
  stat_.Clear();
  parsed_ast_funcs_.clear();
}

uint64_t JitCompiler::ComputeCompileCacheKey(std::string_view source, DType return_type,
                                             const std::vector<DType>& arg_types,
                                             const std::vector<Arg>* dyn_args) const {
  if (dyn_args == nullptr || dyn_args->empty()) {
    return absl::HashOf(source, opts_.optimize_level, opts_.fast_math, opts_.skip_auto_vectorize_passes,
                        opts_.enable_compile_cache, return_type, arg_types);
  }
  std::vector<std::pair<std::string_view, std::string_view>> dyn_arg_views;
  dyn_arg_views.reserve(dyn_args->size());
  for (const Arg& arg : *dyn_args) {
    dyn_arg_views.emplace_back(arg.name, std::string_view(arg.schema));
  }
  return absl::HashOf(source, opts_.optimize_level, opts_.fast_math, opts_.skip_auto_vectorize_passes,
                      opts_.enable_compile_cache, return_type, arg_types, dyn_arg_views);
}

void JitCompiler::StoreCompileCache(std::string_view source, DType return_type, const std::vector<DType>& arg_types,
                                    const std::vector<Arg>* dyn_args, const std::string& function_name) {
  if (!opts_.enable_compile_cache || codegen_ == nullptr) {
    return;
  }
  CompileCacheEntry entry;
  entry.codegen = codegen_;
  entry.parsed_ast_funcs = parsed_ast_funcs_;
  entry.stat = stat_;
  entry.function_name = function_name;
  compile_cache_[ComputeCompileCacheKey(source, return_type, arg_types, dyn_args)] = std::move(entry);
}

bool JitCompiler::LookupCompileCache(uint64_t key, DType return_type, const std::vector<DType>& arg_types,
                                     CachedJitTarget* out) {
  if (!opts_.enable_compile_cache || out == nullptr) {
    return false;
  }
  auto found = compile_cache_.find(key);
  if (found == compile_cache_.end()) {
    return false;
  }
  CompileCacheEntry& entry = found->second;
  parsed_ast_funcs_ = entry.parsed_ast_funcs;
  stat_ = entry.stat;
  codegen_ = entry.codegen;

  std::string err;
  for (auto& func : parsed_ast_funcs_) {
    if (func.name != entry.function_name) {
      continue;
    }
    if (!func.CompareSignature(return_type, arg_types, err)) {
      return false;
    }
    auto func_ptr_result = entry.codegen->GetFunctionPtr(entry.function_name);
    if (!func_ptr_result.ok()) {
      return false;
    }
    out->func_ptr = func_ptr_result.value();
    out->codegen = entry.codegen;
    out->stat = entry.stat;
    out->function_name = entry.function_name;
    return true;
  }
  return false;
}
absl::Status JitCompiler::Compile() { return codegen_->Finish(); }

absl::StatusOr<void*> JitCompiler::GetFunctionPtr(const std::string& name) {
  if (!codegen_) {
    return absl::InvalidArgumentError("null compiled session to get function ptr");
  }
  return codegen_->GetFunctionPtr(name);
}

absl::StatusOr<std::string> JitCompiler::VerifyFunctionSignature(const std::string& name, DType return_type,
                                                                 const std::vector<DType>& args_types) {
  for (auto& func : parsed_ast_funcs_) {
    if (name == func.name) {
      std::string err;
      if (!func.CompareSignature(return_type, args_types, err)) {
      }
      return func.name;
    }
  }
  return absl::NotFoundError(fmt::format("No function:{} found in compiled functions.", name));
}

absl::StatusOr<std::string> JitCompiler::VerifyFunctionSignature(DType return_type,
                                                                 const std::vector<DType>& args_types) {
  return VerifyFunctionSignature(parsed_ast_funcs_[0].name, return_type, args_types);
}

absl::Status JitCompiler::ThrowVectorExprError(const std::string& msg) {
  int line = ast_ctx_.GetLineNo();
  std::string src_line = ast_ctx_.GetSourceLine(line);
  auto src_line_val = codegen_->NewStringView(src_line);
  auto line_val = codegen_->NewU32(line);
  auto msg_val = codegen_->NewStringView(msg);
  auto result = codegen_->CallFunction(functions::kBuiltinThrowVectorExprEx, {line_val, src_line_val, msg_val});
  if (!result.ok()) {
    return result.status();
  }
  return absl::OkStatus();
}

absl::Status JitCompiler::BuildIR(const ast::Function& function) {
  std::vector<std::string> func_arg_names;
  if (function.args.has_value()) {
    for (auto& arg : *function.args) {
      func_arg_names.emplace_back(arg.name);
    }
  }
  FunctionDesc desc = function.ToFuncDesc();
  auto status = codegen_->DefineFunction(desc, func_arg_names);
  if (!status.ok()) {
    return status;
  }

  status = BuildIR(function.body);
  if (!status.ok()) {
    return status;
  }

  return codegen_->FinishFunction();
}

absl::Status JitCompiler::BuildIR(const ast::Block& block) { return BuildIR(block.statements); }
}  // namespace compiler
}  // namespace rapidudf