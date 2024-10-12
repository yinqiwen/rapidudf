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
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/ast/symbols.h"
#include "rapidudf/compiler/codegen.h"
#include "rapidudf/functions/functions.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {
JitCompiler::JitCompiler(const Options& opts) : opts_(opts) {
  functions::init_builtin();
  ast::Symbols::Init();
}

absl::StatusOr<std::vector<std::string>> JitCompiler::CompileSource(const std::string& source, bool dump_asm) {
  std::lock_guard<std::mutex> guard(jit_mutex_);
  NewCodegen(dump_asm);
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

void JitCompiler::NewCodegen(bool print_asm) {
  ast_ctx_.Clear();
  codegen_ = std::make_shared<CodeGen>(opts_, print_asm);
  stat_.Clear();
  parsed_ast_funcs_.clear();
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