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
#include "rapidudf/jit/xbyak/jit.h"
#include <exception>
#include <memory>
#include <vector>

#include "rapidudf/ast/function.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/ast/symbols.h"
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/jit/xbyak/function.h"
#include "rapidudf/jit/xbyak/register.h"
#include "rapidudf/jit/xbyak/value.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

#include "rapidudf/log/log.h"

#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#endif

namespace rapidudf {
namespace xbyak {
JitCompiler::JitCompiler(Options opts) : opts_(opts) {
  init_builtin();
  ast::Symbols::Init();
}

void JitCompiler::AddCompileContex(const ast::Function& func_ast) {
  CompileContext ctx;
  ctx.func_ast = func_ast;
  typename CodeGenerator::Options codegen_opts;
  codegen_opts.max_code_size = opts_.max_code_size;
  codegen_opts.use_registers = opts_.use_registers;
  if (ast_ctx_.GetAllFuncCalls(compile_function_idx_).size() > 0 ||
      ast_ctx_.GetAllImplicitFuncCalls(compile_function_idx_).size() > 0) {
    codegen_opts.use_callee_saved_registers = true;
  } else {
    codegen_opts.use_callee_saved_registers = false;
  }
  ctx.code_gen = std::make_unique<CodeGenerator>(codegen_opts);
  compile_ctxs_.emplace_back(std::move(ctx));
}

const FunctionDesc* JitCompiler::GetFunction(const std::string& name) {
  for (uint32_t i = 0; i < compile_function_idx_; i++) {
    if (compile_ctxs_[i].desc.name == name) {
      return &compile_ctxs_[i].desc;
    }
  }
  if (name == compile_ctxs_[compile_function_idx_].desc.name) {
    // self call
    const uint8_t* code = GetCodeGenerator().GetCodeGen().getCode();
    compile_ctxs_[compile_function_idx_].desc.func = const_cast<void*>(reinterpret_cast<const void*>(code));
    return &compile_ctxs_[compile_function_idx_].desc;
  }
  return FunctionFactory::GetFunction(name);
}

absl::StatusOr<ValuePtr> JitCompiler::GetLocalVar(const std::string& name) {
  auto& local_vars = compile_ctxs_[compile_function_idx_].local_vars;
  auto found = local_vars.find(name);
  if (found == local_vars.end()) {
    RUDF_LOG_ERROR_STATUS(absl::NotFoundError(fmt::format("No var:{} found.", name)));
  }
  return found->second;
}

absl::StatusOr<std::vector<std::string>> JitCompiler::CompileSource(const std::string& source, bool dump_asm) {
  std::lock_guard<std::mutex> guard(jit_mutex_);
  compile_ctxs_.clear();
  auto f = ast::parse_functions_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }
  for (auto& func : f.value()) {
    AddCompileContex(func);
  }
  std::vector<std::string> func_names;
  for (uint32_t i = 0; i < compile_ctxs_.size(); i++) {
    compile_function_idx_ = i;
    auto status = DoCompileFunctionAst(compile_ctxs_[i]);
    if (!status.ok()) {
      return status;
    }
    if (dump_asm) {
      GetCodeGenerator().DumpAsm();
    }
    func_names.emplace_back(compile_ctxs_[i].desc.name);
  }
  return func_names;
}

absl::Status JitCompiler::DoCompileFunctionAst(CompileContext& ctx) {
  ctx.local_vars.clear();
  std::vector<FuncArgRegister> all_func_arg_registers;
  auto all_func_calls = ast_ctx_.GetAllFuncCalls(compile_function_idx_);
  auto implicit_func_calls = ast_ctx_.GetAllImplicitFuncCalls(compile_function_idx_);
  all_func_calls.merge(implicit_func_calls);
  for (const auto& [name, func_call] : all_func_calls) {
    auto arg_registers = GetFuncArgsRegistersByDTypes(func_call->arg_types);
    RUDF_DEBUG("Normal funcation call:{} with registers:{}", name, arg_registers.size());
    if (arg_registers.empty() && !func_call->arg_types.empty()) {
      RUDF_LOG_ERROR_STATUS(absl::NotFoundError(fmt::format("Can NOT allocate registers for func:{}.", name)));
    }
    all_func_arg_registers.insert(all_func_arg_registers.end(), arg_registers.begin(), arg_registers.end());
    if (func_call->attrs.UseArenaAllocator()) {
      ctx.use_arena_allocator = true;
    }
  }

  // for (const auto& builtin_func_call : ast_ctx_.GetAllBuiltinFuncCalls(compile_function_idx_)) {
  //   RUDF_DEBUG("Buitin funcation call:{}", builtin_func_call);
  //   auto* func_call = FunctionFactory::GetFunction(builtin_func_call);
  //   if (nullptr == func_call) {
  //     RUDF_LOG_ERROR_STATUS(absl::NotFoundError(fmt::format("No buitlin func:{} found.", builtin_func_call)));
  //   }
  //   if (func_call->attrs.UseArenaAllocator()) {
  //     ctx.use_arena_allocator = true;
  //   }
  //   auto arg_registers = GetFuncArgsRegistersByDTypes(func_call->arg_types);
  //   if (arg_registers.empty() && !func_call->arg_types.empty()) {
  //     RUDF_LOG_ERROR_STATUS(
  //         absl::NotFoundError(fmt::format("Can NOT allocate registers for func:{}.", builtin_func_call)));
  //   }
  //   all_func_arg_registers.insert(all_func_arg_registers.end(), arg_registers.begin(), arg_registers.end());
  // }

  auto unused_registers = GetUnuseFuncArgsRegisters(all_func_arg_registers);
  GetCodeGenerator().AddFreeRegisters(unused_registers);

  ctx.desc = ctx.func_ast.ToFuncDesc();
  // ctx.has_simd_vector_operations = ast_ctx_.HasSimdVectorOperation(compile_function_idx_);

  auto arg_registers = GetFuncArgsRegistersByDTypes(ctx.desc.arg_types);
  if (!ctx.desc.arg_types.empty() && arg_registers.empty()) {
    return absl::InvalidArgumentError("Can NOT use registers for all func args.");
  }
  std::vector<RegisterId> exclude_regs;
  for (auto& regs : arg_registers) {
    for (auto reg : regs) {
      exclude_regs.emplace_back(RegisterId(*reg));
    }
  }
  for (size_t i = 0; i < ctx.desc.arg_types.size(); i++) {
    auto var = GetCodeGenerator().NewValue(ctx.desc.arg_types[i], exclude_regs, false);
    auto reg_var = Value::New(&GetCodeGenerator(), ctx.desc.arg_types[i], arg_registers[i], false);
    if (0 != var->Copy(*reg_var)) {
      RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(
          fmt::format("Faield to copy arg register for arg:{}", ctx.func_ast.args->at(i).name)));
    }
    var->SetVarName(ctx.func_ast.args->at(i).name);
    ctx.local_vars.emplace(ctx.func_ast.args->at(i).name, var);
  }
  RUDF_DEBUG("Func args compiled success.");
  try {
    auto rc = CompileBody(ctx.func_ast.body);
    if (!rc.ok()) {
      return rc;
    }
  } catch (std::exception& ex) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("Compile function body failed with exception:{}", ex.what())));
  } catch (...) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("Compile function body failed with unknow exception."));
  }

  GetCodeGenerator().Label(std::string(kFuncExistLabel));
  GetCodeGenerator().GetCodeGen().nop();
  GetCodeGenerator().Finish();

  const uint8_t* code = GetCodeGenerator().GetCodeGen().getCode();
  ctx.desc.func = const_cast<void*>(reinterpret_cast<const void*>(code));
  return absl::OkStatus();
}

absl::Status JitCompiler::DoCompileFunction(const std::string& source) {
  auto f = ast::parse_function_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_LOG_ERROR_STATUS(f.status());
  }
  AddCompileContex(f.value());
  // ast_funcs_.emplace_back(f.value());
  return DoCompileFunctionAst(compile_ctxs_[0]);
}

absl::Status JitCompiler::DoCompileExpression(const std::string& source, ast::Function& gen_func_ast) {
  // code_gen_ = std::make_unique<CodeGenerator>(max_code_size_, use_registers_);
  auto f = ast::parse_expression_ast(ast_ctx_, source);
  if (!f.ok()) {
    RUDF_DEBUG("x:{}", ast_ctx_.IsVarExist("x", false).ok());
    RUDF_LOG_ERROR_STATUS(f.status());
  }
  // RUDF_INFO("after ast parse, {}", ast_ctx_.)
  ast::ReturnStatement return_statement;
  return_statement.expr = *f;
  gen_func_ast.body.statements.emplace_back(return_statement);
  AddCompileContex(gen_func_ast);

  return DoCompileFunctionAst(compile_ctxs_[0]);
}

absl::Status JitCompiler::CompileBody(const ast::Block& block) {
  for (auto& statement : block.statements) {
    auto rc = std::visit([&](auto&& arg) { return CompileStatement(arg); }, statement);
    if (!rc.ok()) {
      return rc;
    }
  }
  return absl::OkStatus();
}
}  // namespace xbyak
}  // namespace rapidudf