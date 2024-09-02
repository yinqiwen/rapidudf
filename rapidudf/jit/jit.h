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
#include <setjmp.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "rapidudf/ast/block.h"
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/expression.h"
#include "rapidudf/ast/function.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/codegen/code_generator.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/value.h"
#include "rapidudf/log/log.h"
namespace rapidudf {

class JitCompiler;
template <typename RET, typename... Args>
class JitFunction {
 public:
  JitFunction(const std::string& name, std::unique_ptr<CodeGenerator>&& code_gen,
              std::vector<std::unique_ptr<std::string>>&& const_vals)
      : name_(name), code_gen_(std::move(code_gen)), const_strings_(std::move(const_vals)) {
    f_ = reinterpret_cast<RET (*)(Args...)>(const_cast<uint8_t*>(code_gen_->GetCodeGen().getCode()));
  }
  JitFunction(JitFunction&& other) {
    name_ = std::move(other.name_);
    code_gen_ = std::move(other.code_gen_);
    const_strings_ = std::move(other.const_strings_);
    f_ = other.f_;
    rethrow_ = other.rethrow_;
    unsafe_ = other.unsafe_;
  }
  ~JitFunction() {}
  JitFunction(const JitFunction&) = delete;
  JitFunction& operator=(const JitFunction&) = delete;

  const std::string& GetName() const { return name_; }

  void SetRethrowException(bool v = true) { rethrow_ = v; }
  void SetUnsafe(bool v = true) { unsafe_ = v; }

  RET UnsafeCall(Args... args) {
    if constexpr (std::is_same_v<void, RET>) {
      f_(args...);
    } else {
      RET r = f_(args...);
      return r;
    }
  }

  RET SafeCall(Args... args) {
    auto& func_ctx = FunctionCallContext::Get(true);
    if (func_ctx.invoke_frame_id == 1) {  // first
      if (setjmp(func_ctx.jmp_env) == 0) {
        if constexpr (std::is_same_v<void, RET>) {
          f_(args...);
          func_ctx.invoke_frame_id = 0;
        } else {
          RET r = f_(args...);
          func_ctx.invoke_frame_id = 0;
          return r;
        }
      } else {
        func_ctx.invoke_frame_id = 0;
        if (rethrow_) {
          throw func_ctx.run_ex;
        }
        RUDF_DEBUG("JitFunction exception captured, return default value.");
        if constexpr (std::is_same_v<void, RET>) {
        } else {
          return {};
        }
      }
    } else {
      if constexpr (std::is_same_v<void, RET>) {
        f_(args...);
      } else {
        RET r = f_(args...);
        return r;
      }
    }
  }

  RET operator()(Args... args) {
    if (unsafe_) {
      return UnsafeCall(args...);
    } else {
      return SafeCall(args...);
    }
  }

 private:
  std::string name_;
  std::unique_ptr<CodeGenerator> code_gen_;
  std::vector<std::unique_ptr<std::string>> const_strings_;
  RET (*f_)(Args...) = nullptr;
  bool unsafe_ = false;
  bool rethrow_ = false;
  friend class JitCompiler;
};

class JitCompiler {
 public:
  explicit JitCompiler(size_t max_size = 4096, bool use_register = true);

  absl::StatusOr<std::vector<std::string>> CompileSource(const std::string& source, bool dump_asm = false);

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> LoadFunction(const std::string& name) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    for (auto& ctx : compile_ctxs_) {
      if (ctx.desc.name == name) {
        if (!ctx.code_gen) {
          RUDF_LOG_ERROR_STATUS(absl::NotFoundError(fmt::format("Func:{} already loaded before.", name)));
        }
        std::string err;
        if (!ctx.func_ast.CompareSignature(return_type, arg_types, err)) {
          RUDF_ERROR("{}", err);
          return absl::InvalidArgumentError(err);
        }
        return JitFunction<RET, Args...>(ctx.desc.name, std::move(ctx.code_gen), std::move(ctx.const_strings));
      }
    }
    return absl::NotFoundError(fmt::format("No function:{} found in compuled functions.", name));
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileFunction(const std::string& source, bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    // ast_funcs_.clear();
    compile_ctxs_.clear();
    auto status = DoCompileFunction(source);
    if (!status.ok()) {
      return status;
    }
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    std::string err;
    if (!compile_ctxs_[0].func_ast.CompareSignature(return_type, arg_types, err)) {
      RUDF_ERROR("{}", err);
      return absl::InvalidArgumentError(err);
    }
    if (dump_asm) {
      GetCodeGenerator().DumpAsm();
    }
    auto& ctx = compile_ctxs_[compile_function_idx_];
    return JitFunction<RET, Args...>(ctx.desc.name, std::move(ctx.code_gen), std::move(ctx.const_strings));
  }

  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> CompileExpression(const std::string& source,
                                                              const std::vector<std::string>& arg_names,
                                                              bool dump_asm = false) {
    std::lock_guard<std::mutex> guard(jit_mutex_);
    ast_ctx_.Clear();
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
    for (size_t i = 0; i < arg_names.size(); i++) {
      if (!ast_ctx_.AddLocalVar(arg_names[i], arg_types[i])) {
        return absl::InvalidArgumentError(fmt::format("Duplicate arg name:{}", arg_names[i]));
      }
      ast::FunctionArg ast_arg;
      ast_arg.dtype = arg_types[i];
      ast_arg.name = arg_names[i];
      gen_func_ast.args->emplace_back(ast_arg);
    }
    auto status = DoCompileExpression(source, gen_func_ast);
    if (!status.ok()) {
      return status;
    }
    if (dump_asm) {
      GetCodeGenerator().DumpAsm();
    }
    auto& ctx = compile_ctxs_[compile_function_idx_];
    return JitFunction<RET, Args...>(ctx.desc.name, std::move(ctx.code_gen), std::move(ctx.const_strings));
  }

 private:
  static constexpr std::string_view kFuncExistLabel = "func_exit";
  struct CompileContext {
    ast::Function func_ast;
    std::unique_ptr<CodeGenerator> code_gen;
    std::vector<std::unique_ptr<std::string>> const_strings;
    std::unordered_map<std::string, ValuePtr> local_vars;
    FunctionDesc desc;
  };
  const FunctionDesc* GetFunction(const std::string& name);
  CodeGenerator& GetCodeGenerator() { return *compile_ctxs_[compile_function_idx_].code_gen; }
  CompileContext& GetCompileContext() { return compile_ctxs_[compile_function_idx_]; }

  absl::Status DoCompileFunctionAst(CompileContext& ctx);
  absl::Status DoCompileFunction(const std::string& source);
  absl::Status DoCompileExpression(const std::string& source, ast::Function& gen_func_ast);

  absl::Status CompileBody(const ast::Block& block);
  absl::Status CompileStatement(const ast::ReturnStatement& statement);
  absl::Status CompileStatement(const ast::IfElseStatement& statement);
  absl::Status CompileStatement(const ast::WhileStatement& statement);
  absl::Status CompileStatement(const ast::ExpressionStatement& statement);

  absl::StatusOr<ValuePtr> CompileExpression(ast::BinaryExprPtr expr);
  absl::StatusOr<ValuePtr> CompileExpression(const ast::VarAccessor& expr);
  absl::StatusOr<ValuePtr> CompileExpression(const ast::VarDefine& expr);
  absl::StatusOr<ValuePtr> CompileExpression(ast::UnaryExprPtr expr);
  absl::StatusOr<ValuePtr> CompileOperand(const ast::Operand& expr);

  absl::StatusOr<ValuePtr> CompileConstants(double v);
  absl::StatusOr<ValuePtr> CompileConstants(int64_t v);
  absl::StatusOr<ValuePtr> CompileConstants(bool v);
  absl::StatusOr<ValuePtr> CompileConstants(const std::string& v);

  absl::StatusOr<ValuePtr> CompileFieldAccess(ValuePtr var, const ast::FieldAccess& field);
  absl::StatusOr<ValuePtr> CompileJsonAccess(ValuePtr var, uint32_t idx);
  absl::StatusOr<ValuePtr> CompileJsonAccess(ValuePtr var, const std::string& key);
  absl::StatusOr<ValuePtr> CompileJsonAccess(ValuePtr var, const ast::VarRef& key);
  absl::StatusOr<ValuePtr> GetLocalVar(const std::string& name);

  absl::StatusOr<ValuePtr> CallFunction(const std::string& name, std::vector<ValuePtr>& args);

  void AddCompileContex(const ast::Function& func_ast);

  std::mutex jit_mutex_;
  // std::unique_ptr<CodeGenerator> code_gen_;

  uint32_t max_code_size_ = 4096;
  bool use_registers_ = false;
  ast::ParseContext ast_ctx_;
  std::vector<CompileContext> compile_ctxs_;

  uint32_t compile_function_idx_ = 0;
  int label_cursor_ = 0;
};
}  // namespace rapidudf