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
#include <fmt/core.h>
#include <variant>
#include "absl/cleanup/cleanup.h"
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/ops/cmp.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
namespace rapidudf {

absl::StatusOr<ValuePtr> JitCompiler::CompileOperand(const ast::Operand& expr) {
  // RUDF_DEBUG("CompileOperand readonly:{}", flags);
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, bool> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, std::string>) {
          return CompileConstants(arg);
        } else if constexpr (std::is_same_v<T, ast::VarAccessor> || std::is_same_v<T, ast::VarDefine>) {
          return CompileExpression(arg);
        } else if constexpr (std::is_same_v<T, ast::BinaryExprPtr> || std::is_same_v<T, ast::UnaryExprPtr>) {
          return CompileExpression(arg);
        } else {
          static_assert(sizeof(arg) == -1, "non-exhaustive visitor!");
          ValuePtr empty;
          return empty;
        }
      },
      expr);
}
absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(ast::BinaryExprPtr expr) {
  ast_ctx_.SetPosition(expr->position);
  auto left_result = CompileOperand(expr->left);
  if (!left_result.ok()) {
    return left_result.status();
  }
  auto left = left_result.value();
  for (auto& [op, right_operand] : expr->right) {
    auto right_result = CompileOperand(right_operand);
    if (!right_result.ok()) {
      return right_result.status();
    }
    auto right = right_result.value();
    if (op != OP_ASSIGN) {
      if (left->GetDType().IsVoid() || right->GetDType().IsVoid()) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format(
            "Can NOT do op:{} with void operands, while left:{}, right:{}", op, left->GetDType(), right->GetDType())));
      }
    }
    RUDF_DEBUG("Start compile expression statement:{}", op);
    switch (op) {
      case OP_PLUS:
      case OP_MINUS:
      case OP_MULTIPLY:
      case OP_DIVIDE:
      case OP_MOD: {
        auto result =
            GetCodeGenerator().NewValue(left->GetDType() > right->GetDType() ? left->GetDType() : right->GetDType());
        int rc = left->ArithmeticOp(op, *right, result);
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
        // return result;
        GetCodeGenerator().DropTmpValue(left);
        left = result;
        break;
      }
      case OP_ASSIGN: {
        // if (!left->GetVarName().empty() && right->IsTemp()) {
        //   left->Swap(*right);
        //   code_gen_.DropTmpValue(right);
        //   return left;
        // }
        if (left->GetVarName().empty()) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT assgin value to non var value.")));
        }
        // auto right_val = right->CastTo(left->GetDType());

        int rc = left->CastToInplace(right->GetDType());
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} cast with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
        rc = left->Copy(*right);
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do oassign with left:{}, right:{}", left->GetDType(), right->GetDType())));
        }
        GetCodeGenerator().DropTmpValue(right);
        // GetCodeGenerator().DropTmpValue(right_val);
        // return left;
        break;
      }
      case OP_EQUAL:
      case OP_NOT_EQUAL:
      case OP_LESS:
      case OP_LESS_EQUAL:
      case OP_GREATER:
      case OP_GREATER_EQUAL: {
        auto result = GetCodeGenerator().NewValue(DATA_U8, {}, true);
        int rc = left->Cmp(op, *right, result);
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
        GetCodeGenerator().DropTmpValue(left);
        GetCodeGenerator().DropTmpValue(right);
        // return result;
        left = result;
        break;
      }
      case OP_LOGIC_AND:
      case OP_LOGIC_OR: {
        uint32_t logic_label_cursor = label_cursor_++;
        std::string fast_exit_label = fmt::format("logic_fast_exit_{}", logic_label_cursor);
        std::string normal_exit_label = fmt::format("logic_normal_exit_{}", logic_label_cursor);
        auto result = GetCodeGenerator().NewValue(DATA_U8, {}, true);
        uint64_t fast_exit_val_bin = op == OP_LOGIC_AND ? 0 : 1;
        auto fast_cmp_val = Value::New(&GetCodeGenerator(), DATA_U8, fast_exit_val_bin);
        int rc = left->Cmp(OP_EQUAL, *fast_cmp_val, nullptr);
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
        GetCodeGenerator().Jump(fast_exit_label, OP_EQUAL);
        rc = left->LogicOp(op, *right, result);
        if (0 != rc) {
          RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
              fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
        }
        GetCodeGenerator().Jump(normal_exit_label);

        GetCodeGenerator().Label(fast_exit_label);
        result->Set(fast_exit_val_bin);

        GetCodeGenerator().Label(normal_exit_label);
        GetCodeGenerator().GetCodeGen().nop();
        GetCodeGenerator().DropTmpValue(left);
        GetCodeGenerator().DropTmpValue(right);
        // return result;
        left = result;
        break;
      }
      default: {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
            fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
      }
    }
  }
  return left;

  // if (expr->right.has_value()) {
  //   auto& [op, right_operand] = *(expr->right);
  //   auto right_result = CompileOperand(right_operand);
  //   if (!right_result.ok()) {
  //     return right_result.status();
  //   }
  //   auto right = right_result.value();
  //   if (op != OP_ASSIGN) {
  //     if (left->GetDType().IsVoid() || right->GetDType().IsVoid()) {
  //       RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format(
  //           "Can NOT do op:{} with void operands, while left:{}, right:{}", op, left->GetDType(),
  //           right->GetDType())));
  //     }
  //   }

  //   RUDF_DEBUG("Start compile expression statement:{}", op);
  //   switch (op) {
  //     case OP_PLUS:
  //     case OP_MINUS:
  //     case OP_MULTIPLY:
  //     case OP_DIVIDE:
  //     case OP_MOD: {
  //       auto result =
  //           GetCodeGenerator().NewValue(left->GetDType() > right->GetDType() ? left->GetDType() : right->GetDType());
  //       int rc = left->ArithmeticOp(op, *right, result);
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //       }
  //       return result;
  //     }
  //     case OP_ASSIGN: {
  //       // if (!left->GetVarName().empty() && right->IsTemp()) {
  //       //   left->Swap(*right);
  //       //   code_gen_.DropTmpValue(right);
  //       //   return left;
  //       // }
  //       if (left->GetVarName().empty()) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT assgin value to non var value.")));
  //       }
  //       // auto right_val = right->CastTo(left->GetDType());

  //       int rc = left->CastToInplace(right->GetDType());
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do op:{} cast with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //       }
  //       rc = left->Copy(*right);
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do oassign with left:{}, right:{}", left->GetDType(), right->GetDType())));
  //       }
  //       GetCodeGenerator().DropTmpValue(right);
  //       // GetCodeGenerator().DropTmpValue(right_val);
  //       return left;
  //     }
  //     case OP_EQUAL:
  //     case OP_NOT_EQUAL:
  //     case OP_LESS:
  //     case OP_LESS_EQUAL:
  //     case OP_GREATER:
  //     case OP_GREATER_EQUAL: {
  //       auto result = GetCodeGenerator().NewValue(DATA_U8, {}, true);
  //       int rc = left->Cmp(op, *right, result);
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //       }
  //       GetCodeGenerator().DropTmpValue(left);
  //       GetCodeGenerator().DropTmpValue(right);
  //       return result;
  //     }
  //     case OP_LOGIC_AND:
  //     case OP_LOGIC_OR: {
  //       uint32_t logic_label_cursor = label_cursor_++;
  //       std::string fast_exit_label = fmt::format("logic_fast_exit_{}", logic_label_cursor);
  //       std::string normal_exit_label = fmt::format("logic_normal_exit_{}", logic_label_cursor);
  //       auto result = GetCodeGenerator().NewValue(DATA_U8, {}, true);
  //       uint64_t fast_exit_val_bin = op == OP_LOGIC_AND ? 0 : 1;
  //       auto fast_cmp_val = Value::New(&GetCodeGenerator(), DATA_U8, fast_exit_val_bin);
  //       int rc = left->Cmp(OP_EQUAL, *fast_cmp_val, nullptr);
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //       }
  //       GetCodeGenerator().Jump(fast_exit_label, OP_EQUAL);
  //       rc = left->LogicOp(op, *right, result);
  //       if (0 != rc) {
  //         RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //             fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //       }
  //       GetCodeGenerator().Jump(normal_exit_label);

  //       GetCodeGenerator().Label(fast_exit_label);
  //       result->Set(fast_exit_val_bin);

  //       GetCodeGenerator().Label(normal_exit_label);
  //       GetCodeGenerator().GetCodeGen().nop();
  //       GetCodeGenerator().DropTmpValue(left);
  //       GetCodeGenerator().DropTmpValue(right);
  //       return result;
  //     }
  //     default: {
  //       RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
  //           fmt::format("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
  //     }
  //   }
  // }
  // return left;
}

absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(const ast::VarAccessor& expr) {
  ast_ctx_.SetPosition(expr.position);
  ValuePtr var;
  if (expr.func_args.has_value()) {
    // name is func name
    std::vector<ValuePtr> arg_values;
    if (expr.func_args->args.has_value()) {
      for (auto func_arg_expr : *(expr.func_args->args)) {
        auto arg_val = CompileExpression(func_arg_expr);
        if (!arg_val.ok()) {
          return arg_val.status();
        }
        arg_values.emplace_back(arg_val.value());
      }
    }

    const FunctionDesc* func_desc = GetFunction(expr.name);
    if (nullptr == func_desc) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("No func:{} found", expr.name)));
    }
    for (size_t i = 0; i < arg_values.size(); i++) {
      if (arg_values[i]->GetDType() != func_desc->arg_types[i]) {
        arg_values[i] = arg_values[i]->CastTo(func_desc->arg_types[i]);
      }
    }
    ValuePtr result = GetCodeGenerator().CallFunction(*func_desc, arg_values);
    if (result) {
      for (auto& arg : arg_values) {
        GetCodeGenerator().DropTmpValue(arg);
      }
    }
    return result;
  } else if (expr.access_args.has_value()) {
    // name is var name
    auto var_result = GetLocalVar(expr.name);
    if (!var_result.ok()) {
      return var_result.status();
    }
    var = var_result.value();
    for (auto access_arg : *expr.access_args) {
      auto next_result = std::visit(
          [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ast::FieldAccess>) {
              return CompileFieldAccess(var, arg);
            } else if constexpr (std::is_same_v<T, ast::DynamicParamAccess>) {
              return std::visit([&](auto&& json_arg) { return CompileJsonAccess(var, json_arg); }, arg);
            } else {
              static_assert(sizeof(arg) == -1, "non-exhaustive visitor!");
              return absl::StatusOr<ValuePtr>(absl::OkStatus());
            }
          },
          access_arg);
      if (!next_result.ok()) {
        return next_result.status();
      }
      GetCodeGenerator().DropTmpValue(var);
      var = next_result.value();
    }
    return var;
  } else {
    // name is var name
    return GetLocalVar(expr.name);
  }
}
absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(const ast::VarDefine& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto var = GetCodeGenerator().NewValue(DType(DATA_VOID), {}, false);
  var->SetVarName(expr.name);
  GetCompileContext().local_vars.emplace(expr.name, var);
  return var;
}
absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(ast::UnaryExprPtr expr) {
  ast_ctx_.SetPosition(expr->position);
  auto val_result = CompileOperand(expr->operand);
  if (!val_result.ok()) {
    return val_result.status();
  }
  auto val = val_result.value();
  if (expr->op.has_value()) {
    if (val->GetDType().IsVoid()) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with void operands", *(expr->op))));
    }
    auto result = GetCodeGenerator().NewValue(val->GetDType(), {}, true);
    int rc = val->UnaryOp(*(expr->op), result);
    if (0 != rc) {
      RUDF_LOG_ERROR_STATUS(
          ast_ctx_.GetErrorStatus(fmt::format("Can NOT do op:{} with  operands:{}", *(expr->op), val->GetDType())));
    }
    GetCodeGenerator().DropTmpValue(val);
    return result;
  }
  return val;
}
}  // namespace rapidudf