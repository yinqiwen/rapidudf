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
#include <vector>
#include "absl/cleanup/cleanup.h"
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/ops/cmp.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/simd/simd_ops.h"
#include "rapidudf/codegen/value.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
namespace rapidudf {

absl::StatusOr<ValuePtr> JitCompiler::CompileOperand(const ast::Operand& expr) {
  // RUDF_DEBUG("CompileOperand readonly:{}", flags);
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::string>) {
          return CompileConstants(arg);
        } else if constexpr (std::is_same_v<T, ast::VarAccessor> || std::is_same_v<T, ast::VarDefine>) {
          return CompileExpression(arg);
        } else if constexpr (std::is_same_v<T, ast::BinaryExprPtr> || std::is_same_v<T, ast::UnaryExprPtr> ||
                             std::is_same_v<T, ast::TernaryExprPtr>) {
          return CompileExpression(arg);
        } else if constexpr (std::is_same_v<T, ast::ConstantNumber>) {
          if (arg.dtype.has_value()) {
            return CompileConstants(arg.dv, *arg.dtype);
          } else {
            return CompileConstants(arg.dv);
          }
        } else if constexpr (std::is_same_v<T, ast::Array>) {
          return CompileExpression(arg);
        } else {
          static_assert(sizeof(arg) == -1, "non-exhaustive visitor!");
          ValuePtr empty;
          return empty;
        }
      },
      expr);
}
absl::StatusOr<ValuePtr> JitCompiler::CallFunction(const std::string& name, std::vector<ValuePtr>& arg_values) {
  const FunctionDesc* func_desc = GetFunction(name);
  if (nullptr == func_desc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("No func:{} found", name)));
  }
  // uint32_t simd_vector_reuse_flag = 0;
  for (size_t i = 0; i < arg_values.size(); i++) {
    if (arg_values[i]->GetDType() != func_desc->arg_types[i]) {
      arg_values[i] = arg_values[i]->CastTo(func_desc->arg_types[i]);
    }
    // if (simd_vector_reuse_flag == 0 && arg_values[i]->IsTemp() && arg_values[i]->GetDType().IsSimdVector()) {
    //   simd_vector_reuse_flag = i + 1;
    // }
  }
  // if (func_desc->is_simd_vector_func) {
  //   arg_values.emplace_back(GetCodeGenerator().NewConstValue(DATA_U32, simd_vector_reuse_flag));
  // }
  ValuePtr result = GetCodeGenerator().CallFunction(*func_desc, arg_values);
  if (result) {
    for (auto& arg : arg_values) {
      GetCodeGenerator().DropTmpValue(arg);
    }
  }
  return result;
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
        if (left->GetDType().IsSimdVector() || right->GetDType().IsSimdVector()) {
          auto func_name = GetFunctionName(op, left->GetDType(), right->GetDType());
          std::vector<ValuePtr> args{left, right};
          auto result = CallFunction(func_name, args);
          if (!result.ok()) {
            return result.status();
          }
          left = result.value();
          break;
        }

        auto result =
            GetCodeGenerator().NewValue(left->GetDType() > right->GetDType() ? left->GetDType() : right->GetDType());
        RUDF_DEBUG("before op:{}, left:{}, right:{}", op, left->IsConst(), right->IsRegister());
        if (left->IsRegister() && right->IsRegister()) {
          RUDF_DEBUG("before op:{}, left:{}, right:{}", op, left->GetOperand().toString(),
                     right->GetOperand().toString());
        }

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
        bool need_copy = true;
        if (left->GetDType().IsVoid()) {
          if (right->IsTemp()) {
            left->Swap(*right);
            left->SetTemp(false);
            need_copy = false;
          } else {
            auto new_left = left->CastTo(right->GetDType());
            if (!new_left) {
              RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT assgin value since cast failed.")));
            }
            if (left.get() != new_left.get()) {
              new_left->SetVarName(left->GetVarName());
              new_left->SetTemp(false);
              // GetCompileContext().local_vars[left->GetVarName()] = new_left;
              GetCodeGenerator().DropTmpValue(left);
              GetCompileContext().local_vars[new_left->GetVarName()] = new_left;
            }
            left = new_left;
          }

        } else {
          int rc = left->CastToInplace(right->GetDType());
          if (0 != rc) {
            RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
                fmt::format("Can NOT do op:{} cast with left:{}, right:{}", op, left->GetDType(), right->GetDType())));
          }
        }
        if (need_copy) {
          int rc = left->Copy(*right);
          if (0 != rc) {
            RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
                fmt::format("Can NOT do assign with left:{}, right:{}", left->GetDType(), right->GetDType())));
          }
        }

        if (left->GetDType().IsSimdVector()) {
          int rc = left->SetSimdVectorTemporary(false);
          if (0 != rc) {
            RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("Can NOT clear temporary flag")));
          }
        }
        GetCodeGenerator().DropTmpValue(right);
        break;
      }
      case OP_EQUAL:
      case OP_NOT_EQUAL:
      case OP_LESS:
      case OP_LESS_EQUAL:
      case OP_GREATER:
      case OP_GREATER_EQUAL: {
        if (left->GetDType().IsSimdVector() || right->GetDType().IsSimdVector()) {
          auto func_name = GetFunctionName(op, left->GetDType(), right->GetDType());
          std::vector<ValuePtr> args{left, right};
          auto result = CallFunction(func_name, args);
          if (!result.ok()) {
            return result.status();
          }
          left = result.value();
          break;
        }
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
        if (left->GetDType().IsSimdVector() || right->GetDType().IsSimdVector()) {
          auto func_name = GetFunctionName(op, left->GetDType(), right->GetDType());
          std::vector<ValuePtr> args{left, right};
          auto result = CallFunction(func_name, args);
          if (!result.ok()) {
            return result.status();
          }
          left = result.value();
          break;
        }
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
    auto result = CallFunction(expr.name, arg_values);
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
absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(ast::TernaryExprPtr expr) {
  ast_ctx_.SetPosition(expr->position);
  auto cond_result = CompileOperand(expr->cond);
  if (expr->true_false_operands.has_value()) {
    if (!cond_result.ok()) {
      return cond_result.status();
    }
    auto cond_val = cond_result.value();
    auto [true_expr, false_expr] = *(expr->true_false_operands);
    if (cond_val->GetDType().IsBool()) {
      uint32_t ternary_cursor = GetLabelCursor();
      std::string ternary_final_label = fmt::format("ternary_final_{}", ternary_cursor);
      std::string ternary_true_label = fmt::format("ternary_true_{}", ternary_cursor);
      auto true_bin = Value::New(&GetCodeGenerator(), DATA_U8, static_cast<uint64_t>(1));
      auto result_val = GetCodeGenerator().NewValue(expr->ternary_result_dtype, {}, true);
      int rc = cond_val->Cmp(OP_EQUAL, *true_bin, nullptr);
      if (0 != rc) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("cmp ternary failed.")));
      }
      GetCodeGenerator().Jump(ternary_true_label, OP_EQUAL);
      auto false_expr_result = CompileOperand(false_expr);
      if (!false_expr_result.ok()) {
        return false_expr_result.status();
      }
      auto false_expr_val = false_expr_result.value();
      false_expr_val = false_expr_val->CastTo(result_val->GetDType());
      if (!false_expr_val) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
            fmt::format("ternary cast false expr value:{} to {}", false_expr_val->GetDType(), result_val->GetDType())));
      }
      rc = result_val->Copy(*false_expr_val);
      if (0 != rc) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("ternary copy false expr result.")));
      }
      GetCodeGenerator().DropTmpValue(false_expr_val);
      GetCodeGenerator().Jump(ternary_final_label);
      GetCodeGenerator().Label(ternary_true_label);
      auto true_expr_result = CompileOperand(true_expr);
      if (!true_expr_result.ok()) {
        return true_expr_result.status();
      }
      auto true_expr_val = true_expr_result.value();
      true_expr_val = true_expr_val->CastTo(result_val->GetDType());
      if (!true_expr_val) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
            fmt::format("ternary cast true expr value:{} to {}", true_expr_val->GetDType(), result_val->GetDType())));
      }
      rc = result_val->Copy(*true_expr_val);
      if (0 != rc) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("ternary copy true expr result.")));
      }
      GetCodeGenerator().DropTmpValue(true_expr_val);
      GetCodeGenerator().Label(ternary_final_label);
      GetCodeGenerator().GetCodeGen().nop();
      return result_val;
    } else if (cond_val->GetDType().IsSimdVectorBit()) {
      auto true_expr_result = CompileOperand(true_expr);
      if (!true_expr_result.ok()) {
        return true_expr_result.status();
      }
      auto true_expr_val = true_expr_result.value();
      auto false_expr_result = CompileOperand(false_expr);
      if (!false_expr_result.ok()) {
        return false_expr_result.status();
      }
      auto false_expr_val = false_expr_result.value();
      auto func_name =
          GetFunctionName(OP_CONDITIONAL, cond_val->GetDType(), true_expr_val->GetDType(), false_expr_val->GetDType());
      std::vector<ValuePtr> args{cond_val, true_expr_val, false_expr_val};
      auto result = CallFunction(func_name, args);
      if (!result.ok()) {
        return result.status();
      }
      GetCodeGenerator().DropTmpValue(true_expr_val);
      GetCodeGenerator().DropTmpValue(false_expr_val);
      return result.value();
    }
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("ternary op with cond:{", cond_val->GetDType())));
  } else {
    return cond_result;
  }
}

absl::StatusOr<ValuePtr> JitCompiler::CompileExpression(const ast::Array& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto span_val = GetCodeGenerator().NewValue(expr.dtype, {}, false);
  auto stack_vals = GetCodeGenerator().NewArrayValue(expr.dtype, expr.elements.size());
  for (size_t i = 0; i < expr.elements.size(); i++) {
    auto result = CompileExpression(expr.elements[i]);
    if (!result.ok()) {
      return result.status();
    }
    int rc = stack_vals[i]->Copy(*result.value());
    if (0 != rc) {
      RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("copy array[{}] stack element failed.", i)));
    }
  }
  size_t span_len = expr.elements.size();
  int rc = span_val->SetSpanSize(span_len);
  if (0 == rc) {
    rc = span_val->SetSpanStackPtr(stack_vals[0]->GetStackOffset());
  }
  if (0 != rc) {
    RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(fmt::format("set span size/ptr failed.")));
  }

  return span_val;
}
}  // namespace rapidudf