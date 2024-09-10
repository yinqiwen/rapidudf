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
#include "rapidudf/ast/expression.h"
#include <fmt/core.h>
#include <cstdint>
#include <variant>
#include <vector>
#include "rapidudf/ast/context.h"
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/reflect.h"
namespace rapidudf {
namespace ast {
static absl::StatusOr<VarTag> validate_operand(ParseContext& ctx, Operand& v) {
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
          return absl::StatusOr<VarTag>(get_dtype<T>());
        } else if constexpr (std::is_same_v<T, std::string>) {
          return absl::StatusOr<VarTag>(DATA_STRING_VIEW);
        } else if constexpr (std::is_same_v<T, VarAccessor> || std::is_same_v<T, VarDefine>) {
          return arg.Validate(ctx);
        } else if constexpr (std::is_same_v<T, BinaryExprPtr> || std::is_same_v<T, UnaryExprPtr> ||
                             std::is_same_v<T, TernaryExprPtr>) {
          return arg->Validate(ctx);
        } else if constexpr (std::is_same_v<T, ConstantNumber>) {
          if (arg.dtype.has_value()) {
            return absl::StatusOr<VarTag>(*arg.dtype);
          }
          int64_t iv = static_cast<int64_t>(arg.dv);
          if (static_cast<double>(iv) == arg.dv) {
            if (iv <= INT32_MAX) {
              return absl::StatusOr<VarTag>(get_dtype<int32_t>());
            }
            return absl::StatusOr<VarTag>(get_dtype<int64_t>());
          }
          return absl::StatusOr<VarTag>(get_dtype<double>());
        } else if constexpr (std::is_same_v<T, Array>) {
          return arg.Validate(ctx);
        } else {
          static_assert(sizeof(arg) == -1, "No avaialble!");
          return absl::InvalidArgumentError("No avaialble");
        }
      },
      v);
}
absl::StatusOr<VarTag> Array::Validate(ParseContext& ctx) {
  if (elements.empty()) {
    return absl::InvalidArgumentError("array can not be empty");
  }
  for (size_t i = 0; i < elements.size(); i++) {
    auto result = elements[i]->Validate(ctx);
    if (!result.ok()) {
      return result.status();
    }
    if (i == 0) {
      dtype = result->dtype.ToAbslSpan();
    }
  }
  return dtype;
}

absl::StatusOr<VarTag> VarRef::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  return ctx.IsVarExist(name, false);
}
absl::StatusOr<VarTag> VarDefine::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  auto result = ctx.IsVarExist(name, true);
  if (!result.ok()) {
    return result.status();
  }
  ctx.AddLocalVar(name, DType(DATA_VOID));
  return VarTag(DATA_VOID, name);
}
absl::StatusOr<std::vector<VarTag>> FuncInvokeArgs::Validate(ParseContext& ctx) {
  std::vector<VarTag> arg_dtypes;
  if (args.has_value()) {
    for (auto expr : *args) {
      auto result = expr->Validate(ctx);
      if (!result.ok()) {
        return result.status();
      }
      arg_dtypes.emplace_back(result.value());
    }
  }
  return arg_dtypes;
}
absl::StatusOr<VarTag> FieldAccess::Validate(ParseContext& ctx, DType src_dtype) {
  auto field_accessor = Reflect::GetStructMember(src_dtype.PtrTo(), field);
  if (!field_accessor) {
    return ctx.GetErrorStatus(fmt::format("Can NOT get member:{} accessor for dtype:{}", field, src_dtype.PtrTo()));
  }
  if (!func_args.has_value()) {
    if (!field_accessor->HasField()) {
      return ctx.GetErrorStatus(fmt::format("Can NOT get field:{} accessor for dtype:{}", field, src_dtype));
    }
    DType field_dtype = *field_accessor->member_field_dtype;
    if (!field_dtype.IsPrimitive() && !field_dtype.IsPtr() && !field_dtype.IsSimdVector()) {
      field_dtype = field_dtype.ToPtr();
    }
    return field_dtype;
  } else {
    if (!field_accessor->HasMemberFunc()) {
      return ctx.GetErrorStatus(fmt::format("Can NOT get member func:{} accessor for dtype:{}", field, src_dtype));
    }
    auto func_arg_dtypes = func_args->Validate(ctx);
    if (!func_arg_dtypes.ok()) {
      return func_arg_dtypes.status();
    }
    return field_accessor->member_func->return_type;
  }
}
absl::StatusOr<VarTag> UnaryExpr::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  auto result = validate_operand(ctx, operand);
  if (!result.ok()) {
    return result.status();
  }
  if (op.has_value()) {
    switch (*op) {
      case OP_NOT: {
        if (!result->dtype.IsBool()) {
          return ctx.GetErrorStatus(fmt::format("can NOT do not op on non bool value:{}", result->dtype));
        }
        break;
      }
      case OP_NEGATIVE: {
        if (!result->dtype.IsNumber()) {
          return ctx.GetErrorStatus(fmt::format("can NOT do negative op on non number value:{}", result->dtype));
        }
        break;
      }
      default: {
        break;
      }
    }
  }
  return result;
}
static bool IsValidSimdVectorBinaryOperands(DType left, DType right) {
  if (left.IsSimdVector() || right.IsSimdVector()) {
    if (left.IsSimdVector() && right.IsSimdVector()) {
      if (left != right) {
        return false;
      }
      return false;
    } else {
      DType left_ele_dtype = left.Elem();
      DType right_ele_dtype = left.Elem();
      if (left_ele_dtype.IsNumber() && right_ele_dtype.IsNumber()) {
        return true;
      }
      if (left_ele_dtype.IsStringView() && right_ele_dtype.IsStringView()) {
        return true;
      }
      return false;
    }
  }
  return false;
}
absl::StatusOr<VarTag> BinaryExpr::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  auto left_result = validate_operand(ctx, left);
  if (!left_result.ok()) {
    return left_result.status();
  }
  VarTag left_var = left_result.value();
  for (auto [op, right_operand] : right) {
    auto right_result = validate_operand(ctx, right_operand);
    if (!right_result.ok()) {
      return right_result.status();
    }
    bool can_binary_op = true;
    do {
      if (left_var.dtype == right_result->dtype) {
        can_binary_op = true;
        break;
      }
      if (left_var.dtype.IsVoid() && !left_var.name.empty() && op == OP_ASSIGN) {
        can_binary_op = true;
        break;
      }
      if (IsValidSimdVectorBinaryOperands(left_var.dtype, right_result->dtype)) {
        can_binary_op = true;
        break;
      }

      if (left_var.dtype.IsJsonPtr() && right_result->dtype.IsPrimitive()) {
        can_binary_op = true;
        break;
      }
      if (right_result->dtype.IsJsonPtr() && left_var.dtype.IsPrimitive()) {
        can_binary_op = true;
        break;
      }
      if (!left_var.dtype.CanCastTo(right_result->dtype) && !right_result->dtype.CanCastTo(left_result->dtype)) {
        can_binary_op = false;
        break;
      }
      if (left_var.dtype.CanCastTo(right_result->dtype)) {
        left_var.dtype = right_result->dtype;
      } else {
        right_result->dtype = left_var.dtype;
      }
      if (left_var.dtype.IsStringPtr() || right_result->dtype.IsStringPtr()) {
        ctx.AddBuiltinFuncCall(kBuiltinCastStdStrToStringView);
      } else if (left_var.dtype.IsFlatbuffersStringPtr() || right_result->dtype.IsFlatbuffersStringPtr()) {
        ctx.AddBuiltinFuncCall(kBuiltinCastFbsStrToStringView);
      }
    } while (0);
    if (!can_binary_op) {
      return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{} by expression validate ",
                                            op, left_var.dtype, right_result->dtype));
    }

    switch (op) {
      case OP_PLUS:
      case OP_MINUS:
      case OP_MULTIPLY:
      case OP_DIVIDE:
      case OP_MOD:
      case OP_PLUS_ASSIGN:
      case OP_MINUS_ASSIGN:
      case OP_MULTIPLY_ASSIGN:
      case OP_DIVIDE_ASSIGN:
      case OP_MOD_ASSIGN: {
        if (left_var.dtype.IsSimdVector() || right_result->dtype.IsSimdVector()) {
          if (!left_var.dtype.IsSimdVector()) {
            left_var.dtype = right_result->dtype;
          }
          // ctx.MarkSimdVectorOperation();
          ctx.AddBuiltinFuncCall(GetFunctionName(op, left_var.dtype, right_result->dtype));
          break;
        }
        if (!left_var.dtype.IsNumber() || !right_result->dtype.IsNumber()) {
          return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op,
                                                left_result->dtype, right_result->dtype));
        }
        break;
      }
      case OP_ASSIGN: {
        if (!left_result->name.empty()) {
          ctx.AddLocalVar(left_result->name, right_result->dtype);
        }
        // return right_result;
        left_var = right_result.value();
        break;
      }
      case OP_EQUAL:
      case OP_NOT_EQUAL:
      case OP_LESS:
      case OP_LESS_EQUAL:
      case OP_GREATER:
      case OP_GREATER_EQUAL: {
        bool can_cmp = false;
        if (left_var.dtype.IsNumber() && right_result->dtype.IsNumber()) {
          can_cmp = true;
        } else if (left_var.dtype.IsStringView() && right_result->dtype.IsStringView()) {
          can_cmp = true;
          ctx.AddBuiltinFuncCall(kBuiltinStringViewCmp);
        } else if (left_var.dtype.IsJsonPtr() && right_result->dtype.IsPrimitive()) {
          can_cmp = true;
          if (right_result->dtype.IsFloat()) {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpFloat);
          } else if (right_result->dtype.IsStringView()) {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpString);
          } else {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpInt);
          }
        } else if (left_var.dtype.IsPrimitive() && right_result->dtype.IsJsonPtr()) {
          can_cmp = true;
          if (left_result->dtype.IsFloat()) {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpFloat);
          } else if (left_result->dtype.IsStringView()) {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpString);
          } else {
            ctx.AddBuiltinFuncCall(kBuiltinJsonCmpInt);
          }
        } else if (left_var.dtype.IsJsonPtr() && right_result->dtype.IsJsonPtr()) {
          can_cmp = true;
          ctx.AddBuiltinFuncCall(kBuiltinJsonCmpJson);
        } else if (left_var.dtype.IsSimdVector() || right_result->dtype.IsSimdVector()) {
          ctx.AddBuiltinFuncCall(GetFunctionName(op, left_var.dtype, right_result->dtype));
          // ctx.MarkSimdVectorOperation();
          left_var = VarTag(DType(DATA_BIT).ToSimdVector());
          break;
        }
        if (!can_cmp) {
          // RUDF_DEBUG("#### {}:{} {}", *left_result, left_result->dtype.IsJsonPtr(),
          // right_result->dtype.IsPrimitive());
          return ctx.GetErrorStatus(
              fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op, left_var.dtype, right_result->dtype));
        }
        left_var = VarTag(DType(DATA_U8));
        break;
      }
      case OP_LOGIC_AND:
      case OP_LOGIC_OR: {
        if (left_result->dtype.IsSimdVectorBit() && right_result->dtype.IsSimdVectorBit()) {
          left_var = VarTag(DType(DATA_BIT).ToSimdVector());
          break;
        }
        if (!left_result->dtype.IsBool() || !right_result->dtype.IsBool()) {
          return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op,
                                                left_result->dtype, right_result->dtype));
        }
        // return DType(DATA_U8);
        left_var = VarTag(DType(DATA_U8));
        break;
      }
      default: {
        return ctx.GetErrorStatus(fmt::format("Unimplemented {} with left dtype:{}, right dtype:{}", op,
                                              left_result->dtype, right_result->dtype));
      }
    }
  }

  return left_var;
}

absl::StatusOr<VarTag> TernaryExpr::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  auto cond_result = validate_operand(ctx, cond);
  if (true_false_operands.has_value()) {
    if (!cond_result.ok()) {
      return cond_result.status();
    }
    auto [true_expr, false_expr] = *true_false_operands;
    auto true_expr_result = validate_operand(ctx, true_expr);
    auto false_expr_result = validate_operand(ctx, false_expr);
    if (!true_expr_result.ok()) {
      return true_expr_result.status();
    }
    if (!false_expr_result.ok()) {
      return false_expr_result.status();
    }
    if (cond_result->dtype.IsBool()) {
      if (true_expr_result->dtype.CanCastTo(false_expr_result->dtype)) {
        ternary_result_dtype = false_expr_result->dtype;
        return VarTag(ternary_result_dtype);
      } else if (false_expr_result->dtype.CanCastTo(true_expr_result->dtype)) {
        ternary_result_dtype = true_expr_result->dtype;
        return VarTag(ternary_result_dtype);
      }
    } else if (cond_result->dtype.IsSimdVectorBit()) {
      if (true_expr_result->dtype.IsSimdVector() && false_expr_result->dtype.IsSimdVector()) {
        if (true_expr_result->dtype == false_expr_result->dtype) {
          ternary_result_dtype = true_expr_result->dtype;
          // ctx.MarkSimdVectorOperation();
          ctx.AddBuiltinFuncCall(
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
          return VarTag(ternary_result_dtype);
        }
      } else if (true_expr_result->dtype.IsSimdVector() && !false_expr_result->dtype.IsSimdVector()) {
        if (false_expr_result->dtype.CanCastTo(true_expr_result->dtype.Elem())) {
          ternary_result_dtype = true_expr_result->dtype;
          // ctx.MarkSimdVectorOperation();
          ctx.AddBuiltinFuncCall(
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
          return VarTag(ternary_result_dtype);
        }
      } else if (!true_expr_result->dtype.IsSimdVector() && false_expr_result->dtype.IsSimdVector()) {
        if (true_expr_result->dtype.CanCastTo(false_expr_result->dtype.Elem())) {
          ternary_result_dtype = false_expr_result->dtype;
          // ctx.MarkSimdVectorOperation();
          ctx.AddBuiltinFuncCall(
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
          return VarTag(ternary_result_dtype);
        }
      } else {
        if (true_expr_result->dtype.IsNumber() && false_expr_result->dtype.IsNumber()) {
          ternary_result_dtype =
              true_expr_result->dtype >= false_expr_result->dtype ? true_expr_result->dtype : false_expr_result->dtype;
          // ctx.MarkSimdVectorOperation();
          ctx.AddBuiltinFuncCall(
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
          return VarTag(ternary_result_dtype.ToSimdVector());
        }
      }
    }
    return ctx.GetErrorStatus(
        fmt::format("can NOT do ternary with cond dtype:{}, true_expr_dtype:{}, false_expr_dtype:{}",
                    cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
  } else {
    return cond_result;
  }
}

absl::StatusOr<VarTag> VarAccessor::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  if (access_args.has_value()) {
    auto result = ctx.IsVarExist(name, false);
    if (!result.ok()) {
      return result.status();
    }
    VarTag var_dtype = result.value();
    for (auto& member : *access_args) {
      auto result = std::visit(
          [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, FieldAccess>) {
              return arg.Validate(ctx, var_dtype.dtype);
            } else {
              return std::visit(
                  [&](auto&& dynamic_arg) {
                    if (!var_dtype.dtype.IsPtr() || !var_dtype.dtype.PtrTo().IsJson()) {
                      return absl::StatusOr<VarTag>(absl::InvalidArgumentError(
                          fmt::format("invalid dtype:{} to do json dynamic access", var_dtype.dtype)));
                    }
                    DType json_dtype(DATA_JSON);
                    DType json_ptr_dtype = json_dtype.ToPtr();
                    return absl::StatusOr<VarTag>(json_ptr_dtype);
                  },
                  arg);
            }
          },
          member);
      if (!result.ok()) {
        return result.status();
      }
      var_dtype = result.value();
    }
    return var_dtype;
  } else if (func_args.has_value()) {
    std::vector<DType> arg_dtypes;
    DType largest_dtype;
    bool has_simd_vector = false;
    if (func_args->args.has_value()) {
      for (auto expr : *(func_args->args)) {
        auto result = expr->Validate(ctx);
        if (!result.ok()) {
          return result.status();
        }
        arg_dtypes.emplace_back(result->dtype);
        if (result->dtype > largest_dtype) {
          largest_dtype = result->dtype;
        }
        if (result->dtype.IsSimdVector()) {
          has_simd_vector = true;
        }
      }
    }
    if (is_builtin_function(name)) {
      if (name == kOpTokenStrs[OP_IOTA]) {
        // ctx.MarkSimdVectorOperation();
        name = GetFunctionName(name, arg_dtypes[0]);
      } else if (has_simd_vector) {
        // ctx.MarkSimdVectorOperation();
        // name = "simd_vector_" + name + "_" + largest_dtype.Elem().GetTypeString();
        name = GetFunctionName(name, arg_dtypes);
      } else {
        // name = name + "_" + largest_dtype.GetTypeString();
        name = GetFunctionName(name, largest_dtype);
      }
    }
    auto result = ctx.CheckFuncExist(name);
    if (!result.ok()) {
      return result.status();
    }
    auto* desc = result.value();
    if (!desc->ValidateArgs(arg_dtypes)) {
      return ctx.GetErrorStatus(
          fmt::format("Invalid func call:{} args with invalid args, simd_vector_func:{}", name, has_simd_vector));
    }
    // if (desc->is_simd_vector_func) {
    //   ctx.MarkSimdVectorOperation();
    // }
    return desc->return_type;
  } else {
    return ctx.IsVarExist(name, false);
  }
}

}  // namespace ast
}  // namespace rapidudf