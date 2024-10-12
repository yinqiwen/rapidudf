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
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "fmt/format.h"

#include "rapidudf/ast/context.h"
#include "rapidudf/functions/functions.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/reflect.h"
namespace rapidudf {
namespace ast {
void RPN::SetDType(ParseContext& ctx, DType dtype) {
  this->dtype = dtype;
  if (dtype.IsSimdVector()) {
    std::string fname = GetFunctionName(functions::kBuiltinNewSimdVector, dtype.Elem());
    std::ignore = ctx.CheckFuncExist(fname);
    std::ignore = ctx.CheckFuncExist(functions::kBuiltinThrowVectorExprEx);
  }
  // if (dtype.IsSimdVector()) {
  //   auto fname = GetFunctionName(kBuiltinVectorEval, dtype.Elem());
  //   std::ignore = ctx.CheckFuncExist(fname, false);
  // }
  // if (dtype.IsSimdColumnPtr()) {
  //   std::ignore = ctx.CheckFuncExist(kBuiltinColumnEval, false);
  // }
}
void RPN::Print() {
  std::string info;

  for (auto& e : nodes) {
    std::visit(
        [&](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          absl::StatusOr<VarTag> result;
          if constexpr (std::is_same_v<T, bool>) {
            info.append(std::to_string(arg));
          } else if constexpr (std::is_same_v<T, std::string>) {
            info.append(arg);
          } else if constexpr (std::is_same_v<T, VarDefine>) {
            info.append("var_def");
          } else if constexpr (std::is_same_v<T, VarAccessor>) {
            info.append("var_access");
          } else if constexpr (std::is_same_v<T, ConstantNumber>) {
            info.append(arg.ToString());
          } else if constexpr (std::is_same_v<T, Array>) {
            info.append("array");
          } else {
            info.append(fmt::format("{}", arg));
          }
          info.append(" ");
        },
        e);
  }
  RUDF_ERROR("RPN: {}", info);
}

static absl::StatusOr<VarTag> validate_operand(ParseContext& ctx, Operand& v, RPN& rpn) {
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        absl::StatusOr<VarTag> result;
        if constexpr (std::is_same_v<T, bool>) {
          rpn.nodes.emplace_back(arg);
          return absl::StatusOr<VarTag>(get_dtype<T>());
        } else if constexpr (std::is_same_v<T, std::string>) {
          rpn.nodes.emplace_back(arg);
          return absl::StatusOr<VarTag>(DATA_STRING_VIEW);
        } else if constexpr (std::is_same_v<T, VarDefine>) {
          result = arg.Validate(ctx);
          rpn.nodes.emplace_back(arg);
        } else if constexpr (std::is_same_v<T, VarAccessor>) {
          bool as_builtin_op = false;
          result = arg.Validate(ctx, rpn, as_builtin_op);
          if (!as_builtin_op) {
            rpn.nodes.emplace_back(arg);
          }
        } else if constexpr (std::is_same_v<T, BinaryExprPtr> || std::is_same_v<T, UnaryExprPtr>) {
          result = arg->Validate(ctx, rpn);
        } else if constexpr (std::is_same_v<T, SelectExprPtr>) {
          result = arg->Validate(ctx, rpn);
          // rpn.nodes.emplace_back(arg);
        } else if constexpr (std::is_same_v<T, ConstantNumber>) {
          rpn.nodes.emplace_back(arg);
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
          result = arg.Validate(ctx);
          rpn.nodes.emplace_back(arg);
        } else {
          static_assert(sizeof(arg) == -1, "No avaialble!");
          return absl::InvalidArgumentError("No avaialble");
        }
        if (result.ok()) {
          if (result.value().dtype.IsSimdColumnPtr() || result.value().dtype.IsSimdVector()) {
            ctx.SetVectorExressionFlag(true);
          }
        }
        return result;
      },
      v);
}

std::string ConstantNumber::ToString() const { return std::to_string(dv); }

absl::StatusOr<VarTag> Array::Validate(ParseContext& ctx) {
  if (elements.empty()) {
    return absl::InvalidArgumentError("array can not be empty");
  }
  for (size_t i = 0; i < elements.size(); i++) {
    RPN rpn;
    auto result = elements[i]->Validate(ctx, rpn);
    if (!result.ok()) {
      return result.status();
    }
    // rpn.dtype = result->dtype;
    rpn.SetDType(ctx, result->dtype);
    if (i == 0) {
      dtype = result->dtype.ToAbslSpan();
    }
    rpns.emplace_back(rpn);
  }
  return dtype;
}

absl::StatusOr<VarTag> VarRef::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);
  auto result = ctx.IsVarExist(name, false);
  if (!result.ok()) {
    return result;
  }
  if (result.value().IsSimdColumnPtr() || result.value().IsSimdVector()) {
    ctx.SetVectorExressionFlag(true);
  }
  return VarTag(result.value(), name);
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
absl::StatusOr<std::vector<VarTag>> FuncInvokeArgs::Validate(ParseContext& ctx, RPN* rpn) {
  std::vector<VarTag> arg_dtypes;
  if (args.has_value()) {
    for (auto expr : *args) {
      RPN arg_rpn;
      auto result = expr->Validate(ctx, rpn == nullptr ? arg_rpn : *rpn);
      if (!result.ok()) {
        return result.status();
      }
      // rpn.dtype = result->dtype;
      if (nullptr == rpn) {
        arg_rpn.SetDType(ctx, result->dtype);
        rpns.emplace_back(arg_rpn);
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
  struct_member = field_accessor.value();
  if (!func_args.has_value()) {
    if (!struct_member.HasField()) {
      return ctx.GetErrorStatus(fmt::format("Can NOT get field:{} accessor for dtype:{}", field, src_dtype));
    }
    DType field_dtype = *struct_member.member_field_dtype;
    if (!field_dtype.IsPrimitive() && !field_dtype.IsPtr() && !field_dtype.IsSimdVector() &&
        !field_dtype.IsStdStringView() && !field_dtype.IsStringView() && !field_dtype.IsAbslSpan()) {
      field_dtype = field_dtype.ToPtr();
    }
    return field_dtype;
  } else {
    if (!struct_member.HasMemberFunc()) {
      return ctx.GetErrorStatus(fmt::format("Can NOT get member func:{} accessor for dtype:{}", field, src_dtype));
    }
    auto func_arg_dtypes = func_args->Validate(ctx);
    if (!func_arg_dtypes.ok()) {
      return func_arg_dtypes.status();
    }

    ctx.AddMemberFuncCall(src_dtype.PtrTo(), field, *struct_member.member_func);
    return struct_member.member_func->return_type;
  }
}
absl::StatusOr<VarTag> UnaryExpr::Validate(ParseContext& ctx, RPN& rpn) {
  ctx.SetPosition(position);
  auto result = validate_operand(ctx, operand, rpn);
  if (!result.ok()) {
    return result.status();
  }
  if (op.has_value()) {
    rpn.nodes.emplace_back(*op);
    switch (*op) {
      case OP_NOT: {
        bool can_not = false;
        if (result->dtype.IsBool()) {
          can_not = true;
        } else if (result->dtype.IsSimdVector() && result->dtype.Elem().IsBool()) {
          can_not = true;
          std::string fname = GetFunctionName(*op, result->dtype);
          auto check_result = ctx.CheckFuncExist(fname, true);
          if (!check_result.ok()) {
            return check_result.status();
          }
        } else if (result->dtype.IsSimdColumnPtr()) {
          can_not = true;
          // todo
          std::string fname = GetFunctionName(*op, result->dtype);
          auto check_result = ctx.CheckFuncExist(fname, true);
          if (!check_result.ok()) {
            return check_result.status();
          }
        }
        if (!can_not) {
          return ctx.GetErrorStatus(fmt::format("can NOT do not op on non bool value:{}", result->dtype));
        }
        break;
      }
      case OP_NEGATIVE: {
        bool can_neg = false;
        if (result->dtype.IsNumber() && result->dtype.IsSigned()) {
          can_neg = true;
        } else if (result->dtype.IsSimdVector() && result->dtype.Elem().IsSigned()) {
          can_neg = true;
          std::string fname = GetFunctionName(*op, result->dtype);
          auto check_result = ctx.CheckFuncExist(fname, true);
          if (!check_result.ok()) {
            return check_result.status();
          }
        } else if (result->dtype.IsSimdColumnPtr()) {
          can_neg = true;
          std::string fname = GetFunctionName(*op, result->dtype);
          auto check_result = ctx.CheckFuncExist(fname, true);
          if (!check_result.ok()) {
            return check_result.status();
          }
        }
        if (!can_neg) {
          return ctx.GetErrorStatus(fmt::format("can NOT do negative op on non number value:{}", result->dtype));
        }
        break;
      }
      default: {
        break;
      }
    }
  }
  // rpn.dtype = result->dtype;
  rpn.SetDType(ctx, result->dtype);
  return result;
}

static bool IsValidSimdVectorBinaryOperands(ParseContext& ctx, DType left, DType right) {
  if (left.IsSimdVector() || right.IsSimdVector()) {
    if (left.IsSimdVector() && right.IsSimdVector()) {
      if (left != right) {
        return false;
      }
      return true;
    } else {
      DType left_ele_dtype = left.Elem();
      DType right_ele_dtype = right.Elem();
      if (left_ele_dtype.IsNumber() && right_ele_dtype.IsNumber()) {
        return true;
      }
      if (left_ele_dtype.CanCastTo(DATA_STRING_VIEW) && right_ele_dtype.CanCastTo(DATA_STRING_VIEW)) {
        return true;
      }
      return false;
    }
  }
  return false;
}
absl::StatusOr<VarTag> BinaryExpr::Validate(ParseContext& ctx, RPN& rpn) {
  ctx.SetPosition(position);
  auto left_result = validate_operand(ctx, left, rpn);
  if (!left_result.ok()) {
    return left_result.status();
  }
  VarTag left_var = left_result.value();
  for (auto [op, right_operand] : right) {
    auto right_result = validate_operand(ctx, right_operand, rpn);
    if (!right_result.ok()) {
      return right_result.status();
    }
    rpn.nodes.emplace_back(op);
    bool can_binary_op = true;
    std::string implicit_func_call;
    do {
      if (left_var.dtype == right_result->dtype) {
        can_binary_op = true;
        break;
      }
      if (left_var.dtype.IsVoid() && !left_var.name.empty() && op == OP_ASSIGN) {
        can_binary_op = true;
        break;
      }
      if (left_var.dtype.IsSimdColumnPtr() || right_result->dtype.IsSimdColumnPtr()) {
        can_binary_op = true;
        if (!left_var.dtype.IsSimdColumnPtr() && !left_var.dtype.IsPrimitive()) {
          can_binary_op = false;
        }
        if (!right_result->dtype.IsSimdColumnPtr() && !right_result->dtype.IsPrimitive()) {
          can_binary_op = false;
        }
        break;
      }
      if (left_var.dtype.IsSimdVector() || right_result->dtype.IsSimdVector()) {
        if (IsValidSimdVectorBinaryOperands(ctx, left_var.dtype, right_result->dtype)) {
          can_binary_op = true;
        } else {
          can_binary_op = false;
        }
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
      if (!ctx.CanCastTo(left_var.dtype, right_result->dtype) && !ctx.CanCastTo(right_result->dtype, left_var.dtype)) {
        can_binary_op = false;
        break;
      }
      if (left_var.dtype.IsNumber() && right_result->dtype.IsNumber()) {
        if (left_var.dtype < right_result->dtype) {
          left_var.dtype = right_result->dtype;
        }
      } else {
        if (left_var.dtype.CanCastTo(right_result->dtype)) {
          left_var.dtype = right_result->dtype;
        } else {
          right_result->dtype = left_var.dtype;
        }
      }

    } while (0);
    if (!can_binary_op) {
      return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{} by expression validate ",
                                            op, left_var.dtype, right_result->dtype));
    }
    // if (implicit_func_call.empty()) {
    //   if (left_var.dtype.IsStringPtr() || right_result->dtype.IsStringPtr()) {
    //     implicit_func_call = functions::kBuiltinCastStdStrToStringView;
    //   } else if (left_var.dtype.IsFlatbuffersStringPtr() || right_result->dtype.IsFlatbuffersStringPtr()) {
    //     implicit_func_call = functions::kBuiltinCastFbsStrToStringView;
    //   }
    // }
    if (!implicit_func_call.empty()) {
      auto result = ctx.CheckFuncExist(implicit_func_call, true);
      if (!result.ok()) {
        return result.status();
      }
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
      case OP_MOD_ASSIGN:
      case OP_POW: {
        if (!left_var.dtype.Elem().IsNumber() || !right_result->dtype.Elem().IsNumber()) {
          return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op,
                                                left_result->dtype, right_result->dtype));
        }
        if (right_result->dtype.IsSimdVector()) {
          left_var.dtype = right_result->dtype;
        }
        // if (op == OP_MOD || op == OP_MOD_ASSIGN) {
        //   if (left_var.dtype.Elem().IsFloat() || right_result->dtype.Elem().IsFloat()) {
        //     return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op,
        //                                           left_result->dtype, right_result->dtype));
        //   }
        // }
        break;
      }
      case OP_ASSIGN: {
        if (!left_var.name.empty()) {
          ctx.AddLocalVar(left_var.name, right_result->dtype);
        } else {
          // return ctx.GetErrorStatus(fmt::format("can NOT do {} on non var.", op));
        }
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
        std::string implicit_func_call;
        if (left_var.dtype.IsNumber() && right_result->dtype.IsNumber()) {
          can_cmp = true;
        } else if (left_var.dtype.IsStringView() && right_result->dtype.IsStringView()) {
          can_cmp = true;
          implicit_func_call = functions::kBuiltinStringViewCmp;
        } else if (left_var.dtype.IsJsonPtr() && right_result->dtype.IsPrimitive()) {
          can_cmp = true;
          if (right_result->dtype.IsFloat()) {
            implicit_func_call = functions::kBuiltinJsonCmpFloat;
          } else if (right_result->dtype.IsStringView()) {
            implicit_func_call = functions::kBuiltinJsonCmpString;
          } else {
            implicit_func_call = functions::kBuiltinJsonCmpInt;
          }
        } else if (left_var.dtype.IsPrimitive() && right_result->dtype.IsJsonPtr()) {
          can_cmp = true;
          if (left_result->dtype.IsFloat()) {
            implicit_func_call = functions::kBuiltinJsonCmpFloat;
          } else if (left_result->dtype.IsStringView()) {
            implicit_func_call = functions::kBuiltinJsonCmpString;
          } else {
            implicit_func_call = functions::kBuiltinJsonCmpInt;
          }
        } else if (left_var.dtype.IsJsonPtr() && right_result->dtype.IsJsonPtr()) {
          can_cmp = true;
          implicit_func_call = functions::kBuiltinJsonCmpJson;
        } else if (left_var.dtype.IsSimdVector() || right_result->dtype.IsSimdVector()) {
          can_cmp = true;
          auto left_ele_dtype = left_var.dtype.Elem();
          auto right_ele_dtype = right_result->dtype.Elem();
          if (!left_ele_dtype.IsNumber() && !left_ele_dtype.IsStringView()) {
            can_cmp = false;
          }
          if (!right_ele_dtype.IsNumber() && !right_ele_dtype.IsStringView()) {
            can_cmp = false;
          }
          implicit_func_call = GetFunctionName(op, left_var.dtype, right_result->dtype);
          // left_var = VarTag(DType(DATA_BIT).ToSimdVector());
        } else if (left_var.dtype.IsSimdColumnPtr() || right_result->dtype.IsSimdColumnPtr()) {
          can_cmp = true;
          implicit_func_call = GetFunctionName(op, left_var.dtype, right_result->dtype);
          // left_var = VarTag(DType(DATA_SIMD_COLUMN).ToPtr());
        }
        if (!can_cmp) {
          return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{} for cmp expression",
                                                op, left_var.dtype, right_result->dtype));
        }
        if (!implicit_func_call.empty()) {
          auto result = ctx.CheckFuncExist(implicit_func_call, true);
          if (!result.ok()) {
            return result.status();
          }
        }
        if (left_var.dtype.IsSimdVector() || right_result->dtype.IsSimdVector()) {
          left_var = VarTag(DType(DATA_BIT).ToSimdVector());
        } else if (left_var.dtype.IsSimdColumnPtr() || right_result->dtype.IsSimdColumnPtr()) {
          left_var = VarTag(DType(DATA_SIMD_COLUMN).ToPtr());
        } else {
          left_var = VarTag(DType(DATA_BIT));
        }
        break;
      }
      case OP_LOGIC_AND:
      case OP_LOGIC_OR: {
        if (left_var.dtype.IsSimdVectorBit() || right_result->dtype.IsSimdVectorBit()) {
          std::string implicit_func_call = GetFunctionName(op, left_var.dtype, right_result->dtype);
          auto result = ctx.CheckFuncExist(implicit_func_call, true);
          if (!result.ok()) {
            return result.status();
          }
          left_var = VarTag(DType(DATA_BIT).ToSimdVector());
        } else if (left_var.dtype.IsSimdColumnPtr() || right_result->dtype.IsSimdColumnPtr()) {
          std::string implicit_func_call = GetFunctionName(op, left_var.dtype, right_result->dtype);
          auto result = ctx.CheckFuncExist(implicit_func_call, true);
          if (!result.ok()) {
            return result.status();
          }
          left_var = VarTag(DType(DATA_SIMD_COLUMN).ToPtr());
        } else {
          if (!left_var.dtype.IsBit() || !right_result->dtype.IsBit()) {
            return ctx.GetErrorStatus(fmt::format("can NOT do {} with left dtype:{}, right dtype:{}", op,
                                                  left_var.dtype, right_result->dtype));
          }
          left_var = VarTag(DType(DATA_BIT));
        }
        break;
      }
      default: {
        return ctx.GetErrorStatus(fmt::format("Unimplemented {} with left dtype:{}, right dtype:{}", op, left_var.dtype,
                                              right_result->dtype));
      }
    }
  }
  // rpn.dtype = left_var.dtype;
  rpn.SetDType(ctx, left_var.dtype);
  return left_var;
}

absl::StatusOr<VarTag> SelectExpr::Validate(ParseContext& ctx, RPN& rpn) {
  ctx.SetPosition(position);
  // if (true_false_operands.has_value()) {
  //   select_rpn = std::make_shared<SelectRPNNode>();
  //   rpn.nodes.emplace_back(select_rpn);
  // }
  auto cond_result = validate_operand(ctx, cond, rpn);
  if (true_false_operands.has_value()) {
    if (!cond_result.ok()) {
      return cond_result.status();
    }
    // select_rpn->cond_rpn.SetDType(ctx, cond_result->dtype);
    auto [true_expr, false_expr] = *true_false_operands;
    auto true_expr_result = validate_operand(ctx, true_expr, rpn);
    auto false_expr_result = validate_operand(ctx, false_expr, rpn);
    if (!true_expr_result.ok()) {
      return true_expr_result.status();
    }
    if (!false_expr_result.ok()) {
      return false_expr_result.status();
    }
    rpn.nodes.emplace_back(OP_CONDITIONAL);
    rpn.SetDType(ctx, true_expr_result->dtype);
    // select_rpn->false_rpn.SetDType(ctx, false_expr_result->dtype);
    if (cond_result->dtype.IsBool()) {
      if (ctx.CanCastTo(true_expr_result->dtype, false_expr_result->dtype)) {
        ternary_result_dtype = false_expr_result->dtype;
        return VarTag(ternary_result_dtype);
      } else if (ctx.CanCastTo(false_expr_result->dtype, true_expr_result->dtype)) {
        ternary_result_dtype = true_expr_result->dtype;
        return VarTag(ternary_result_dtype);
      }
    } else if (cond_result->dtype.IsSimdVectorBit()) {
      bool can_select = false;
      std::string implicit_func_call;
      if (true_expr_result->dtype.IsSimdVector() && false_expr_result->dtype.IsSimdVector()) {
        if (true_expr_result->dtype == false_expr_result->dtype) {
          ternary_result_dtype = true_expr_result->dtype;
          can_select = true;
          implicit_func_call =
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype);
        }
      } else if (true_expr_result->dtype.IsSimdVector() && !false_expr_result->dtype.IsSimdVector()) {
        if (ctx.CanCastTo(false_expr_result->dtype, true_expr_result->dtype.Elem())) {
          ternary_result_dtype = true_expr_result->dtype;
          can_select = true;
          implicit_func_call =
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype);
        }
      } else if (!true_expr_result->dtype.IsSimdVector() && false_expr_result->dtype.IsSimdVector()) {
        if (ctx.CanCastTo(true_expr_result->dtype, false_expr_result->dtype.Elem())) {
          ternary_result_dtype = false_expr_result->dtype;
          can_select = true;
          implicit_func_call =
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype);
        }
      } else {
        if (true_expr_result->dtype.IsNumber() && false_expr_result->dtype.IsNumber()) {
          ternary_result_dtype =
              true_expr_result->dtype >= false_expr_result->dtype ? true_expr_result->dtype : false_expr_result->dtype;
          can_select = true;
          implicit_func_call =
              GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype);
        }
      }
      if (can_select) {
        if (!implicit_func_call.empty()) {
          auto result = ctx.CheckFuncExist(implicit_func_call, true);
          if (!result.ok()) {
            return result.status();
          }
        }
        return VarTag(ternary_result_dtype.ToSimdVector());
      }
    } else if (cond_result->dtype.IsSimdColumnPtr()) {
      bool valid_true_dtype = false;
      std::vector<std::string> implicit_func_calls;
      if (true_expr_result->dtype.IsSimdColumnPtr() || true_expr_result->dtype.IsPrimitive()) {
        valid_true_dtype = true;
      }
      bool valid_false_dtype = false;
      if (false_expr_result->dtype.IsSimdColumnPtr() || false_expr_result->dtype.IsPrimitive()) {
        valid_false_dtype = true;
      }

      if (valid_true_dtype && valid_false_dtype) {
        implicit_func_calls.emplace_back(
            GetFunctionName(OP_CONDITIONAL, cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
        for (auto implicit_func_call : implicit_func_calls) {
          auto result = ctx.CheckFuncExist(implicit_func_call, true);
          if (!result.ok()) {
            return result.status();
          }
        }
        DType result_dtype(DATA_SIMD_COLUMN);
        return VarTag(result_dtype.ToPtr());
      }
    }
    return ctx.GetErrorStatus(
        fmt::format("can NOT do ternary with cond dtype:{}, true_expr_dtype:{}, false_expr_dtype:{}",
                    cond_result->dtype, true_expr_result->dtype, false_expr_result->dtype));
  } else {
    if (cond_result.ok()) {
      // cond_rpn.dtype = cond_result->dtype;
      rpn.SetDType(ctx, cond_result->dtype);
    }
    return cond_result;
  }
}

absl::StatusOr<VarTag> VarAccessor::Validate(ParseContext& ctx, RPN& rpn, bool& as_builtin_op) {
  as_builtin_op = false;
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
                    bool can_brackets_op = false;
                    if (!var_dtype.dtype.IsPtr()) {
                      can_brackets_op = false;
                    } else if (var_dtype.dtype.IsJsonPtr()) {
                      can_brackets_op = true;
                    } else if (var_dtype.dtype.IsVectorPtr()) {
                      can_brackets_op = true;
                    } else if (var_dtype.dtype.IsMapPtr() || var_dtype.dtype.IsUnorderedMapPtr()) {
                      can_brackets_op = true;
                    } else if (var_dtype.dtype.IsSimdTablePtr()) {
                      can_brackets_op = true;
                    }
                    if (!can_brackets_op) {
                      return absl::StatusOr<VarTag>(absl::InvalidArgumentError(
                          fmt::format("invalid dtype:{} to do json dynamic access", var_dtype.dtype)));
                    }

                    std::string_view implicit_func_call;
                    DType ret_dtype;
                    using T = std::decay_t<decltype(dynamic_arg)>;
                    if (var_dtype.dtype.IsJsonPtr()) {
                      bool get_by_member = true;
                      if constexpr (std::is_same_v<uint32_t, T>) {
                        get_by_member = false;
                      } else if constexpr (std::is_same_v<std::string, T>) {
                      } else {
                        auto result = dynamic_arg.Validate(ctx);
                        if (!result.ok()) {
                          return absl::StatusOr<VarTag>(result.status());
                        }
                        auto var_ref_dtype = result.value().dtype;
                        if (var_ref_dtype.IsInteger()) {
                          get_by_member = false;
                        } else if (var_ref_dtype.IsStringView() || var_ref_dtype.IsStdStringView()) {
                          get_by_member = true;
                        } else {
                          return absl::StatusOr<VarTag>(absl::InvalidArgumentError(
                              fmt::format("Can NOT do json get by dtype:{}", var_ref_dtype)));
                        }
                      }
                      if (get_by_member) {
                        implicit_func_call = functions::kBuiltinJsonMemberGet;
                      } else {
                        implicit_func_call = functions::kBuiltinJsonArrayGet;
                      }
                      DType json_dtype(DATA_JSON);
                      ret_dtype = json_dtype.ToPtr();
                    } else if (var_dtype.dtype.IsVectorPtr() || var_dtype.dtype.IsMapPtr() ||
                               var_dtype.dtype.IsUnorderedMapPtr() || var_dtype.dtype.IsSimdTablePtr()) {
                      std::string member_func = "get";
                      auto field_accessor = Reflect::GetStructMember(var_dtype.dtype.PtrTo(), member_func);
                      if (!field_accessor) {
                        return absl::StatusOr<VarTag>(ctx.GetErrorStatus(fmt::format(
                            "Can NOT get member:{} accessor for dtype:{}", member_func, var_dtype.dtype.PtrTo())));
                      }
                      if (!field_accessor->member_func.has_value()) {
                        return absl::StatusOr<VarTag>(ctx.GetErrorStatus(fmt::format(
                            "Can NOT get member func:{} accessor for dtype:{}", member_func, var_dtype.dtype.PtrTo())));
                      }
                      ret_dtype = field_accessor->member_func->return_type;
                      if (ret_dtype.IsPtr()) {
                        auto ret_ptr_to = ret_dtype.PtrTo();
                        if (ret_ptr_to.IsSimdColumnPtr() || ret_ptr_to.IsInteger() || ret_ptr_to.IsFloat()) {
                          ret_dtype = ret_dtype.PtrTo();
                        }
                      }
                      ctx.AddMemberFuncCall(var_dtype.dtype.PtrTo(), member_func, *field_accessor->member_func);
                      access_func_names.emplace_back(GetMemberFuncName(var_dtype.dtype.PtrTo(), member_func));
                    }
                    if (!implicit_func_call.empty()) {
                      auto result = ctx.CheckFuncExist(implicit_func_call, true);
                      if (!result.ok()) {
                        return absl::StatusOr<VarTag>(result.status());
                      }
                      access_func_names.emplace_back(std::string(implicit_func_call));
                    }
                    return absl::StatusOr<VarTag>(ret_dtype);
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
    DType compute_dtype;
    bool has_simd_vector = false;
    bool has_simd_column = false;
    builtin_op = functions::get_buitin_func_op(name);
    int operand_count = get_operand_count(builtin_op);
    DType first_number_dtype;
    auto validate_result = func_args->Validate(ctx, operand_count > 0 ? &rpn : nullptr);
    if (!validate_result.ok()) {
      return validate_result.status();
    }

    std::vector<VarTag> arg_tag_dtypes = validate_result.value();
    std::vector<DType> arg_dtypes;
    for (auto arg_dtype : arg_tag_dtypes) {
      if (!has_simd_vector && arg_dtype.dtype > compute_dtype) {
        compute_dtype = arg_dtype.dtype;
      }
      if (arg_dtype.dtype.IsSimdVector()) {
        has_simd_vector = true;
        compute_dtype = arg_dtype.dtype;
      }
      if (arg_dtype.dtype.IsNumber() && first_number_dtype.IsInvalid()) {
        first_number_dtype = arg_dtype.dtype;
      }
      arg_dtypes.emplace_back(arg_dtype.dtype);
    }
    if (builtin_op != OP_INVALID) {
      if (operand_count > 0) {
        rpn.nodes.emplace_back(builtin_op);
        as_builtin_op = true;
        DType ret_dtype = compute_dtype;
        rpn.SetDType(ctx, ret_dtype);
        return ret_dtype;
      }

      if (name == kOpTokenStrs[OP_IOTA]) {
        name = GetFunctionName(OP_IOTA, first_number_dtype);
      } else if (has_simd_vector || has_simd_column) {
        name = GetFunctionName(builtin_op, arg_dtypes);
      } else {
        name = GetFunctionName(builtin_op, compute_dtype);
      }
    }
    auto result = ctx.CheckFuncExist(name);
    if (!result.ok()) {
      return result.status();
    }
    auto* desc = result.value();
    if (operand_count > 0) {
      rpn.nodes.emplace_back(builtin_op);
      as_builtin_op = true;
      return desc->return_type;
    }

    if (desc->context_arg_idx >= 0) {
      if (ctx.GetFuncContextArgIdx() >= 0 && arg_dtypes.size() == desc->arg_types.size() - 1) {
        DType ctx_ptr_dtype = DType(DATA_CONTEXT).ToPtr();
        arg_dtypes.insert(arg_dtypes.begin() + desc->context_arg_idx, ctx_ptr_dtype);
      }
    }
    if (!desc->ValidateArgs(arg_dtypes)) {
      if (desc->context_arg_idx >= 0 && ctx.GetFuncContextArgIdx() < 0) {
        return ctx.GetErrorStatus(
            fmt::format("Function:{} need `rapidudf::Context` arg, missing in expression/udf args.", name));
      }
      return ctx.GetErrorStatus(fmt::format("Invalid func call:{} args with invalid args", name));
    }

    return desc->return_type;
  } else {
    auto result = ctx.IsVarExist(name, false);
    if (!result.ok()) {
      return result;
    }
    return VarTag(result.value(), name);
  }
}

}  // namespace ast
}  // namespace rapidudf