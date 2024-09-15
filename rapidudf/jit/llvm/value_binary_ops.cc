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

#include <llvm/IR/Use.h>
#include <string_view>
#include <vector>
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {
ValuePtr Value::BinaryOp(OpToken op, ValuePtr right) {
  ValuePtr left = SelfPtr();
  DType dst_dtype = dtype_;
  DType right_dtype = right->dtype_;
  if (right->dtype_ != dtype_) {
    if (right->dtype_.IsNumber() && dtype_.IsNumber()) {
      if (right->dtype_ > dtype_) {
        dst_dtype = right->dtype_;
        left = left->CastTo(dst_dtype);
      } else {
        dst_dtype = dtype_;
        right = right->CastTo(dst_dtype);
      }
    } else {
      if (dtype_.CanCastTo(right->dtype_)) {
        dst_dtype = right->dtype_;
        left = left->CastTo(dst_dtype);
      } else if (right->dtype_.CanCastTo(dtype_)) {
        right = right->CastTo(dst_dtype);
      } else {
        if ((left->GetDType().IsJsonPtr() || right->GetDType().IsJsonPtr()) &&
            (op >= OP_EQUAL && op <= OP_GREATER_EQUAL)) {
          // continue cmp json
        } else {
          RUDF_ERROR("Can NOT do {} for left:{}, right:{}", op, dtype_, right_dtype);
          return {};
        }
      }
    }

    if (!left || !right) {
      RUDF_ERROR("Can NOT do {} for left:{}, right:{}", op, dtype_, right_dtype);
      return {};
    }
  }
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  DType ret_dtype = dst_dtype;
  ::llvm::Value* result_val = nullptr;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Type* intrinsic_arg_type = get_type(ir_builder_->getContext(), dst_dtype);
  switch (op) {
    case OP_PLUS:
    case OP_PLUS_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateAdd(left->GetValue(), right->GetValue());
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFAdd(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_MINUS:
    case OP_MINUS_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateSub(left->GetValue(), right->GetValue());
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFSub(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_MULTIPLY:
    case OP_MULTIPLY_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateMul(left->GetValue(), right->GetValue());
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFMul(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_DIVIDE:
    case OP_DIVIDE_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateSDiv(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateUDiv(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFDiv(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_MOD:
    case OP_MOD_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateSRem(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateURem(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFRem(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_LOGIC_OR: {
      if (dst_dtype.IsBool()) {
        result_val = ir_builder_->CreateLogicalOr(left->GetValue(), right->GetValue());
        ret_dtype = DATA_BIT;
      }
      break;
    }
    case OP_LOGIC_AND: {
      RUDF_INFO("{} {}", left->GetDType(), right->GetDType());
      if (dst_dtype.IsBool()) {
        result_val = ir_builder_->CreateLogicalAnd(left->GetValue(), right->GetValue());
        ret_dtype = DATA_BIT;
      }
      break;
    }
    case OP_EQUAL: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateICmpEQ(left->GetValue(), right->GetValue());
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOEQ(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_NOT_EQUAL: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateICmpNE(left->GetValue(), right->GetValue());
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpONE(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_GREATER: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSGT(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateICmpUGT(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOGT(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_GREATER_EQUAL: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSGE(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateICmpUGE(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOGE(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_LESS: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSLT(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateICmpULT(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOLT(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_LESS_EQUAL: {
      ret_dtype = DATA_BIT;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSLE(left->GetValue(), right->GetValue());
        } else {
          result_val = ir_builder_->CreateICmpULE(left->GetValue(), right->GetValue());
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOLE(left->GetValue(), right->GetValue());
      }
      break;
    }
    case OP_POW: {
      builtin_intrinsic = ::llvm::Intrinsic::pow;
      if (dtype_.IsF32()) {
        intrinsic_arg_type = ::llvm::Type::getFloatTy(ir_builder_->getContext());
      } else if (dtype_.IsF64()) {
        intrinsic_arg_type = ::llvm::Type::getDoubleTy(ir_builder_->getContext());
      } else if (dtype_.IsInteger()) {
        left = left->CastTo(DATA_F64);
        right = right->CastTo(DATA_F64);
        intrinsic_arg_type = ::llvm::Type::getDoubleTy(ir_builder_->getContext());
        ret_dtype = DATA_F64;
      }
      builtin_intrinsic_args.emplace_back(left->GetValue());
      builtin_intrinsic_args.emplace_back(right->GetValue());
      break;
    }
    case OP_MAX: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          builtin_intrinsic = ::llvm::Intrinsic::smax;
        } else {
          builtin_intrinsic = ::llvm::Intrinsic::umax;
        }
      } else if (dst_dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::maximum;
      }
      builtin_intrinsic_args.emplace_back(left->GetValue());
      builtin_intrinsic_args.emplace_back(right->GetValue());
      break;
    }
    case OP_MIN: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          builtin_intrinsic = ::llvm::Intrinsic::smin;
        } else {
          builtin_intrinsic = ::llvm::Intrinsic::umin;
        }
      } else if (dst_dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::minimum;
      }
      builtin_intrinsic_args.emplace_back(left->GetValue());
      builtin_intrinsic_args.emplace_back(right->GetValue());
      break;
    }
    default: {
      break;
    }
  }

  if (dst_dtype.IsStringView() && op >= OP_EQUAL && op <= OP_GREATER_EQUAL) {
    auto op_arg = New(DATA_U32, compiler_, ir_builder_->getInt32(op));
    std::vector<ValuePtr> args{op_arg, left, right};
    auto result = compiler_->CallFunction(kBuiltinStringViewCmp, args);
    if (result.ok()) {
      return result.value();
    } else {
      RUDF_ERROR("Can NOT do {} for left:{}, right:{} with error:{}", op, left->GetDType(), right->GetDType(),
                 result.status().ToString());
      return {};
    }
  }
  if ((left->GetDType().IsJsonPtr() || right->GetDType().IsJsonPtr()) && (op >= OP_EQUAL && op <= OP_GREATER_EQUAL)) {
    return left->JsonCmp(op, right, false);
  }

  if (builtin_intrinsic != 0) {
    if (nullptr != intrinsic_arg_type) {
      ::llvm::Function* intrinsic_func = ::llvm::Intrinsic::getDeclaration(ir_builder_->GetInsertBlock()->getModule(),
                                                                           builtin_intrinsic, {intrinsic_arg_type});
      result_val = ir_builder_->CreateCall(intrinsic_func, builtin_intrinsic_args);
    }
  }

  if (!result_val) {
    RUDF_ERROR("Can NOT do {} for left:{}, right:{}", op, left->GetDType(), right->GetDType());
    return {};
  }
  return New(ret_dtype, compiler_, result_val);
}

ValuePtr Value::JsonCmp(OpToken op, ValuePtr right, bool reverse) {
  auto& other = *right;
  std::string_view cmp_func;
  ValuePtr other_val = other.SelfPtr();
  std::vector<ValuePtr> cmp_args;
  cmp_args.emplace_back(New(DATA_U32, compiler_, ir_builder_->getInt32(op)));

  if (dtype_.IsJsonPtr()) {
    if (other.GetDType().IsNumber() || other.GetDType().IsStringView()) {
      if (other.GetDType().IsStringView() || other.GetDType().IsF64() || other.GetDType().IsBool()) {
        // do nothing
      } else {
        if (other.GetDType().IsF32()) {
          other_val = other.CastTo(DATA_F64);
        } else {
          other_val = other.CastTo(DATA_I64);
        }
      }
      cmp_args.emplace_back(SelfPtr());
      cmp_args.emplace_back(other_val);
      cmp_args.emplace_back(New(DATA_BIT, compiler_, ir_builder_->getInt1(reverse)));
      switch (other_val->GetDType().GetFundamentalType()) {
        case DATA_STRING_VIEW: {
          cmp_func = kBuiltinJsonCmpString;
          break;
        }
        case DATA_I32:
        case DATA_I64: {
          cmp_func = kBuiltinJsonCmpInt;
          break;
        }
        case DATA_F64: {
          cmp_func = kBuiltinJsonCmpFloat;
          break;
        }
        case DATA_BIT: {
          cmp_func = kBuiltinJsonCmpBool;
          break;
        }
        default: {
          break;
        }
      }
    } else if (other.GetDType().IsJsonPtr()) {
      cmp_func = kBuiltinJsonCmpJson;
      cmp_args.emplace_back(SelfPtr());
      cmp_args.emplace_back(other_val);
    } else {
      RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
      return {};
    }
  } else if (GetDType().IsNumber() || GetDType().IsStringView()) {
    if (other.GetDType().IsJsonPtr()) {
      return other.JsonCmp(op, SelfPtr(), true);
    } else {
      RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
      return {};
    }
  } else {
    RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
    return {};
  }
  auto result = compiler_->CallFunction(cmp_func, cmp_args);
  if (!result.ok()) {
    return {};
  }
  return result.value();
}

}  // namespace llvm
}  // namespace rapidudf