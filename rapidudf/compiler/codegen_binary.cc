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
#include <functional>
#include "rapidudf/compiler/codegen.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "fmt/format.h"

#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::BinaryOp(OpToken op, DType dtype, ::llvm::Value* left, ::llvm::Value* right) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* ret_type = left->getType();
  ::llvm::Type* element_type = GetElementType(left->getType());
  DType element_dtype = dtype.Elem();
  std::function<::llvm::Value*(::llvm::Value*)> intrinsic_post_transformer;
  switch (op) {
    case OP_PLUS: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFAdd(left, right);
      } else {
        ret_value = builder_->CreateAdd(left, right);
      }
      break;
    }
    case OP_MINUS: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFSub(left, right);
      } else {
        ret_value = builder_->CreateSub(left, right);
      }
      break;
    }
    case OP_MULTIPLY: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFMul(left, right);
      } else {
        ret_value = builder_->CreateMul(left, right);
      }
      break;
    }
    case OP_DIVIDE: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFDiv(left, right);
      } else {
        if (dtype.IsSigned()) {
          ret_value = builder_->CreateSDiv(left, right);
        } else {
          ret_value = builder_->CreateUDiv(left, right);
        }
      }
      break;
    }
    case OP_MOD: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFRem(left, right);
      } else {
        if (dtype.IsSigned()) {
          ret_value = builder_->CreateSRem(left, right);
        } else {
          ret_value = builder_->CreateURem(left, right);
        }
      }
      break;
    }
    case OP_LOGIC_OR: {
      if (element_dtype.IsBool()) {
        ret_value = builder_->CreateLogicalOr(left, right);
      }
      break;
    }
    case OP_LOGIC_AND: {
      if (element_dtype.IsBool()) {
        ret_value = builder_->CreateLogicalAnd(left, right);
      }
      break;
    }
    case OP_LOGIC_XOR: {
      if (element_dtype.IsBool()) {
        ret_value = builder_->CreateXor(left, right);
      }
      break;
    }
    case OP_EQUAL: {
      if (element_dtype.IsInteger()) {
        ret_value = builder_->CreateICmpEQ(left, right);
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpOEQ(left, right);
      }
      break;
    }
    case OP_NOT_EQUAL: {
      if (element_dtype.IsInteger()) {
        ret_value = builder_->CreateICmpNE(left, right);
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpONE(left, right);
      }
      break;
    }
    case OP_GREATER: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          ret_value = builder_->CreateICmpSGT(left, right);
        } else {
          ret_value = builder_->CreateICmpUGT(left, right);
        }
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpOGT(left, right);
      }
      break;
    }
    case OP_GREATER_EQUAL: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          ret_value = builder_->CreateICmpSGE(left, right);
        } else {
          ret_value = builder_->CreateICmpUGE(left, right);
        }
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpOGE(left, right);
      }
      break;
    }
    case OP_LESS: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          ret_value = builder_->CreateICmpSLT(left, right);
        } else {
          ret_value = builder_->CreateICmpULT(left, right);
        }
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpOLT(left, right);
      }
      break;
    }
    case OP_LESS_EQUAL: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          ret_value = builder_->CreateICmpSLE(left, right);
        } else {
          ret_value = builder_->CreateICmpULE(left, right);
        }
      } else if (element_dtype.IsFloat()) {
        ret_value = builder_->CreateFCmpOLE(left, right);
      }
      break;
    }
    case OP_POW: {
      if (element_dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::pow;
        builtin_intrinsic_args.emplace_back(left);
        builtin_intrinsic_args.emplace_back(right);
      } else if (element_dtype.IsInteger()) {
        builtin_intrinsic = ::llvm::Intrinsic::pow;
        auto new_left = CastTo(left, dtype, DATA_F64);
        auto new_right = CastTo(right, dtype, DATA_F64);
        if (!new_left.ok()) {
          return new_left.status();
        }
        if (!new_right.ok()) {
          return new_right.status();
        }
        builtin_intrinsic_args.emplace_back(new_left.value());
        builtin_intrinsic_args.emplace_back(new_right.value());
        ret_type = builder_->getDoubleTy();
        intrinsic_post_transformer = [&](::llvm::Value* v) { return CastTo(v, DATA_F64, dtype).value(); };
      }
      break;
    }
    case OP_MAX: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          builtin_intrinsic = ::llvm::Intrinsic::smax;
        } else {
          builtin_intrinsic = ::llvm::Intrinsic::umax;
        }
      } else if (element_dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::maximum;
      }
      builtin_intrinsic_args.emplace_back(left);
      builtin_intrinsic_args.emplace_back(right);
      break;
    }
    case OP_MIN: {
      if (element_dtype.IsInteger()) {
        if (element_dtype.IsSigned()) {
          builtin_intrinsic = ::llvm::Intrinsic::smin;
        } else {
          builtin_intrinsic = ::llvm::Intrinsic::umin;
        }
      } else if (element_dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::minimum;
      }
      builtin_intrinsic_args.emplace_back(left);
      builtin_intrinsic_args.emplace_back(right);
      break;
    }
    default: {
      return absl::InvalidArgumentError(fmt::format("Unsupported op:{}", op));
    }
  }
  if (ret_value != nullptr) {
    return ret_value;
  }
  if (builtin_intrinsic > 0) {
    ::llvm::Value* v = builder_->CreateIntrinsic(ret_type, builtin_intrinsic, builtin_intrinsic_args);
    if (intrinsic_post_transformer) {
      v = intrinsic_post_transformer(v);
    }
    return v;
  }
  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}

absl::StatusOr<ValuePtr> CodeGen::BinaryOp(OpToken op, ValuePtr left, ValuePtr right) {
  bool need_assign = false;
  switch (op) {
    case OP_ASSIGN: {
      need_assign = true;
      break;
    }
    case OP_PLUS_ASSIGN: {
      op = OP_PLUS;
      need_assign = true;
      break;
    }
    case OP_MINUS_ASSIGN: {
      op = OP_MINUS;
      need_assign = true;
      break;
    }
    case OP_MULTIPLY_ASSIGN: {
      op = OP_MULTIPLY;
      need_assign = true;
      break;
    }
    case OP_DIVIDE_ASSIGN: {
      op = OP_DIVIDE;
      need_assign = true;
      break;
    }
    case OP_MOD_ASSIGN: {
      op = OP_MOD;
      need_assign = true;
      break;
    }
    default: {
      break;
    }
  }
  ValuePtr result_val;
  if (op != OP_ASSIGN) {
    DType compyte_dtype = left->GetDType();
    if (left->GetDType() != right->GetDType()) {
      auto normalize_dtype_result = NormalizeDType({left->GetDType(), right->GetDType()});
      if (!normalize_dtype_result.ok()) {
        return normalize_dtype_result.status();
      }
      compyte_dtype = normalize_dtype_result.value();
      auto cast_result = CastTo(left, compyte_dtype);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      left = cast_result.value();
      cast_result = CastTo(right, compyte_dtype);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      right = cast_result.value();
    }
    DType result_dtype = compyte_dtype;
    if (is_compare_op(op)) {
      result_dtype = DType(DATA_BIT);
    }
    if (compyte_dtype.IsStringView()) {
      if (op >= OP_EQUAL && op <= OP_GREATER_EQUAL) {
        auto op_arg = NewU32(static_cast<uint32_t>(op));
        std::vector<ValuePtr> args{op_arg, left, right};
        auto result = CallFunction(functions::kBuiltinStringViewCmp, args);
        if (result.ok()) {
          return result.value();
        } else {
          return result.status();
        }
      }
    }

    auto result = BinaryOp(op, compyte_dtype, left->LoadValue(), right->LoadValue());
    if (!result.ok()) {
      std::string extern_func_name = GetFunctionName(op, left->GetDType());
      auto val_result = CallFunction(extern_func_name, {left, right});
      if (val_result.ok()) {
        result_val = val_result.value();
      } else {
        return result.status();
      }
    } else {
      result_val = NewValue(result_dtype, result.value());
    }
  } else {
    result_val = right;
  }
  if (need_assign) {
    auto status = left->CopyFrom(result_val);
    if (!status.ok()) {
      return status;
    }
    return left;
  } else {
    return result_val;
  }
}

}  // namespace compiler
}  // namespace rapidudf