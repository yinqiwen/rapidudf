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
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
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
    if (dtype_.CanCastTo(right->dtype_)) {
      dst_dtype = right->dtype_;
      left = left->CastTo(dst_dtype);
    } else if (right->dtype_.CanCastTo(dtype_)) {
      right = right->CastTo(dst_dtype);
    }
    if (!left || !right) {
      RUDF_ERROR("Can NOT do {} for left:{}, right:{}", op, dtype_, right_dtype);
      return {};
    }
  }
  DType ret_dtype = dst_dtype;
  ::llvm::Value* result_val = nullptr;
  switch (op) {
    case OP_PLUS:
    case OP_PLUS_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateAdd(left->GetValue(), right->val_);
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFAdd(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_MINUS:
    case OP_MINUS_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateSub(left->GetValue(), right->val_);
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFSub(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_MULTIPLY:
    case OP_MULTIPLY_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateMul(left->GetValue(), right->val_);
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFMul(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_DIVIDE:
    case OP_DIVIDE_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateSDiv(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateUDiv(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFDiv(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_MOD:
    case OP_MOD_ASSIGN: {
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateSRem(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateURem(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFRem(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_LOGIC_OR: {
      if (dst_dtype.IsBool()) {
        result_val = ir_builder_->CreateLogicalOr(left->GetValue(), right->val_);
        ret_dtype = DATA_U8;
      }
      break;
    }
    case OP_LOGIC_AND: {
      if (dst_dtype.IsBool()) {
        result_val = ir_builder_->CreateLogicalAnd(left->GetValue(), right->val_);
        ret_dtype = DATA_U8;
      }
      break;
    }
    case OP_EQUAL: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateICmpEQ(left->GetValue(), right->val_);
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOEQ(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_NOT_EQUAL: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        result_val = ir_builder_->CreateICmpNE(left->GetValue(), right->val_);
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpONE(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_GREATER: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSGT(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateICmpUGT(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOGT(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_GREATER_EQUAL: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSGE(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateICmpUGE(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOGE(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_LESS: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSLT(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateICmpULT(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOLT(left->GetValue(), right->val_);
      }
      break;
    }
    case OP_LESS_EQUAL: {
      ret_dtype = DATA_U8;
      if (dst_dtype.IsInteger()) {
        if (dst_dtype.IsSigned()) {
          result_val = ir_builder_->CreateICmpSLE(left->GetValue(), right->val_);
        } else {
          result_val = ir_builder_->CreateICmpULE(left->GetValue(), right->val_);
        }
      } else if (dst_dtype.IsFloat()) {
        result_val = ir_builder_->CreateFCmpOLE(left->GetValue(), right->val_);
      }
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
  if (!result_val) {
    RUDF_ERROR("Can NOT do {} for left:{}, right:{}", op, left->GetDType(), right->GetDType());
    return {};
  }
  return New(ret_dtype, compiler_, result_val);
}
}  // namespace llvm
}  // namespace rapidudf