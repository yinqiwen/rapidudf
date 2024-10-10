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
#include "rapidudf/jit/llvm/codegen.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

static ::llvm::Type* get_element_type(::llvm::Type* t) {
  if (t->isVectorTy()) {
    ::llvm::VectorType* vtype = reinterpret_cast<::llvm::VectorType*>(t);
    return vtype->getElementType();
  } else {
    return t;
  }
}

CodeGen::CodeGen(::llvm::IRBuilderBase* builder, uint32_t& label_cursor)
    : builder_(builder), label_cursor_(label_cursor) {}

absl::StatusOr<::llvm::Value*> CodeGen::CastTo(::llvm::Value* val, DType src_dtype, DType dst_dtype) {
  ::llvm::Value* new_val = nullptr;
  ::llvm::Type* dst_type = get_type(builder_->getContext(), dst_dtype);
  if (val->getType()->isVectorTy()) {
    ::llvm::VectorType* vtype = reinterpret_cast<::llvm::VectorType*>(val->getType());
    dst_type = ::llvm::VectorType::get(dst_type, vtype->getElementCount());
  }
  if (dst_dtype.IsFloat()) {
    if (src_dtype.IsInteger()) {
      if (src_dtype.IsSigned()) {
        new_val = builder_->CreateSIToFP(val, dst_type);
      } else {
        new_val = builder_->CreateUIToFP(val, dst_type);
      }
    } else {
      if (src_dtype.Bits() > dst_dtype.Bits()) {
        new_val = builder_->CreateFPExt(val, dst_type);
      } else {
        new_val = builder_->CreateFPTrunc(val, dst_type);
      }
    }
  } else {
    if (src_dtype.IsFloat()) {
      if (dst_dtype.IsSigned()) {
        new_val = builder_->CreateFPToSI(val, dst_type);
      } else {
        new_val = builder_->CreateFPToUI(val, dst_type);
      }
    } else {
      if (dst_dtype.Bits() > src_dtype.Bits()) {
        if (dst_dtype.IsSigned()) {
          new_val = builder_->CreateZExt(val, dst_type);
        } else {
          new_val = builder_->CreateZExt(val, dst_type);
        }
      } else {
        new_val = builder_->CreateTrunc(val, dst_type);
      }
    }
  }
  if (nullptr != new_val) {
    return new_val;
  }
  return absl::InvalidArgumentError(fmt::format("Can NOT cast from {} to {}", src_dtype, dst_dtype));
}
absl::StatusOr<::llvm::Value*> CodeGen::CastTo(::llvm::Value* val, DType src_dtype, ::llvm::Value* dst_dtype) {
  //  builder_->CreateSwitch(dst_dtype, BasicBlock *Dest)
  return absl::UnimplementedError("####CastTo");
}

absl::StatusOr<::llvm::Value*> CodeGen::UnaryOp(OpToken op, DType dtype, ::llvm::Value* val) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* element_type = get_element_type(val->getType());
  switch (op) {
    case OP_NEGATIVE: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFNeg(val);
      } else {
        ret_value = builder_->CreateNeg(val);
      }
      break;
    }
    case OP_NOT: {
      ret_value = builder_->CreateNot(val);
      break;
    }
    case OP_SIN: {
      builtin_intrinsic = ::llvm::Intrinsic::sin;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_COS: {
      builtin_intrinsic = ::llvm::Intrinsic::cos;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_FLOOR: {
      builtin_intrinsic = ::llvm::Intrinsic::floor;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_ABS: {
      builtin_intrinsic_args.emplace_back(val);
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::fabs;
      } else {
        builtin_intrinsic = ::llvm::Intrinsic::abs;
        builtin_intrinsic_args.emplace_back(builder_->getInt1(0));
      }
      break;
    }
    case OP_SQRT: {
      builtin_intrinsic = ::llvm::Intrinsic::sqrt;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_CEIL: {
      builtin_intrinsic = ::llvm::Intrinsic::ceil;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_ROUND: {
      builtin_intrinsic = ::llvm::Intrinsic::round;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_EXP: {
      builtin_intrinsic = ::llvm::Intrinsic::exp;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_EXP2: {
      builtin_intrinsic = ::llvm::Intrinsic::exp2;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_LOG: {
      builtin_intrinsic = ::llvm::Intrinsic::log;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_LOG2: {
      builtin_intrinsic = ::llvm::Intrinsic::log2;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_LOG10: {
      builtin_intrinsic = ::llvm::Intrinsic::log10;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_RINT: {
      builtin_intrinsic = ::llvm::Intrinsic::rint;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_TRUNC: {
      builtin_intrinsic = ::llvm::Intrinsic::trunc;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
#if LLVM_VERSION_MAJOR >= 19
    case OP_TAN: {
      builtin_intrinsic = ::llvm::Intrinsic::tan;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_ASIN: {
      builtin_intrinsic = ::llvm::Intrinsic::asin;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_ACOS: {
      builtin_intrinsic = ::llvm::Intrinsic::acos;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_ATAN: {
      builtin_intrinsic = ::llvm::Intrinsic::atan;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_SINH: {
      builtin_intrinsic = ::llvm::Intrinsic::sinh;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_COSH: {
      builtin_intrinsic = ::llvm::Intrinsic::cosh;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
    case OP_TANH: {
      builtin_intrinsic = ::llvm::Intrinsic::tanh;
      builtin_intrinsic_args.emplace_back(val);
      break;
    }
#endif
    default: {
      return absl::InvalidArgumentError(fmt::format("Unsupported op:{}", op));
    }
  }
  if (ret_value != nullptr) {
    return ret_value;
  }
  if (builtin_intrinsic > 0) {
    return builder_->CreateIntrinsic(val->getType(), builtin_intrinsic, builtin_intrinsic_args);
  }
  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}

absl::StatusOr<::llvm::Value*> CodeGen::BinaryOp(OpToken op, DType dtype, ::llvm::Value* left, ::llvm::Value* right) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* ret_type = left->getType();
  ::llvm::Type* element_type = get_element_type(left->getType());
  DType element_dtype = dtype.Elem();
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
    return builder_->CreateIntrinsic(ret_type, builtin_intrinsic, builtin_intrinsic_args);
  }
  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}

absl::StatusOr<::llvm::Value*> CodeGen::TernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                                  ::llvm::Value* c) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* ret_type = b->getType();
  ::llvm::Type* element_type = get_element_type(b->getType());
  DType element_dtype = dtype.Elem();
  switch (op) {
    case OP_CONDITIONAL: {
      ret_value = builder_->CreateSelect(a, b, c);
      break;
    }
    case OP_FMA: {
      if (dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::fma;
        builtin_intrinsic_args.emplace_back(a);
        builtin_intrinsic_args.emplace_back(b);
        builtin_intrinsic_args.emplace_back(c);
      }
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
    return builder_->CreateIntrinsic(ret_type, builtin_intrinsic, builtin_intrinsic_args);
  }
  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}

absl::StatusOr<::llvm::Value*> CodeGen::NewConstVectorValue(DType dtype, ::llvm::Value* val) {
  if (::llvm::isa<::llvm::Constant>(val)) {
    auto vector_type = get_vector_type(builder_->getContext(), dtype);
    return ::llvm::ConstantVector::getSplat(vector_type->getElementCount(), reinterpret_cast<::llvm::Constant*>(val));
  }

  return builder_->CreateVectorSplat(k_vector_size, val);
}

absl::StatusOr<::llvm::Value*> CodeGen::LoadVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateInBoundsGEP(ele_type, ptr, {idx});
  return builder_->CreateLoad(vector_type, offset_ptr);
}
absl::StatusOr<::llvm::Value*> CodeGen::LoadNVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx,
                                                    ::llvm::Value* n) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});

  auto vector_ptr_value = builder_->CreateAlloca(vector_type);
  ::llvm::MaybeAlign align(1);
  builder_->CreateMemCpy(vector_ptr_value, align, offset_ptr, align,
                         builder_->CreateMul(n, builder_->getInt32(ele_type->getScalarSizeInBits() / 8)));
  auto vector_value = builder_->CreateLoad(vector_type, vector_ptr_value);
  return vector_value;
}

absl::Status CodeGen::StoreVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateInBoundsGEP(ele_type, ptr, {idx});
  if (val->getType()->isPointerTy()) {
    builder_->CreateStore(builder_->CreateLoad(vector_type, val), offset_ptr);
  } else {
    auto* inst = builder_->CreateStore(val, offset_ptr);
    ::llvm::Align align(1);
    inst->setAlignment(align);
  };

  return absl::OkStatus();
}
absl::Status CodeGen::StoreNVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx,
                                   ::llvm::Value* n) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});

  auto vector_ptr_value = builder_->CreateAlloca(vector_type);
  builder_->CreateStore(val, vector_ptr_value);
  ::llvm::MaybeAlign align(1);
  builder_->CreateMemCpy(offset_ptr, align, vector_ptr_value, align,
                         builder_->CreateMul(n, builder_->getInt32(ele_type->getScalarSizeInBits() / 8)));

  return absl::OkStatus();
}

}  // namespace llvm
}  // namespace rapidudf