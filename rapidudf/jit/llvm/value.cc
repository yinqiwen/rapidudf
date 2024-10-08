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
#include "rapidudf/jit/llvm/value.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include "fmt/core.h"
#include "llvm/IR/Use.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/jit_session.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace llvm {
Value::Value(Private, DType dtype, JitCompiler* c, ::llvm::Value* val, ::llvm::Type* t)
    : dtype_(dtype), compiler_(c), val_(val), type_(t) {
  ir_builder_ = compiler_->GetSession()->GetIRBuilder();
}
bool Value::IsWritable() const { return dtype_.IsVoid() || type_ != nullptr; }
::llvm::Value* Value::GetValue() {
  if (type_ != nullptr) {
    return ir_builder_->CreateLoad(type_, val_);
  }
  return val_;
}

::llvm::Value* Value::GetPtrValue() {
  if (type_ != nullptr) {
    return val_;
  }
  return nullptr;
}

absl::Status Value::CopyFrom(ValuePtr other) {
  if (!dtype_.IsVoid()) {
    if (dtype_ != other->dtype_) {
      return absl::InvalidArgumentError(
          fmt::format("Can not copy from dtype:{} while current dtype:{}", other->dtype_, dtype_));
    }
  } else {
    if (other->type_ == nullptr) {
      if (type_ == nullptr) {
        type_ = get_type(ir_builder_->getContext(), other->GetDType());
        if (type_ == nullptr) {
          return absl::InvalidArgumentError(fmt::format("Can not alloca for dtype:{}", other->GetDType()));
        }
        val_ = ir_builder_->CreateAlloca(type_);
      }
    } else {
      type_ = other->type_;
      val_ = other->val_;
      dtype_ = other->dtype_;
      return absl::OkStatus();
    }
  }
  dtype_ = other->dtype_;
  if (type_ != nullptr) {
    ir_builder_->CreateStore(other->val_, val_);
  } else {
    val_ = other->val_;
  }
  return absl::OkStatus();
}
ValuePtr Value::Select(ValuePtr true_val, ValuePtr false_val) {
  if (true_val->dtype_ != false_val->dtype_) {
    RUDF_ERROR("Can NOT select since true_val dtype:{} is not eqaul with false_val dtype:{}", true_val->GetDType(),
               false_val->GetDType());
    return {};
  }
  auto new_val = ir_builder_->CreateSelect(val_, true_val->GetValue(), false_val->GetValue());
  return New(true_val->GetDType(), compiler_, new_val);
}
absl::Status Value::SetSimdVectorTemporary(bool v) {
  if (!dtype_.IsSimdVector()) {
    return absl::InvalidArgumentError(fmt::format("Can not set temprary flag on {}", dtype_));
  }
  if (type_ == nullptr) {
    return absl::InvalidArgumentError(
        fmt::format("Can not set temprary flag on {} with empty type with val:{}", dtype_, v));
  }

  // ::llvm::Value* result = nullptr;
  // ::llvm::ConstantInt* neg_one =
  //     ::llvm::ConstantInt::get(::llvm::Type::getInt128Ty(ir_builder_->getContext()), -1, true);
  // ::llvm::ConstantInt* one = ::llvm::ConstantInt::get(::llvm::Type::getInt128Ty(ir_builder_->getContext()), 1, true);

  // ::llvm::ConstantInt* index = ::llvm::ConstantInt::get(::llvm::Type::getInt128Ty(ir_builder_->getContext()), 0,
  // true); if (v) {
  //   //  %mask = shl i128 1, %index
  //   //  %set_bit_value = or i128 %value, %mask
  //   auto mask = ir_builder_->CreateShl(one, index);
  //   result = ir_builder_->CreateOr({GetValue(), mask});
  // } else {
  //   //   %mask = xor i128 -1, shl i128 1, %index
  //   // %cleared_value = and i128 %value, %mask
  //   auto mask = ir_builder_->CreateShl(one, index);
  //   mask = ir_builder_->CreateXor(mask, neg_one);
  //   result = ir_builder_->CreateAnd({GetValue(), mask});
  // }
  // if (type_ != nullptr) {
  //   ir_builder_->CreateStore(result, val_);
  // } else {
  //   val_ = result;
  // }

  // DType simd_vector_dtype(DATA_U8);
  // simd_vector_dtype = simd_vector_dtype.ToSimdVector();
  // ::llvm::StructType* simd_vector_type =
  //     static_cast<::llvm::StructType*>(get_type(ir_builder_->getContext(), simd_vector_dtype));
  // auto size_field_ptr =
  //     ir_builder_->CreateInBoundsGEP(simd_vector_type, val_, {ir_builder_->getInt32(0), ir_builder_->getInt32(0)});
  // auto size_field_val = ir_builder_->CreateLoad(ir_builder_->getInt64Ty(), size_field_ptr);
  // ::llvm::Value* result = nullptr;
  // if (v) {
  //   uint64_t mask_v = 1ULL;
  //   auto mask = ir_builder_->getInt64(mask_v);
  //   result = ir_builder_->CreateOr({size_field_val, mask});
  // } else {
  //   uint64_t mask_v = ~(1ULL << 0);
  //   auto mask = ir_builder_->getInt64(mask_v);
  //   result = ir_builder_->CreateAnd({size_field_val, mask});
  // }
  // ir_builder_->CreateStore(result, size_field_ptr);
  return absl::OkStatus();
}

absl::StatusOr<::llvm::Value*> Value::GetStructPtrValue() {
  if (!dtype_.IsSimdVector() && !dtype_.IsStringView()) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector ptr field on {}", dtype_));
  }
  if (type_ == nullptr) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector ptr field  with empty type"));
  }
  auto ptr_field_ptr = ir_builder_->CreateInBoundsGEP(
      type_, val_, std::vector<::llvm::Value*>{ir_builder_->getInt32(0), ir_builder_->getInt32(1)});
  return ir_builder_->CreateLoad(::llvm::PointerType::get(ir_builder_->getInt64Ty(), 0), ptr_field_ptr);
}
absl::StatusOr<::llvm::Value*> Value::GetStructSizeValue() {
  if (!dtype_.IsSimdVector() && !dtype_.IsStringView()) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector size field on {}", dtype_));
  }
  if (type_ == nullptr) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector size field  with empty type"));
  }
  auto size_field_ptr = ir_builder_->CreateInBoundsGEP(
      type_, val_, std::vector<::llvm::Value*>{ir_builder_->getInt32(0), ir_builder_->getInt32(0)});
  return ir_builder_->CreateLoad(ir_builder_->getInt64Ty(), size_field_ptr);
}

}  // namespace llvm
}  // namespace rapidudf