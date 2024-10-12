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
#include "rapidudf/compiler/value.h"

#include "fmt/format.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"

#include "rapidudf/compiler/type.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"

namespace rapidudf {
namespace compiler {
Value::Value(Private, DType dtype, ::llvm::IRBuilderBase* ir_builder, ::llvm::Value* val,
             ::llvm::Type* ptr_element_type)
    : dtype_(dtype), ir_builder_(ir_builder), val_(val), ptr_element_type_(ptr_element_type) {}
bool Value::IsWritable() const { return dtype_.IsVoid() || ptr_element_type_ != nullptr; }
::llvm::Value* Value::LoadValue() {
  if (ptr_element_type_ != nullptr) {
    return ir_builder_->CreateLoad(ptr_element_type_, val_);
  }
  return val_;
}
::llvm::Value* Value::GetPtrValue() {
  if (ptr_element_type_ != nullptr) {
    return val_;
  }
  if (val_->getType()->isPointerTy()) {
    return val_;
  }
  return nullptr;
}

absl::Status Value::CopyFrom(ValuePtr other) {
  if (dtype_.IsVoid()) {
    if (other->ptr_element_type_ == nullptr) {
      ptr_element_type_ = get_type(ir_builder_->getContext(), other->GetDType());
    } else {
      ptr_element_type_ = other->ptr_element_type_;
      val_ = other->val_;
      dtype_ = other->dtype_;
      return absl::OkStatus();
    }
  } else {
    if (dtype_ != other->dtype_) {
      return absl::InvalidArgumentError(
          fmt::format("Can not copy from dtype:{} while current dtype:{}", other->dtype_, dtype_));
    }
  }
  if (ptr_element_type_ == nullptr) {
    return absl::InvalidArgumentError(
        fmt::format("Can not alloca for dtype:{} while ptr element type is null.", dtype_));
  }
  dtype_ = other->dtype_;
  if (nullptr == val_) {
    val_ = ir_builder_->CreateAlloca(ptr_element_type_);
  }
  ir_builder_->CreateStore(other->LoadValue(), val_);

  return absl::OkStatus();
}

absl::StatusOr<::llvm::Value*> Value::GetStructPtrValue() {
  if (!dtype_.IsSimdVector() && !dtype_.IsStringView()) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector ptr field on {}", dtype_));
  }
  if (ptr_element_type_ == nullptr) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector ptr field  with empty type"));
  }
  auto ptr_field_ptr = ir_builder_->CreateInBoundsGEP(
      ptr_element_type_, val_, std::vector<::llvm::Value*>{ir_builder_->getInt32(0), ir_builder_->getInt32(1)});
  return ir_builder_->CreateLoad(::llvm::PointerType::get(ir_builder_->getInt64Ty(), 0), ptr_field_ptr);
}
absl::StatusOr<::llvm::Value*> Value::GetStructSizeValue() {
  if (!dtype_.IsSimdVector() && !dtype_.IsStringView()) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector size field on {}", dtype_));
  }
  if (ptr_element_type_ == nullptr) {
    return absl::InvalidArgumentError(fmt::format("Can not retirve simd vector size field  with empty type"));
  }
  auto size_field_ptr = ir_builder_->CreateInBoundsGEP(
      ptr_element_type_, val_, std::vector<::llvm::Value*>{ir_builder_->getInt32(0), ir_builder_->getInt32(0)});
  return ir_builder_->CreateLoad(ir_builder_->getInt64Ty(), size_field_ptr);
}
absl::StatusOr<ValuePtr> Value::GetVectorSizeValue() {
  auto size_result = GetStructSizeValue();
  if (!size_result.ok()) {
    return size_result.status();
  }
  ::llvm::Value* size_capacity_val = size_result.value();
  size_capacity_val = ir_builder_->CreateLShr(size_capacity_val, ir_builder_->getInt64(1));
  auto mask = ir_builder_->getInt64(0x7FFFFFFFLL);
  size_capacity_val = ir_builder_->CreateAnd(size_capacity_val, mask);
  size_capacity_val = ir_builder_->CreateTrunc(size_capacity_val, ir_builder_->getInt32Ty());
  return New(DATA_I32, ir_builder_, size_capacity_val);
}

absl::Status Value::Inc(uint64_t v) {
  if (ptr_element_type_ == nullptr) {
    return absl::InvalidArgumentError(fmt::format("Can not int  with empty type"));
  }
  if (dtype_.IsI64() || dtype_.IsU64()) {
    auto* inc_v = ir_builder_->CreateAdd(LoadValue(), ir_builder_->getInt64(v));
    ir_builder_->CreateStore(inc_v, val_);
  } else if (dtype_.IsI32() || dtype_.IsU32()) {
    auto* inc_v = ir_builder_->CreateAdd(LoadValue(), ir_builder_->getInt32(v));
    ir_builder_->CreateStore(inc_v, val_);
  } else {
    return absl::InvalidArgumentError(fmt::format("Can not inc on dtype:{}", dtype_));
  }
  return absl::OkStatus();
}

}  // namespace compiler
}  // namespace rapidudf