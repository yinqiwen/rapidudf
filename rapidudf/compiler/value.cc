/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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