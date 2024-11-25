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

#include <utility>
#include "rapidudf/compiler/codegen.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "fmt/format.h"

#include "rapidudf/compiler/type.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::NewConstantVectorValue(DType dtype, ::llvm::Value* val) {
  if (::llvm::isa<::llvm::Constant>(val)) {
    auto vector_type = get_vector_type(builder_->getContext(), dtype);
    return ::llvm::ConstantVector::getSplat(vector_type->getElementCount(), reinterpret_cast<::llvm::Constant*>(val));
  }
  return builder_->CreateVectorSplat(simd::kVectorUnitSize, val);
}
absl::StatusOr<::llvm::Value*> CodeGen::NewConstantVectorValue(ValuePtr val) {
  return NewConstantVectorValue(val->GetDType(), val->LoadValue());
}

::llvm::Value* CodeGen::NewVectorVar(DType dtype) {
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto* ptr = builder_->CreateAlloca(vector_type);
  return ptr;
}

absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> CodeGen::NewStackConstantVector(ValuePtr constant) {
  auto result = NewConstantVectorValue(constant);
  if (!result.ok()) {
    return result.status();
  }
  auto* val = result.value();
  auto* ptr = NewVectorVar(constant->GetDType());
  builder_->CreateStore(val, ptr);
  return std::make_pair(ptr, val);
}

absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> CodeGen::LoadVector(DType dtype, ::llvm::Value* ptr,
                                                                              ::llvm::Value* idx) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateInBoundsGEP(ele_type, ptr, {idx});
  auto* load = builder_->CreateLoad(vector_type, offset_ptr);
  ::llvm::Align align(1);
  load->setAlignment(align);
  return std::make_pair(offset_ptr, load);
}

absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> CodeGen::LoadNVector(DType dtype, ::llvm::Value* ptr,
                                                                               ::llvm::Value* idx, ::llvm::Value* n) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});

  auto vector_ptr_value = builder_->CreateAlloca(vector_type);
  ::llvm::Constant* fill_v = nullptr;
  switch (dtype.GetFundamentalType()) {
    case DATA_F64: {
      ::llvm::APFloat fv(static_cast<double>(1.0));
      fill_v = ::llvm::ConstantFP::get(builder_->getContext(), fv);
      break;
    }
    case DATA_F32: {
      ::llvm::APFloat fv(static_cast<float>(1.0));
      fill_v = ::llvm::ConstantFP::get(builder_->getContext(), fv);
      break;
    }
    case DATA_I64:
    case DATA_U64: {
      fill_v = builder_->getInt64(1);
      break;
    }
    case DATA_I32:
    case DATA_U32: {
      fill_v = builder_->getInt32(1);
      break;
    }
    case DATA_I16:
    case DATA_U16: {
      fill_v = builder_->getInt16(1);
      break;
    }
    case DATA_I8:
    case DATA_U8: {
      fill_v = builder_->getInt8(1);
      break;
    }
    case DATA_STRING_VIEW: {
      ::llvm::APInt zero(128, {0, 0});
      fill_v = ::llvm::ConstantInt::get(builder_->getContext(), zero);
      break;
    }
    default: {
      break;
    }
  }
  if (nullptr != fill_v) {
    auto* fill_vec_v = ::llvm::ConstantVector::getSplat(vector_type->getElementCount(), fill_v);
    builder_->CreateStore(fill_vec_v, vector_ptr_value);
  }
  if (ele_type->getScalarSizeInBits() >= 8) {
    ::llvm::MaybeAlign align(1);
    builder_->CreateMemCpy(vector_ptr_value, align, offset_ptr, align,
                           builder_->CreateMul(n, builder_->getInt32(ele_type->getScalarSizeInBits() / 8)));
  } else {
    builder_->CreateStore(builder_->CreateLoad(vector_type, offset_ptr), vector_ptr_value);
  }
  auto vector_value = builder_->CreateLoad(vector_type, vector_ptr_value);
  return std::make_pair(vector_ptr_value, vector_value);
}

absl::Status CodeGen::StoreVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);

  if (val->getType()->isPointerTy()) {
    auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});
    auto* inst = builder_->CreateStore(builder_->CreateLoad(vector_type, val), offset_ptr);
    ::llvm::Align align(1);
    inst->setAlignment(align);
  } else {
    if (ele_type->getScalarSizeInBits() == 1) {
      auto offset_ptr =
          builder_->CreateGEP(builder_->getInt8Ty(), ptr, {builder_->CreateUDiv(idx, builder_->getInt32(8))});
      builder_->CreateStore(val, offset_ptr);
    } else {
      auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});
      auto* inst = builder_->CreateStore(val, offset_ptr);
      ::llvm::Align align(1);
      inst->setAlignment(align);
    }
  };

  return absl::OkStatus();
}
absl::Status CodeGen::StoreNVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx,
                                   ::llvm::Value* n) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);

  if (ele_type->getScalarSizeInBits() == 1) {
    auto offset_ptr =
        builder_->CreateGEP(builder_->getInt8Ty(), ptr, {builder_->CreateUDiv(idx, builder_->getInt32(8))});
    builder_->CreateStore(val, offset_ptr);
  } else {
    auto offset_ptr = builder_->CreateGEP(ele_type, ptr, {idx});
    auto vector_ptr_value = builder_->CreateAlloca(vector_type);
    builder_->CreateStore(val, vector_ptr_value);
    ::llvm::MaybeAlign align(1);
    builder_->CreateMemCpy(offset_ptr, align, vector_ptr_value, align,
                           builder_->CreateMul(n, builder_->getInt32(ele_type->getScalarSizeInBits() / 8)));
  }

  return absl::OkStatus();
}
}  // namespace compiler
}  // namespace rapidudf