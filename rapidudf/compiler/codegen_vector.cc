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

#include <llvm/ADT/APFloat.h>
#include <tuple>
#include "rapidudf/compiler/codegen.h"

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
absl::StatusOr<::llvm::Value*> CodeGen::NewConstVectorValue(DType dtype, ::llvm::Value* val) {
  if (::llvm::isa<::llvm::Constant>(val)) {
    auto vector_type = get_vector_type(builder_->getContext(), dtype);
    return ::llvm::ConstantVector::getSplat(vector_type->getElementCount(), reinterpret_cast<::llvm::Constant*>(val));
  }

  return builder_->CreateVectorSplat(k_vector_size, val);
}
absl::StatusOr<::llvm::Value*> CodeGen::NewConstVectorValue(ValuePtr val) {
  return NewConstVectorValue(val->GetDType(), val->LoadValue());
}

absl::StatusOr<::llvm::Value*> CodeGen::LoadVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx) {
  auto ele_type = get_type(builder_->getContext(), dtype.Elem());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto offset_ptr = builder_->CreateInBoundsGEP(ele_type, ptr, {idx});
  auto* load = builder_->CreateLoad(vector_type, offset_ptr);
  ::llvm::Align align(1);
  load->setAlignment(align);
  return load;
}

absl::StatusOr<::llvm::Value*> CodeGen::LoadNVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx,
                                                    ::llvm::Value* n) {
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
      // fill_v = builder_->getInt128Ty(1);
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
    auto* inst = builder_->CreateStore(builder_->CreateLoad(vector_type, val), offset_ptr);
    ::llvm::Align align(1);
    inst->setAlignment(align);
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
}  // namespace compiler
}  // namespace rapidudf