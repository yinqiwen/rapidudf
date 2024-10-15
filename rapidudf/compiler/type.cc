/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include "rapidudf/compiler/type.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/simd/vector.h"

namespace rapidudf {
namespace compiler {

// static const cpu_features::X86Features features = cpu_features::GetX86Info().features;

void init_buitin_types(::llvm::LLVMContext& ctx) {
  auto size_type = ::llvm::IntegerType::get(ctx, 64);
  auto pointer_type = ::llvm::PointerType::getUnqual(::llvm::Type::getInt8Ty(ctx));
  // auto* string_view_type = ::llvm::StructType::create(ctx, "string_view");
  // string_view_type->setBody(size_type, pointer_type);

  auto* std_string_view_type = ::llvm::StructType::create(ctx, "std_string_view");
  std_string_view_type->setBody({size_type, pointer_type});

  auto* absl_span_type = ::llvm::StructType::create(ctx, "absl_span");
  absl_span_type->setBody({pointer_type, size_type});

  auto* simd_vector_type = ::llvm::StructType::create(ctx, "simd_vector");
  simd_vector_type->setBody({size_type, pointer_type});
  // auto* simd_vector_type = ::llvm::IntegerType::get(ctx, 128);
}
::llvm::Type* get_type(::llvm::LLVMContext& ctx, DType dtype) {
  if (dtype.IsPtr()) {
    if (dtype.IsContextPtr()) {
      return ::llvm::PointerType::get(ctx, 0);
    }
    auto base_type = get_type(ctx, dtype.PtrTo());
    if (!base_type) {
      // return ::llvm::Type::getInt8PtrTy(ctx);
      return ::llvm::PointerType::get(ctx, 0);
    }
    return ::llvm::PointerType::getUnqual(base_type);
  }

  if (dtype.IsAbslSpan()) {
    return ::llvm::StructType::getTypeByName(ctx, "absl_span");
    // return ::llvm::IntegerType::get(ctx, 128);
  }
  if (dtype.IsSimdVector()) {
    return ::llvm::StructType::getTypeByName(ctx, "simd_vector");
    // return ::llvm::IntegerType::get(ctx, 128);
  }

  switch (dtype.GetFundamentalType()) {
    case DATA_VOID: {
      return ::llvm::Type::getVoidTy(ctx);
    }
    case DATA_BIT: {
      return ::llvm::Type::getIntNTy(ctx, 1);
    }
    case DATA_U8:
    case DATA_I8:
    case DATA_U16:
    case DATA_I16:
    case DATA_U32:
    case DATA_I32:
    case DATA_U64:
    case DATA_I64: {
      return ::llvm::IntegerType::get(ctx, dtype.Bits());
    }
    case DATA_F16: {
      return ::llvm::Type::getHalfTy(ctx);
    }
    case DATA_F32: {
      return ::llvm::Type::getFloatTy(ctx);
    }
    case DATA_F64: {
      return ::llvm::Type::getDoubleTy(ctx);
    }
    case DATA_F80: {
      return ::llvm::Type::getX86_FP80Ty(ctx);
    }
    case DATA_STD_STRING_VIEW: {
      return ::llvm::StructType::getTypeByName(ctx, "std_string_view");
      // return ::llvm::IntegerType::get(ctx, 128);
    }
    case DATA_STRING_VIEW: {
      // return ::llvm::StructType::getTypeByName(ctx, "string_view");
      return ::llvm::IntegerType::get(ctx, 128);
    }
    default: {
      // RUDF_ERROR("Unsupported dtype:{} to get llvm type.", dtype);
      return nullptr;
    }
  }
}

::llvm::VectorType* get_vector_type(::llvm::LLVMContext& ctx, DType dtype) {
  DType ele_dtype = dtype.Elem();
  uint32_t n = 0;
  if (ele_dtype.Bits() > 0 && ele_dtype.Bits() <= 128) {
    n = simd::kVectorUnitSize;
  }
  ::llvm::Type* ele_type = get_type(ctx, ele_dtype);
  if (n > 0 && ele_type != nullptr) {
    return ::llvm::VectorType::get(ele_type, n, false);
  }
  return nullptr;
}

}  // namespace compiler
}  // namespace rapidudf