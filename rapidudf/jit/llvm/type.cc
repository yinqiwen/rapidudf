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
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
namespace rapidudf {
namespace llvm {
void init_buitin_types(::llvm::LLVMContext& ctx) {
  auto* string_view_type = ::llvm::StructType::create(ctx, "string_view");
  auto size_type = ::llvm::IntegerType::get(ctx, 64);
  auto pointer_type = ::llvm::PointerType::getUnqual(::llvm::Type::getInt8Ty(ctx));
  string_view_type->setBody(size_type, pointer_type);

  auto* std_string_view_type = ::llvm::StructType::create(ctx, "std_string_view");
  std_string_view_type->setBody(size_type, pointer_type);

  auto* absl_span_type = ::llvm::StructType::create(ctx, "absl_span");
  absl_span_type->setBody(pointer_type, size_type);
}
::llvm::Type* get_type(::llvm::LLVMContext& ctx, DType dtype) {
  if (dtype.IsPtr()) {
    auto base_type = get_type(ctx, dtype.PtrTo());
    if (!base_type) {
      return ::llvm::Type::getInt8PtrTy(ctx);
    }
    return ::llvm::PointerType::getUnqual(base_type);
  }
  if (dtype.IsAbslSpan()) {
    return ::llvm::StructType::getTypeByName(ctx, "absl_span");
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
    case DATA_F32: {
      return ::llvm::Type::getFloatTy(ctx);
    }
    case DATA_F64: {
      return ::llvm::Type::getDoubleTy(ctx);
    }
    case DATA_STD_STRING_VIEW: {
      return ::llvm::StructType::getTypeByName(ctx, "std_string_view");
    }
    case DATA_STRING_VIEW: {
      return ::llvm::StructType::getTypeByName(ctx, "string_view");
    }
    default: {
      // RUDF_ERROR("Unsupported dtype:{} to get llvm type.", dtype);
      return nullptr;
    }
  }
}

}  // namespace llvm
}  // namespace rapidudf