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

#include "rapidudf/compiler/codegen.h"

#include <cstdint>
#include <cstring>

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
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
#include "rapidudf/types/string_view.h"
namespace rapidudf {
namespace compiler {
ValuePtr CodeGen::NewValue(DType dtype, ::llvm::Value* val, ::llvm::Type* type) {
  return Value::New(dtype, builder_.get(), val, type);
}

ValuePtr CodeGen::NewU32Var(uint32_t init) {
  ::llvm::Value* cursor_val = builder_->CreateAlloca(builder_->getInt32Ty());
  builder_->CreateStore(builder_->getInt32(init), cursor_val);
  return Value::New(DATA_U32, builder_.get(), cursor_val, builder_->getInt32Ty());
}
ValuePtr CodeGen::NewI32Var(uint32_t init) {
  ::llvm::Value* cursor_val = builder_->CreateAlloca(builder_->getInt32Ty());
  builder_->CreateStore(builder_->getInt32(init), cursor_val);
  return Value::New(DATA_I32, builder_.get(), cursor_val, builder_->getInt32Ty());
}

ValuePtr CodeGen::NewVar(DType dtype) {
  auto typ = get_type(builder_->getContext(), dtype);
  ::llvm::Value* cursor_val = builder_->CreateAlloca(typ);
  return Value::New(dtype, builder_.get(), cursor_val, typ);
}

ValuePtr CodeGen::NewU32(uint32_t v) {
  auto val = builder_->getInt32(v);
  return NewValue(DATA_U32, val);
}
ValuePtr CodeGen::NewI32(uint32_t v) {
  auto val = builder_->getInt32(v);
  return NewValue(DATA_I32, val);
}
ValuePtr CodeGen::NewBool(bool v) {
  auto val = builder_->getInt1(v);
  return NewValue(DATA_BIT, val);
}
ValuePtr CodeGen::NewF32(float v) {
  ::llvm::APFloat fv(v);
  auto val = ::llvm::ConstantFP::get(builder_->getContext(), fv);
  return NewValue(DATA_F32, val);
}
ValuePtr CodeGen::NewF64(double v) {
  ::llvm::APFloat fv(v);
  auto val = ::llvm::ConstantFP::get(builder_->getContext(), fv);
  return NewValue(DATA_F64, val);
}
ValuePtr CodeGen::NewStringView(const std::string& v) {
  // StringView layout (16 bytes total, little-endian on x86-64):
  //   [0..3]  uint32_t size_
  //   [4..7]  char     prefix_[4]
  //   [8..15] union { char inlined[8]; const char* data; } value_
  //
  // For inline strings (size <= kInlineSize == 12) the payload is fully
  // contained in prefix_ + value_.inlined; we materialize the 16 bytes as a
  // single i128 immediate constant so neither the host process nor any global
  // is referenced from the JIT'd code.
  //
  // For non-inline strings we emit a private constant [N x i8] global into
  // the JIT module (linked into the JIT-managed memory by ORC), then build
  // the StringView at runtime: the low 8 bytes (size + prefix) are an
  // immediate, the high 8 bytes are a runtime pointer obtained via GEP on
  // that global.
  ::llvm::IntegerType* string_view_type = static_cast<::llvm::IntegerType*>(GetType(DATA_STRING_VIEW).value());
  auto* str_val = builder_->CreateAlloca(string_view_type);

  if (StringView::isInline(v.size())) {
    StringView view(v.data(), static_cast<int32_t>(v.size()));
    static_assert(sizeof(StringView) == 16, "StringView must be 16 bytes");
    uint64_t words[2];
    std::memcpy(words, &view, sizeof(view));
    ::llvm::APInt sv_bits(128, ::llvm::ArrayRef<uint64_t>(words, 2));
    builder_->CreateStore(::llvm::ConstantInt::get(string_view_type, sv_bits), str_val);
    return NewValue(DATA_STRING_VIEW, str_val, string_view_type);
  }

  // Long path: dedup by content; emit one private global per distinct literal.
  ::llvm::GlobalVariable* gv = nullptr;
  auto it = string_globals_.find(v);
  if (it != string_globals_.end()) {
    gv = it->second;
  } else {
    auto* str_const = ::llvm::ConstantDataArray::getString(*context_, v, /*AddNull=*/false);
    gv = new ::llvm::GlobalVariable(*module_, str_const->getType(), /*isConstant=*/true,
                                    ::llvm::GlobalValue::PrivateLinkage, str_const, ".rudf_str");
    gv->setUnnamedAddr(::llvm::GlobalValue::UnnamedAddr::Global);
    string_globals_.emplace(v, gv);
  }

  // Compose the StringView in-place via two stores. After SROA + InstCombine
  // this collapses to a single i128 store sequence, but the data pointer
  // remains a relocation against the in-module global rather than a host
  // immediate.
  //
  // Low 8 bytes: { uint32 size, char prefix[4] } known at compile time.
  uint64_t low = static_cast<uint64_t>(static_cast<uint32_t>(v.size()));
  uint64_t prefix_bits = 0;
  std::memcpy(&prefix_bits, v.data(), StringView::kPrefixSize);
  low |= prefix_bits << 32;

  auto* i64_ty = builder_->getInt64Ty();
  auto* low_ptr = builder_->CreateBitCast(str_val, i64_ty->getPointerTo());
  builder_->CreateStore(::llvm::ConstantInt::get(i64_ty, low), low_ptr);

  auto* high_ptr = builder_->CreateConstInBoundsGEP1_32(i64_ty, low_ptr, 1);
  // GEP to first byte of the global → const char* equivalent. Sized to i64.
  auto* zero32 = builder_->getInt32(0);
  auto* data_ptr = builder_->CreateInBoundsGEP(gv->getValueType(), gv, {zero32, zero32});
  auto* data_as_int = builder_->CreatePtrToInt(data_ptr, i64_ty);
  builder_->CreateStore(data_as_int, high_ptr);

  return NewValue(DATA_STRING_VIEW, str_val, string_view_type);
}

ValuePtr CodeGen::NewVoid(const std::string& name) {
  auto var = NewValue(DATA_VOID, nullptr);
  current_func_->named_values.emplace(name, var);
  return var;
}

absl::StatusOr<ValuePtr> CodeGen::NewArray(DType dtype, const std::vector<ValuePtr>& elements) {
  auto element_type_result = GetType(dtype);
  if (!element_type_result.ok()) {
    return element_type_result.status();
  }
  auto* element_type = element_type_result.value();
  auto* stack_val = builder_->CreateAlloca(element_type, builder_->getInt64(elements.size()));

  for (size_t i = 0; i < elements.size(); i++) {
    auto element_val = elements[i];
    if (element_val->GetDType() != dtype) {
      auto orig_element_dtype = element_val->GetDType();
      auto cast_result = CastTo(element_val, dtype);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      element_val = cast_result.value();
    }
    auto element_ptr =
        builder_->CreateInBoundsGEP(element_type, stack_val, std::vector<::llvm::Value*>{builder_->getInt32(i)});
    builder_->CreateStore(element_val->LoadValue(), element_ptr);
  }
  auto* span_type = ::llvm::StructType::getTypeByName(builder_->getContext(), "absl_span");
  auto* span_val = builder_->CreateAlloca(span_type);
  auto size_val = builder_->getInt64(elements.size());
  ::llvm::Value* zero = builder_->getInt32(0);
  ::llvm::Value* offset = builder_->getInt32(1);
  auto size_field_ptr = builder_->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  builder_->CreateStore(size_val, size_field_ptr);
  offset = builder_->getInt32(0);
  auto ptr_field_ptr = builder_->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  builder_->CreateStore(stack_val, ptr_field_ptr);
  return NewValue(dtype.ToAbslSpan(), builder_->CreateLoad(span_type, span_val));
}

absl::StatusOr<ValuePtr> CodeGen::GetStructField(ValuePtr obj, DType field_dtype, uint32_t offset) {
  ::llvm::Value* offset_val = builder_->getInt32(offset);
  auto field_ptr = builder_->CreateInBoundsGEP(builder_->getInt8Ty(), obj->LoadValue(), {offset_val});
  if (field_dtype.IsNumber() || field_dtype.IsStringView() || field_dtype.IsStdStringView() ||
      field_dtype.IsSimdVector() || field_dtype.IsAbslSpan() || field_dtype.IsPtr()) {
    auto dst_type_result = GetType(field_dtype);
    if (!dst_type_result.ok()) {
      return dst_type_result.status();
    }
    return NewValue(field_dtype, field_ptr, dst_type_result.value());
  } else {
    return NewValue(field_dtype.ToPtr(), field_ptr);
  }
}
}  // namespace compiler
}  // namespace rapidudf