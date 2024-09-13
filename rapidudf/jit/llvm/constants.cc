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

#include <fmt/core.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <vector>
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"

namespace rapidudf {
namespace llvm {
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, double v, DType dtype) {
  if (dtype.IsInteger()) {
    uint64_t uiv = static_cast<uint64_t>(v);
    bool is_signed = dtype.IsSigned();
    ::llvm::APInt iv(dtype.Bits(), uiv, is_signed);
    auto val = ::llvm::ConstantInt::get(ir_builder_->getContext(), iv);
    return NewValue(dtype, val);
  } else if (dtype.IsF32()) {
    ::llvm::APFloat fv(static_cast<float>(v));
    auto val = ::llvm::ConstantFP::get(ir_builder_->getContext(), fv);
    return NewValue(dtype, val);
  } else {
    ::llvm::APFloat fv(v);
    auto val = ::llvm::ConstantFP::get(ir_builder_->getContext(), fv);
    return NewValue(dtype, val);
  }
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, double v) {
  int64_t iv = static_cast<int64_t>(v);
  if (static_cast<double>(iv) == v) {
    DType int_dtype(iv <= INT32_MAX ? DATA_I32 : DATA_I64);
    auto val = ::llvm::ConstantInt::getSigned(GetType(int_dtype).value(), iv);
    return NewValue(int_dtype, val);
  }
  ::llvm::APFloat fv(v);
  auto val = ::llvm::ConstantFP::get(ir_builder_->getContext(), fv);
  return NewValue(DATA_F64, val);
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, bool v) {
  auto val = ::llvm::ConstantInt::getBool(GetType(DATA_U8).value(), v);
  return NewValue(DATA_U8, val);
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::Array& expr) {
  ast_ctx_.SetPosition(expr.position);
  auto element_dtype = expr.dtype.Elem();
  auto element_type_result = GetType(element_dtype);
  if (!element_type_result.ok()) {
    return element_type_result.status();
  }
  auto* element_type = element_type_result.value();
  auto* stack_val = ir_builder_->CreateAlloca(element_type, ir_builder_->getInt64(expr.elements.size()));

  for (size_t i = 0; i < expr.elements.size(); i++) {
    auto result = BuildIR(ctx, expr.elements[i]);
    if (!result.ok()) {
      return result.status();
    }
    auto element_val = result.value();
    if (element_val->GetDType() != element_dtype) {
      auto orig_element_dtype = element_val->GetDType();
      element_val = element_val->CastTo(element_dtype);
      if (!element_val) {
        RUDF_LOG_ERROR_STATUS(ast_ctx_.GetErrorStatus(
            fmt::format("Cast array:{} element from {} to {} failed.", i, orig_element_dtype, element_dtype)));
      }
    }
    auto element_ptr =
        ir_builder_->CreateInBoundsGEP(element_type, stack_val, std::vector<::llvm::Value*>{ir_builder_->getInt64(i)});
    ir_builder_->CreateStore(element_val->GetValue(), element_ptr);
  }
  auto* span_type = ::llvm::StructType::getTypeByName(ir_builder_->getContext(), "absl_span");
  auto* span_val = ir_builder_->CreateAlloca(span_type);
  auto size_val = ir_builder_->getInt64(expr.elements.size());
  ::llvm::Value* zero = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 0);
  ::llvm::Value* offset = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 1);
  auto size_field_ptr = ir_builder_->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  ir_builder_->CreateStore(size_val, size_field_ptr);
  offset = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 0);
  auto ptr_field_ptr = ir_builder_->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  ir_builder_->CreateStore(stack_val, ptr_field_ptr);
  return NewValue(expr.dtype, ir_builder_->CreateLoad(span_type, span_val));
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const std::string& v) {
  std::unique_ptr<std::string> str = std::make_unique<std::string>(v);
  StringView view(*str);
  uint64_t* uv = reinterpret_cast<uint64_t*>(&view);
  GetSession().const_strings.emplace_back(std::move(str));
  ::llvm::StructType* string_view_type = static_cast<::llvm::StructType*>(GetType(DATA_STRING_VIEW).value());
  auto* str_val = ir_builder_->CreateAlloca(string_view_type);
  ::llvm::Value* zero = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 0);
  ::llvm::Value* offset = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 0);
  auto size_field_ptr =
      ir_builder_->CreateInBoundsGEP(string_view_type, str_val, std::vector<::llvm::Value*>{zero, offset});
  auto size_val = ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ir_builder_->getContext()), uv[0]);
  ir_builder_->CreateStore(size_val, size_field_ptr);

  offset = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(ir_builder_->getContext()), 1);
  auto ptr_field_ptr =
      ir_builder_->CreateInBoundsGEP(string_view_type, str_val, std::vector<::llvm::Value*>{zero, offset});
  auto ptr_val = ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ir_builder_->getContext()), uv[1]);
  ir_builder_->CreateStore(ptr_val, ptr_field_ptr);

  return NewValue(DATA_STRING_VIEW, ir_builder_->CreateLoad(string_view_type, str_val));
}

}  // namespace llvm
}  // namespace rapidudf