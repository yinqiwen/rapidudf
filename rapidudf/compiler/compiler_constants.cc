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

#include "rapidudf/compiler/codegen.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::ConstantNumber& expr) {
  if (expr.dtype.has_value()) {
    return BuildIR(expr.dv, *expr.dtype);
  } else {
    return BuildIR(expr.dv);
  }
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(double v, DType dtype) {
  if (dtype.IsInteger()) {
    uint64_t uiv = static_cast<uint64_t>(v);
    bool is_signed = dtype.IsSigned();
    ::llvm::APInt iv(dtype.Bits(), uiv, is_signed);
    auto val = ::llvm::ConstantInt::get(codegen_->GetContext(), iv);
    return codegen_->NewValue(dtype, val);
  } else if (dtype.IsF32()) {
    ::llvm::APFloat fv(static_cast<float>(v));
    auto val = ::llvm::ConstantFP::get(codegen_->GetContext(), fv);
    return codegen_->NewValue(dtype, val);
  } else {
    ::llvm::APFloat fv(v);
    auto val = ::llvm::ConstantFP::get(codegen_->GetContext(), fv);
    return codegen_->NewValue(dtype, val);
  }
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(double v) {
  int64_t iv = static_cast<int64_t>(v);
  if (static_cast<double>(iv) == v) {
    DType int_dtype(iv <= INT32_MAX ? DATA_I32 : DATA_I64);
    return BuildIR(v, int_dtype);
  }
  return codegen_->NewF64(v);
}
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(bool v) { return codegen_->NewBool(v); }
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(uint32_t v) { return codegen_->NewU32(v); }
absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const std::string& v) { return codegen_->NewStringView(v); }
}  // namespace compiler
}  // namespace rapidudf