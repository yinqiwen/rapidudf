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