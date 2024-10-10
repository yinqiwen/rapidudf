/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once

#include "absl/status/statusor.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace llvm {
class IRBuilderBase;
class Value;
class Type;
class Constant;
}  // namespace llvm

namespace rapidudf {
namespace llvm {
class CodeGen {
 public:
  explicit CodeGen(::llvm::IRBuilderBase* builder, uint32_t& label_cursor);
  absl::StatusOr<::llvm::Value*> NewConstVectorValue(DType dtype, ::llvm::Value* val);
  absl::StatusOr<::llvm::Value*> LoadVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx);
  absl::StatusOr<::llvm::Value*> LoadNVector(DType dtype, ::llvm::Value* ptr, ::llvm::Value* idx, ::llvm::Value* n);
  absl::Status StoreVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx);
  absl::Status StoreNVector(DType dtype, ::llvm::Value* val, ::llvm::Value* ptr, ::llvm::Value* idx, ::llvm::Value* n);

  absl::StatusOr<::llvm::Value*> CastTo(::llvm::Value* val, DType src_dtype, DType dst_dtype);
  absl::StatusOr<::llvm::Value*> CastTo(::llvm::Value* val, DType src_dtype, ::llvm::Value* dst_dtype);
  absl::StatusOr<::llvm::Value*> UnaryOp(OpToken op, DType dtype, ::llvm::Value* val);
  absl::StatusOr<::llvm::Value*> BinaryOp(OpToken op, DType dtype, ::llvm::Value* left, ::llvm::Value* right);
  absl::StatusOr<::llvm::Value*> TernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                           ::llvm::Value* c);

 private:
  uint32_t GetLabelCursor() { return label_cursor_++; }
  ::llvm::IRBuilderBase* builder_ = nullptr;
  uint32_t& label_cursor_;
};
}  // namespace llvm
}  // namespace rapidudf