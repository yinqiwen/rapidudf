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

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::TernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                                  ::llvm::Value* c) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* ret_type = b->getType();
  // ::llvm::Type* element_type = GetElementType(b->getType());
  DType element_dtype = dtype.Elem();
  switch (op) {
    case OP_CONDITIONAL: {
      ret_value = builder_->CreateSelect(a, b, c);
      break;
    }
    case OP_FMA: {
      if (dtype.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::fma;
        builtin_intrinsic_args.emplace_back(a);
        builtin_intrinsic_args.emplace_back(b);
        builtin_intrinsic_args.emplace_back(c);
      }
      break;
    }

    default: {
      return absl::InvalidArgumentError(fmt::format("Unsupported op:{}", op));
    }
  }
  if (ret_value != nullptr) {
    return ret_value;
  }
  if (builtin_intrinsic > 0) {
    return builder_->CreateIntrinsic(ret_type, builtin_intrinsic, builtin_intrinsic_args);
  }
  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}
absl::StatusOr<ValuePtr> CodeGen::TernaryOp(OpToken op, ValuePtr a, ValuePtr b, ValuePtr c) {
  DType compute_dtype;
  if (op == OP_CONDITIONAL) {
    if (!a->GetDType().IsBit()) {
      RUDF_LOG_RETURN_FMT_ERROR("Can not select on cond dtype:{}", a->GetDType());
    }
    if (b->GetDType() != c->GetDType()) {
      auto normalize_dtype_result = NormalizeDType({b->GetDType(), c->GetDType()});
      if (!normalize_dtype_result.ok()) {
        return normalize_dtype_result.status();
      }
      compute_dtype = normalize_dtype_result.value();
      auto cast_result = CastTo(b, compute_dtype);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      b = cast_result.value();
      cast_result = CastTo(c, compute_dtype);
      if (!cast_result.ok()) {
        return cast_result.status();
      }
      c = cast_result.value();
    } else {
      compute_dtype = b->GetDType();
    }
  } else {
    auto normalize_dtype_result = NormalizeDType({a->GetDType(), b->GetDType(), c->GetDType()});
    if (!normalize_dtype_result.ok()) {
      return normalize_dtype_result.status();
    }
    compute_dtype = normalize_dtype_result.value();
    auto cast_result = CastTo(b, compute_dtype);
    if (!cast_result.ok()) {
      return cast_result.status();
    }
    b = cast_result.value();
    cast_result = CastTo(c, compute_dtype);
    if (!cast_result.ok()) {
      return cast_result.status();
    }
    c = cast_result.value();
    cast_result = CastTo(a, compute_dtype);
    if (!cast_result.ok()) {
      return cast_result.status();
    }
    a = cast_result.value();
  }
  auto result = TernaryOp(op, compute_dtype, a->LoadValue(), b->LoadValue(), c->LoadValue());
  if (!result.ok()) {
    return result.status();
  }
  return NewValue(compute_dtype, result.value());
}
}  // namespace compiler
}  // namespace rapidudf