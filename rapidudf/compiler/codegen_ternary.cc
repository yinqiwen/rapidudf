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
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::VectorTernaryOp(OpToken op, DType dtype, ::llvm::Value* a, ::llvm::Value* b,
                                                        ::llvm::Value* c, ::llvm::Value* output) {
  std::string fname = GetFunctionName(op, dtype.ToSimdVector());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto result = CallFunction(fname, std::vector<::llvm::Value*>{a, b, c, output});
  if (!result.ok()) {
    return result.status();
  }
  return builder_->CreateLoad(vector_type, output);
}

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
  ValuePtr result_val;
  if (!result.ok()) {
    std::string extern_func_name = GetFunctionName(op, compute_dtype);
    auto val_result = CallFunction(extern_func_name, {a, b, c});
    if (val_result.ok()) {
      result_val = val_result.value();
    } else {
      return result.status();
    }
  } else {
    result_val = NewValue(compute_dtype, result.value());
  }
  // if (!result.ok()) {
  //   return result.status();
  // }
  // return NewValue(compute_dtype, result_val);
  return result_val;
}
}  // namespace compiler
}  // namespace rapidudf