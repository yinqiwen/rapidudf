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

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "fmt/format.h"

#include "rapidudf/compiler/type.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::CastTo(::llvm::Value* val, DType src_dtype, DType dst_dtype) {
  ::llvm::Value* new_val = nullptr;
  ::llvm::Type* dst_type = get_type(builder_->getContext(), dst_dtype);
  if (val->getType()->isVectorTy()) {
    ::llvm::VectorType* vtype = reinterpret_cast<::llvm::VectorType*>(val->getType());
    dst_type = ::llvm::VectorType::get(dst_type, vtype->getElementCount());
  }
  if (dst_dtype.IsFloat()) {
    if (src_dtype.IsInteger()) {
      if (src_dtype.IsSigned()) {
        new_val = builder_->CreateSIToFP(val, dst_type);
      } else {
        new_val = builder_->CreateUIToFP(val, dst_type);
      }
    } else {
      if (src_dtype.Bits() > dst_dtype.Bits()) {
        new_val = builder_->CreateFPExt(val, dst_type);
      } else {
        new_val = builder_->CreateFPTrunc(val, dst_type);
      }
    }
  } else {
    if (src_dtype.IsFloat()) {
      if (dst_dtype.IsSigned()) {
        new_val = builder_->CreateFPToSI(val, dst_type);
      } else {
        new_val = builder_->CreateFPToUI(val, dst_type);
      }
    } else {
      if (dst_dtype.Bits() > src_dtype.Bits()) {
        if (dst_dtype.IsSigned()) {
          new_val = builder_->CreateZExt(val, dst_type);
        } else {
          new_val = builder_->CreateZExt(val, dst_type);
        }
      } else {
        new_val = builder_->CreateTrunc(val, dst_type);
      }
    }
  }
  if (nullptr != new_val) {
    return new_val;
  }
  return absl::InvalidArgumentError(fmt::format("Can NOT cast from {} to {}", src_dtype, dst_dtype));
}
// absl::StatusOr<::llvm::Value*> CodeGen::CastTo(::llvm::Value* val, DType src_dtype, ::llvm::Value* dst_dtype) {
//   //  builder_->CreateSwitch(dst_dtype, BasicBlock *Dest)
//   return absl::UnimplementedError("####CastTo");
// }

absl::StatusOr<ValuePtr> CodeGen::CastTo(ValuePtr val, DType dst_dtype) {
  DType src_dtype = val->GetDType();
  if (src_dtype == dst_dtype) {
    return val;
  }
  if (!src_dtype.CanCastTo(dst_dtype)) {
    RUDF_LOG_RETURN_FMT_ERROR("Can NOT cast from {} to {}:{}", src_dtype, dst_dtype);
  }
  if (src_dtype.IsJsonPtr() && dst_dtype.IsPrimitive()) {
    std::string func_name = GetFunctionName(functions::kBuiltinJsonExtract, dst_dtype);
    return CallFunction(func_name, {val});
  }
  // auto* dst_type = get_type(builder_->getContext(), dst_dtype);
  if (dst_dtype.IsNumber()) {
    auto cast_result = CastTo(val->LoadValue(), src_dtype, dst_dtype);
    if (!cast_result.ok()) {
      return cast_result.status();
    }
    return Value::New(dst_dtype, builder_.get(), cast_result.value());
  } else if (dst_dtype.IsStringView()) {
    std::string_view cast_func;
    if (src_dtype.IsStringPtr()) {
      cast_func = functions::kBuiltinCastStdStrToStringView;
    } else if (src_dtype.IsFlatbuffersStringPtr()) {
      cast_func = functions::kBuiltinCastFbsStrToStringView;
    }
    if (!cast_func.empty()) {
      auto result = CallFunction(cast_func, std::vector<ValuePtr>{val});
      if (result.ok()) {
        return result.value();
      }
      return result.status();
    }
  } else if (dst_dtype.IsVoidPtr() && src_dtype.IsPtr()) {
    return val;
  }

  RUDF_LOG_RETURN_FMT_ERROR("Can NOT cast from {} to {}", src_dtype, dst_dtype);
}

}  // namespace compiler
}  // namespace rapidudf