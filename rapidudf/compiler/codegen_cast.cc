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
    RUDF_LOG_RETURN_FMT_ERROR("Can NOT cast from {} to {}", src_dtype, dst_dtype);
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
  }

  RUDF_LOG_RETURN_FMT_ERROR("Can NOT cast from {} to {}", src_dtype, dst_dtype);
}

}  // namespace compiler
}  // namespace rapidudf