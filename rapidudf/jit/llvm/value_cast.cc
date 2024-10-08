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
#include "llvm/IR/Use.h"
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace llvm {
ValuePtr Value::CastTo(DType dtype) {
  if (!dtype_.CanCastTo(dtype)) {
    RUDF_ERROR("Can NOT cast from {} to {}", dtype_, dtype);
    return {};
  }

  if (dtype == dtype_) {
    return SelfPtr();
  }

  auto* dst_type = get_type(ir_builder_->getContext(), dtype);
  if (dtype.IsNumber()) {
    ::llvm::Value* new_val = nullptr;
    if (dtype.IsFloat()) {
      if (dtype_.IsInteger()) {
        if (dtype_.IsSigned()) {
          new_val = ir_builder_->CreateSIToFP(GetValue(), dst_type);
        } else {
          new_val = ir_builder_->CreateUIToFP(GetValue(), dst_type);
        }
      } else {
        if (dtype.Bits() > dtype_.Bits()) {
          new_val = ir_builder_->CreateFPExt(GetValue(), dst_type);
        } else {
          new_val = ir_builder_->CreateFPTrunc(GetValue(), dst_type);
        }
      }
    } else {
      if (dtype_.IsFloat()) {
        if (dtype.IsSigned()) {
          new_val = ir_builder_->CreateFPToSI(GetValue(), dst_type);
        } else {
          new_val = ir_builder_->CreateFPToUI(GetValue(), dst_type);
        }
      } else {
        if (dtype.Bits() > dtype_.Bits()) {
          if (dtype.IsSigned()) {
            new_val = ir_builder_->CreateZExt(GetValue(), dst_type);
          } else {
            new_val = ir_builder_->CreateZExt(GetValue(), dst_type);
          }
        } else {
          new_val = ir_builder_->CreateTrunc(GetValue(), dst_type);
        }
      }
    }
    return New(dtype, compiler_, new_val);
  } else if (dtype.IsStringView()) {
    std::string_view cast_func;
    if (dtype_.IsStringPtr()) {
      cast_func = kBuiltinCastStdStrToStringView;
    } else if (dtype_.IsFlatbuffersStringPtr()) {
      cast_func = kBuiltinCastFbsStrToStringView;
    }
    if (!cast_func.empty()) {
      auto result = compiler_->CallFunction(cast_func, std::vector<ValuePtr>{SelfPtr()});
      if (result.ok()) {
        return result.value();
      }
      RUDF_ERROR("Can NOT cast from {} to {} with error:{}", dtype_, dtype, result.status().ToString());
      return {};
    }
  }
  RUDF_ERROR("Can NOT cast from {} to {}", dtype_, dtype);
  return {};
}
}  // namespace llvm
}  // namespace rapidudf