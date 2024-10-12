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

#include "rapidudf/compiler/type.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {

absl::StatusOr<::llvm::Value*> CodeGen::UnaryOp(OpToken op, DType dtype, ::llvm::Value* val) {
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Value* ret_value = nullptr;
  ::llvm::Type* element_type = GetElementType(val->getType());

  switch (op) {
    case OP_NEGATIVE: {
      if (element_type->isFloatingPointTy()) {
        ret_value = builder_->CreateFNeg(val);
      } else {
        ret_value = builder_->CreateNeg(val);
      }
      break;
    }
    case OP_NOT: {
      ret_value = builder_->CreateNot(val);
      break;
    }
    case OP_SIN: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::sin;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_COS: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::cos;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_FLOOR: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::floor;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_ABS: {
      builtin_intrinsic_args.emplace_back(val);
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::fabs;
      } else {
        builtin_intrinsic = ::llvm::Intrinsic::abs;
        builtin_intrinsic_args.emplace_back(builder_->getInt1(0));
      }
      break;
    }
    case OP_SQRT: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::sqrt;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_CEIL: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::ceil;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_ROUND: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::round;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_EXP: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::exp;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_EXP2: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::exp2;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_LOG: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::log;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_LOG2: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::log2;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_LOG10: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::log10;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_RINT: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::rint;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_TRUNC: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::trunc;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
#if LLVM_VERSION_MAJOR >= 19
    case OP_TAN: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::tan;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_ASIN: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::asin;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_ACOS: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::acos;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_ATAN: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::atan;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_SINH: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::sinh;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_COSH: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::cosh;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
    case OP_TANH: {
      if (element_type->isFloatingPointTy()) {
        builtin_intrinsic = ::llvm::Intrinsic::tanh;
        builtin_intrinsic_args.emplace_back(val);
      }
      break;
    }
#endif
    default: {
      return absl::InvalidArgumentError(fmt::format("Unsupported op:{}", op));
    }
  }
  if (ret_value != nullptr) {
    return ret_value;
  }
  if (builtin_intrinsic > 0) {
    return builder_->CreateIntrinsic(val->getType(), builtin_intrinsic, builtin_intrinsic_args);
  }

  return absl::InvalidArgumentError(fmt::format("Unsupported op:{} for dtype:{}", op, dtype));
}

absl::StatusOr<ValuePtr> CodeGen::UnaryOp(OpToken op, ValuePtr val) {
  auto result = UnaryOp(op, val->GetDType(), val->LoadValue());
  if (!result.ok()) {
    std::string extern_func_name = GetFunctionName(op, val->GetDType());
    auto val_result = CallFunction(extern_func_name, {val});
    if (val_result.ok()) {
      return val_result.value();
    } else {
      return result.status();
    }
  }
  return NewValue(val->GetDType(), result.value());
}

}  // namespace compiler
}  // namespace rapidudf