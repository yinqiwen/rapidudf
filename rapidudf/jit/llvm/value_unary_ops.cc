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
#include <vector>
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

ValuePtr Value::UnaryOp(OpToken op) {
  ::llvm::Value* result_val = nullptr;
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  std::vector<::llvm::Value*> builtin_intrinsic_args;
  ::llvm::Type* arg_type = get_type(ir_builder_->getContext(), dtype_);
  switch (op) {
    case OP_NEGATIVE: {
      if (!dtype_.IsNumber()) {
        RUDF_ERROR("Can NOT do negative op on non number value:{}", dtype_);
        return {};
      }
      if (dtype_.IsFloat()) {
        result_val = ir_builder_->CreateFNeg(GetValue());
      } else {
        result_val = ir_builder_->CreateNeg(GetValue());
      }
      break;
    }
    case OP_NOT: {
      if (!dtype_.IsBool()) {
        RUDF_ERROR("Can NOT do negative op on non u8 value:{}", dtype_);
        return {};
      }
      result_val = ir_builder_->CreateNot(GetValue());
      break;
    }
    case OP_SIN: {
      builtin_intrinsic = ::llvm::Intrinsic::sin;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_COS: {
      builtin_intrinsic = ::llvm::Intrinsic::cos;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_FLOOR: {
      builtin_intrinsic = ::llvm::Intrinsic::floor;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_ABS: {
      builtin_intrinsic_args.emplace_back(GetValue());
      if (dtype_.IsFloat()) {
        builtin_intrinsic = ::llvm::Intrinsic::fabs;
      } else {
        builtin_intrinsic = ::llvm::Intrinsic::abs;
        builtin_intrinsic_args.emplace_back(ir_builder_->getInt1(0));
      }
      break;
    }
    case OP_SQRT: {
      builtin_intrinsic = ::llvm::Intrinsic::sqrt;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_CEIL: {
      builtin_intrinsic = ::llvm::Intrinsic::ceil;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_ROUND: {
      builtin_intrinsic = ::llvm::Intrinsic::round;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_EXP: {
      builtin_intrinsic = ::llvm::Intrinsic::exp;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_EXP2: {
      builtin_intrinsic = ::llvm::Intrinsic::exp2;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_LOG: {
      builtin_intrinsic = ::llvm::Intrinsic::log;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_LOG2: {
      builtin_intrinsic = ::llvm::Intrinsic::log2;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_LOG10: {
      builtin_intrinsic = ::llvm::Intrinsic::log10;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_RINT: {
      builtin_intrinsic = ::llvm::Intrinsic::rint;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_TRUNC: {
      builtin_intrinsic = ::llvm::Intrinsic::trunc;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
#if LLVM_VERSION_MAJOR >= 19
    case OP_TAN: {
      builtin_intrinsic = ::llvm::Intrinsic::tan;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_ASIN: {
      builtin_intrinsic = ::llvm::Intrinsic::asin;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_ACOS: {
      builtin_intrinsic = ::llvm::Intrinsic::acos;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_ATAN: {
      builtin_intrinsic = ::llvm::Intrinsic::atan;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_SINH: {
      builtin_intrinsic = ::llvm::Intrinsic::sinh;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_COSH: {
      builtin_intrinsic = ::llvm::Intrinsic::cosh;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
    case OP_TANH: {
      builtin_intrinsic = ::llvm::Intrinsic::tanh;
      builtin_intrinsic_args.emplace_back(GetValue());
      break;
    }
#endif
    default: {
      // RUDF_ERROR("Unsupported unary op:{}", op);
      return {};
    }
  }

  if (builtin_intrinsic != 0) {
    if (nullptr != arg_type) {
      ::llvm::Function* intrinsic_func =
          ::llvm::Intrinsic::getDeclaration(ir_builder_->GetInsertBlock()->getModule(), builtin_intrinsic, {arg_type});
      result_val = ir_builder_->CreateCall(intrinsic_func, builtin_intrinsic_args);
    }
  }
  if (!result_val) {
    RUDF_ERROR("Can NOT do {} for val:{}", op, dtype_);
  }
  return New(dtype_, compiler_, result_val);
}
}  // namespace llvm
}  // namespace rapidudf