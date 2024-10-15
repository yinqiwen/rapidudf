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
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace compiler {
absl::StatusOr<::llvm::Value*> CodeGen::VectorUnaryOp(OpToken op, DType dtype, ::llvm::Value* input,
                                                      ::llvm::Value* output) {
  std::string fname = GetFunctionName(op, dtype.ToSimdVector());
  auto vector_type = get_vector_type(builder_->getContext(), dtype);
  auto result = CallFunction(fname, std::vector<::llvm::Value*>{input, output});
  if (!result.ok()) {
    return result.status();
  }
  return builder_->CreateLoad(vector_type, output);
}
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