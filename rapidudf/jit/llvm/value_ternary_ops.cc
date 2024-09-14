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

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Use.h>
#include <algorithm>
#include <string_view>
#include <vector>
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {
ValuePtr Value::TernaryOp(OpToken op, ValuePtr second, ValuePtr third) {
  ValuePtr _1st = SelfPtr();
  ValuePtr _2nd = second;
  ValuePtr _3rd = third;
  DType _1st_dtype = dtype_;
  DType _2nd_dtype = _2nd->dtype_;
  DType _3rd_dtype = _3rd->dtype_;
  DType dst_dtype;
  if (_1st->dtype_.IsNumber() && _2nd->dtype_.IsNumber() && _3rd->dtype_.IsNumber()) {
    if (_1st_dtype > _2nd_dtype) {
      dst_dtype = _1st_dtype;
    } else {
      dst_dtype = _2nd_dtype;
    }
    if (dst_dtype < _3rd_dtype) {
      dst_dtype = _3rd_dtype;
    }
    _1st = _1st->CastTo(dst_dtype);
    _2nd = _2nd->CastTo(dst_dtype);
    _3rd = _3rd->CastTo(dst_dtype);
  } else {
    RUDF_ERROR("Can NOT do {} for 1st:{}, 2nd:{}, 3rd:{}", op, _1st_dtype, _2nd_dtype, _3rd_dtype);
    return {};
  }
  if (!_1st || !_2nd || !_3rd) {
    RUDF_ERROR("Can NOT do {} for 1st:{}, 2nd:{}, 3rd:{}", op, _1st_dtype, _2nd_dtype, _3rd_dtype);
    return {};
  }
  ::llvm::Intrinsic::ID builtin_intrinsic = 0;
  DType ret_dtype = dst_dtype;
  ::llvm::Value* result_val = nullptr;
  switch (op) {
    case OP_FMA: {
      builtin_intrinsic = ::llvm::Intrinsic::fma;
      break;
    }
    default: {
      break;
    }
  }

  if (builtin_intrinsic != 0) {
    ::llvm::Type* arg_type = nullptr;
    if (dst_dtype.IsF32()) {
      arg_type = ::llvm::Type::getFloatTy(ir_builder_->getContext());
    } else if (dst_dtype.IsF64()) {
      arg_type = ::llvm::Type::getDoubleTy(ir_builder_->getContext());
    } else if (dst_dtype.IsInteger()) {
      _1st = _1st->CastTo(DATA_F64);
      _2nd = _2nd->CastTo(DATA_F64);
      _3rd = _3rd->CastTo(DATA_F64);
      arg_type = ::llvm::Type::getDoubleTy(ir_builder_->getContext());
      ret_dtype = DATA_F64;
    }
    if (nullptr != arg_type) {
      ::llvm::Function* intrinsic_func =
          ::llvm::Intrinsic::getDeclaration(ir_builder_->GetInsertBlock()->getModule(), builtin_intrinsic, {arg_type});
      result_val = ir_builder_->CreateCall(intrinsic_func, {_1st->GetValue(), _2nd->GetValue(), _3rd->GetValue()});
    }
  }

  if (!result_val) {
    RUDF_ERROR("Can NOT do {} for 1st:{}, 2nd:{}, 3rd:{}", op, _1st->GetDType(), _2nd->GetDType(), _3rd->GetDType());
    return {};
  }
  return New(ret_dtype, compiler_, result_val);
}

}  // namespace llvm
}  // namespace rapidudf