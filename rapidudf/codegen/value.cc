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
#include "rapidudf/codegen/value.h"

#include <atomic>
#include <memory>
#include <type_traits>
#include <vector>
#include "rapidudf/codegen/ops/arithmetic_ops.h"
#include "rapidudf/codegen/optype.h"
#include "xbyak/xbyak_util.h"

#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/code_generator.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/ops/cast.h"
#include "rapidudf/codegen/ops/cmp.h"
#include "rapidudf/codegen/ops/copy.h"
#include "rapidudf/codegen/ops/logic_ops.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
using namespace Xbyak::util;
static std::atomic<uint64_t> g_uniq_id_seed{0};
Value::Value(Private, CodeGenerator* c, DType dtype, uint32_t stack_offset, uint32_t stack_len, bool temp) {
  c_ = c;
  dtype_ = dtype;
  stack_offset_ = stack_offset;
  stack_len_ = stack_len;
  temp_ = temp;
  for (uint32_t i = 0; i < dtype_.QwordSize(); i++) {
    stack_addrs_.emplace_back(
        c_->GetStackAddr(dtype_.QwordSize() == 1 ? dtype_ : DType(DATA_U64), stack_offset - i * 8));
  }
  uniq_id_ = g_uniq_id_seed.fetch_add(1);
}
Value::Value(Private, CodeGenerator* c, DType dtype, std::vector<const Xbyak::Reg*> regs, bool temp) {
  c_ = c;
  dtype_ = dtype;
  temp_ = temp;
  registers_ = regs;
  registers_.resize(dtype_.QwordSize());
  uniq_id_ = g_uniq_id_seed.fetch_add(1);
}
Value::Value(Private, CodeGenerator* c, DType dtype, const Xbyak::Reg* reg, bool temp)
    : Value(Private(), c, dtype, std::vector<const Xbyak::Reg*>{reg}, temp) {}

Value::Value(Private, CodeGenerator* c, DType dtype, std::vector<uint64_t> const_values) {
  c_ = c;
  dtype_ = dtype;
  const_values_ = const_values;
  const_values_.resize(dtype_.QwordSize());
  uniq_id_ = g_uniq_id_seed.fetch_add(1);
}
Value::Value(Private, CodeGenerator* c, DType dtype, uint64_t const_value)
    : Value(Private(), c, dtype, std::vector<uint64_t>{const_value}) {}
Value::~Value() {}

const Xbyak::Operand& Value::GetOperand(size_t i) const {
  if (IsRegister()) {
    return GetRegister(i);
  } else if (IsStack()) {
    return GetStackAddress(i);
  } else {
    RUDF_CRITICAL("unreadble");
    return GetRegister(i);
  }
}

void Value::Swap(Value& other) {
  std::swap(dtype_, other.dtype_);
  std::swap(registers_, other.registers_);
  std::swap(stack_addrs_, other.stack_addrs_);
  std::swap(const_values_, other.const_values_);
  std::swap(stack_offset_, other.stack_offset_);
  std::swap(stack_len_, other.stack_len_);
}

void Value::Drop() {
  if (droped_) {
    return;
  }
  if (temp_) {
    for (auto reg : registers_) {
      if (nullptr != reg) {
        c_->RecycleRegister(reg);
      }
    }
  }
  registers_.clear();
  if (stack_offset_ > 0) {
    if (temp_) {
      c_->RecycleStack(stack_offset_, stack_len_);
    }
    stack_offset_ = 0;
    stack_len_ = 0;
  }
  droped_ = true;
}

int Value::ToStack() {
  if (IsRegister()) {
    auto [stack_offset, stack_len] = c_->AllocateStack(dtype_, dtype_.ByteSize());
    if (dtype_.QwordSize() == 1) {
      int rc = copy_value(c_->GetCodeGen(), dtype_, GetRegister(), c_->GetStackAddr(dtype_, stack_offset));
      if (0 != rc) {
        c_->RecycleStack(stack_offset, stack_len);
        return rc;
      }
    } else {
      uint32_t cursor = 0;
      for (uint32_t i = 0; i < registers_.size(); i++) {
        int rc =
            copy_value(c_->GetCodeGen(), DATA_U64, *registers_[i], c_->GetStackAddr(dtype_, stack_offset - cursor));
        if (0 != rc) {
          c_->RecycleStack(stack_offset, stack_len);
          return rc;
        }
        cursor += 8;
      }
    }
    stack_offset_ = stack_offset;
    stack_len_ = stack_len;
    stack_addrs_.clear();
    for (uint32_t i = 0; i < dtype_.QwordSize(); i++) {
      stack_addrs_.emplace_back(c_->GetStackAddr(i == 0 ? dtype_ : DType(DATA_U64), stack_offset - i * 8));
    }
    registers_.clear();
  }
  return 0;
}

int Value::UnaryOp(OpToken op, ValuePtr result) {
  auto left = New(c_, dtype_, static_cast<uint64_t>(0));
  switch (op) {
    case OP_NEGATIVE: {
      if (!dtype_.IsNumber()) {
        RUDF_ERROR("Can NOT do negative op on non number value:{}", dtype_);
        return -1;
      }

      left->Set(0);
      return left->ArithmeticOp(OP_MINUS, *this, result);
    }
    case OP_NOT: {
      if (!dtype_.IsBool()) {
        RUDF_ERROR("Can NOT do negative op on non u8 value:{}", dtype_);
        return -1;
      }
      left->Set(1);
      return left->ArithmeticOp(OP_MINUS, *this, result);
      break;
    }
    default: {
      RUDF_ERROR("Unsupported unary op:{}", op);
      return -1;
    }
  }
}

int Value::LogicOp(OpToken op, Value& other, ValuePtr result) {
  if (!dtype_.IsNumber() || !other.GetDType().IsNumber()) {
    RUDF_ERROR("Can NOT do logic op:{} with left dtype:{}, right dtype:{}", dtype_, other.GetDType());
    return -1;
  }
  if (result->GetDType() != DATA_U8) {
    RUDF_ERROR("Logic op result value MUST be bool/u8, but the result dtype:{}", result->GetDType());
    return -1;
  }
  if (result->IsConst()) {
    RUDF_ERROR("Logic op result value MUST not be const");
    return -1;
  }
  auto left = SelfPtr();
  auto right = other.SelfPtr();
  left = left->CastTo(DATA_U8);
  if (!left) {
    return -1;
  }
  right = right->CastTo(DATA_U8);
  if (!right) {
    return -1;
  }
  // int rc = CastTo(DATA_U8);
  // if (0 == rc) {
  //   rc = other.CastTo(DATA_U8);
  // }
  // if (0 != rc) {
  //   return rc;
  // }
  if (left->IsConst() && right->IsConst()) {
    if (result->IsTemp()) {
      bool result_bin = false;
      int rc = logic_op(c_->GetCodeGen(), op, DATA_U8, left->GetConstBin(), right->GetConstBin(), result_bin);
      if (0 != rc) {
        return rc;
      }
      auto const_bin = New(c_, DATA_U8, result_bin);
      result->Swap(*const_bin);
      return 0;
    } else {
      return logic_op(c_->GetCodeGen(), op, DATA_U8, left->GetConstBin(), right->GetConstBin(), result->GetOperand());
    }
  }
  if (left->IsConst() && !right->IsConst()) {
    return logic_op(c_->GetCodeGen(), op, DATA_U8, left->GetConstBin(), right->GetOperand(), result->GetOperand());
  }
  if (!left->IsConst() && right->IsConst()) {
    return logic_op(c_->GetCodeGen(), op, DATA_U8, left->GetOperand(), right->GetConstBin(), result->GetOperand());
  }
  if (!left->IsConst() && !right->IsConst()) {
    return logic_op(c_->GetCodeGen(), op, DATA_U8, left->GetOperand(), right->GetOperand(), result->GetOperand());
  }
  RUDF_CRITICAL("Unreacable");
  return -1;
}

int Value::ArithmeticOp(OpToken op, Value& other, ValuePtr result) {
  auto left = SelfPtr();
  auto right = other.SelfPtr();
  if (left->GetDType().IsSimdVector() || right->GetDType().IsSimdVector()) {
    // do simd binary op

    if (left->GetDType().IsSimdVector() && right->GetDType().IsSimdVector()) {
      DType left_ele_dtype = left->GetDType().Elem();
      DType right_ele_dtype = right->GetDType().Elem();
      if (left_ele_dtype != right_ele_dtype) {
        RUDF_ERROR("Can NOT do op:{} with left:{}, right:{}", op, left->GetDType(), right->GetDType());
        return -1;
      }
      if (left->IsConst()) {
        left = left->CastTo(right->GetDType().Elem());
      }
      if (right->IsConst()) {
        right = right->CastTo(left->GetDType().Elem());
      }
    }
  }

  DType dst_dtype = dtype_;
  if (other.dtype_ != dtype_) {
    dst_dtype = other.dtype_ > dtype_ ? other.dtype_ : dtype_;
    left = left->CastTo(dst_dtype);
    if (!left) {
      return -1;
    }
    right = right->CastTo(dst_dtype);
    if (!right) {
      return -1;
    }
  }

  if (!left->IsConst() && !right->IsConst()) {
    return arithmetic_op(c_->GetCodeGen(), op, dst_dtype, left->GetOperand(), right->GetOperand(),
                         result->GetOperand());
  } else if (!left->IsConst() && right->IsConst()) {
    return arithmetic_op(c_->GetCodeGen(), op, dst_dtype, left->GetOperand(), right->GetConstBin(),
                         result->GetOperand());
  } else if (left->IsConst() && !right->IsConst()) {
    return arithmetic_op(c_->GetCodeGen(), op, dst_dtype, left->GetConstBin(), right->GetOperand(),
                         result->GetOperand());
  } else if (left->IsConst() && right->IsConst()) {
    if (result->IsTemp()) {
      uint64_t result_bin = false;
      int rc = arithmetic_op(c_->GetCodeGen(), op, dst_dtype, left->GetConstBin(), right->GetConstBin(), result_bin);
      if (0 != rc) {
        return rc;
      }
      auto const_bin = New(c_, dtype_, result_bin);
      result->Swap(*const_bin);
      return 0;
    } else {
      return arithmetic_op(c_->GetCodeGen(), op, dst_dtype, left->GetConstBin(), right->GetConstBin(),
                           result->GetOperand());
    }
  } else {
    abort();
  }
  return 0;
}

const Xbyak::Address& Value::GetStackAddress(size_t i) const { return stack_addrs_[i]; }

int Value::Write(const Value& other) {
  if (!dtype_.IsPtr()) {
    RUDF_ERROR("Can NOT write into non ptr value:{}", dtype_);
    return -1;
  }
  if (dtype_.PtrTo() != other.dtype_) {
    RUDF_ERROR("Can NOT write into ptr value:{} with value dtype:{}", dtype_, other.dtype_);
    return -1;
  }
  if (IsConst()) {
    RUDF_ERROR("Can NOT write to const value");
    return -1;
  }

  if (other.IsConst()) {
    return DoSetValue(other.const_values_, true);
  } else if (other.IsStack() || other.IsRegister()) {
    for (uint32_t i = 0; i < dtype_.QwordSize(); i++) {
      int rc = copy_value(c_->GetCodeGen(), dtype_.QwordSize() == 1 ? dtype_ : DATA_U64, other.GetOperand(i),
                          GetOperand(i), true);
      if (0 != rc) {
        RUDF_ERROR(".copy_value faie");
        return rc;
      }
    }
    return 0;
  }
  RUDF_CRITICAL("Not reachable");
  return -1;
  return 0;
}

int Value::Cmp(OpToken op, Value& other, ValuePtr result) {
  if (result && result->IsConst()) {
    RUDF_ERROR("Can NOT store cmp result into const value.");
    return -1;
  }

  if (dtype_.IsJsonPtr() || other.dtype_.IsJsonPtr()) {
    return CmpJson(op, other, result, false);
  }

  DType cmp_dtype = dtype_;
  auto left = SelfPtr();
  auto right = other.SelfPtr();
  if (other.dtype_ != dtype_) {
    if (dtype_.CanCastTo(other.dtype_)) {
      cmp_dtype = other.dtype_;
      left = CastTo(cmp_dtype);
      if (!left) {
        return -1;
      }
    } else if (other.dtype_.CanCastTo(dtype_)) {
      cmp_dtype = dtype_;
      right = other.CastTo(cmp_dtype);
      if (!right) {
        return -1;
      }
    } else {
      RUDF_ERROR("Invalid src:dtype:{} and dst dtype:{} to cmp.", dtype_, other.dtype_);
      return -1;
    }
  }
  if (cmp_dtype.IsStringView()) {
    auto op_arg = New(c_, DATA_U32, static_cast<uint64_t>(op));
    std::vector<ValuePtr> args{op_arg, left, right};
    auto cmp_result = c_->CallFunction(std::string(kBuiltinStringViewCmp), args);
    if (result) {
      result->Copy(*cmp_result);
      c_->DropTmpValue(cmp_result);
      c_->DropTmpValue(args);
      return 0;
    } else {
      RUDF_CRITICAL("Can not save cmp string_view result!");
    }
  }
  if (!cmp_dtype.IsNumber()) {
    RUDF_ERROR("Invalid dtype:{} to cmp.", cmp_dtype);
    return -1;
  }
  if (left->IsConst() && right->IsConst()) {
    if (result) {
      if (result->IsTemp()) {
        bool result_b = false;
        int rc = cmp_value(c_->GetCodeGen(), op, cmp_dtype, left->GetConstBin(), right->GetConstBin(), result_b);
        if (0 != rc) {
          return rc;
        }
        auto const_b = New(c_, DATA_U8, static_cast<uint64_t>(result_b ? 1 : 0));
        result->Swap(*const_b);
        return 0;
      } else {
        return cmp_value(c_->GetCodeGen(), op, cmp_dtype, left->GetConstBin(), right->GetConstBin(),
                         result->GetOperand());
      }
    } else {
      return cmp_value(c_->GetCodeGen(), cmp_dtype, left->GetConstBin(), right->GetConstBin());
    }
  }
  int rc = 0;
  if (right->IsConst()) {
    rc = left->CmpConst(*right);
  } else if (other.IsRegister()) {
    rc = left->CmpRegister(*right);
  } else if (other.IsStack()) {
    rc = left->CmpStack(*right);
  } else {
    RUDF_ERROR("Invalid state to cmp");
    rc = -1;
  }
  if (0 != rc) {
    return rc;
  }
  if (!result) {
    return 0;
  }
  RUDF_DEBUG("Start to store cmp result.");
  return store_cmp_result(c_->GetCodeGen(), op, result->GetOperand());
}

int Value::Mov(const Xbyak::Reg& dst) { return MovValue(dst); }

int Value::Mov(const Xbyak::Reg& dst0, const Xbyak::Reg& dst1) {
  if (dtype_.ByteSize() != 16) {
    RUDF_CRITICAL("Only move 16bytes to double registers, while dtype:{}", dtype_);
    abort();
    return -1;
  }
  return MovValue(dst0, dst1);
}

int Value::Copy(const Value& other) {
  if (IsConst()) {
    RUDF_ERROR("Can NOT copy to const value");
    return -1;
  }
  if (dtype_ != other.GetDType()) {
    RUDF_ERROR("Can NOT copy from dtype:{} to dtype:{}", other.dtype_, dtype_);
    return -1;
  }
  // RUDF_DEBUG("src_stack:{}, src_reg:{}, dst_stack:{}, dst_reg:{}", IsStack(), IsRegister(), other.IsStack(),
  //            other.IsRegister());
  if (other.IsConst()) {
    return DoSetValue(other.const_values_);
  } else if (other.IsStack() || other.IsRegister()) {
    for (uint32_t i = 0; i < dtype_.QwordSize(); i++) {
      int rc =
          copy_value(c_->GetCodeGen(), dtype_.QwordSize() == 1 ? dtype_ : DATA_U64, other.GetOperand(i), GetOperand(i));
      if (0 != rc) {
        RUDF_ERROR("copy_value fail");
        return rc;
      }
    }
    return 0;
  }
  RUDF_CRITICAL("Not reachable");
  return -1;
}

ValuePtr Value::CastTo(DType dtype) {
  if (dtype_ == dtype) {
    return SelfPtr();
  }
  if (IsTemp() || IsConst()) {
    int rc = CastToInplace(dtype);
    if (rc != 0) {
      return {};
    }
    return SelfPtr();
  }
  auto tmp = c_->NewValue(dtype_, {}, true);
  int rc = tmp->Copy(*this);
  if (rc != 0) {
    return {};
  }
  rc = tmp->CastToInplace(dtype);
  if (rc != 0) {
    return {};
  }
  return tmp;
}

int Value::CastToInplace(DType dtype) {
  if (dtype_ == dtype) {
    return 0;
  }

  if (dtype_.IsVoid() && !dtype.IsVoid()) {
    if (dtype_.QwordSize() < dtype.QwordSize()) {
      RUDF_ERROR("Invalid src:dtype:{} and dst dtype:{} to cast.", dtype_, dtype);
      return -1;
    }
    dtype_ = dtype;
    if (IsStack()) {
      stack_addrs_[0] = c_->GetStackAddr(dtype_, stack_offset_);
    }
    return 0;
  }
  if (!dtype_.CanCastTo(dtype)) {
    RUDF_ERROR("Can NOT cast from {} to {}", dtype_, dtype);
    return -1;
  }
  if (dtype.IsStringView()) {
    ValuePtr cast_result;
    if (dtype_.IsPtr()) {
      auto tmp = c_->NewValue(dtype);
      if (dtype_.PtrTo().IsString()) {
        cast_result = c_->CallFunction(std::string(kBuiltinCastStdStrToStringView), {SelfPtr()});
      } else if (dtype_.PtrTo().IsFlatbuffersString()) {
        cast_result = c_->CallFunction(std::string(kBuiltinCastFbsStrToStringView), {SelfPtr()});
      } else if (dtype_.PtrTo().IsStdStringView()) {
        cast_result = c_->CallFunction(std::string(kBuiltinCastStdStrViewToStringView), {SelfPtr()});
      }
      if (!cast_result) {
        RUDF_ERROR("Cast from {} to {} failed", dtype_, dtype);
        return -1;
      }
      tmp->Copy(*cast_result);
      dtype_ = dtype;
      Swap(*tmp);
      c_->DropTmpValue(tmp);
      return 0;
    } else {
      RUDF_ERROR("Can NOT cast from {} to {}", dtype_, dtype);
      return -1;
    }
  }

  if (!dtype_.IsNumber() || !dtype.IsNumber()) {
    RUDF_ERROR("Invalid src:dtype:{} and dst dtype:{} to cast.", dtype_, dtype);
    return -1;
  }
  if (IsConst()) {
    const_values_[0] = convert_to(const_values_[0], dtype_, dtype);
    dtype_ = dtype;
    return 0;
  } else if (IsStack()) {
    uint32_t src_len = dtype_.ByteSize();
    uint32_t dst_len = dtype.ByteSize();
    int rc = 0;
    if (src_len != dst_len) {
      auto tmp = c_->AllocateValue(dtype, dst_len, {}, false, temp_);
      rc = static_cast_value(c_->GetCodeGen(), GetStackAddress(), dtype_, tmp->GetStackAddress(), dtype);
      if (rc == 0) {
        std::swap(tmp->stack_len_, stack_len_);
        std::swap(tmp->stack_offset_, stack_offset_);
        tmp->Drop();
      }
    } else {
      rc = static_cast_value(c_->GetCodeGen(), GetStackAddress(), dtype_, GetStackAddress(), dtype);
    }
    if (0 == rc) {
      dtype_ = dtype;
    }
    return 0;
  } else if (IsRegister()) {
    int rc = static_cast_value(c_->GetCodeGen(), *registers_[0], dtype_, dtype);
    if (0 == rc) {
      dtype_ = dtype;
    }
    return rc;
  }
  RUDF_ERROR("Invalid state to cast.");
  return -1;
}

int Value::CmpJson(OpToken op, Value& other, ValuePtr result, bool reverse) {
  if (dtype_.IsJsonPtr()) {
    if (other.GetDType().IsNumber() || other.GetDType().IsStringView()) {
      auto other_val = other.SelfPtr();
      if (other.GetDType().IsStringView() || other.GetDType().IsF64() || other.GetDType().IsBool()) {
        // do nothing
      } else {
        if (other.GetDType().IsF32()) {
          other_val = other.CastTo(DATA_F64);
        } else {
          other_val = other.CastTo(DATA_I64);
        }
      }
      auto op_arg = New(c_, DATA_U32, static_cast<uint64_t>(op));
      auto reverse_arg = New(c_, DATA_U8, static_cast<uint64_t>(0));
      std::vector<ValuePtr> args{op_arg, SelfPtr(), other_val, reverse_arg};
      ValuePtr cmp_result;
      switch (other_val->GetDType().GetFundamentalType()) {
        case DATA_STRING_VIEW: {
          cmp_result = c_->CallFunction(std::string(kBuiltinJsonCmpString), args);
          break;
        }
        case DATA_I32:
        case DATA_I64: {
          cmp_result = c_->CallFunction(std::string(kBuiltinJsonCmpInt), args);
          break;
        }
        case DATA_F64: {
          cmp_result = c_->CallFunction(std::string(kBuiltinJsonCmpFloat), args);
          break;
        }
        case DATA_U8: {
          cmp_result = c_->CallFunction(std::string(kBuiltinJsonCmpBool), args);
          break;
        }
        default: {
          break;
        }
      }
      if (result) {
        result->Copy(*cmp_result);
        c_->DropTmpValue(cmp_result);
        return 0;
      } else {
        RUDF_CRITICAL("Can not save cmp json result!");
      }
    } else if (other.GetDType().IsJsonPtr()) {
      auto op_arg = New(c_, DATA_U32, static_cast<uint64_t>(op));
      std::vector<ValuePtr> args{op_arg, SelfPtr(), other.SelfPtr()};
      ValuePtr cmp_result = c_->CallFunction(std::string(kBuiltinJsonCmpJson), args);
      if (result) {
        result->Copy(*cmp_result);
        c_->DropTmpValue(cmp_result);
        return 0;
      } else {
        RUDF_CRITICAL("Can not save cmp json result!");
      }
    } else {
      RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
      return -1;
    }
  } else if (GetDType().IsNumber() || GetDType().IsStringView()) {
    if (other.GetDType().IsJsonPtr()) {
      return other.CmpJson(op, *this, result, true);
    } else {
      RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
      return -1;
    }
  } else {
    RUDF_ERROR("Can NOT cmp json with left:{}, right:{}", dtype_, other.dtype_);
    return -1;
  }
  return -1;
}

int Value::CmpConst(const Value& other) {
  if (IsStack()) {
    return cmp_value(c_->GetCodeGen(), dtype_, GetStackAddress(), other.GetConstBin());
  } else if (IsRegister()) {
    return cmp_value(c_->GetCodeGen(), dtype_, GetRegister(), other.GetConstBin());
  } else {
    RUDF_ERROR("Invalid state to cmp.");
    return -1;
  }
}
int Value::CmpStack(const Value& other) {
  if (IsConst()) {
    return cmp_value(c_->GetCodeGen(), dtype_, other.GetStackAddress(), GetConstBin(), true);
  } else if (IsStack()) {
    return cmp_value(c_->GetCodeGen(), dtype_, GetStackAddress(), other.GetStackAddress());
  } else if (IsRegister()) {
    return cmp_value(c_->GetCodeGen(), dtype_, GetRegister(), other.GetStackAddress());
  } else {
    RUDF_ERROR("Invalid state to cmp.");
    return -1;
  }
}
int Value::CmpRegister(const Value& other) {
  if (IsConst()) {
    return cmp_value(c_->GetCodeGen(), dtype_, other.GetRegister(), GetConstBin(), true);
  } else if (IsStack()) {
    return cmp_value(c_->GetCodeGen(), dtype_, other.GetRegister(), GetStackAddress(), true);
  } else if (IsRegister()) {
    return cmp_value(c_->GetCodeGen(), dtype_, GetRegister(), other.GetRegister());
  } else {
    RUDF_ERROR("Invalid state to cmp.");
    return -1;
  }
}

int Value::MovValue(std::vector<const Xbyak::Reg*> dsts) {
  if (dtype_.QwordSize() != dsts.size()) {
    RUDF_ERROR("Can NOT mov values for dtype:{} with dst register count:{}", dtype_, dsts.size());
    return -1;
  }
  if (IsConst()) {
    for (size_t i = 0; i < dsts.size(); i++) {
      int rc = copy_value(c_->GetCodeGen(), i == 0 ? dtype_ : DATA_U64, const_values_[i], *dsts[i]);
      if (0 != rc) {
        return rc;
      }
    }
    return 0;
  } else if (IsStack()) {
    for (size_t i = 0; i < dsts.size(); i++) {
      int rc = copy_value(c_->GetCodeGen(), i == 0 ? dtype_ : DATA_U64, GetStackAddress(i), *dsts[i]);
      if (0 != rc) {
        return rc;
      }
    }
    return 0;
  } else if (IsRegister()) {
    for (size_t i = 0; i < dsts.size(); i++) {
      int rc = copy_value(c_->GetCodeGen(), i == 0 ? dtype_ : DATA_U64, GetRegister(i), *dsts[i]);
      if (0 != rc) {
        return rc;
      }
    }
    return 0;
  } else {
    RUDF_ERROR("Can NOT mov unreable!!!", dtype_, dsts.size());
    return -1;
  }
}

int Value::MovValue(const Xbyak::Reg& dst) { return MovValue(std::vector<const Xbyak::Reg*>{&dst}); }
int Value::MovValue(const Xbyak::Reg& dst0, const Xbyak::Reg& dst1) {
  return MovValue(std::vector<const Xbyak::Reg*>{&dst0, &dst1});
}

int Value::DoSetValue(std::vector<uint64_t> vals, bool is_ptr) {
  if (dtype_.QwordSize() != vals.size()) {
    RUDF_ERROR("Can NOT set values for dtype:{} with values count:{}", dtype_, vals.size());
    return -1;
  }
  if (IsConst()) {
    for (size_t i = 0; i < vals.size(); i++) {
      const_values_[i] = vals[i];
    }
    return 0;
  } else if (IsRegister()) {
    for (size_t i = 0; i < vals.size(); i++) {
      int rc = copy_value(c_->GetCodeGen(), i == 0 ? dtype_ : DATA_U64, vals[i], GetRegister(i), is_ptr);
      if (0 != rc) {
        return rc;
      }
    }
    return 0;
  } else if (IsStack()) {
    for (size_t i = 0; i < vals.size(); i++) {
      int rc = copy_value(c_->GetCodeGen(), i == 0 ? dtype_ : DATA_U64, vals[i], GetStackAddress(i), is_ptr);
      if (0 != rc) {
        return rc;
      }
    }
    return 0;
  }
  RUDF_CRITICAL("Not reachable!!!");
  return -1;
}

int Value::DoSetValue(uint64_t val) { return DoSetValue(std::vector<uint64_t>{val}); }

int Value::DoSetStringView(StringView str) {
  if (!dtype_.IsStringView()) {
    RUDF_ERROR("Invalid dtype:{} to set string_view", dtype_);
    return -1;
  }
  uint64_t* data = reinterpret_cast<uint64_t*>(&str);
  return DoSetValue(std::vector<uint64_t>{data[0], data[1]});
}

}  // namespace rapidudf
