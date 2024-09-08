/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once
#include <string.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/ops/copy.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/register.h"
#include "xbyak/xbyak.h"

namespace rapidudf {

class Value;
using ValuePtr = std::shared_ptr<Value>;
class CodeGenerator;
class Value : public std::enable_shared_from_this<Value> {
 private:
  struct Private {
    explicit Private() = default;
  };

 public:
  Value(Private, CodeGenerator* c, DType dtype, uint32_t stack_offset, uint32_t stack_len, bool temp);
  Value(Private, CodeGenerator* c, DType dtype, const Xbyak::Reg* reg, bool temp = false);
  Value(Private, CodeGenerator* c, DType dtype, std::vector<const Xbyak::Reg*> regs, bool temp = false);
  Value(Private, CodeGenerator* c, DType dtype, uint64_t const_value);
  Value(Private, CodeGenerator* c, DType dtype, std::vector<uint64_t> const_values);

  static std::shared_ptr<Value> New(CodeGenerator* c, DType dtype, uint32_t stack_offset, uint32_t stack_len,
                                    bool temp = false) {
    return std::make_shared<Value>(Private(), c, dtype, stack_offset, stack_len, temp);
  }
  static std::shared_ptr<Value> New(CodeGenerator* c, DType dtype, const Xbyak::Reg* reg, bool temp = false) {
    return std::make_shared<Value>(Private(), c, dtype, reg, temp);
  }
  static std::shared_ptr<Value> New(CodeGenerator* c, DType dtype, std::vector<const Xbyak::Reg*> regs,
                                    bool temp = false) {
    return std::make_shared<Value>(Private(), c, dtype, regs, temp);
  }
  static std::shared_ptr<Value> New(CodeGenerator* c, DType dtype, uint64_t const_value) {
    return std::make_shared<Value>(Private(), c, dtype, const_value);
  }
  static std::shared_ptr<Value> New(CodeGenerator* c, DType dtype, std::vector<uint64_t> const_values) {
    return std::make_shared<Value>(Private(), c, dtype, const_values);
  }
  ~Value();

  const std::string& GetVarName() const { return var_name_; }
  void SetVarName(const std::string& name) { var_name_ = name; }
  uint64_t Id() const { return uniq_id_; }
  bool IsTemp() const { return temp_; }
  void SetTemp(bool v) { temp_ = v; }
  uint32_t GetStackOffset() const { return stack_offset_; }
  const Xbyak::Address& GetStackAddress(size_t i = 0) const;
  uint64_t GetConstBin(size_t i = 0) const { return const_values_[i]; }
  const Xbyak::Reg& GetRegister(size_t i = 0) const { return *registers_[i]; }
  const Xbyak::Operand& GetOperand(size_t i = 0) const;
  int ToStack();

  bool IsConst() const { return !const_values_.empty(); }
  bool IsStack() const { return stack_offset_ > 0; }
  bool IsRegister() const { return !registers_.empty(); }

  DType GetDType() const { return dtype_; }

  int Mov(const Xbyak::Reg& dst);
  int Mov(const Xbyak::Reg& dst0, const Xbyak::Reg& dst1);
  int Cmp(OpToken op, Value& other, ValuePtr result);
  ValuePtr CastTo(DType dtype);
  int CastToInplace(DType dtype);
  int Copy(const Value& other);
  int LogicOp(OpToken op, Value& other, ValuePtr result);
  int ArithmeticOp(OpToken op, Value& other, ValuePtr result);
  int UnaryOp(OpToken op, ValuePtr result);
  int Write(const Value& other);
  // int VectorGet(uint32_t idx, ValuePtr result);

  int SetSimdVectorTemporary(bool v);

  int Copy(const Xbyak::Reg* reg) {
    auto other = New(c_, dtype_, reg);
    return Copy(*other);
  }

  void Swap(Value& other);

  void Drop();

  int SetSpanSize(uint64_t size);
  int SetSpanStackPtr(uint32_t offset);

  std::string StorageInfo() const;

  template <typename T>
  int Set(T val) {
    if constexpr (std::is_same_v<double, T>) {
      if (dtype_ == DATA_F64) {
        uint64_t int_val = 0;
        memcpy(&int_val, &val, sizeof(double));
        return DoSetValue(int_val);
      } else if (dtype_ == DATA_F32) {
        float fv = static_cast<float>(val);
        uint32_t int_val = 0;
        memcpy(&int_val, &fv, sizeof(float));
        return DoSetValue(int_val);
      } else {
        uint64_t int_val = static_cast<uint64_t>(val);
        return DoSetValue(int_val);
      }
    } else if constexpr (std::is_same_v<float, T>) {
      if (dtype_ == DATA_F64) {
        double dv = static_cast<double>(val);
        uint64_t int_val = 0;
        memcpy(&int_val, &dv, sizeof(double));
        return DoSetValue(int_val);
      } else if (dtype_ == DATA_F32) {
        uint32_t int_val = 0;
        memcpy(&int_val, &val, sizeof(float));
        return DoSetValue(int_val);
      } else {
        uint64_t int_val = static_cast<uint64_t>(val);
        return DoSetValue(int_val);
      }
    } else if constexpr (std::is_pointer_v<T>) {
      uint64_t int_val = reinterpret_cast<uint64_t>(val);
      return DoSetValue(int_val);
    } else if constexpr (std::is_same_v<StringView, T>) {
      return DoSetStringView(val);
    } else {
      uint64_t int_val = static_cast<uint64_t>(val);
      return DoSetValue(int_val);
    }
  }

 private:
  ValuePtr SelfPtr() { return shared_from_this(); }

  int MovRegister8(const Xbyak::Reg& dst, const Xbyak::Reg& src);
  int MovRegister4(const Xbyak::Reg& dst, const Xbyak::Reg& src);
  int MovValue(const Xbyak::Reg& dst0, const Xbyak::Reg& dst1);
  int MovValue(const Xbyak::Reg& dst);
  int MovValue(std::vector<const Xbyak::Reg*> dsts);
  int DoSetValue(std::vector<uint64_t> vals, bool is_ptr = false);
  int DoSetValue(uint64_t val);
  int CmpConst(const Value& other);
  int CmpStack(const Value& other);
  int CmpRegister(const Value& other);
  int CmpJson(OpToken op, Value& other, ValuePtr result, bool reverse);

  int DoSetStringView(StringView str);

  std::string var_name_;
  uint64_t uniq_id_ = 0;
  CodeGenerator* c_ = nullptr;
  // std::shared_ptr<bool> codegen_available_;
  DType dtype_;
  uint32_t stack_offset_ = 0;
  uint32_t stack_len_ = 0;

  std::vector<uint64_t> const_values_;
  std::vector<Xbyak::Address> stack_addrs_;
  std::vector<const Xbyak::Reg*> registers_;

  // std::optional<uint64_t> const_value_;
  // std::optional<uint64_t> const_value1_;

  // const Xbyak::Reg* reg0_ = nullptr;
  // const Xbyak::Reg* reg1_ = nullptr;
  // std::optional<Xbyak::Address> stack_addr_;

  bool temp_ = false;
  bool droped_ = false;

  friend class codegen;
};

// static_assert(sizeof(Value) == 16, "sizeof(Value) != 16");
}  // namespace rapidudf