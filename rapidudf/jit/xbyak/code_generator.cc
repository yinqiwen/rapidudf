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
#include "rapidudf/jit/xbyak/code_generator.h"
#include <stdio.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "rapidudf/jit/xbyak/register.h"
#include "xbyak/xbyak_util.h"

#include "rapidudf/jit/xbyak/function.h"
#include "rapidudf/jit/xbyak/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace xbyak {
// #define ADD_REG(z, n, text) tmp_xmm_regs_.emplace_back(std::make_shared<Xbyak::Xmm>(BOOST_PP_CAT(text, n)));
#define ADD_FLOAT_ARG_REG(z, n, text) func_arg_xmm_regs_.emplace_back(&BOOST_PP_CAT(text, n));
#define ADD_TMP_REG(z, n, text) free_tmp_regs.emplace_back(&BOOST_PP_CAT(text, n));
// #define ADD_TMP_128_REG(z, n, text) tmp_xmm_regs_.emplace_back(std::make_shared<Xbyak::Xmm>(BOOST_PP_CAT(text, n)));

// #define ADD_TMP_256_REG(z, n, text) tmp_ymm_regs_.emplace_back(std::make_shared<Xbyak::Ymm>(BOOST_PP_CAT(text, n)));
using namespace Xbyak::util;
static inline uint32_t allgin16(uint32_t x) { return (x + 15) & ~15; }

CodeGenerator::CodeGenerator(Options opts) : opts_(opts) {
  RUDF_DEBUG("New CodeGenerator with max_code_size:{}, use_registers:{}, use_callee_saved_registers:{}",
             opts_.max_code_size, opts_.use_registers, opts_.use_callee_saved_registers);
  jit_ = std::make_unique<Xbyak::CodeGenerator>(opts_.max_code_size);
  if (opts_.use_registers) {
    InitFreeRegisters();
  }

  jit_->push(rbp);
  jit_->mov(rbp, rsp);

  if (opts_.use_registers && opts_.use_callee_saved_registers) {
    for (auto reg : callee_saved_registers_) {
      SaveRegister(reg);
    }
  }
}
int CodeGenerator::Finish() {
  RestoreCalleeSavedRegisters();
  jit_->leave();
  jit_->ret();
  return 0;
}

CodeGenerator::~CodeGenerator() {}

int CodeGenerator::ReturnValue(ValuePtr val) {
  RUDF_INFO("return dtype:{} from {}", val->GetDType(), val->StorageInfo());
  if (val->GetDType().IsFloat()) {
    val->Mov(xmm0);
  } else {
    if (val->GetDType().Bits() <= 64) {
      val->Mov(rax.changeBit(val->GetDType().Bits()));
    } else {
      val->Mov(rax, rdx);
    }
  }
  return 0;
}

int CodeGenerator::Label(const std::string& label) {
  jit_->L(label);
  return 0;
}
int CodeGenerator::Jump(const std::string& label) {
  jit_->jmp(label, jit_->T_NEAR);
  return 0;
}
int CodeGenerator::Jump(const std::string& label, OpToken cmp) {
  switch (cmp) {
    case OP_EQUAL: {
      jit_->je(label, jit_->T_NEAR);
      break;
    }
    case OP_NOT_EQUAL: {
      jit_->jne(label, jit_->T_NEAR);
      break;
    }
    case OP_LESS: {
      jit_->jb(label, jit_->T_NEAR);
      break;
    }
    case OP_LESS_EQUAL: {
      jit_->jbe(label, jit_->T_NEAR);
      break;
    }
    case OP_GREATER: {
      jit_->ja(label, jit_->T_NEAR);
      break;
    }
    case OP_GREATER_EQUAL: {
      jit_->jae(label, jit_->T_NEAR);
      break;
    }
    default: {
      RUDF_ERROR("Unsupported cmp result:{}", static_cast<int>(cmp));
      return -1;
    }
  }
  return 0;
}

void CodeGenerator::InitFreeRegisters() {
  // free_registers_.emplace_back(&rbx);
  // free_registers_.emplace_back(&r12);
  // free_registers_.emplace_back(&r13);
  // free_registers_.emplace_back(&r14);
  // free_registers_.emplace_back(&r15);
#define ADD_FREE_XMM_REG_HIGH(z, n, text) free_registers_.emplace_back(&BOOST_PP_CAT(text, n));
#define ADD_FREE_XMM_REG_LOW(z, n, text) free_registers_.emplace_back(&BOOST_PP_CAT(text, n));
  BOOST_PP_REPEAT_FROM_TO(8, 16, ADD_FREE_XMM_REG_LOW, xmm)
#undef ADD_FREE_XMM_REG_HIGH
#undef ADD_FREE_XMM_REG_LOW

  callee_saved_registers_.insert(&rbx);
  callee_saved_registers_.insert(&r12);
  callee_saved_registers_.insert(&r13);
  callee_saved_registers_.insert(&r14);
  callee_saved_registers_.insert(&r15);
}

void CodeGenerator::AddFreeRegisters(const std::vector<const Xbyak::Reg*>& regs, bool head) {
  if (!opts_.use_registers) {
    return;
  }
  if (head) {
    for (auto iter = regs.rbegin(); iter != regs.rend(); iter++) {
      free_registers_.emplace_front(*iter);
    }
  } else {
    for (auto reg : regs) {
      free_registers_.emplace_back(reg);
    }
  }
  std::sort(free_registers_.begin(), free_registers_.end(), [](const Xbyak::Reg* left, const Xbyak::Reg* right) {
    if (left->getKind() < right->getKind()) {
      return true;
    }
    if (left->getKind() > right->getKind()) {
      return false;
    }
    return left->getIdx() < right->getIdx();
  });
  // std::random_device rd;
  // std::mt19937 g(rd());
  // std::shuffle(free_registers_.begin(), free_registers_.end(), g);

  if (opts_.use_callee_saved_registers) {
    for (auto reg : callee_saved_registers_) {
      free_registers_.emplace_front(reg);
    }
  }
  std::string reg_strs;
  for (auto reg : free_registers_) {
    reg_strs.append(reg->toString()).append(",");
  }
  RUDF_INFO("Total [{}] free registers:{}", free_registers_.size(), reg_strs);
}

void CodeGenerator::SaveRegister(const Xbyak::Reg* reg) {
  stack_cursor_ += 8;
  auto addr = GetStackAddr(DATA_U64, stack_cursor_);
  if (reg->isXMM()) {
    jit_->movq(rax, reinterpret_cast<const Xbyak::Xmm&>(*reg));
    jit_->mov(addr, rax);
  } else {
    jit_->mov(addr, *reg);
  }
  saved_registers_.emplace(reg, stack_cursor_);
  RUDF_DEBUG("save register:{} with stack offset:{}", reg->toString(), stack_cursor_);
}
void CodeGenerator::LoadRegister(const Xbyak::Reg* reg) {
  auto found = saved_registers_.find(reg);
  if (found != saved_registers_.end()) {
    uint32_t offset = found->second;
    auto addr = GetStackAddr(DATA_U64, offset);
    if (reg->isXMM()) {
      jit_->mov(rcx, addr);
      jit_->movq(reinterpret_cast<const Xbyak::Xmm&>(*reg), rcx);
    } else {
      jit_->mov(*reg, addr);
    }
    saved_registers_.erase(found);
    RecycleStack(offset, 8);
  }
}

void CodeGenerator::SaveInuseRegisters() {
  for (auto iter = inuse_registers_.begin(); iter != inuse_registers_.end(); iter++) {
    auto reg = *iter;
    if (callee_saved_registers_.count(reg) == 0) {
      RUDF_DEBUG("Save caller saved register:{}", reg->toString());
      SaveRegister(reg);
    } else {
      RUDF_DEBUG("No need to save register:{}", reg->toString());
    }
  }
}
void CodeGenerator::RestoreInuseRegisters() {
  for (auto iter = inuse_registers_.rbegin(); iter != inuse_registers_.rend(); iter++) {
    auto reg = *iter;
    if (callee_saved_registers_.count(reg) == 0) {
      RUDF_DEBUG("Restore caller saved register:{}", reg->toString());
      LoadRegister(reg);
    } else {
      RUDF_DEBUG("No need to restore callee saved register:{}", reg->toString());
    }
  }
}
void CodeGenerator::RestoreCalleeSavedRegisters() {
  // for (auto reg : used_callee_saved_registers_) {
  //   RUDF_DEBUG("Restore callee saved register:{}", reg->toString());
  //   LoadRegister(reg);
  // }
  if (opts_.use_registers && opts_.use_callee_saved_registers) {
    for (auto reg : callee_saved_registers_) {
      LoadRegister(reg);
    }
  }
}

const Xbyak::Reg* CodeGenerator::AllocateRegister(const std::vector<RegisterId>& exclude_regs) {
  if (free_registers_.empty()) {
    return nullptr;
  }
  std::vector<const Xbyak::Reg*> allocated_not_used;
  const Xbyak::Reg* allocated_reg = nullptr;
  while (!free_registers_.empty()) {
    auto op = free_registers_.front();
    free_registers_.pop_front();
    if (!exclude_regs.empty()) {
      RegisterId id(*op);
      for (auto exclude_reg : exclude_regs) {
        if (id == exclude_reg) {
          allocated_not_used.emplace_back(op);
          op = nullptr;
          break;
        }
      }
    }
    if (nullptr != op) {
      allocated_reg = op;
      break;
    }
  }
  for (auto reg : allocated_not_used) {
    free_registers_.emplace_back(reg);
  }

  if (nullptr == allocated_reg) {
    return nullptr;
  }

  inuse_registers_.insert(allocated_reg);
  // if (callee_saved_registers_.count(allocated_reg->toString()) == 1) {
  //   if (used_callee_saved_registers_.insert(allocated_reg).second) {
  //     RUDF_DEBUG("Save callee saved register:{} when first allocated", allocated_reg->toString());
  //     // SaveRegister(allocated_reg);
  //   }
  // }
  RUDF_DEBUG("Allocate register:{}, {} free left.", allocated_reg->toString(), free_registers_.size());
  return allocated_reg;
}

void CodeGenerator::RecycleRegister(const Xbyak::Reg* reg) {
  if (reg != nullptr) {
    auto n = inuse_registers_.erase(reg);
    RUDF_DEBUG("Recycle reg:{}, inuse:{}", reg->toString(), n);
    if (n == 1) {
      free_registers_.emplace_back(reg);
    }
  }
}

void CodeGenerator::RecycleStack(uint32_t offset, uint32_t len) {
  recycled_stacks_[offset] = len;
  while (!recycled_stacks_.empty()) {
    auto last = *recycled_stacks_.rbegin();
    if (stack_cursor_ == last.first) {
      stack_cursor_ -= last.second;
      recycled_stacks_.erase(last.first);
      continue;
    } else {
      break;
    }
  }
}

void CodeGenerator::DropTmpValue(ValuePtr tmp) {
  if (!tmp || !tmp->IsTemp()) {
    return;
  }
  tmp->Drop();
}
void CodeGenerator::DropTmpValue(const std::vector<ValuePtr> tmps) {
  for (auto p : tmps) {
    DropTmpValue(p);
  }
}

std::pair<uint32_t, uint32_t> CodeGenerator::AllocateStack(uint32_t len) {
  uint32_t allign = 8;
  uint32_t allign_rest = stack_cursor_ % allign;
  uint32_t stack_offset_inc = 0;
  if (allign_rest != 0) {
    stack_offset_inc = allign_rest;
  }
  stack_offset_inc += len;
  stack_cursor_ += stack_offset_inc;
  return {stack_cursor_, stack_offset_inc};
}

ValuePtr CodeGenerator::AllocateValue(DType dtype, uint32_t len, const std::vector<RegisterId>& exlucde_regs,
                                      bool with_register, bool temp) {
  ValuePtr value;
  switch (len) {
    case 8:
    case 4:
    case 2:
    case 1: {
      const Xbyak::Reg* reg = with_register ? AllocateRegister(exlucde_regs) : nullptr;
      if (reg != nullptr) {
        value = Value::New(this, dtype, reg, temp);
        break;
      } else {
        auto [stack_offset, stack_len] = AllocateStack(len);
        value = Value::New(this, dtype, stack_offset, stack_len, temp);
        break;
      }
    }
    case 16: {
      const Xbyak::Reg* reg0 = with_register ? AllocateRegister(exlucde_regs) : nullptr;
      const Xbyak::Reg* reg1 = with_register ? AllocateRegister(exlucde_regs) : nullptr;
      if (reg0 != nullptr && reg1 != nullptr) {
        value = Value::New(this, dtype, {reg0, reg1}, temp);
        break;
      } else {
        RecycleRegister(reg0);
        RecycleRegister(reg1);
        auto [stack_offset, stack_len] = AllocateStack(len);
        value = Value::New(this, dtype, stack_offset, stack_len, temp);
        break;
      }
    }
    default: {
      RUDF_ERROR("Can NOT allocate value with len:{} for dtype:{}", len, dtype);
      return nullptr;
    }
  }
  RUDF_DEBUG("Allocate {} for value with dtype:{}, temp:{}", value->StorageInfo(), dtype, temp);
  return value;
}

std::vector<ValuePtr> CodeGenerator::NewArrayValue(DType dtype, size_t n) {
  size_t stack_len = dtype.Elem().ByteSize() * n;
  std::vector<ValuePtr> vals(n);
  auto [stack_offset, actual_stack_len] = AllocateStack(stack_len);
  for (size_t i = 0; i < n; i++) {
    auto value = Value::New(this, dtype.Elem(), stack_offset, dtype.Elem().ByteSize(), false);
    stack_offset -= dtype.Elem().ByteSize();
    vals[i] = value;
  }
  return vals;
}

ValuePtr CodeGenerator::NewValueByRegister(DType dtype, const Xbyak::Reg& reg) {
  auto value = Value::New(this, dtype, &reg);
  return value;
}
ValuePtr CodeGenerator::NewValueByRegister(DType dtype, const std::vector<const Xbyak::Reg*>& regs) {
  auto value = Value::New(this, dtype, regs, false);
  return value;
}
ValuePtr CodeGenerator::NewConstValue(DType dtype, uint64_t val) {
  auto value = Value::New(this, dtype, val);
  return value;
}

ValuePtr CodeGenerator::NewValue(DType dtype, const std::vector<RegisterId>& exlucde_regs, bool temp) {
  if (dtype.IsVoid()) {
    return NewConstValue(dtype, 0);
  } else {
    auto val = AllocateValue(dtype, dtype.ByteSize(), exlucde_regs, true, temp);

    return val;
  }
}

ValuePtr CodeGenerator::CallFunction(const FunctionDesc& desc, const std::vector<const Value*>& args) {
  SaveInuseRegisters();
  RUDF_DEBUG("Func:{} call args:{}, return {}", desc.name, args.size(), desc.return_type);
  if (desc.arg_types.size() != args.size()) {
    RUDF_ERROR("Func:{} mismatch args size {}/{}", desc.arg_types.size(), args.size());
    return {};
  }
  std::vector<DType> arg_dtypes;
  for (auto arg : args) {
    arg_dtypes.emplace_back(arg->GetDType());
  }
  std::vector<FuncArgRegister> arg_registers = GetFuncArgsRegistersByDTypes(arg_dtypes);
  if (arg_registers.size() != args.size()) {
    RUDF_ERROR("Can NOT allocate arg registers for func:{} with args size {}", args.size());
    return {};
  }

  if (stack_cursor_ > 0) {
    if (last_rsp_stack_cursor_ < stack_cursor_) {
      jit_->sub(rsp, allgin16(stack_cursor_ - last_rsp_stack_cursor_));
      last_rsp_stack_cursor_ = stack_cursor_;
    }
  }
  for (size_t i = 0; i < args.size(); i++) {
    // RUDF_DEBUG("Func:{} call need {} registers for arg:{}", desc.name, arg_registers[i].size(), i);
    auto arg_value = Value::New(this, args[i]->GetDType(), arg_registers[i], false);
    int rc = arg_value->Copy(*args[i]);
    if (0 != rc) {
      return {};
    }
  }

  jit_->mov(rax, (size_t)desc.func);
  jit_->call(rax);
  RestoreInuseRegisters();

  if (!desc.return_type.IsVoid()) {
    uint32_t total_bits = 0;
    auto return_value_regs = GetFuncReturnValueRegisters(desc.return_type, total_bits);
    uint32_t stack_bytes = (total_bits / 8);
    if (return_value_regs.empty()) {
      stack_cursor_ += stack_bytes;
      auto return_value = Value::New(this, desc.return_type, stack_cursor_, stack_bytes, true);
      return return_value;
    } else {
      auto register_val = Value::New(this, desc.return_type, return_value_regs, false);
      auto return_value = AllocateValue(desc.return_type, stack_bytes, {}, true, true);
      int rc = return_value->Copy(*register_val);
      if (rc != 0) {
        return {};
      }
      return return_value;
    }
  } else {
    return Value::New(this, DATA_VOID, static_cast<uint64_t>(0));
  }
}

ValuePtr CodeGenerator::CallFunction(const std::string& name, const std::vector<const Value*>& args) {
  const FunctionDesc* desc = FunctionFactory::GetFunction(name);
  if (nullptr == desc) {
    RUDF_ERROR("No func found for {}", name);
    return {};
  }
  return CallFunction(*desc, args);

  // return {};
}

void CodeGenerator::DumpAsm() {
  //   c.getCode(), c.getSize()
  std::string dest = "/tmp/lse_tmp.bin";
  FILE* f = fopen(dest.c_str(), "w");
  if (nullptr == f) {
    return;
  }
  fwrite(jit_->getCode(), jit_->getSize(), 1, f);
  fclose(f);
  std::string exec_cmd = "objdump -M x86-64 -D -b binary -m i386 " + dest;
  exec_cmd.append(" 2>&1");
  FILE* compile = popen(exec_cmd.c_str(), "r");
  char buf[1024];
  std::string err;
  while (fgets(buf, sizeof(buf), compile) != 0) {
    err.append(buf);
  }
  pclose(compile);
  if (!err.empty()) {
    printf("%s\n", err.c_str());
  }
}
Xbyak::Address CodeGenerator::GetStackAddr(DType dtype, uint32_t offset) {
  uint32_t len = dtype.ByteSize();
  switch (len) {
    case 16: {
      return xword[rbp - offset];
    }
    case 8: {
      return qword[rbp - offset];
    }
    case 4: {
      return dword[rbp - offset];
    }
    case 2: {
      return word[rbp - offset];
    }
    case 1: {
      return byte[rbp - offset];
    }
    default: {
      RUDF_ERROR("Unsupported len:{} to get address", len);
      abort();
    }
  }
}
}  // namespace xbyak
}  // namespace rapidudf