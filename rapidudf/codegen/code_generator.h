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
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "xbyak/xbyak.h"

#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/register.h"
#include "rapidudf/codegen/value.h"

namespace rapidudf {

class Value;

class CodeGenerator {
 public:
  explicit CodeGenerator(size_t max_size = 4096, bool use_register = false);
  ~CodeGenerator();

  void AddFreeRegisters(const std::vector<const Xbyak::Reg*>& regs, bool head = true);
  std::vector<ValuePtr> NewArrayValue(DType dtype, size_t n);
  ValuePtr NewValue(DType dtype, const std::vector<RegisterId>& exlucde_regs = {}, bool temp = true);
  ValuePtr NewValueByRegister(DType dtype, const Xbyak::Reg& reg);
  ValuePtr NewValueByRegister(DType dtype, const std::vector<const Xbyak::Reg*>& regs);
  ValuePtr NewConstValue(DType dtype, uint64_t val = 0);
  void DropTmpValue(ValuePtr tmp);
  void DropTmpValue(const std::vector<ValuePtr> tmps);

  int Label(const std::string& label);
  int Jump(const std::string& label);
  int Jump(const std::string& label, OpToken cmp);

  ValuePtr CallFunction(const std::string& name, const std::vector<const Value*>& args);
  ValuePtr CallFunction(const std::string& name, const std::vector<ValuePtr>& args) {
    std::vector<const Value*> arg_vals;
    for (auto& p : args) {
      arg_vals.emplace_back(p.get());
    }
    return CallFunction(name, arg_vals);
  }
  ValuePtr CallFunction(const FunctionDesc& f, const std::vector<const Value*>& args);
  ValuePtr CallFunction(const FunctionDesc& f, const std::vector<ValuePtr>& args) {
    std::vector<const Value*> arg_vals;
    for (auto& p : args) {
      arg_vals.emplace_back(p.get());
    }
    return CallFunction(f, arg_vals);
  }
  int ReturnValue(ValuePtr val);

  int Finish();

  template <typename RET, typename... Args>
  auto GetFunc() {
    return jit_->getCode<RET (*)(Args...)>();
  }

  const uint8_t* GetRawCodePtr() const { return jit_->getCode(); }

  void DumpAsm();

  Xbyak::CodeGenerator& GetCodeGen() { return *jit_; }
  Xbyak::Address GetStackAddr(DType dtype, uint32_t offset);

 private:
  std::pair<uint32_t, uint32_t> AllocateStack(uint32_t len);
  ValuePtr AllocateValue(DType dtype, uint32_t len, const std::vector<RegisterId>& exlucde_regs, bool with_register,
                         bool temp);

  void RecycleStack(uint32_t offset, uint32_t len);

  void SaveRegister(const Xbyak::Reg* reg);
  void LoadRegister(const Xbyak::Reg* reg);

  void InitFreeRegisters();
  void SaveInuseRegisters();
  void RestoreInuseRegisters();
  void RestoreCalleeSavedRegisters();
  const Xbyak::Reg* AllocateRegister(const std::vector<RegisterId>& exlucde_regs);
  void RecycleRegister(const Xbyak::Reg* reg);

  std::unique_ptr<Xbyak::CodeGenerator> jit_;

  uint32_t stack_cursor_ = 0;
  uint32_t last_rsp_stack_cursor_ = 0;

  std::map<uint32_t, uint32_t> recycled_stacks_;

  std::deque<const Xbyak::Reg*> free_registers_;
  std::set<const Xbyak::Reg*> inuse_registers_;

  std::set<const Xbyak::Reg*> callee_saved_registers_;
  std::set<const Xbyak::Reg*> used_callee_saved_registers_;

  std::unordered_map<const Xbyak::Reg*, uint32_t> saved_registers_;
  bool use_register_ = true;

  friend class Value;
};

}  // namespace rapidudf