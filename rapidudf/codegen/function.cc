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
#include "rapidudf/codegen/function.h"
#include <array>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/log/log.h"
namespace rapidudf {
using namespace Xbyak::util;
using FuncRegMap = std::unordered_map<std::string, FuncDesc>;
static std::unique_ptr<FuncRegMap> g_regs = nullptr;

static inline uint32_t allgin_n(uint32_t x, uint32_t n) { return (x + n - 1) & ~(n - 1); }

std::vector<const Xbyak::Reg*> GetFuncReturnValueRegisters(DType return_type, uint32_t& total_bits) {
  std::vector<const Xbyak::Reg*> regs;
  uint32_t bits = 0;
  auto return_types = return_type.ExtractTupleDtypes();
  for (auto dtype : return_types) {
    bits = allgin_n(bits, dtype.Bits() >= 64 ? 64 : dtype.Bits());

    bits += dtype.Bits();
  }
  if (bits > 64) {
    bits = allgin_n(bits, 64);
  }
  total_bits = bits;
  RUDF_DEBUG("total {}bits for {}", total_bits, return_type);
  if (bits <= 128) {  // only use registers less_eq 128bits
    uint32_t register_used_bits = 0;
    bool previous_all_floats = true;
    std::deque<const Xbyak::Reg*> int_regs{&rax, &rdx};
    std::deque<const Xbyak::Reg*> float_regs{&xmm0, &xmm1};
    bits = 0;
    for (auto dtype : return_types) {
      bits = allgin_n(bits, dtype.Bits() >= 64 ? 64 : dtype.Bits());
      bits += dtype.Bits();
      register_used_bits += dtype.Bits();
      if (!dtype.IsFloat()) {
        previous_all_floats = false;
      }
      while (register_used_bits >= 64) {
        if (previous_all_floats) {
          regs.push_back(float_regs.front());
          float_regs.pop_front();
        } else {
          regs.push_back(int_regs.front());
          int_regs.pop_front();
        }
        register_used_bits -= 64;
      }
      if (register_used_bits == 0) {
        previous_all_floats = true;
      }
    }
    if (register_used_bits > 0) {
      if (previous_all_floats) {
        regs.push_back(float_regs.front());
        float_regs.pop_front();
      } else {
        regs.push_back(int_regs.front());
        int_regs.pop_front();
      }
    }
  }
  return regs;
}

std::vector<FuncArgRegister> GetFuncArgsRegistersByDTypes(const std::vector<DType>& arg_types) {
  std::vector<FuncArgRegister> args_storage;
  std::deque<const Xbyak::Reg*> int_regs{&rdi, &rsi, &rdx, &rcx, &r8, &r9};
  std::deque<const Xbyak::Reg*> float_regs{&xmm0, &xmm1, &xmm2, &xmm3, &xmm4, &xmm5, &xmm6, &xmm7};

  for (size_t i = 0; i < arg_types.size(); i++) {
    auto arg_dtype = arg_types[i];
    if (arg_dtype.IsFloat()) {
      if (!float_regs.empty()) {
        const Xbyak::Reg* reg = float_regs.front();
        float_regs.pop_front();
        args_storage.emplace_back(FuncArgRegister{reg});
        continue;
      } else {
        RUDF_ERROR("Too many func args:{}, while can NOT use registers.", arg_types.size());
        return {};
      }
    } else if (arg_dtype.IsNumber() || arg_dtype.IsPtr()) {
      if (!int_regs.empty()) {
        const Xbyak::Reg* reg = int_regs.front();
        int_regs.pop_front();
        args_storage.emplace_back(FuncArgRegister{reg});
        continue;
      } else {
        RUDF_ERROR("Too many func args:{}, while can NOT use registers.", arg_types.size());
        return {};
      }
    } else if (arg_dtype.IsAbslSpan() || arg_dtype.IsStringView()) {
      if (int_regs.size() >= 2) {
        auto* reg0 = int_regs.front();
        int_regs.pop_front();
        auto* reg1 = int_regs.front();
        int_regs.pop_front();
        args_storage.emplace_back(FuncArgRegister{reg0, reg1});
        continue;
      }
    } else {
      RUDF_ERROR("Unsupported dtype:{} as func arg at {}", arg_dtype, i);
      return {};
    }
  }
  return args_storage;
}
std::vector<const Xbyak::Reg*> GetUnuseFuncArgsRegisters(const std::vector<FuncArgRegister>& used_regs) {
  // exclude rdx/rcx xmm0/xmm1
  std::set<const Xbyak::Reg*> all_arg_registers{&rdi,  &rsi,  &r8,   &r9,   &r10,  &r11,
                                                &xmm2, &xmm3, &xmm4, &xmm5, &xmm6, &xmm7};
  for (auto& arg_regs : used_regs) {
    for (auto reg : arg_regs) {
      all_arg_registers.erase(reg);
    }
  }
  std::vector<const Xbyak::Reg*> unused_regs;
  for (auto reg : all_arg_registers) {
    unused_regs.emplace_back(reg);
  }
  RUDF_INFO("Total {} unused func arg registers.", unused_regs.size());
  return unused_regs;
}

bool FuncDesc::ValidateArgs(const std::vector<DType>& ts) const {
  if (arg_types.size() != ts.size()) {
    return false;
  }
  for (size_t i = 0; i < ts.size(); i++) {
    if (!ts[i].CanCastTo(arg_types[i])) {
      return false;
    }
  }
  return true;
}

std::vector<const Xbyak::Reg*> FuncDesc::GetReturnValueRegisters(uint32_t& total_bits) const {
  return GetFuncReturnValueRegisters(return_type, total_bits);
}

std::vector<FuncArgRegister> FuncDesc::GetArgsRegisters() const { return GetFuncArgsRegistersByDTypes(arg_types); }

void FuncFactory::Register(FuncDesc&& desc) {
  if (!g_regs) {
    g_regs = std::make_unique<FuncRegMap>();
  }
  if (g_regs->count(desc.name) > 0) {
    RUDF_CRITICAL("Duplicate func name:{}", desc.name);
    // abort();
  }
  RUDF_DEBUG("Registe func name:{}", desc.name);
  g_regs->emplace(desc.name, desc);
}
const FuncDesc* FuncFactory::GetFunc(const std::string& name) {
  if (!g_regs) {
    return nullptr;
  }
  auto found = g_regs->find(name);
  if (found == g_regs->end()) {
    return nullptr;
  }
  return &(found->second);
}

}  // namespace rapidudf