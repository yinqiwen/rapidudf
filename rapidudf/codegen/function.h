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
#include <setjmp.h>
#include <array>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <exception>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"

#include "xbyak/xbyak.h"

namespace rapidudf {

using FuncArgRegister = std::vector<const Xbyak::Reg*>;

std::vector<const Xbyak::Reg*> GetFuncReturnValueRegisters(DType dtype, uint32_t& total_bits);
std::vector<FuncArgRegister> GetFuncArgsRegistersByDTypes(const std::vector<DType>& arg_types);
std::vector<const Xbyak::Reg*> GetUnuseFuncArgsRegisters(const std::vector<FuncArgRegister>& used_regs);

struct FunctionDesc {
  std::string name;
  // return types
  DType return_type;
  // args types
  std::vector<DType> arg_types;
  void* func = nullptr;

  bool ValidateArgs(const std::vector<DType>& ts) const;
  std::vector<const Xbyak::Reg*> GetReturnValueRegisters(uint32_t& total_bits) const;
  std::vector<FuncArgRegister> GetArgsRegisters() const;
  // std::vector<std::variant<const Xbyak::Reg*, uint32_t>> GetArgsRegisters() const;
};

class FunctionFactory {
 public:
  template <typename RET, typename... Args>
  bool Register(std::string_view name, RET (*f)(Args...), bool safe = true) {
    FunctionDesc desc;
    desc.name = std::string(name);
    desc.func = reinterpret_cast<void*>(f);
    desc.return_type = get_dtype<RET>();
    (desc.arg_types.emplace_back(get_dtype<Args>()), ...);
    return Register(std::move(desc));
  }

  static bool Register(FunctionDesc&& desc);
  static const FunctionDesc* GetFunction(const std::string& name);
};

class FuncRegister {
 public:
  FuncRegister(std::function<void()>&& f) { f(); }

  template <typename RET, typename... Args>
  FuncRegister(std::string_view name, RET (*f)(Args...)) {
    FunctionDesc desc;
    desc.name = std::string(name);

    desc.func = reinterpret_cast<void*>(f);
    desc.return_type = get_dtype<RET>();
    (desc.arg_types.emplace_back(get_dtype<Args>()), ...);
    FunctionFactory::Register(std::move(desc));
  }
};

std::string GetFunctionName(std::string_view prefix, OpToken op, DType dtype);

}  // namespace rapidudf

#define RUDF_FUNC_REGISTER(f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(f), f);

#define RUDF_SAFE_FUNC_REGISTER(f)                                                  \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)([]() { \
    auto ff = f;                                                                    \
    SAFE_FUNC_WRAPPER(BOOST_PP_STRINGIZE(f), ff);                                   \
    ::rapidudf::FuncRegister reg(BOOST_PP_STRINGIZE(f), ff);                        \
  });

#define RUDF_FUNC_REGISTER_WITH_NAME(NAME, f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);

#define RUDF_SAFE_FUNC_REGISTER_WITH_NAME(NAME, f)                                  \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)([]() { \
    auto ff = f;                                                                    \
    SAFE_FUNC_WRAPPER(NAME, ff);                                                    \
    ::rapidudf::FuncRegister reg(NAME, ff);                                         \
  });
