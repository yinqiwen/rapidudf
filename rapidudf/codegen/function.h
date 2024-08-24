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
#include "rapidudf/log/log.h"

#include "xbyak/xbyak.h"

namespace rapidudf {

using FuncArgRegister = std::vector<const Xbyak::Reg*>;

std::vector<const Xbyak::Reg*> GetFuncReturnValueRegisters(DType dtype, uint32_t& total_bits);
std::vector<FuncArgRegister> GetFuncArgsRegistersByDTypes(const std::vector<DType>& arg_types);
std::vector<const Xbyak::Reg*> GetUnuseFuncArgsRegisters(const std::vector<FuncArgRegister>& used_regs);

struct FuncDesc {
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

struct FunctionCallContext {
  jmp_buf jmp_env;
  std::exception run_ex;
  uint32_t invoke_frame_id = 0;
};

class FuncFactory {
 public:
  static void Register(FuncDesc&& desc);
  static const FuncDesc* GetFunc(const std::string& name);
  static FunctionCallContext& GetFunctionCallContext(bool start) {
    static thread_local FunctionCallContext ctx;
    if (start) {
      ctx.invoke_frame_id++;
    }
    return ctx;
  }
};

constexpr uint64_t fnv1a_hash(const char* str) {
  uint64_t hash = 14695981039346656037ULL;
  while (*str) {
    hash = (hash ^ static_cast<uint64_t>(*str)) * 1099511628211ULL;
    ++str;
  }
  return hash;
}

template <typename F>
struct FunctionTraits;

template <typename R, typename... Args>
struct FunctionTraits<R (*)(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename R, typename... Args>
struct FunctionTraits<R(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename R, typename... Args>
struct FunctionTraits<std::function<R(Args...)>> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename C, typename R, typename... Args>
struct FunctionTraits<R (C::*)(Args...)> {
  using object_type = C;
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename C, typename R, typename... Args>
struct FunctionTraits<R (C::*)(Args...) const> {
  using object_type = C;
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename F, std::size_t N>
using function_argument_type_t = typename std::tuple_element<N, typename FunctionTraits<F>::arguments>::type;

template <typename F>
using first_function_argument_type_t = function_argument_type_t<F, 0>;

template <uint64_t, uint32_t, typename F>
struct SafeFunctionWrapper;

template <uint64_t H, uint32_t N, typename R, typename... Args>
struct SafeFunctionWrapper<H, N, R(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
  using func_t = R (*)(Args...);
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static R SafeCall(Args... args) {
    try {
      if constexpr (std::is_same_v<void, R>) {
        GetFunc()(args...);
      } else {
        return GetFunc()(args...);
      }
    } catch (...) {
      auto& func_ctx = FuncFactory::GetFunctionCallContext(false);
      try {
        throw;
      } catch (const std::exception& e) {
        RUDF_ERROR("func:{} invoke exception type:[{}], msg:{}", GetFuncName(), DType::Demangle(typeid(e).name()),
                   e.what());
        func_ctx.run_ex = e;
      } catch (...) {
        RUDF_ERROR("func:{} invoke unknown eception cpature!", GetFuncName());
      }
      if (func_ctx.invoke_frame_id > 0) {
        longjmp(func_ctx.jmp_env, 1);  // Jump out of deep nested calls
      }
      if constexpr (std::is_same_v<void, R>) {
        // nothing
      } else {
        return {};
      }
    }
  }
};

template <uint64_t, uint32_t, typename F>
class SafeFuncRegister;
template <uint64_t hash, uint32_t line, typename R, typename... Args>
class SafeFuncRegister<hash, line, R(Args...)> {
 public:
  SafeFuncRegister(std::string_view name, R (*f)(Args...)) {
    FuncDesc desc;
    desc.name = std::string(name);
    SafeFunctionWrapper<hash, line, R(Args...)>::GetFuncName() = name;
    SafeFunctionWrapper<hash, line, R(Args...)>::GetFunc() = f;
    desc.func = reinterpret_cast<void*>(&SafeFunctionWrapper<hash, line, R(Args...)>::SafeCall);
    desc.return_type = get_dtype<R>();
    (desc.arg_types.emplace_back(get_dtype<Args>()), ...);
    FuncFactory::Register(std::move(desc));
  }
};

class FuncRegister {
 public:
  template <typename RET, typename... Args>
  FuncRegister(std::string_view name, RET (*f)(Args...), bool safe = true) {
    FuncDesc desc;
    desc.name = std::string(name);
    desc.func = reinterpret_cast<void*>(f);
    desc.return_type = get_dtype<RET>();
    (desc.arg_types.emplace_back(get_dtype<Args>()), ...);
    FuncFactory::Register(std::move(desc));
  }
};

}  // namespace rapidudf

#define RUDF_FUNC_REGISTER(f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(f), f);

#define RUDF_SAFE_FUNC_REGISTER(f)                                                                          \
  static ::rapidudf::SafeFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(f)), __LINE__, decltype(f)> \
      BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(f), f);

#define RUDF_FUNC_REGISTER_WITH_NAME(NAME, f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);

#define RUDF_SAFE_FUNC_REGISTER_WITH_NAME(NAME, f)                                                             \
  static ::rapidudf::SafeFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(NAME)), __LINE__, decltype(f)> \
      BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(NAME), f);
