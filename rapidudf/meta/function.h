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
#include <cstdint>
#include <functional>
#include "rapidudf/log/log.h"

namespace rapidudf {

struct FunctionCallContext {
  jmp_buf jmp_env;
  std::exception run_ex;
  uint32_t invoke_frame_id = 0;
  static FunctionCallContext& Get(bool start) {
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
constexpr uint64_t fnv1a_hash(std::string_view str) { return fnv1a_hash(str.data()); }

template <uint64_t, uint32_t, uint64_t, typename F>
struct MemberFunctionWrapper;

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct MemberFunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...)> {
  using return_type = R;
  using func_t = R (T::*)(Args...);
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static R Call(T* p, Args... args) {
    auto func = GetFunc();
    if constexpr (std::is_same_v<void, R>) {
      (p->*func)(std::forward<Args>(args)...);
    } else {
      return (p->*func)(std::forward<Args>(args)...);
    }
  }
};
template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct MemberFunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...) const> {
  using return_type = R;
  using func_t = R (T::*)(Args...) const;
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static R Call(const T* p, Args... args) {
    auto func = GetFunc();
    if constexpr (std::is_same_v<void, R>) {
      (p->*func)(std::forward<Args>(args)...);
    } else {
      return (p->*func)(std::forward<Args>(args)...);
    }
  }
};

template <uint64_t, uint32_t, uint64_t, typename F>
struct SafeFunctionWrapper;

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename R, typename... Args>
struct SafeFunctionWrapper<SOURCE, LINE, HASH, R(Args...)> {
  using return_type = R;
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
    auto func = GetFunc();
    try {
      if constexpr (std::is_same_v<void, R>) {
        func(std::forward<Args>(args)...);
      } else {
        return func(std::forward<Args>(args)...);
      }
    } catch (...) {
      auto& func_ctx = FunctionCallContext::Get(false);
      try {
        throw;
      } catch (const std::exception& e) {
        RUDF_ERROR("func:{} invoke exception type:[{}], msg:{}", GetFuncName(), typeid(e).name(), e.what());
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

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename R, typename... Args>
struct SafeFunctionWrapper<SOURCE, LINE, HASH, R (*)(Args...)> {
  using return_type = R;
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
    auto func = GetFunc();
    try {
      if constexpr (std::is_same_v<void, R>) {
        func(std::forward<Args>(args)...);
      } else {
        return func(std::forward<Args>(args)...);
      }
    } catch (...) {
      auto& func_ctx = FunctionCallContext::Get(false);
      try {
        throw;
      } catch (const std::exception& e) {
        RUDF_ERROR("func:{} invoke exception type:[{}], msg:{}", GetFuncName(), typeid(e).name(), e.what());
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

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct SafeFunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
  using func_t = R (T::*)(Args...);
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static R SafeCall(T* p, Args... args) {
    try {
      auto func = GetFunc();
      if constexpr (std::is_same_v<void, R>) {
        (p->*func)(std::forward<Args>(args)...);
      } else {
        return (p->*func)(std::forward<Args>(args)...);
      }
    } catch (...) {
      auto& func_ctx = FunctionCallContext::Get(false);
      try {
        throw;
      } catch (const std::exception& e) {
        RUDF_ERROR("func:{} invoke exception type:[{}], msg:{}", GetFuncName(), typeid(e).name(), e.what());
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

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct SafeFunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...) const> {
  using return_type = R;
  using func_t = R (T::*)(Args...) const;
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static R SafeCall(const T* p, Args... args) {
    try {
      auto func = GetFunc();
      if constexpr (std::is_same_v<void, R>) {
        (p->*func)(std::forward<Args>(args)...);
      } else {
        return (p->*func)(std::forward<Args>(args)...);
      }
    } catch (...) {
      auto& func_ctx = FunctionCallContext::Get(false);
      try {
        throw;
      } catch (const std::exception& e) {
        RUDF_ERROR("func:{} invoke exception type:[{}], msg:{}", GetFuncName(), typeid(e).name(), e.what());
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
}  // namespace rapidudf

#define MEMBER_FUNC_WRAPPER(name, func)                                                            \
  do {                                                                                             \
    using wrapper_t = rapidudf::MemberFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                      rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                   \
    wrapper_t::GetFuncName() = name;                                                               \
    rapidudf::ReflectFactory::AddStructMethodAccessor(name, &wrapper_t::Call);                     \
  } while (0)

#define SAFE_MEMBER_FUNC_WRAPPER(name, func)                                                     \
  do {                                                                                           \
    using wrapper_t = rapidudf::SafeFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                    rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                 \
    wrapper_t::GetFuncName() = name;                                                             \
    rapidudf::ReflectFactory::AddStructMethodAccessor(name, &wrapper_t::SafeCall);               \
  } while (0)

#define SAFE_FUNC_WRAPPER(name, func)                                                            \
  do {                                                                                           \
    using wrapper_t = rapidudf::SafeFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                    rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                 \
    wrapper_t::GetFuncName() = name;                                                             \
    func = wrapper_t::SafeCall;                                                                  \
  } while (0)
