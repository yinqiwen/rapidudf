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
#include <cstdint>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {

constexpr uint64_t fnv1a_hash(const char* str) {
  uint64_t hash = 14695981039346656037ULL;
  while (*str) {
    hash = (hash ^ static_cast<uint64_t>(*str)) * 1099511628211ULL;
    ++str;
  }
  return hash;
}
constexpr uint64_t fnv1a_hash(std::string_view str) { return fnv1a_hash(str.data()); }

struct FunctionDesc {
  std::string name;
  // return types
  DType return_type;
  // args types
  std::vector<DType> arg_types;
  void* func = nullptr;
  int context_arg_idx = -1;

  void Init();
  bool ValidateArgs(const std::vector<DType>& ts) const;
  bool CompareSignature(DType rtype, const std::vector<DType>& args_types) const;
};

class FunctionFactory {
 public:
  static constexpr std::string_view kSimdVectorUnaryFuncPrefix = "simd_vector_unary";
  static constexpr std::string_view kSimdVectorBinaryFuncPrefix = "simd_vector_binary";
  static constexpr std::string_view kSimdVectorTernaryFuncPrefix = "simd_vector_ternary";
  static constexpr std::string_view kSimdVectorFuncPrefix = "simd_vector";
  template <typename RET, typename... Args>
  bool Register(std::string_view name, RET (*f)(Args...)) {
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

template <typename SAFE_WRAPPER = void>
class FuncRegister {
 public:
  template <typename RET, typename... Args>
  FuncRegister(std::string_view name, RET (*f)(Args...)) {
    FunctionDesc desc;
    desc.name = std::string(name);
    if constexpr (std::is_void_v<SAFE_WRAPPER>) {
      desc.func = reinterpret_cast<void*>(f);
    } else {
      SAFE_WRAPPER::GetFunc() = f;
      SAFE_WRAPPER::GetFuncName() = std::string(name);
      desc.func = reinterpret_cast<void*>(SAFE_WRAPPER::SafeCall);
    }
    desc.return_type = get_dtype<RET>();

    (desc.arg_types.emplace_back(get_dtype<Args>()), ...);
    FunctionFactory::Register(std::move(desc));
  }
};
std::string GetMemberFuncName(DType dtype, const std::string& member);
std::string GetFunctionName(OpToken op, DType dtype);
std::string GetFunctionName(OpToken op, DType left_dtype, DType right_dtype);
std::string GetFunctionName(OpToken op, DType a, DType b, DType c);
std::string GetFunctionName(std::string_view op, DType dtype);
std::string GetFunctionName(std::string_view op, DType left_dtype, DType right_dtype);
std::string GetFunctionName(std::string_view op, DType a, DType b, DType c);
std::string GetFunctionName(std::string_view op, const std::vector<DType>& arg_dtypes);

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
    if (nullptr == p) {
      throw std::logic_error(fmt::format("NULL object pointer to call member func:{}", GetFuncName()));
    }
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
    if (nullptr == p) {
      throw std::logic_error(fmt::format("NULL object pointer to call member func:{}", GetFuncName()));
    }
    auto func = GetFunc();
    if constexpr (std::is_same_v<void, R>) {
      (p->*func)(std::forward<Args>(args)...);
    } else {
      return (p->*func)(std::forward<Args>(args)...);
    }
  }
};

template <uint64_t, uint32_t, uint64_t, typename F>
struct FunctionWrapper;

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename R, typename... Args>
struct FunctionWrapper<SOURCE, LINE, HASH, R(Args...)> {
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
    if constexpr (std::is_same_v<void, R>) {
      func(std::forward<Args>(args)...);
    } else {
      return func(std::forward<Args>(args)...);
    }
  }
};

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename R, typename... Args>
struct FunctionWrapper<SOURCE, LINE, HASH, R (*)(Args...)> {
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
    if constexpr (std::is_same_v<void, R>) {
      func(std::forward<Args>(args)...);
    } else {
      return func(std::forward<Args>(args)...);
    }
  }
};

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct FunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...)> {
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
    if (nullptr == p) {
      throw std::logic_error(fmt::format("NULL object pointer to call member func:{}", GetFuncName()));
    }
    auto func = GetFunc();
    if constexpr (std::is_same_v<void, R>) {
      (p->*func)(std::forward<Args>(args)...);
    } else {
      return (p->*func)(std::forward<Args>(args)...);
    }
  }
};

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T, typename R, typename... Args>
struct FunctionWrapper<SOURCE, LINE, HASH, R (T::*)(Args...) const> {
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
    if (nullptr == p) {
      throw std::logic_error(fmt::format("NULL object pointer to call member func:{}", GetFuncName()));
    }
    auto func = GetFunc();
    if constexpr (std::is_same_v<void, R>) {
      (p->*func)(std::forward<Args>(args)...);
    } else {
      return (p->*func)(std::forward<Args>(args)...);
    }
  }
};
}  // namespace rapidudf

#define MEMBER_FUNC_WRAPPER(name, func)                                                                     \
  do {                                                                                                      \
    using wrapper_t = typename rapidudf::MemberFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                               rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                            \
    wrapper_t::GetFuncName() = name;                                                                        \
    rapidudf::Reflect::AddStructMethodAccessor(name, &wrapper_t::Call);                                     \
  } while (0)

#define SAFE_MEMBER_FUNC_WRAPPER(name, func)                                                              \
  do {                                                                                                    \
    using wrapper_t = typename rapidudf::SafeFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                             rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                          \
    wrapper_t::GetFuncName() = name;                                                                      \
    rapidudf::Reflect::AddStructMethodAccessor(name, &wrapper_t::SafeCall);                               \
  } while (0)

#define SAFE_FUNC_WRAPPER(name, func)                                                                     \
  do {                                                                                                    \
    using wrapper_t = typename rapidudf::SafeFunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__,    \
                                                             rapidudf::fnv1a_hash(name), decltype(func)>; \
    wrapper_t::GetFunc() = func;                                                                          \
    wrapper_t::GetFuncName() = name;                                                                      \
    func = wrapper_t::SafeCall;                                                                           \
  } while (0)

#define RUDF_FUNC_REGISTER(f) \
  static ::rapidudf::FuncRegister<void> BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(f), f);

#define RUDF_FUNC_REGISTER_WITH_NAME(NAME, f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);

#define RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(hash, NAME, f)                                  \
  static ::rapidudf::FuncRegister<                                                            \
      rapidudf::FunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__, hash, decltype(f)>> \
      BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);
