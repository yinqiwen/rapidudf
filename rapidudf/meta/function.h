/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cstdint>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "rapidudf/log/log.h"
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
  bool is_vector_func = false;

  void Init();
  bool ValidateArgs(const std::vector<DType>& ts) const;
  bool CompareSignature(DType rtype, const std::vector<DType>& args_types) const;

  bool PassArgByValue(size_t i) const;
  const DType& LastArg() const;
  uint32_t GetOperandCount() const;
};

template <typename T>
DType get_function_arg_dtype() {
  DType dtype = get_dtype<T>();
  return dtype;
}

class FunctionFactory {
 public:
  static constexpr std::string_view kSimdVectorFuncPrefix = "simd_vector";
  template <typename RET, typename... Args>
  bool Register(std::string_view name, RET (*f)(Args...)) {
    FunctionDesc desc;
    desc.name = std::string(name);
    desc.func = reinterpret_cast<void*>(f);
    desc.return_type = get_dtype<RET>();
    (desc.arg_types.emplace_back(get_function_arg_dtype<Args>()), ...);
    return Register(std::move(desc));
  }

  static bool Register(FunctionDesc&& desc);
  static const FunctionDesc* GetFunction(const std::string& name);

  template <typename... Args>
  static absl::Status RegisterVectorFunction(std::string_view name, void (*f)(Args...)) {
    FunctionDesc desc;
    desc.name = std::string(name);
    desc.func = reinterpret_cast<void*>(f);
    desc.return_type = DType(DATA_VOID);
    (desc.arg_types.emplace_back(get_function_arg_dtype<Args>()), ...);
    return RegisterVectorFunction(std::move(desc));
  }
  static absl::Status RegisterVectorFunction(FunctionDesc&& desc);
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

    (desc.arg_types.emplace_back(get_function_arg_dtype<Args>()), ...);
    FunctionFactory::Register(std::move(desc));
  }
};

class VectorFuncRegister {
 public:
  template <typename... Args>
  VectorFuncRegister(std::string_view name, void (*f)(Args...)) {
    auto status = FunctionFactory::RegisterVectorFunction(name, f);
    if (!status.ok()) {
      RUDF_CRITICAL("Invalid func:{} with reason:{}", name, status.ToString());
    }
  }
};

std::string GetMemberFuncName(DType dtype, const std::string& member);
std::string GetFunctionName(OpToken op, DType dtype);
std::string GetFunctionName(OpToken op, DType dtype0, DType dtype1);
std::string GetFunctionName(std::string_view op, DType dtype);

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

#define RUDF_VECTOR_FUNC_REGISTER(f) \
  static ::rapidudf::VectorFuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(f), f);

#define RUDF_FUNC_REGISTER_WITH_NAME(NAME, f) \
  static ::rapidudf::FuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);

#define RUDF_VECTOR_FUNC_REGISTER_WITH_NAME(NAME, f) \
  static ::rapidudf::VectorFuncRegister BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);

#define RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(hash, NAME, f)                                  \
  static ::rapidudf::FuncRegister<                                                            \
      rapidudf::FunctionWrapper<rapidudf::fnv1a_hash(__FILE__), __LINE__, hash, decltype(f)>> \
      BOOST_PP_CAT(rudf_reg_funcs_, __COUNTER__)(NAME, f);
