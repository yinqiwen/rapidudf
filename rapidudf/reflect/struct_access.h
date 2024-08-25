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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "rapidudf/codegen/code_generator.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/value.h"
#include "rapidudf/reflect/reflect.h"
#include "rapidudf/reflect/reflect_value.h"
#include "xbyak/xbyak.h"

namespace rapidudf {

template <typename T>
class StructFieldAccess {
 public:
  static std::optional<StructMember> GetStructMember(const std::string& name) {
    return ReflectFactory::GetStructMember(get_dtype<T>(), name);
  }
};

template <typename T>
class StructMethodAccess {
 public:
  static std::optional<StructMember> GetStructMember(DType dtype, const std::string& name) {
    return ReflectFactory::GetStructMember(get_dtype<T>(), name);
  }
};

template <typename T>
class StructFieldAccessRegister {
 public:
  StructFieldAccessRegister() { StructFieldAccess<T>::Init(); }
};
template <typename T>
class StructMethodAccessRegister {
 public:
  StructMethodAccessRegister() { StructMethodAccess<T>::Init(); }
};

template <typename T>
class StructAccessHelperRegister {
 public:
  StructAccessHelperRegister(std::function<void()>&& f) { f(); }
};
template <uint64_t, uint32_t, typename T, typename F>
class MemberFuncRegister;
template <uint64_t hash, uint32_t line, typename T, typename R, typename... Args>
class MemberFuncRegister<hash, line, T, R (T::*)(Args...)> {
 public:
  MemberFuncRegister(std::string_view name, R (T::*f)(Args...)) {
    MemberFunctionWrapper<hash, line, R (T::*)(Args...)>::GetFuncName() = name;
    MemberFunctionWrapper<hash, line, R (T::*)(Args...)>::GetFunc() = f;
    ReflectFactory::AddStructMethodAccessor(std::string(name),
                                            &MemberFunctionWrapper<hash, line, R (T::*)(Args...)>::Call);
    DTypeFactory::Add<T>();
  }
};
template <uint64_t hash, uint32_t line, typename T, typename R, typename... Args>
class MemberFuncRegister<hash, line, T, R (T::*)(Args...) const> {
 public:
  MemberFuncRegister(std::string_view name, R (T::*f)(Args...) const) {
    MemberFunctionWrapper<hash, line, R (T::*)(Args...) const>::GetFuncName() = name;
    MemberFunctionWrapper<hash, line, R (T::*)(Args...) const>::GetFunc() = f;
    ReflectFactory::AddStructMethodAccessor(std::string(name),
                                            &MemberFunctionWrapper<hash, line, R (T::*)(Args...) const>::Call);
    DTypeFactory::Add<T>();
  }
};

template <uint64_t, uint32_t, typename T, typename F>
class SafeMemberFuncRegister;
template <uint64_t hash, uint32_t line, typename T, typename R, typename... Args>
class SafeMemberFuncRegister<hash, line, T, R (T::*)(Args...)> {
 public:
  SafeMemberFuncRegister(std::string_view name, R (T::*f)(Args...)) {
    SafeFunctionWrapper<hash, line, R (T::*)(Args...)>::GetFuncName() = name;
    SafeFunctionWrapper<hash, line, R (T::*)(Args...)>::GetFunc() = f;
    ReflectFactory::AddStructMethodAccessor(std::string(name),
                                            &SafeFunctionWrapper<hash, line, R (T::*)(Args...)>::SafeCall);
    DTypeFactory::Add<T>();
  }
};
template <uint64_t hash, uint32_t line, typename T, typename R, typename... Args>
class SafeMemberFuncRegister<hash, line, T, R (T::*)(Args...) const> {
 public:
  SafeMemberFuncRegister(std::string_view name, R (T::*f)(Args...) const) {
    SafeFunctionWrapper<hash, line, R (T::*)(Args...) const>::GetFuncName() = name;
    SafeFunctionWrapper<hash, line, R (T::*)(Args...) const>::GetFunc() = f;
    ReflectFactory::AddStructMethodAccessor(std::string(name),
                                            &SafeFunctionWrapper<hash, line, R (T::*)(Args...) const>::SafeCall);
    DTypeFactory::Add<T>();
  }
};

template <uint64_t hash, uint32_t line, typename T, typename R, typename... Args>
class SafeMemberFuncRegister<hash, line, T, R(T*, Args...)> {
 public:
  SafeMemberFuncRegister(std::string_view name, R (*f)(T*, Args...)) {
    SafeFunctionWrapper<hash, line, R(T*, Args...)>::GetFuncName() = name;
    SafeFunctionWrapper<hash, line, R(T*, Args...)>::GetFunc() = f;
    ReflectFactory::AddStructMethodAccessor(std::string(name),
                                            &SafeFunctionWrapper<hash, line, R(T*, Args...)>::SafeCall);
    DTypeFactory::Add<T>();
  }
};

}  // namespace rapidudf

#define RUDF_STRUCT_FILED_OFFSETOF(TYPE, ELEMENT) ((size_t) & (((TYPE*)0)->ELEMENT))
#define RUDF_STRUCT_ADD_FIELD_ACCESS_CODE(r, TYPE, i, member)                                                          \
  ::rapidudf::ReflectFactory::AddStructField(::rapidudf::get_dtype<TYPE>(),                                            \
                                             BOOST_PP_STRINGIZE(member),                                               \
                                                                ::rapidudf::get_dtype<decltype(((TYPE*)0)->member)>(), \
                                                                RUDF_STRUCT_FILED_OFFSETOF(TYPE, member));

#define RUDF_STRUCT_FIELDS(st, ...)                                                                         \
  namespace rapidudf {                                                                                      \
  template <>                                                                                               \
  class StructFieldAccess<st> {                                                                             \
   private:                                                                                                 \
   public:                                                                                                  \
    static void Init() {                                                                                    \
      static bool inited = false;                                                                           \
      if (inited) {                                                                                         \
        return;                                                                                             \
      }                                                                                                     \
      BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_FIELD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
      ::rapidudf::DTypeFactory::Add<st>();                                                                  \
      inited = true;                                                                                        \
    }                                                                                                       \
    static std::optional<rapidudf::StructMember> GetStructMember(const std::string& name) {                 \
      Init();                                                                                               \
      return rapidudf::ReflectFactory::GetStructMember(rapidudf::get_dtype<st>(), name);                    \
    }                                                                                                       \
  };                                                                                                        \
  static ::rapidudf::StructFieldAccessRegister<st> BOOST_PP_CAT(rudf_struct_access_, __COUNTER__);          \
  }

#define RUDF_STRUCT_ADD_SAFE_C_METHOD_ACCESS_CODE(r, TYPE, i, member)                                        \
  {                                                                                                          \
    using func_type = decltype(TYPE::member);                                                                \
    using first_arg_type = std::remove_pointer_t<::rapidudf::first_function_argument_type_t<func_type>>;     \
    static ::rapidudf::SafeMemberFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(member)), __LINE__,  \
                                                                     first_arg_type, decltype(TYPE::member)> \
        BOOST_PP_CAT(rudf_safe_member_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(member), &TYPE::member);       \
  }

#define RUDF_STRUCT_ADD_C_METHOD_ACCESS_CODE(r, TYPE, i, member) \
  ::rapidudf::ReflectFactory::AddStructMethodAccessor(BOOST_PP_STRINGIZE(member), &TYPE::member);

#define RUDF_STRUCT_ADD_SAFE_METHOD_ACCESS_CODE(r, TYPE, i, member)                                           \
  static ::rapidudf::SafeMemberFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(NAME)), __LINE__, TYPE, \
                                                                   decltype(&TYPE::member)>                   \
      BOOST_PP_CAT(rudf_safe_member_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(member), &TYPE::member);

#define RUDF_STRUCT_ADD_METHOD_ACCESS_CODE(r, TYPE, i, member)                                                \
  {                                                                                                           \
    static ::rapidudf::MemberFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(member)), __LINE__, TYPE, \
                                                                 decltype(&TYPE::member)>                     \
        BOOST_PP_CAT(rudf_member_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(member), &TYPE::member);             \
  }

#define RUDF_STRUCT_MEMBER_METHODS(st, ...)                                                                  \
  namespace rapidudf {                                                                                       \
  template <>                                                                                                \
  class StructMethodAccess<st> {                                                                             \
   public:                                                                                                   \
    static void Init() {                                                                                     \
      static bool inited = false;                                                                            \
      if (inited) {                                                                                          \
        return;                                                                                              \
      }                                                                                                      \
      BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_METHOD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
      ::rapidudf::DTypeFactory::Add<st>();                                                                   \
      inited = true;                                                                                         \
    }                                                                                                        \
                                                                                                             \
    static std::optional<rapidudf::StructMember> GetStructMember(const std::string& name) {                  \
      Init();                                                                                                \
      return rapidudf::ReflectFactory::GetStructMember(rapidudf::get_dtype<st>(), name);                     \
    }                                                                                                        \
  };                                                                                                         \
  static ::rapidudf::StructMethodAccessRegister<st> BOOST_PP_CAT(rudf_struct_method_access_, __COUNTER__);   \
  }

#define RUDF_STRUCT_SAFE_MEMBER_METHODS(st, ...)                                                                  \
  namespace rapidudf {                                                                                            \
  template <>                                                                                                     \
  class StructMethodAccess<st> {                                                                                  \
   public:                                                                                                        \
    static void Init() {                                                                                          \
      static bool inited = false;                                                                                 \
      if (inited) {                                                                                               \
        return;                                                                                                   \
      }                                                                                                           \
      BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_SAFE_METHOD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
      ::rapidudf::DTypeFactory::Add<st>();                                                                        \
      inited = true;                                                                                              \
    }                                                                                                             \
                                                                                                                  \
    static std::optional<rapidudf::StructMember> GetStructMember(const std::string& name) {                       \
      Init();                                                                                                     \
      return rapidudf::ReflectFactory::GetStructMember(rapidudf::get_dtype<st>(), name);                          \
    }                                                                                                             \
  };                                                                                                              \
  static ::rapidudf::StructMethodAccessRegister<st> BOOST_PP_CAT(rudf_struct_method_access_, __COUNTER__);        \
  }

#define RUDF_STRUCT_METHODS_BIND(helper, ...)                                                                        \
  static ::rapidudf::StructAccessHelperRegister<helper> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_C_METHOD_ACCESS_CODE, helper, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))     \
  });

#define RUDF_STRUCT_SAFE_METHODS_BIND(helper, ...)                                                                    \
  static ::rapidudf::StructAccessHelperRegister<helper> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)([]() {  \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_SAFE_C_METHOD_ACCESS_CODE, helper, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
  });

#define RUDF_STRUCT_SAFE_MEMBER_METHOD_BIND(st, NAME, member_method)                                                   \
  {                                                                                                                    \
    using func_type = decltype(member_method);                                                                         \
    using object_type = ::rapidudf::FunctionTraits<func_type>::object_type;                                            \
    static ::rapidudf::SafeMemberFuncRegister<::rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(NAME)), __LINE__, object_type, \
                                                                     func_type>                                        \
        BOOST_PP_CAT(rudf_safe_member_funcs_, __COUNTER__)(BOOST_PP_STRINGIZE(NAME), member_method);                   \
  }

#define RUDF_STRUCT_MEMBER_METHOD_BIND(st, NAME, member_method)                                                        \
  {                                                                                                                    \
    static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)(           \
        []() { ::rapidudf::ReflectFactory::AddStructMemberMethodAccessor(BOOST_PP_STRINGIZE(NAME), member_method); }); \
  }
