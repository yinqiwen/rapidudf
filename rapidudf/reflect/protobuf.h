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
#include <type_traits>
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/reflect/reflect.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

template <typename T, typename Enable = void>
struct PBGetterReturnType {
  using return_type = T;
  static return_type value(const T& v) { return v; }
  static return_type default_value() { return {}; }
};
template <typename T>
struct PBGetterReturnType<T, typename std::enable_if<std::is_base_of<google::protobuf::Message, T>::value>::type> {
  using return_type = const T*;
  static return_type value(const T& v) { return &v; }
  static return_type default_value() {
    static T empty;
    return &empty;
  }
};
template <>
struct PBGetterReturnType<std::string> {
  using return_type = const std::string*;
  static return_type value(const std::string& v) { return &v; }
};

template <typename K, typename V>
struct PBMapHelper {
  using return_type_t = typename PBGetterReturnType<V>::return_type;
  static return_type_t Get(const ::google::protobuf::Map<K, V>* pb_map, K k) {
    if (nullptr == pb_map) {
      return PBGetterReturnType<V>::default_value();
    }
    auto found = pb_map->find(k);
    if (found == pb_map->end()) {
      return PBGetterReturnType<V>::default_value();
    }
    return PBGetterReturnType<V>::value(found->second);
  }
  static size_t Size(const ::google::protobuf::Map<K, V>* pb_map) {
    if (nullptr == pb_map) {
      return 0;
    }
    return pb_map->size();
  }
};

template <typename T>
struct PBRepeatedPtrFieldHelper {
  using return_type_t = typename PBGetterReturnType<T>::return_type;
  static return_type_t Get(const ::google::protobuf::RepeatedPtrField<T>* pb_vec, int i) {
    if (nullptr == pb_vec) {
      return PBGetterReturnType<T>::default_value();
    }
    const auto& val = pb_vec->Get(i);
    return PBGetterReturnType<T>::value(val);
  }
  static size_t Size(const ::google::protobuf::RepeatedPtrField<T>* pb_vec) {
    if (nullptr == pb_vec) {
      return 0;
    }
    return pb_vec->size();
  }
};

template <typename T>
struct PBRepeatedFieldHelper {
  using return_type_t = typename PBGetterReturnType<T>::return_type;
  static return_type_t Get(const ::google::protobuf::RepeatedField<T>* pb_vec, int i) {
    if (nullptr == pb_vec) {
      return PBGetterReturnType<T>::default_value();
    }
    const auto& val = pb_vec->Get(i);
    return PBGetterReturnType<T>::value(val);
  }
  static size_t Size(const ::google::protobuf::RepeatedField<T>* pb_vec) {
    if (nullptr == pb_vec) {
      return 0;
    }
    return pb_vec->size();
  }
};

template <typename V>
struct PBMapHelper<std::string, V> {
  using return_type_t = typename PBGetterReturnType<V>::return_type;
  static return_type_t Get(const ::google::protobuf::Map<std::string, V>* pb_map, StringView k) {
    if (nullptr == pb_map) {
      return PBGetterReturnType<V>::default_value();
    }
    auto found = pb_map->find(k);
    if (found == pb_map->end()) {
      return PBGetterReturnType<V>::default_value();
    }
    return PBGetterReturnType<V>::value(found->second);
  }
  static size_t Size(const ::google::protobuf::Map<std::string, V>* pb_map) {
    if (nullptr == pb_map) {
      return 0;
    }
    return pb_map->size();
  }
};

template <uint64_t, uint32_t, uint64_t, typename F>
struct PBSetStringHelper;

template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T>
struct PBSetStringHelper<SOURCE, LINE, HASH, void (T::*)(std::string&&)> {
  using func_t = void (T::*)(std::string&&);
  static std::string& GetFuncName() {
    static std::string func_name;
    return func_name;
  }
  static func_t& GetFunc() {
    static func_t func = nullptr;
    return func;
  }
  static void Call(T* p, StringView str) {
    auto func = GetFunc();
    (p->*func)(str.str());
  }
};

template <typename T>
void try_register_pb_container_member_funcs(std::string_view member) {
  using remove_ptr_t = std::remove_pointer_t<T>;
  using remove_reference_t = std::remove_reference_t<remove_ptr_t>;
  using remove_cv_t = std::remove_cv_t<remove_reference_t>;
  if constexpr (is_specialization<remove_cv_t, google::protobuf::Map>::value) {
    using key_type = typename remove_cv_t::key_type;
    using mapped_type = typename remove_cv_t::mapped_type;
    Reflect::AddStructMethodAccessor("get", &PBMapHelper<key_type, mapped_type>::Get);
    Reflect::AddStructMethodAccessor("size", &PBMapHelper<key_type, mapped_type>::Size);
  } else if constexpr (is_specialization<remove_cv_t, google::protobuf::RepeatedPtrField>::value) {
    using value_type = typename remove_cv_t::value_type;
    Reflect::AddStructMethodAccessor("get", &PBRepeatedPtrFieldHelper<value_type>::Get);
    Reflect::AddStructMethodAccessor("size", &PBRepeatedPtrFieldHelper<value_type>::Size);
  } else if constexpr (is_specialization<remove_cv_t, google::protobuf::RepeatedField>::value) {
    using value_type = typename remove_cv_t::value_type;
    Reflect::AddStructMethodAccessor("get", &PBRepeatedFieldHelper<value_type>::Get);
    Reflect::AddStructMethodAccessor("size", &PBRepeatedFieldHelper<value_type>::Size);
  }
}

}  // namespace rapidudf

#define RUDF_PB_SET_STRING_HELPER(st, member)                                                                     \
  static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_method_access_, __COUNTER__)([]() {  \
    using set_func_t = void (st::*)(std::string&&);                                                               \
    set_func_t set_func = &st::BOOST_PP_CAT(set_, member);                                                        \
    using wrapper_t =                                                                                             \
        rapidudf::PBSetStringHelper<rapidudf::fnv1a_hash(__FILE__), __LINE__,                                     \
                                    rapidudf::fnv1a_hash(BOOST_PP_STRINGIZE(member)), decltype(set_func)>;        \
    wrapper_t::GetFunc() = set_func;                                                                              \
    wrapper_t::GetFuncName() = BOOST_PP_STRINGIZE(BOOST_PP_CAT(set_, member));                                    \
    rapidudf::Reflect::AddStructMethodAccessor(BOOST_PP_STRINGIZE(BOOST_PP_CAT(set_, member)), &wrapper_t::Call); \
  });

#define RUDF_PB_ADD_FIELD_ACCESS_CODE(r, TYPE, i, member)                                          \
  {                                                                                                \
    using func_return_t = decltype(((TYPE*)1)->member());                                          \
    using getter_func_t = func_return_t (TYPE::*)() const;                                         \
    getter_func_t func = &TYPE::member;                                                            \
    MEMBER_FUNC_WRAPPER(BOOST_PP_STRINGIZE(member), func);                                         \
    ::rapidudf::try_register_pb_container_member_funcs<func_return_t>(BOOST_PP_STRINGIZE(member)); \
  }

#define RUDF_PB_FIELDS(st, ...)                                                                                  \
  namespace rapidudf {                                                                                           \
  template <>                                                                                                    \
  struct ReflectRegisterHelper<rapidudf::fnv1a_hash(__FILE__), __LINE__, 0, st> {                                \
    template <typename T = void>                                                                                 \
    static void Init() {                                                                                         \
      BOOST_PP_SEQ_FOR_EACH_I(RUDF_PB_ADD_FIELD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))          \
      ::rapidudf::DTypeFactory::Add<st>();                                                                       \
    }                                                                                                            \
  };                                                                                                             \
  static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_method_access_, __COUNTER__)([]() { \
    ReflectRegisterHelper<rapidudf::fnv1a_hash(__FILE__), __LINE__, 0, st>::Init<void>();                        \
  });                                                                                                            \
  }
