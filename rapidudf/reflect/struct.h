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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <functional>
#include <string>

#include "rapidudf/meta/dtype.h"
#include "rapidudf/reflect/flatbuffers.h"
#include "rapidudf/reflect/protobuf.h"
#include "rapidudf/reflect/reflect.h"

namespace rapidudf {
namespace reflect {
struct Field {
  DType dtype;
  uint32_t bytes_offset = 0;
};
}  // namespace reflect
}  // namespace rapidudf

namespace rapidudf {

template <typename T>
class StructAccess {
 public:
  static std::optional<StructMember> GetStructMember(const std::string& name) {
    return Reflect::GetStructMember(get_dtype<T>(), name);
  }
};

template <typename T>
class StructAccessHelperRegister {
 public:
  StructAccessHelperRegister(std::function<void()>&& f) { f(); }
};

}  // namespace rapidudf

#define RUDF_STRUCT_FILED_OFFSETOF(TYPE, ELEMENT) ((size_t) & (((TYPE*)0)->ELEMENT))
#define RUDF_STRUCT_ADD_FIELD_ACCESS_CODE(r, TYPE, i, member)                                                   \
  ::rapidudf::Reflect::AddStructField(::rapidudf::get_dtype<TYPE>(),                                            \
                                      BOOST_PP_STRINGIZE(member),                                               \
                                                         ::rapidudf::get_dtype<decltype(((TYPE*)0)->member)>(), \
                                                         RUDF_STRUCT_FILED_OFFSETOF(TYPE, member));

#define RUDF_STRUCT_FIELDS(st, ...)                                                                             \
  static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_field_access_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_FIELD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))       \
    ::rapidudf::DTypeFactory::Add<st>();                                                                        \
  });

#define RUDF_STRUCT_ADD_SAFE_C_METHOD_ACCESS_CODE(r, TYPE, i, member) \
  SAFE_MEMBER_FUNC_WRAPPER(BOOST_PP_STRINGIZE(member), &TYPE::member);

#define RUDF_STRUCT_ADD_C_METHOD_ACCESS_CODE(r, TYPE, i, member) \
  ::rapidudf::Reflect::AddStructMethodAccessor(BOOST_PP_STRINGIZE(member), &TYPE::member);

#define RUDF_STRUCT_ADD_SAFE_METHOD_ACCESS_CODE(r, TYPE, i, member) \
  SAFE_MEMBER_FUNC_WRAPPER(BOOST_PP_STRINGIZE(member), &TYPE::member);

#define RUDF_STRUCT_ADD_METHOD_ACCESS_CODE(r, TYPE, i, member)                         \
  {                                                                                    \
    MEMBER_FUNC_WRAPPER(BOOST_PP_STRINGIZE(member), &TYPE::member);                    \
    using ret_type = ::rapidudf::FunctionTraits<decltype(&TYPE::member)>::return_type; \
    ::rapidudf::try_register_fbs_vector_member_funcs<ret_type>();                      \
  }

#define RUDF_STRUCT_MEMBER_METHODS(st, ...)                                                                      \
  static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_member_access_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_METHOD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))       \
    ::rapidudf::DTypeFactory::Add<st>();                                                                         \
  });

#define RUDF_STRUCT_HELPER_METHODS_BIND(helper, ...)                                                                 \
  static ::rapidudf::StructAccessHelperRegister<helper> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_C_METHOD_ACCESS_CODE, helper, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))     \
  });

#define RUDF_STRUCT_HELPER_METHOD_BIND(NAME, func)                  \
  static ::rapidudf::StructAccessHelperRegister<void> BOOST_PP_CAT( \
      rudf_struct_access_helper_, __COUNTER__)([]() { ::rapidudf::Reflect::AddStructMethodAccessor(NAME, func); });

#define RUDF_STRUCT_MEMBER_METHOD_BIND(NAME, member_method)                                         \
  static ::rapidudf::StructAccessHelperRegister BOOST_PP_CAT(rudf_member_bind_, __COUNTER__)([]() { \
    using obj_t = FunctionTraits<decltype(member_method)>::object_type;                             \
    MEMBER_FUNC_WRAPPER(NAME, member_method);                                                       \
    DTypeFactory::Add<obj_t>();                                                                     \
  });
