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
#include <string>

#include "rapidudf/codegen/dtype.h"
#include "rapidudf/reflect/flatbuffers.h"
#include "rapidudf/reflect/protobuf.h"
#include "rapidudf/reflect/reflect.h"

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

#define RUDF_STRUCT_SAFE_MEMBER_METHODS(st, ...)                                                                 \
  static ::rapidudf::StructAccessHelperRegister<st> BOOST_PP_CAT(rudf_struct_member_access_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_SAFE_METHOD_ACCESS_CODE, st, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))  \
    ::rapidudf::DTypeFactory::Add<st>();                                                                         \
  });

#define RUDF_STRUCT_HELPER_METHODS_BIND(helper, ...)                                                                 \
  static ::rapidudf::StructAccessHelperRegister<helper> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)([]() { \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_C_METHOD_ACCESS_CODE, helper, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))     \
  });

#define RUDF_STRUCT_SAFE_HELPER_METHODS_BIND(helper, ...)                                                             \
  static ::rapidudf::StructAccessHelperRegister<helper> BOOST_PP_CAT(rudf_struct_access_helper_, __COUNTER__)([]() {  \
    BOOST_PP_SEQ_FOR_EACH_I(RUDF_STRUCT_ADD_SAFE_C_METHOD_ACCESS_CODE, helper, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
  });

#define RUDF_STRUCT_SAFE_MEMBER_METHOD_BIND(NAME, member_method)                                         \
  static ::rapidudf::StructAccessHelperRegister BOOST_PP_CAT(rudf_safe_member_bind_, __COUNTER__)([]() { \
    using obj_t = FunctionTraits<decltype(member_method)>::object_type;                                  \
    SAFE_MEMBER_FUNC_WRAPPER(NAME, member_method);                                                       \
    DTypeFactory::Add<obj_t>();                                                                          \
  });

#define RUDF_STRUCT_MEMBER_METHOD_BIND(NAME, member_method)                                         \
  static ::rapidudf::StructAccessHelperRegister BOOST_PP_CAT(rudf_member_bind_, __COUNTER__)([]() { \
    using obj_t = FunctionTraits<decltype(member_method)>::object_type;                             \
    MEMBER_FUNC_WRAPPER(NAME, member_method);                                                       \
    DTypeFactory::Add<obj_t>();                                                                     \
  });
