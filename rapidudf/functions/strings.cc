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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <string_view>

#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace functions {

#define REGISTER_STRING_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();
#define REGISTER_STRING_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_STRING_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

bool compare_string_view(uint32_t op, StringView left, StringView right) {
  bool result = false;
  switch (op) {
    case OP_EQUAL: {
      result = left == right;
      break;
    }
    case OP_NOT_EQUAL: {
      result = left != right;
      break;
    }
    case OP_GREATER: {
      result = left > right;
      break;
    }
    case OP_GREATER_EQUAL: {
      result = left >= right;
      break;
    }
    case OP_LESS: {
      result = left < right;
      break;
    }
    case OP_LESS_EQUAL: {
      result = left <= right;
      break;
    }
    default: {
      RUDF_CRITICAL("Unsupported string_view compare op:{}", op);
      return false;
    }
  }
  RUDF_DEBUG("cmp string:{} & {} with op:{} result:{}", left, right, op, result);
  return result;
}

StringView cast_stdstr_to_string_view(const std::string& str) {
  RUDF_DEBUG("cast_stdstr_to_string_view #{}#", str);
  return StringView(str);
}
StringView cast_fbsstr_to_string_view(const flatbuffers::String& str) { return StringView(str.c_str(), str.size()); }
StringView cast_stdstrview_to_string_view(std::string_view str) { return StringView(str); }

struct StringViewHelper {
  static size_t size(StringView s) { return s.size(); }
  static bool contains(StringView s, StringView part) { return StringView::contains(s, part); }
  static bool starts_with(StringView s, StringView part) { return StringView::starts_with(s, part); }
  static bool ends_with(StringView s, StringView part) { return StringView::ends_with(s, part); }
  static bool contains_ignore_case(StringView s, StringView part) { return StringView::contains_ignore_case(s, part); }
  static bool starts_with_ignore_case(StringView s, StringView part) {
    return StringView::starts_with_ignore_case(s, part);
  }
  static bool ends_with_ignore_case(StringView s, StringView part) {
    return StringView::ends_with_ignore_case(s, part);
  }
};

struct StdStringViewHelper {
  static size_t size(std::string_view s) { return s.size(); }
};

void init_builtin_strings_funcs() {
  RUDF_STRUCT_HELPER_METHODS_BIND(StringViewHelper, size, contains, starts_with, ends_with, contains_ignore_case,
                                  starts_with_ignore_case, ends_with_ignore_case)
  RUDF_STRUCT_HELPER_METHODS_BIND(StdStringViewHelper, size)
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinStringViewCmp, compare_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastStdStrToStringView, cast_stdstr_to_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastFbsStrToStringView, cast_fbsstr_to_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastStdStrViewToStringView, cast_stdstrview_to_string_view);
}
}  // namespace functions

}  // namespace rapidudf