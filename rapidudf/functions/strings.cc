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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <string_view>

#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace functions {

#define REGISTER_STRING_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();
#define REGISTER_STRING_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_STRING_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

template <OpToken op>
void compare_string_views(const StringView* left, const StringView* right, uint8_t* ret) {
  for (size_t i = 0; i < simd::kVectorUnitSize; i++) {
    bool v = compare_string_view(static_cast<uint32_t>(op), left[i], right[i]);
    uint32_t byte_idx = i / 8;
    uint8_t bit_cursor = i % 8;
    if (v) {
      ret[byte_idx] = bit_set(ret[byte_idx], bit_cursor);
    } else {
      ret[byte_idx] = bit_clear(ret[byte_idx], bit_cursor);
    }
  }
}

static void register_string_view_vector_cmp_func() {
  DType simd_vector_string = DType(DATA_STRING_VIEW).ToSimdVector();
  std::string func_name = GetFunctionName(OP_EQUAL, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_EQUAL>);
  func_name = GetFunctionName(OP_NOT_EQUAL, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_NOT_EQUAL>);
  func_name = GetFunctionName(OP_GREATER_EQUAL, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_GREATER_EQUAL>);
  func_name = GetFunctionName(OP_LESS_EQUAL, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_LESS_EQUAL>);
  func_name = GetFunctionName(OP_GREATER, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_GREATER>);
  func_name = GetFunctionName(OP_LESS, simd_vector_string);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), compare_string_views<OP_LESS>);
}

StringView cast_stdstr_to_string_view(const std::string& str) { return StringView(str); }
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

  register_string_view_vector_cmp_func();
}
}  // namespace functions

}  // namespace rapidudf