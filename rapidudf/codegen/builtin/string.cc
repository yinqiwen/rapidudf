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
#include "absl/strings/match.h"
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/builtin/builtin_symbols.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/simd/simd_ops.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/types/simd.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

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

StringView cast_stdstr_to_string_view(const std::string& str) { return StringView(str); }
StringView cast_fbsstr_to_string_view(const flatbuffers::String& str) { return StringView(str.c_str(), str.size()); }
StringView cast_stdstrview_to_string_view(std::string_view str) { return StringView(str); }

template <OpToken op>
static void register_simd_vector_string_cmp() {
  simd::Vector<Bit> (*simd_f0)(simd::Vector<StringView>, simd::Vector<StringView>, uint32_t) =
      simd::simd_vector_string_cmp<op>;
  std::string func_name =
      GetFunctionName(op, DType(DATA_STRING_VIEW).ToSimdVector(), DType(DATA_STRING_VIEW).ToSimdVector());
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, func_name.c_str(), simd_f0);

  simd::Vector<Bit> (*simd_f1)(simd::Vector<StringView>, StringView, bool, uint32_t) =
      simd::simd_vector_string_cmp_scalar<op>;
  func_name = GetFunctionName(op, DType(DATA_STRING_VIEW).ToSimdVector(), DType(DATA_STRING_VIEW));
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, func_name.c_str(), simd_f1);
}

struct StringViewHelper {
  static size_t size(StringView s) { return s.size(); }
  static bool contains(StringView s, StringView part) {
    return absl::StrContains(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool starts_with(StringView s, StringView part) {
    return absl::StartsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool ends_with(StringView s, StringView part) {
    return absl::EndsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool contains_ignore_case(StringView s, StringView part) {
    return absl::StrContainsIgnoreCase(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool starts_with_ignore_case(StringView s, StringView part) {
    return absl::StartsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool ends_with_ignore_case(StringView s, StringView part) {
    return absl::EndsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
};

void init_builtin_string_funcs() {
  RUDF_STRUCT_HELPER_METHODS_BIND(StringViewHelper, size, contains, starts_with, ends_with, contains_ignore_case,
                                  starts_with_ignore_case, ends_with_ignore_case)
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinStringViewCmp, compare_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastStdStrToStringView, cast_stdstr_to_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastFbsStrToStringView, cast_fbsstr_to_string_view);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinCastStdStrViewToStringView, cast_stdstrview_to_string_view);
  REGISTER_STRING_FUNCS(register_simd_vector_string_cmp, OP_GREATER, OP_GREATER_EQUAL, OP_LESS_EQUAL, OP_LESS, OP_EQUAL,
                        OP_NOT_EQUAL)
}

}  // namespace rapidudf