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
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace functions {

const JsonObject& json_member_get(const JsonObject& json, StringView key) { return json[key]; }
const JsonObject& json_array_get(const JsonObject& json, size_t idx) { return json[idx]; }

template <typename T>
static bool json_cmp(uint32_t op, T left, T right, bool reverse) {
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
      if (reverse) {
        return right > left;
      } else {
        result = left > right;
      }
      break;
    }
    case OP_GREATER_EQUAL: {
      if (reverse) {
        result = left >= right;
      } else {
        result = right >= left;
      }

      break;
    }
    case OP_LESS: {
      if (reverse) {
        result = left < right;
      } else {
        result = right < left;
      }
      break;
    }
    case OP_LESS_EQUAL: {
      if (reverse) {
        result = left <= right;
      } else {
        result = right <= left;
      }
      break;
    }
    default: {
      RUDF_CRITICAL("Unsupported string_view compare op:{}", op);
      return false;
    }
  }
  RUDF_DEBUG("cmp {} & {} with op:{} result:{}", left, right, op, result);
  return result;
}

template <typename T>
static T json_extract(JsonObject& json) {
  T v = {};
  if constexpr (std::is_same_v<StringView, T>) {
    v = StringView(json.get<std::string_view>());
  } else {
    v = json.get<T>();
  }
  return v;
}

template <typename T>
static void register_json_extract() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(kBuiltinJsonExtract, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), json_extract<T>);
}

// bool json_cmp_string(uint32_t op, const JsonObject& json, StringView right, bool reverse) {
//   RUDF_DEBUG("json_cmp_string {}", right);
//   if (!json.is_string()) {
//     RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
//     return false;
//   }
//   StringView left = StringView(json.get<std::string_view>());
//   return json_cmp(op, left, right, reverse);
// }

// bool json_cmp_int(uint32_t op, const JsonObject& json, int64_t right, bool reverse) {
//   RUDF_DEBUG("json_cmp_int {}", right);
//   if (!json.is_number_integer()) {
//     RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
//     return false;
//   }
//   int64_t left = json.get<int64_t>();
//   return json_cmp(op, left, right, reverse);
// }
// bool json_cmp_bool(uint32_t op, const JsonObject& json, bool right, bool reverse) {
//   RUDF_DEBUG("json_cmp_bool {}", right);
//   if (!json.is_boolean()) {
//     RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
//     return false;
//   }
//   bool left = json.get<bool>();
//   return json_cmp(op, left, right, reverse);
// }
// bool json_cmp_float(uint32_t op, const JsonObject& json, double right, bool reverse) {
//   RUDF_DEBUG("json_cmp_float {}", right);
//   if (!json.is_number_float()) {
//     RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
//     return false;
//   }
//   double left = json.get<double>();
//   return json_cmp(op, left, right, reverse);
// }
// bool json_cmp_json(uint32_t op, const JsonObject& left, const JsonObject& right) {
//   if (left.is_boolean() && right.is_boolean()) {
//     return json_cmp_bool(op, left, right.get<bool>(), false);
//   } else if (left.is_string() && right.is_string()) {
//     return json_cmp_string(op, left, StringView(right.get<std::string_view>()), false);
//   } else if (left.is_number_float() && right.is_number_float()) {
//     return json_cmp_float(op, left, right.get<double>(), false);
//   } else if (left.is_number_integer() && right.is_number_integer()) {
//     return json_cmp_int(op, left, right.get<int64_t>(), false);
//   } else {
//     RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with left json:{}, right json:{} ", op,
//     left.type_name(),
//                   right.type_name());
//     return false;
//   }
// }

#define REGISTER_JSON_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_JSON_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_JSON_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

void init_builtin_json_funcs() {
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonMemberGet, json_member_get);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonArrayGet, json_array_get);

  REGISTER_JSON_FUNCS(register_json_extract, bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t,
                      uint64_t, float, double, StringView)
  // RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpString, json_cmp_string);
  // RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpInt, json_cmp_int);
  // RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpFloat, json_cmp_float);
  // RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpBool, json_cmp_bool);
  // RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpJson, json_cmp_json);
}
}  // namespace functions
}  // namespace rapidudf