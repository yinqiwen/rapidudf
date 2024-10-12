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

#include <cstdint>
#include <string_view>

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

bool json_cmp_string(uint32_t op, const JsonObject& json, StringView right, bool reverse) {
  RUDF_DEBUG("json_cmp_string {}", right);
  if (!json.is_string()) {
    RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
    return false;
  }
  StringView left = StringView(json.get<std::string_view>());
  return json_cmp(op, left, right, reverse);
}

bool json_cmp_int(uint32_t op, const JsonObject& json, int64_t right, bool reverse) {
  RUDF_DEBUG("json_cmp_int {}", right);
  if (!json.is_number_integer()) {
    RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
    return false;
  }
  int64_t left = json.get<int64_t>();
  return json_cmp(op, left, right, reverse);
}
bool json_cmp_bool(uint32_t op, const JsonObject& json, bool right, bool reverse) {
  RUDF_DEBUG("json_cmp_bool {}", right);
  if (!json.is_boolean()) {
    RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
    return false;
  }
  bool left = json.get<bool>();
  return json_cmp(op, left, right, reverse);
}
bool json_cmp_float(uint32_t op, const JsonObject& json, double right, bool reverse) {
  RUDF_DEBUG("json_cmp_float {}", right);
  if (!json.is_number_float()) {
    RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with non string json object:{} ", op, json.type_name());
    return false;
  }
  double left = json.get<double>();
  return json_cmp(op, left, right, reverse);
}
bool json_cmp_json(uint32_t op, const JsonObject& left, const JsonObject& right) {
  if (left.is_boolean() && right.is_boolean()) {
    return json_cmp_bool(op, left, right.get<bool>(), false);
  } else if (left.is_string() && right.is_string()) {
    return json_cmp_string(op, left, StringView(right.get<std::string_view>()), false);
  } else if (left.is_number_float() && right.is_number_float()) {
    return json_cmp_float(op, left, right.get<double>(), false);
  } else if (left.is_number_integer() && right.is_number_integer()) {
    return json_cmp_int(op, left, right.get<int64_t>(), false);
  } else {
    RUDF_CRITICAL("Unsupported json_cmp_string compare op:{} with left json:{}, right json:{} ", op, left.type_name(),
                  right.type_name());
    return false;
  }
}
void init_builtin_json_funcs() {
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonMemberGet, json_member_get);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonArrayGet, json_array_get);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpString, json_cmp_string);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpInt, json_cmp_int);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpFloat, json_cmp_float);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpBool, json_cmp_bool);
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinJsonCmpJson, json_cmp_json);
}
}  // namespace functions
}  // namespace rapidudf