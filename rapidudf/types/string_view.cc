/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include "rapidudf/types/string_view.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
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
}  // namespace rapidudf