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
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
template <typename T>
constexpr bool is_valid_operand(OpToken op) {
  switch (op) {
    case OP_ASSIGN: {
      return true;
    }
    case OP_PLUS:
    case OP_MINUS:
    case OP_MULTIPLY:
    case OP_DIVIDE:
    case OP_PLUS_ASSIGN:
    case OP_MINUS_ASSIGN:
    case OP_MULTIPLY_ASSIGN:
    case OP_DIVIDE_ASSIGN:
    case OP_IOTA:
    case OP_CLAMP:
    case OP_FMA:
    case OP_FMS:
    case OP_FNMA:
    case OP_FNMS:
    case OP_SUM:
    case OP_MAX:
    case OP_MIN: {
      if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_MOD:
    case OP_MOD_ASSIGN: {
      if constexpr (std::is_integral_v<T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_ABS:
    case OP_NEGATIVE: {
      if constexpr (std::is_signed_v<T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_NOT:
    case OP_LOGIC_AND:
    case OP_LOGIC_OR: {
      if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, Bit>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_CLONE:
    case OP_EQUAL:
    case OP_NOT_EQUAL:
    case OP_LESS:
    case OP_LESS_EQUAL:
    case OP_GREATER:
    case OP_GREATER_EQUAL: {
      if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, StringView>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_POW:
    case OP_SQRT:
    case OP_FLOOR:
    case OP_CEIL:
    case OP_RINT:
    case OP_ERF:
    case OP_ERFC:
    case OP_SIN:
    case OP_COS:
    case OP_TAN:
    case OP_ASIN:
    case OP_ACOS:
    case OP_ATAN:
    case OP_ATANH:
    case OP_ATAN2:
    case OP_SINH:
    case OP_COSH:
    case OP_TANH:
    case OP_ASINH:
    case OP_ACOSH:
    case OP_EXP:
    case OP_EXP2:
    case OP_EXPM1:
    case OP_LOG:
    case OP_LOG2:
    case OP_LOG10:
    case OP_LOG1P:
    case OP_HYPOT:
    case OP_DOT:
    case OP_ROUND: {
      if constexpr (std::is_floating_point_v<T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_CONDITIONAL: {
      if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_SORT:
    case OP_SELECT:
    case OP_TOPK:
    case OP_ARG_SORT:
    case OP_ARG_SELECT: {
      if constexpr (std::is_floating_point_v<T> || std::is_same_v<int64_t, T> || std::is_same_v<int32_t, T> ||
                    std::is_same_v<int16_t, T> || std::is_same_v<uint64_t, T> || std::is_same_v<uint32_t, T> ||
                    std::is_same_v<uint16_t, T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_SORT_KV:
    case OP_SELECT_KV:
    case OP_TOPK_KV: {
      if constexpr (std::is_floating_point_v<T> || std::is_same_v<int64_t, T> || std::is_same_v<int32_t, T> ||
                    std::is_same_v<uint64_t, T> || std::is_same_v<uint32_t, T>) {
        return true;
      } else {
        return false;
      }
    }
    case OP_FILTER:
    case OP_GATHER: {
      if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, StringView>) {
        return true;
      } else {
        return false;
      }
    }
    default: {
      return false;
    }
  }
}

inline bool is_arithmetic_op(OpToken op) {
  switch (op) {
    case OP_PLUS:
    case OP_MINUS:
    case OP_MULTIPLY:
    case OP_DIVIDE:
    case OP_PLUS_ASSIGN:
    case OP_MINUS_ASSIGN:
    case OP_MULTIPLY_ASSIGN:
    case OP_DIVIDE_ASSIGN:
    case OP_CLAMP:
    case OP_FMA:
    case OP_FMS:
    case OP_FNMA:
    case OP_FNMS:
    case OP_SUM:
    case OP_MAX:
    case OP_MIN:
    case OP_MOD:
    case OP_MOD_ASSIGN:
    case OP_ABS:
    case OP_NEGATIVE:
    case OP_POW:
    case OP_SQRT:
    case OP_FLOOR:
    case OP_CEIL:
    case OP_RINT:
    case OP_ERF:
    case OP_ERFC:
    case OP_SIN:
    case OP_COS:
    case OP_TAN:
    case OP_ASIN:
    case OP_ACOS:
    case OP_ATAN:
    case OP_ATANH:
    case OP_ATAN2:
    case OP_SINH:
    case OP_COSH:
    case OP_TANH:
    case OP_ASINH:
    case OP_ACOSH:
    case OP_EXP:
    case OP_EXP2:
    case OP_EXPM1:
    case OP_LOG:
    case OP_LOG2:
    case OP_LOG10:
    case OP_LOG1P:
    case OP_HYPOT:
    case OP_DOT:
    case OP_ROUND: {
      return true;
    }
    default: {
      return false;
    }
  }
}
inline bool is_compare_op(OpToken op) { return op >= OP_EQUAL && op <= OP_GREATER_EQUAL; }
inline bool is_logic_op(OpToken op) {
  return op == OP_LOGIC_AND || op == OP_LOGIC_OR || op == OP_LOGIC_XOR || op == OP_NOT;
}

inline bool is_unary_op(OpToken op) { return op > OP_UNARY_BEGIN && op < OP_UNARY_END; }
inline bool is_binary_op(OpToken op) { return op > OP_BINARY_BEGIN && op < OP_BINARY_END; }
inline bool is_ternary_op(OpToken op) { return op > OP_TERNARY_BEGIN && op < OP_TERNARY_END; }

inline int get_operand_count(OpToken op) {
  if (is_unary_op(op)) {
    return 1;
  }
  if (is_binary_op(op)) {
    return 2;
  }
  if (is_ternary_op(op)) {
    return 3;
  }
  return -1;
  // switch (op) {
  //   case OP_DOT: {
  //     return 2;
  //   }
  //   case OP_IOTA: {
  //     return 3;
  //   }
  //   case OP_ARG_SORT:
  //   case OP_SORT: {
  //     return 2;
  //   }
  //   case OP_SELECT:
  //   case OP_TOPK:
  //   case OP_ARG_SELECT: {
  //     return 3;
  //   }
  //   case OP_SORT_KV: {
  //     return 3;
  //   }

  //   case OP_SELECT_KV:
  //   case OP_TOPK_KV: {
  //     return 4;
  //   }
  //   case OP_SUM:
  //   case OP_CLONE:
  //   case OP_SCALAR_CAST:
  //   case OP_FILTER: {
  //     return 1;
  //   }
  //   case OP_GATHER: {
  //     return 2;
  //   }
  //   default: {
  //     return -1;
  //   }
  // }
}
}  // namespace rapidudf