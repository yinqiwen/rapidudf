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
    default: {
      return false;
    }
  }
}
}  // namespace rapidudf