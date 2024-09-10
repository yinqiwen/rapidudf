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
#include "rapidudf/jit/xbyak/ops/cmp.h"
#include <atomic>
#include "rapidudf/jit/xbyak/ops/copy.h"
#include "rapidudf/meta/optype.h"
#include "xbyak/xbyak_util.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
using namespace Xbyak::util;
static std::atomic<uint32_t> g_label_cursor{0};

int cmp_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Operand& left, const Xbyak::Operand& right,
              bool reverse) {
  uint32_t bits = dtype.Bits();
  if (left.isXMM()) {
    copy_value(c, dtype, left, rax);
    return cmp_value(c, dtype, rax.changeBit(bits), right, reverse);
  }
  if (right.isXMM()) {
    copy_value(c, dtype, right, rcx);
    return cmp_value(c, dtype, left, rcx.changeBit(bits), reverse);
  }
  if (left.isREG() && left.getBit() != bits) {
    return cmp_value(c, dtype, reinterpret_cast<const Xbyak::Reg&>(left).changeBit(bits), right, reverse);
  }
  if (right.isREG() && right.getBit() != bits) {
    return cmp_value(c, dtype, left, reinterpret_cast<const Xbyak::Reg&>(right).changeBit(bits), reverse);
  }
  if (dtype == DATA_F32) {
    copy_value(c, dtype, left, xmm0);
    copy_value(c, dtype, right, xmm1);
    if (reverse) {
      c.comiss(xmm1, xmm0);
    } else {
      c.comiss(xmm0, xmm1);
    }
  } else if (dtype == DATA_F64) {
    copy_value(c, dtype, left, xmm0);
    copy_value(c, dtype, right, xmm1);
    if (reverse) {
      c.comisd(xmm1, xmm0);
    } else {
      c.comisd(xmm0, xmm1);
    }
  } else {
    if (left.isMEM() && right.isMEM()) {
      copy_value(c, dtype, left, rax.changeBit(bits));
      return cmp_value(c, dtype, rax.changeBit(bits), right, reverse);
    } else {
      if (reverse) {
        c.cmp(right, left);
      } else {
        c.cmp(left, right);
      }
    }
  }
  return 0;
}
int cmp_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Operand& left, uint64_t right, bool reverse) {
  uint32_t bits = dtype.Bits();
  c.mov(rdx.changeBit(bits), right);

  int rc = cmp_value(c, dtype, left, rdx.changeBit(bits), reverse);

  return rc;
}

int cmp_value(Xbyak::CodeGenerator& c, DType dtype, uint64_t left, uint64_t right) {
  uint32_t bits = dtype.Bits();
  c.mov(rax.changeBit(bits), left);
  c.mov(rdx.changeBit(bits), right);
  return cmp_value(c, dtype, rax.changeBit(bits), rdx.changeBit(bits));
}

template <typename T>
static std::optional<bool> do_constant_cmp_op(OpToken op, DType dtype, uint64_t left_bin, uint64_t right_bin) {
  auto left_v = dtype.ToPrimitiveValue<T>(left_bin);
  auto right_v = dtype.ToPrimitiveValue<T>(right_bin);
  if (!left_v || !right_v) {
    RUDF_ERROR("Failed to convert to primitive value for dtype:{}", dtype);
    return {};
  }
  auto left = *left_v;
  auto right = *right_v;
  switch (op) {
    case OP_EQUAL: {
      return left == right;
    }
    case OP_NOT_EQUAL: {
      return left != right;
    }
    case OP_LESS: {
      return left < right;
    }
    case OP_LESS_EQUAL: {
      return left <= right;
    }
    case OP_GREATER: {
      return left > right;
    }
    case OP_GREATER_EQUAL: {
      return left >= right;
    }
    default: {
      RUDF_ERROR("Unsupported op:{} on dtype:{}", op, dtype);
      return {};
    }
  }
  return {};
}

int cmp_value(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right, bool& result) {
  std::optional<bool> result_bin;
  switch (dtype.GetFundamentalType()) {
    case DATA_F32: {
      result_bin = do_constant_cmp_op<float>(op, dtype, left, right);

      break;
    }
    case DATA_F64: {
      result_bin = do_constant_cmp_op<double>(op, dtype, left, right);
      break;
    }
    case DATA_U64: {
      result_bin = do_constant_cmp_op<uint64_t>(op, dtype, left, right);
      break;
    }
    case DATA_U32: {
      result_bin = do_constant_cmp_op<uint32_t>(op, dtype, left, right);

      break;
    }
    case DATA_I32: {
      result_bin = do_constant_cmp_op<int32_t>(op, dtype, left, right);
      break;
    }
    case DATA_U16: {
      result_bin = do_constant_cmp_op<uint16_t>(op, dtype, left, right);

      break;
    }
    case DATA_I16: {
      result_bin = do_constant_cmp_op<int16_t>(op, dtype, left, right);
      break;
    }
    case DATA_U8: {
      result_bin = do_constant_cmp_op<uint8_t>(op, dtype, left, right);
      break;
    }
    case DATA_I8: {
      result_bin = do_constant_cmp_op<int8_t>(op, dtype, left, right);
      break;
    }
    default: {
      RUDF_ERROR("Unsupported dtype:{} for constants cmp op.", dtype);
      return -1;
    }
  }
  if (!result_bin) {
    RUDF_ERROR("Convert from primitive value to binary failed with dtype:{}", dtype);
    return -1;
  }
  result = *result_bin;
  return 0;
}

int cmp_value(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right,
              const Xbyak::Operand& result) {
  bool result_b = false;
  if (0 != cmp_value(c, op, dtype, left, right, result_b)) {
    return -1;
  }
  return copy_value(c, DATA_U8, result_b ? 1 : 0, result);
}

static std::string get_cmp_label() { return "cmp_" + std::to_string(g_label_cursor.fetch_add(1)); }
static std::string get_exit_label() { return "cmp_exit_" + std::to_string(g_label_cursor.fetch_add(1)); }
int store_cmp_result(Xbyak::CodeGenerator& c, OpToken op, const Xbyak::Operand& result) {
  std::string label = get_cmp_label();
  std::string exit_label = get_exit_label();
  switch (op) {
    case OP_EQUAL: {
      c.je(label, c.T_NEAR);
      break;
    }
    case OP_NOT_EQUAL: {
      c.jne(label, c.T_NEAR);
      break;
    }
    case OP_LESS: {
      c.jb(label, c.T_NEAR);
      break;
    }
    case OP_LESS_EQUAL: {
      c.jbe(label, c.T_NEAR);
      break;
    }
    case OP_GREATER: {
      c.ja(label, c.T_NEAR);
      break;
    }
    case OP_GREATER_EQUAL: {
      c.jae(label, c.T_NEAR);
      break;
    }
    default: {
      RUDF_ERROR("Unsuooprted cmp op:{}", static_cast<int>(op));
      return -1;
    }
  }
  copy_value(c, DATA_U8, 0, result);
  // c.mov(result, 0);
  c.jmp(exit_label);
  c.L(label);
  // c.mov(result, 1);
  copy_value(c, DATA_U8, 1, result);
  c.L(exit_label);
  c.nop();
  return 0;
}

}  // namespace rapidudf