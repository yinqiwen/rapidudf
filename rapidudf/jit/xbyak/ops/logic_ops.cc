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
#include "rapidudf/jit/xbyak/ops/logic_ops.h"
#include <atomic>
#include "xbyak/xbyak_util.h"

#include "rapidudf/jit/xbyak/ops/cmp.h"
#include "rapidudf/jit/xbyak/ops/copy.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
using namespace Xbyak::util;
static std::atomic<uint32_t> g_label_cursor{0};
static bool verify_logic_op_args(OpToken op, DType dtype) {
  if (dtype.GetFundamentalType() != DATA_U8) {
    RUDF_ERROR("Only support u8 type for logic ops, but got:{}", dtype);
    return false;
  }
  if (op != OP_LOGIC_AND && op != OP_LOGIC_OR) {
    RUDF_ERROR("Only support and/or or for logic ops, but got:{}", static_cast<int>(op));
    return false;
  }
  return true;
}
static std::string get_and_label() { return "logic_and_" + std::to_string(g_label_cursor.fetch_add(1)); }
static std::string get_or_label() { return "logic_or_" + std::to_string(g_label_cursor.fetch_add(1)); }
static std::string get_exit_label() { return "logic_exit_" + std::to_string(g_label_cursor.fetch_add(1)); }

int logic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Operand& left, const Xbyak::Operand& right,
             const Xbyak::Operand& result) {
  if (!verify_logic_op_args(op, dtype)) {
    return -1;
  }
  std::string cmp_label = op == OP_LOGIC_AND ? get_and_label() : get_or_label();
  std::string exit_label = get_exit_label();
  uint64_t cmp_val = op == OP_LOGIC_AND ? 0 : 1;
  int rc = cmp_value(c, dtype, left, cmp_val);
  if (0 != rc) {
    return rc;
  }
  c.je(cmp_label, c.T_NEAR);
  rc = cmp_value(c, dtype, right, cmp_val);
  if (0 != rc) {
    return rc;
  }
  c.je(cmp_label, c.T_NEAR);
  rc = copy_value(c, DATA_U8, 1 - cmp_val, result);
  if (0 != rc) {
    return rc;
  }
  c.jmp(exit_label, c.T_NEAR);
  c.L(cmp_label);
  rc = copy_value(c, DATA_U8, cmp_val, result);
  if (0 != rc) {
    return rc;
  }
  c.L(exit_label);
  c.nop();
  return 0;
}
int logic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, const Xbyak::Operand& right,
             const Xbyak::Operand& result) {
  c.mov(rax.changeBit(right.getBit()), left);
  return logic_op(c, op, dtype, rax.changeBit(right.getBit()), right, result);
}
int logic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Operand& left, uint64_t right,
             const Xbyak::Operand& result) {
  c.mov(rax.changeBit(left.getBit()), right);
  return logic_op(c, op, dtype, left, rax.changeBit(left.getBit()), result);
}

static std::optional<bool> do_constant_logic_op(OpToken op, DType dtype, uint64_t left_bin, uint64_t right_bin) {
  auto left_v = dtype.ToPrimitiveValue<bool>(left_bin);
  auto right_v = dtype.ToPrimitiveValue<bool>(right_bin);
  if (!left_v || !right_v) {
    RUDF_ERROR("Failed to convert to primitive value for dtype:{}", dtype);
    return {};
  }
  auto left = *left_v;
  auto right = *right_v;
  switch (op) {
    case OP_LOGIC_AND: {
      return left && right;
    }
    case OP_LOGIC_OR: {
      return left || right;
    }
    default: {
      RUDF_ERROR("Unsupported op:{} on dtype:{}", op, dtype);
      return {};
    }
  }
  return {};
}

int logic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right, bool& result) {
  auto result_b = do_constant_logic_op(op, dtype, left, right);
  if (!result_b) {
    return -1;
  }
  result = *result_b;
  return 0;
}
int logic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right,
             const Xbyak::Operand& result) {
  bool result_b = false;
  if (0 != logic_op(c, op, dtype, left, right, result_b)) {
    return -1;
  }
  return copy_value(c, DATA_U8, result_b ? 1 : 0, result);
}
}  // namespace rapidudf