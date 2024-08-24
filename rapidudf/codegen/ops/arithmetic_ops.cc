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
#include "rapidudf/codegen/ops/arithmetic_ops.h"
#include <atomic>
#include <optional>
#include <type_traits>
#include <typeindex>
#include "rapidudf/codegen/dtype.h"
#include "xbyak/xbyak_util.h"

#include "rapidudf/codegen/ops/copy.h"
#include "rapidudf/log/log.h"
namespace rapidudf {
using namespace Xbyak::util;

static bool is_valid_operand(const Xbyak::Operand& operand) {
  if (operand.isMEM()) {
    return true;
  }
  if (operand.isREG() || operand.isXMM()) {
    return true;
  }
  RUDF_ERROR("Invalid operand kind:{}", static_cast<int>(operand.getKind()));
  return false;
}

static bool is_rax(const Xbyak::Operand& reg) {
  if (reg.isREG() && reg.getKind() == rax.getKind() && reg.getIdx() == rax.getIdx()) {
    return true;
  }
  return false;
}

static int do_xmm_register_arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Xmm& left,
                                         const Xbyak::Operand& right) {
  switch (op) {
    case OP_PLUS: {
      switch (dtype.GetFundamentalType()) {
        case DATA_F64: {
          c.addsd(left, right);
          break;
        }
        case DATA_F32: {
          c.addss(left, right);
          break;
        }
        case DATA_U8:
        case DATA_I8: {
          c.paddb(left, right);
          break;
        }
        case DATA_U16:
        case DATA_I16: {
          c.paddw(left, right);
          break;
        }
        case DATA_U32:
        case DATA_I32: {
          c.paddd(left, right);
          break;
        }
        case DATA_U64:
        case DATA_I64: {
          c.paddq(left, right);
          break;
        }
        default: {
          RUDF_ERROR("Unsupported dtype:{}", dtype);
          return -1;
        }
      }
      break;
    }
    case OP_MINUS: {
      switch (dtype.GetFundamentalType()) {
        case DATA_F64: {
          c.subsd(left, right);
          break;
        }
        case DATA_F32: {
          c.subss(left, right);
          break;
        }
        case DATA_U8:
        case DATA_I8: {
          c.psubb(left, right);
          break;
        }
        case DATA_U16:
        case DATA_I16: {
          c.psubw(left, right);
          break;
        }
        case DATA_U32:
        case DATA_I32: {
          c.psubd(left, right);
          break;
        }
        case DATA_U64:
        case DATA_I64: {
          c.psubq(left, right);
          break;
        }
        default: {
          RUDF_ERROR("Unsupported dtype:{}", dtype);
          return -1;
        }
      }
      break;
    }
    case OP_MULTIPLY: {
      switch (dtype.GetFundamentalType()) {
        case DATA_F64: {
          c.mulsd(left, right);
          break;
        }
        case DATA_F32: {
          c.mulss(left, right);
          break;
        }
        case DATA_U8:
        case DATA_I8:
        case DATA_U16:
        case DATA_I16:
        case DATA_U32:
        case DATA_I32: {
          c.pmulld(left, right);
          break;
        }
        case DATA_U64:
        case DATA_I64: {
          c.pmuldq(left, right);
          break;
        }
        default: {
          RUDF_ERROR("Unsupported dtype:{}", dtype);
          return -1;
        }
      }
      break;
    }
    case OP_DIVIDE: {
      switch (dtype.GetFundamentalType()) {
        case DATA_F64: {
          c.divsd(left, right);
          break;
        }
        case DATA_F32: {
          c.divss(left, right);
          break;
        }
        default: {
          RUDF_ERROR("Unsupported dtype:{}", dtype);
          return -1;
        }
      }
      break;
    }
    default: {
      RUDF_ERROR("Unsupported op:{}", static_cast<int>(op));
      return -1;
    }
  }
  return 0;
}

static int do_regular_register_arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Reg& left,
                                             const Xbyak::Operand& right) {
  if (dtype == DATA_F64 || dtype == DATA_F32) {
    RUDF_ERROR("Unsupported dtype:{} on regular registers to do arithmetic", dtype);
    return -1;
  }
  uint32_t bits = dtype.Bits();

  switch (op) {
    case OP_PLUS: {
      c.add(left, right);
      break;
    }
    case OP_MINUS: {
      c.sub(left, right);
      break;
    }
    case OP_MULTIPLY: {
      c.imul(left, right);
      break;
    }
    case OP_MOD:
    case OP_DIVIDE: {
      if (!is_rax(left)) {
        c.mov(rax, left);
      }

      if (bits <= 32) {
        c.cdq();
      } else {
        c.cqo();
      }
      c.idiv(right);
      // div:rax mod:rdx
      if (op == OP_DIVIDE) {
        if (!is_rax(left)) {
          c.mov(left, rax.changeBit(bits));
        }
      } else {
        c.mov(left, rdx.changeBit(bits));
      }
      break;
    }

    default: {
      RUDF_ERROR("Unsupported op:{}", static_cast<int>(op));
      return -1;
    }
  }
  return 0;
}

template <typename T>
static std::optional<T> do_constant_arithmetic_op(OpToken op, DType dtype, uint64_t left_bin, uint64_t right_bin) {
  auto left_v = dtype.ToPrimitiveValue<T>(left_bin);
  auto right_v = dtype.ToPrimitiveValue<T>(right_bin);
  if (!left_v || !right_v) {
    RUDF_ERROR("Failed to convert to primitive value for dtype:{}", dtype);
    return {};
  }
  auto left = *left_v;
  auto right = *right_v;
  switch (op) {
    case OP_PLUS: {
      return left + right;
    }
    case OP_MINUS: {
      return left - right;
    }
    case OP_MULTIPLY: {
      return left * right;
    }
    case OP_MOD: {
      if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
        return {};
      } else {
        return left % right;
      }
    }
    case OP_DIVIDE: {
      return left / right;
    }
    default: {
      RUDF_ERROR("Unsupported op:{} on dtype:{}", op, dtype);
      return {};
    }
  }
  return {};
}

int arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Operand& left,
                  const Xbyak::Operand& right, const Xbyak::Operand& result) {
  if (!is_valid_operand(left) || !is_valid_operand(right) || !is_valid_operand(result)) {
    return -1;
  }
  uint32_t bits = dtype.Bits();
  RUDF_DEBUG("op:{}, result bits:{}  left_reg:{}, left_xmm:{},right_reg:{}, right_xmm:{}", op, bits, left.isREG(),
             left.isXMM(), right.isREG(), right.isXMM());
  if (dtype == DATA_F32 || dtype == DATA_F64) {
    copy_value(c, dtype, left, xmm0);
    copy_value(c, dtype, right, xmm1);
    int rc = do_xmm_register_arithmetic_op(c, op, dtype, xmm0, xmm1);
    if (rc == 0) {
      copy_value(c, dtype, xmm0, result);
    }
    return rc;
  } else {
    if (right.isXMM()) {
      copy_value(c, dtype, right, rcx.changeBit(bits));
      return arithmetic_op(c, op, dtype, left, rcx.changeBit(bits), result);
    }
    if (!is_rax(left)) {
      copy_value(c, dtype, left, rax.changeBit(bits));
    }
    int rc = 0;
    if (right.getBit() != bits && right.isREG()) {
      rc = do_regular_register_arithmetic_op(c, op, dtype, rax.changeBit(bits),
                                             reinterpret_cast<const Xbyak::Reg&>(right).changeBit(bits));
    } else {
      rc = do_regular_register_arithmetic_op(c, op, dtype, rax.changeBit(bits), right);
    }
    if (rc != 0) {
      return rc;
    }
    copy_value(c, dtype, rax.changeBit(bits), result);
    return 0;
  }
}

int arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, const Xbyak::Operand& left, uint64_t right,
                  const Xbyak::Operand& result) {
  uint32_t bits = dtype.Bits();
  c.mov(rcx.changeBit(bits), right);
  return arithmetic_op(c, op, dtype, left, rcx.changeBit(bits), result);
}

int arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, const Xbyak::Operand& right,
                  const Xbyak::Operand& result) {
  uint32_t bits = dtype.Bits();
  c.mov(rcx.changeBit(bits), left);
  return arithmetic_op(c, op, dtype, rcx.changeBit(bits), right, result);
}

int arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right, uint64_t& result) {
  std::optional<uint64_t> result_bin;
  switch (dtype.GetFundamentalType()) {
    case DATA_F32: {
      auto v = do_constant_arithmetic_op<float>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_F64: {
      auto v = do_constant_arithmetic_op<double>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_U64: {
      auto v = do_constant_arithmetic_op<uint64_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_U32: {
      auto v = do_constant_arithmetic_op<uint32_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_I32: {
      auto v = do_constant_arithmetic_op<int32_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_U16: {
      auto v = do_constant_arithmetic_op<uint16_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_I16: {
      auto v = do_constant_arithmetic_op<int16_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_U8: {
      auto v = do_constant_arithmetic_op<uint8_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    case DATA_I8: {
      auto v = do_constant_arithmetic_op<int8_t>(op, dtype, left, right);
      if (!v) {
        return -1;
      }
      result_bin = dtype.FromPrimitiveValue(*v);
      break;
    }
    default: {
      RUDF_ERROR("Unsupported dtype:{} for constants arithmetic op.", dtype);
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

int arithmetic_op(Xbyak::CodeGenerator& c, OpToken op, DType dtype, uint64_t left, uint64_t right,
                  const Xbyak::Operand& result) {
  uint64_t result_bin;
  if (0 != arithmetic_op(c, op, dtype, left, right, result_bin)) {
    return -1;
  }
  return copy_value(c, dtype, result_bin, result);
}

}  // namespace rapidudf