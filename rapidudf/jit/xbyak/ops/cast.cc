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
#include "rapidudf/jit/xbyak/ops/cast.h"
#include <cassert>

#include "xbyak/xbyak_util.h"

#include "rapidudf/jit/xbyak/ops/copy.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"

namespace rapidudf {
namespace xbyak {
using namespace Xbyak::util;
static void stack_cast_int_to_int(Xbyak::CodeGenerator& c, Xbyak::Address src_addr, Xbyak::Address dst_addr) {
  c.mov(rax.changeBit(src_addr.getBit()), src_addr);
  if (dst_addr.getBit() > src_addr.getBit()) {
    if (src_addr.getBit() == 8) {
      c.cbw();
    }
    if (src_addr.getBit() <= 16 && dst_addr.getBit() >= 32) {
      c.cwde();
    }
    if (src_addr.getBit() <= 32 && dst_addr.getBit() == 64) {
      c.cdqe();
    }
  }
  c.mov(dst_addr, rax.changeBit(dst_addr.getBit()));
}

static void stack_cast_float_to_int(Xbyak::CodeGenerator& c, Xbyak::Address src_addr, Xbyak::Address dst_addr) {
  if (dst_addr.getBit() < 64) {
    if (src_addr.getBit() == 64) {
      c.cvttsd2si(eax, src_addr);
    } else {
      c.cvttss2si(eax, src_addr);
    }
  } else {
    if (src_addr.getBit() == 64) {
      c.cvttsd2si(rax, src_addr);
    } else {
      c.cvttss2si(rax, src_addr);
    }
  }
  c.mov(dst_addr, rax.changeBit(dst_addr.getBit()));
}

static void stack_cast_float_to_float(Xbyak::CodeGenerator& c, Xbyak::Address src_addr, Xbyak::Address dst_addr) {
  c.pxor(xmm0, xmm0);
  if (src_addr.getBit() == 32) {
    c.cvtss2sd(xmm0, src_addr);
    c.movsd(dst_addr, xmm0);
  } else {
    c.cvtsd2ss(xmm0, src_addr);
    c.movss(dst_addr, xmm0);
  }
}
static void stack_cast_int_to_float(Xbyak::CodeGenerator& c, Xbyak::Address src_addr, Xbyak::Address dst_addr) {
  c.pxor(xmm0, xmm0);
  if (src_addr.getBit() < 32) {
    c.mov(rax.changeBit(src_addr.getBit()), src_addr);
    if (src_addr.getBit() < 16) {
      c.cbw();
    }
    c.cwde();
  }

  if (dst_addr.getBit() == 32) {
    if (src_addr.getBit() < 32) {
      c.cvtsi2ss(xmm0, eax);
    } else {
      c.cvtsi2ss(xmm0, src_addr);
    }
    c.movss(dst_addr, xmm0);
  } else {
    if (src_addr.getBit() < 32) {
      c.cvtsi2sd(xmm0, eax);
    } else {
      c.cvtsi2sd(xmm0, src_addr);
    }
    c.movsd(dst_addr, xmm0);
  }
}

static void register_cast_int_to_float(Xbyak::CodeGenerator& c, const Xbyak::Reg& reg, DType src_dtype,
                                       DType dst_dtype) {
  uint32_t dst_len = dst_dtype.ByteSize();
  uint32_t dst_bits = dst_len * 8;
  uint32_t src_len = src_dtype.ByteSize();
  uint32_t src_bits = src_len * 8;
  c.mov(rax.changeBit(src_bits), reg);
  if (dst_len > src_len) {
    if (dst_bits >= 16 && src_bits == 8) {
      c.cbw();
    }
    if (dst_bits >= 32 && src_bits <= 16) {
      c.cwde();
    }
    if (dst_bits >= 64 && src_bits <= 32) {
      c.cdqe();
    }
  }
  if (src_bits < 64) {
    src_bits = 32;
  }
  if (dst_bits == 32) {
    c.cvtsi2ss(xmm0, rax.changeBit(src_bits));
    auto new_reg = reg.changeBit(32);
    c.movd(reinterpret_cast<const Xbyak::Reg32&>(new_reg), xmm0);
  } else {
    c.cvtsi2sd(xmm0, rax.changeBit(src_bits));
    auto new_reg = reg.changeBit(64);
    c.movq(reinterpret_cast<const Xbyak::Reg64&>(new_reg), xmm0);
  }
}

static void register_cast_int_to_int(Xbyak::CodeGenerator& c, const Xbyak::Reg& reg, DType src_dtype, DType dst_dtype) {
  uint32_t dst_len = dst_dtype.ByteSize();
  uint32_t dst_bits = dst_len * 8;
  uint32_t src_len = src_dtype.ByteSize();
  uint32_t src_bits = src_len * 8;
  if (dst_len > src_len) {
    c.mov(rax.changeBit(src_bits), reg);
    if (dst_bits >= 16 && src_bits == 8) {
      c.cbw();
    }
    if (dst_bits >= 32 && src_bits <= 16) {
      c.cwde();
    }
    if (dst_bits >= 64 && src_bits <= 32) {
      c.cdqe();
    }
    c.mov(reg.changeBit(dst_bits), rax.changeBit(dst_bits));
    return;
  }
}

static void register_cast_float_to_int(Xbyak::CodeGenerator& c, const Xbyak::Reg& reg, DType src_dtype,
                                       DType dst_dtype) {
  uint32_t dst_bits = dst_dtype.Bits();
  uint32_t src_bits = src_dtype.Bits();
  auto tmp_reg = reg.changeBit(src_bits);

  if (src_dtype == DATA_F32) {
    c.movd(xmm0, reinterpret_cast<const Xbyak::Reg32&>(tmp_reg));
  } else {
    c.movq(xmm0, reinterpret_cast<const Xbyak::Reg64&>(tmp_reg));
  }
  auto tmp_dst_reg = reg.changeBit(dst_bits < 64 ? 32 : 64);
  if (src_bits < 64) {
    c.cvttss2si(tmp_dst_reg, xmm0);
  } else {
    c.cvttsd2si(tmp_dst_reg, xmm0);
  }
}

static void register_cast_float_to_float(Xbyak::CodeGenerator& c, const Xbyak::Reg& reg, DType src_dtype,
                                         DType dst_dtype) {
  c.pxor(xmm0, xmm0);
  if (reg.getBit() == 32) {
    c.movd(xmm1, reinterpret_cast<const Xbyak::Reg32&>(reg));
    c.cvtss2sd(xmm0, xmm1);
    auto tmp_dst_reg = reg.changeBit(64);
    c.movq(reinterpret_cast<const Xbyak::Reg64&>(tmp_dst_reg), xmm0);
  } else {
    c.movq(xmm1, reinterpret_cast<const Xbyak::Reg64&>(reg));
    c.cvtsd2ss(xmm0, xmm1);
    auto tmp_dst_reg = reg.changeBit(32);
    c.movd(reinterpret_cast<const Xbyak::Reg32&>(tmp_dst_reg), xmm0);
  }
}

int static_cast_value(Xbyak::CodeGenerator& c, const Xbyak::Reg& reg, DType src_dtype, DType dst_dtype) {
  if (!reg.isREG()) {
    // RUDF_ERROR("Can NOT 'static_cast' on non regular register:{}", static_cast<int>(reg.getKind()));
    // return -1;
    copy_value(c, src_dtype, reg, rax.changeBit(src_dtype.Bits()));
    int rc = static_cast_value(c, rax.changeBit(src_dtype.Bits()), src_dtype, dst_dtype);
    if (0 != rc) {
      return rc;
    }
    copy_value(c, dst_dtype, rax.changeBit(dst_dtype.Bits()), reg);
    return 0;
  }
  if (src_dtype == dst_dtype) {
    return 0;
  }
  switch (src_dtype.GetFundamentalType()) {
    case DATA_U8:
    case DATA_U16:
    case DATA_U32:
    case DATA_U64:
    case DATA_I8:
    case DATA_I16:
    case DATA_I32:
    case DATA_I64: {
      if (dst_dtype == DATA_F32 || dst_dtype == DATA_F64) {
        register_cast_int_to_float(c, reg, src_dtype, dst_dtype);
      } else {
        register_cast_int_to_int(c, reg, src_dtype, dst_dtype);
      }
      break;
    }
    case DATA_F32:
    case DATA_F64: {
      if (dst_dtype == DATA_F64 || dst_dtype == DATA_F32) {
        register_cast_float_to_float(c, reg, src_dtype, dst_dtype);
      } else {
        register_cast_float_to_int(c, reg, src_dtype, dst_dtype);
      }

      break;
    }
    default: {
      abort();
    }
  }
  return 0;
}
int static_cast_value(Xbyak::CodeGenerator& c, Xbyak::Address src_addr, DType src_dtype, Xbyak::Address dst_addr,
                      DType dst_dtype) {
  if (src_dtype == dst_dtype) {
    return 0;
  }
  switch (src_dtype.GetFundamentalType()) {
    case DATA_U8:
    case DATA_U16:
    case DATA_U32:
    case DATA_U64:
    case DATA_I8:
    case DATA_I16:
    case DATA_I32:
    case DATA_I64: {
      if (dst_dtype == DATA_F32 || dst_dtype == DATA_F64) {
        stack_cast_int_to_float(c, src_addr, dst_addr);
      } else {
        stack_cast_int_to_int(c, src_addr, dst_addr);
      }
      break;
    }
    case DATA_F32:
    case DATA_F64: {
      if (dst_dtype == DATA_F64 || dst_dtype == DATA_F32) {
        stack_cast_float_to_float(c, src_addr, dst_addr);
      } else {
        stack_cast_float_to_int(c, src_addr, dst_addr);
      }
      break;
    }
    default: {
      abort();
    }
  }
  return 0;
}
}  // namespace xbyak
}  // namespace rapidudf