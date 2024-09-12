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
#include "rapidudf/jit/xbyak/ops/bits_ops.h"
#include <cstdint>
#include "rapidudf/log/log.h"
namespace rapidudf {
namespace xbyak {
using namespace Xbyak::util;
int bits_set(Xbyak::CodeGenerator& c, const Xbyak::Operand& op, uint8_t n) {
  if (n >= 64) {
    RUDF_ERROR("Can NOT bits set with bit offset:{}", n);
    return -1;
  }
  uint64_t or_val = (1 << n);
  if (op.isREG()) {
    const Xbyak::Reg& reg = reinterpret_cast<const Xbyak::Reg&>(op);
    if (or_val <= INT32_MAX) {
      c.or_(reg.changeBit(32), or_val);
    } else {
      c.mov(rax, or_val);
      c.or_(reg.changeBit(64), rax);
    }
    return 0;
  } else if (op.isXMM()) {
    const Xbyak::Xmm& xmm_reg = reinterpret_cast<const Xbyak::Xmm&>(op);
    c.movq(rax, xmm_reg);
    if (or_val <= INT32_MAX) {
      c.or_(rax.changeBit(32), or_val);
    } else {
      c.mov(rdx, or_val);
      c.or_(rax, rdx);
    }
    c.movq(xmm_reg, rax);
    return 0;
  } else if (op.isMEM()) {
    const Xbyak::Address& addr = reinterpret_cast<const Xbyak::Address&>(op);
    if (or_val <= INT32_MAX) {
      c.or_(addr, or_val);
    } else {
      c.mov(rax, or_val);
      c.or_(addr, rax);
    }
    return 0;
  } else {
    RUDF_ERROR("Can NOT bits set");
    return -1;
  }
}
int bits_clear(Xbyak::CodeGenerator& c, const Xbyak::Operand& op, uint8_t n) {
  if (n >= 64) {
    RUDF_ERROR("Can NOT bits clear with bit offset:{}", n);
    return -1;
  }
  uint64_t mask = ~(1ULL << n);
  if (op.isREG()) {
    const Xbyak::Reg& reg = reinterpret_cast<const Xbyak::Reg&>(op);
    c.mov(rax, mask);
    c.and_(reg.changeBit(64), rax);
    return 0;
  } else if (op.isXMM()) {
    const Xbyak::Xmm& xmm_reg = reinterpret_cast<const Xbyak::Xmm&>(op);
    c.movq(rax, xmm_reg);
    c.mov(rdx, mask);
    c.and_(rax, rdx);
    c.movq(xmm_reg, rax);
    return 0;
  } else if (op.isMEM()) {
    const Xbyak::Address& addr = reinterpret_cast<const Xbyak::Address&>(op);
    c.mov(rax, mask);
    c.and_(addr, rax);
    return 0;
  } else {
    RUDF_ERROR("Can NOT bits clear");
    return -1;
  }
}
}  // namespace xbyak
}  // namespace rapidudf