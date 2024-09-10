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
#include "rapidudf/jit/xbyak/ops/copy.h"
#include <cassert>

#include "xbyak/xbyak_util.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"

namespace rapidudf {
using namespace Xbyak::util;
int copy_value(Xbyak::CodeGenerator& c, DType dtype, uint64_t bin, const Xbyak::Operand& dst, bool is_ptr) {
  uint32_t bits = dtype.Bits();

  if (dst.isXMM()) {
    if (is_ptr) {
      c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(dst));
      c.mov(qword[rax], bin);
    } else {
      c.mov(rax, bin);
      if (bits == 64) {
        c.movq(reinterpret_cast<const Xbyak::Xmm&>(dst), rax);
      } else {
        c.movd(reinterpret_cast<const Xbyak::Xmm&>(dst), eax);
      }
    }
    return 0;
  } else if (dst.isREG()) {
    if (is_ptr) {
      c.mov(ptr[reinterpret_cast<const Xbyak::Reg&>(dst)], bin);
    } else {
      c.mov(dst, bin);
    }
    return 0;
  } else if (dst.isMEM()) {
    if (is_ptr) {
      c.mov(rax, dst);
      c.mov(ptr[rax], bin);
    } else {
      c.mov(rax, bin);
      if (dst.getBit() < 64) {
        c.mov(dst, rax.changeBit(dst.getBit()));
      } else {
        c.mov(dst, rax);
      }
    }
    return 0;
  } else {
    RUDF_ERROR("Unsupporte dst register kind:{} to mov value.", static_cast<int>(dst.getKind()));
    return -1;
  }
}
static int register_copy_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Reg& src, const Xbyak::Reg& dst,
                               bool is_ptr) {
  uint32_t src_bits = dtype.Bits();
  if (dst.isREG()) {
    if (src.isREG()) {
      if (is_ptr) {
        c.mov(ptr[dst], src);
      } else {
        c.mov(dst.changeBit(src.getBit()), src);
      }
    } else if (src.isXMM()) {
      if (src_bits == 64) {
        if (is_ptr) {
          c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(src));
          c.mov(ptr[dst], rax);
        } else {
          auto dst_tmp = dst.changeBit(64);
          c.movq(reinterpret_cast<const Xbyak::Reg64&>(dst_tmp), reinterpret_cast<const Xbyak::Xmm&>(src));
        }
      } else {
        if (is_ptr) {
          c.movd(eax, reinterpret_cast<const Xbyak::Xmm&>(src));
          c.mov(ptr[dst], eax);
        } else {
          auto dst_tmp = dst.changeBit(32);
          c.movd(reinterpret_cast<const Xbyak::Reg32&>(dst_tmp), reinterpret_cast<const Xbyak::Xmm&>(src));
        }
      }
    } else {
      RUDF_ERROR("Source reg invalid kind:{}.", static_cast<int>(src.getKind()));
      return -1;
    }
    return 0;
  } else if (dst.isXMM()) {
    if (src.isREG()) {
      if (is_ptr) {
        c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(dst));
        c.mov(ptr[rax], src);
      } else {
        if (src.getBit() == 64) {
          c.movq(reinterpret_cast<const Xbyak::Xmm&>(dst), reinterpret_cast<const Xbyak::Reg64&>(src));
        } else {
          auto tmp = src.changeBit(32);
          c.movd(reinterpret_cast<const Xbyak::Xmm&>(dst), reinterpret_cast<const Xbyak::Reg32&>(tmp));
        }
      }
    } else if (src.isXMM()) {
      if (is_ptr) {
        c.movq(rcx, reinterpret_cast<const Xbyak::Xmm&>(src));
        c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(dst));
        c.mov(ptr[rax], rcx.changeBit(dtype.Bits()));
      } else {
        c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(src));
        c.movq(reinterpret_cast<const Xbyak::Xmm&>(dst), rax);
      }
    } else {
      RUDF_ERROR("Source reg invalid kind:{}.", static_cast<int>(src.getKind()));
      return -1;
    }
    return 0;
  } else {
    RUDF_ERROR("Unsupporte dst register kind:{} to mov value.", static_cast<int>(dst.getKind()));
    return -1;
  }
}
static int stack_register_copy_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Address& src_addr,
                                     const Xbyak::Reg& dst, bool is_ptr) {
  if (dst.isREG()) {
    if (is_ptr) {
      c.mov(ptr[dst], src_addr);
    } else {
      c.mov(dst.changeBit(src_addr.getBit()), src_addr);
    }
    return 0;
  } else if (dst.isXMM()) {
    if (is_ptr) {
      c.movq(rax, reinterpret_cast<const Xbyak::Xmm&>(dst));
      c.mov(ptr[rax], src_addr);
    } else {
      if (src_addr.getBit() == 64) {
        c.movq(reinterpret_cast<const Xbyak::Xmm&>(dst), src_addr);
      } else {
        c.movd(reinterpret_cast<const Xbyak::Xmm&>(dst), src_addr);
      }
    }

    return 0;
  } else {
    RUDF_ERROR("Unsupporte dst register kind:{} to mov value.", static_cast<int>(dst.getKind()));
    return -1;
  }
}
static int register_stack_copy_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Reg& src,
                                     const Xbyak::Address& dst, bool is_ptr) {
  if (src.isREG()) {
    if (is_ptr) {
      c.mov(rax, dst);
      c.mov(ptr[rax], src.changeBit(dst.getBit()));
    } else {
      c.mov(dst, src.changeBit(dst.getBit()));
    }
    return 0;
  } else if (src.isXMM()) {
    if (is_ptr) {
      c.mov(rax, dst);
      c.movq(rcx, reinterpret_cast<const Xbyak::Xmm&>(src));
      c.mov(ptr[rax], rcx.changeBit(dtype.Bits()));
    } else {
      if (dst.getBit() == 64) {
        c.movq(dst, reinterpret_cast<const Xbyak::Xmm&>(src));
      } else {
        c.movd(dst, reinterpret_cast<const Xbyak::Xmm&>(src));
      }
    }

    return 0;
  } else {
    RUDF_ERROR("Unsupporte dst register kind:{} to mov value.", static_cast<int>(dst.getKind()));
    return -1;
  }
}

static int stack_stack_copy_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Address& src_addr,
                                  const Xbyak::Address& dst, bool is_ptr) {
  if (is_ptr) {
    c.mov(rax, dst);
    c.mov(ptr[rax], src_addr);
  } else {
    c.mov(rax.changeBit(src_addr.getBit()), src_addr);
    c.mov(dst, rax.changeBit(src_addr.getBit()));
  }
  return 0;
}

int copy_value(Xbyak::CodeGenerator& c, DType dtype, const Xbyak::Operand& src, const Xbyak::Operand& dst,
               bool is_ptr) {
  if ((src.isREG() || src.isXMM()) && (dst.isREG() || dst.isXMM())) {
    return register_copy_value(c, dtype, reinterpret_cast<const Xbyak::Reg&>(src),
                               reinterpret_cast<const Xbyak::Reg&>(dst), is_ptr);
  }
  if ((src.isMEM()) && (dst.isREG() || dst.isXMM())) {
    return stack_register_copy_value(c, dtype, reinterpret_cast<const Xbyak::Address&>(src),
                                     reinterpret_cast<const Xbyak::Reg&>(dst), is_ptr);
  }
  if ((src.isREG() || src.isXMM()) && (dst.isMEM())) {
    return register_stack_copy_value(c, dtype, reinterpret_cast<const Xbyak::Reg&>(src),
                                     reinterpret_cast<const Xbyak::Address&>(dst), is_ptr);
  }
  if (src.isMEM() && dst.isMEM()) {
    return stack_stack_copy_value(c, dtype, reinterpret_cast<const Xbyak::Address&>(src),
                                  reinterpret_cast<const Xbyak::Address&>(dst), is_ptr);
  }
  RUDF_ERROR("Unsupported src kind:{}, dst kind:{} comination.", static_cast<int>(src.getKind()),
             static_cast<int>(dst.getKind()));
  return -1;
}
}  // namespace rapidudf