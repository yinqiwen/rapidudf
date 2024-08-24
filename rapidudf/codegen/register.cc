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
#include "rapidudf/codegen/register.h"
#include "rapidudf/log/log.h"
namespace rapidudf {
using namespace Xbyak::util;

static void mov_reg_to_reg(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  code_gen->mov(*dst, *src);
}
static void mov_reg_to_xmm_low_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  code_gen->movq(reinterpret_cast<const Xbyak::Xmm&>(*dst), reinterpret_cast<const Xbyak::Reg64&>(*src));
}
static void mov_reg_to_xmm_low_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  auto tmp = src->changeBit(32);
  code_gen->movd(reinterpret_cast<const Xbyak::Xmm&>(*dst), reinterpret_cast<const Xbyak::Reg32&>(tmp));
}
static void mov_reg_to_xmm_high_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  code_gen->pinsrq(reinterpret_cast<const Xbyak::Xmm&>(*dst), reinterpret_cast<const Xbyak::Reg64&>(*src), 1);
}
static void mov_reg_to_xmm_high_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  auto tmp = src->changeBit(32);
  code_gen->pinsrd(reinterpret_cast<const Xbyak::Xmm&>(*dst), reinterpret_cast<const Xbyak::Reg32&>(tmp), 2);
}
static void mov_xmm_low_to_reg_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  auto tmp = dst->changeBit(32);
  code_gen->movd(reinterpret_cast<const Xbyak::Reg32&>(tmp), reinterpret_cast<const Xbyak::Xmm&>(*src));
}
static void mov_xmm_low_to_reg_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  code_gen->movq(reinterpret_cast<const Xbyak::Reg64&>(*dst), reinterpret_cast<const Xbyak::Xmm&>(*src));
}
static void mov_xmm_high_to_reg_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  code_gen->pextrq(reinterpret_cast<const Xbyak::Reg64&>(*dst), reinterpret_cast<const Xbyak::Xmm&>(*src), 1);
}
static void mov_xmm_high_to_reg_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  auto tmp = dst->changeBit(32);

  code_gen->pextrd(reinterpret_cast<const Xbyak::Reg32&>(tmp), reinterpret_cast<const Xbyak::Xmm&>(*src), 2);
}
static void mov_xmm_low_to_xmm_low_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_low_to_reg_32(code_gen, &rax, src);
  mov_reg_to_xmm_low_32(code_gen, dst, &rax);
}
static void mov_xmm_low_to_xmm_low_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_low_to_reg_64(code_gen, &rax, src);
  mov_reg_to_xmm_low_64(code_gen, dst, &rax);
}
static void mov_xmm_low_to_xmm_high_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_low_to_reg_32(code_gen, &rax, src);
  mov_reg_to_xmm_high_32(code_gen, dst, &rax);
}
static void mov_xmm_low_to_xmm_high_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_low_to_reg_64(code_gen, &rax, src);
  mov_reg_to_xmm_high_64(code_gen, dst, &rax);
}

static void mov_xmm_high_to_xmm_high_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_high_to_reg_32(code_gen, &rax, src);
  mov_reg_to_xmm_high_32(code_gen, dst, &rax);
}
static void mov_xmm_high_to_xmm_high_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_high_to_reg_64(code_gen, &rax, src);
  mov_reg_to_xmm_high_64(code_gen, dst, &rax);
}
static void mov_xmm_high_to_xmm_low_32(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_high_to_reg_32(code_gen, &rax, src);
  mov_reg_to_xmm_low_32(code_gen, dst, &rax);
}
static void mov_xmm_high_to_xmm_low_64(Xbyak::CodeGenerator* code_gen, const Xbyak::Reg* dst, const Xbyak::Reg* src) {
  mov_xmm_high_to_reg_64(code_gen, &rax, src);
  mov_reg_to_xmm_low_64(code_gen, dst, &rax);
}

Register::Register(Xbyak::CodeGenerator& code_gen, const Xbyak::Reg* reg, bool high) : code_gen_(code_gen) {
  reg_ = reg;
  if (reg->isXMM()) {
    if (high) {
      is_xmm_high_ = true;
      is_xmm_low_ = false;
    } else {
      is_xmm_high_ = false;
      is_xmm_low_ = true;
    }
  }
}
int Register::Mov(const Register& src, uint32_t bits) {
  printf("###mov src xmm_h:%d, xmm_L:%d, regular:%d\n", src.IsXmmHigh(), src.IsXmmLow(), src.IsRegular());
  printf("###mov dst xmm_h:%d, xmm_L:%d, regular:%d\n", IsXmmHigh(), IsXmmLow(), IsRegular());
  if (bits == 64) {
    if (IsRegular()) {
      if (src.IsRegular()) {
        mov_reg_to_reg(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        printf("###mov_xmm_high_to_reg_64\n");
        mov_xmm_high_to_reg_64(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_reg_64(&code_gen_, reg_, src.reg_);
      } else {
        printf("####00\n");
        return -1;
      }
    } else if (IsXmmHigh()) {
      if (src.IsRegular()) {
        mov_reg_to_xmm_high_64(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        mov_xmm_high_to_xmm_high_64(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_xmm_high_64(&code_gen_, reg_, src.reg_);
      } else {
        printf("####01\n");
        return -1;
      }
    } else if (IsXmmLow()) {
      if (src.IsRegular()) {
        mov_reg_to_xmm_low_64(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        mov_xmm_high_to_xmm_low_64(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_xmm_low_64(&code_gen_, reg_, src.reg_);
      } else {
        printf("####02\n");
        return -1;
      }
    } else {
      printf("####03\n");
      return -1;
    }
  } else if (bits == 32 || bits == 16 || bits == 8) {
    if (IsRegular()) {
      if (src.IsRegular()) {
        mov_reg_to_reg(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        printf("###movmov_xmm_high_to_reg_32\n");
        mov_xmm_high_to_reg_32(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_reg_32(&code_gen_, reg_, src.reg_);
      } else {
        printf("####10\n");
        return -1;
      }
    } else if (IsXmmHigh()) {
      if (src.IsRegular()) {
        mov_reg_to_xmm_high_32(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        mov_xmm_high_to_xmm_high_32(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_xmm_high_32(&code_gen_, reg_, src.reg_);
      } else {
        printf("####11\n");
        return -1;
      }
    } else if (IsXmmLow()) {
      if (src.IsRegular()) {
        mov_reg_to_xmm_low_32(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmHigh()) {
        mov_xmm_high_to_xmm_low_32(&code_gen_, reg_, src.reg_);
      } else if (src.IsXmmLow()) {
        mov_xmm_low_to_xmm_low_32(&code_gen_, reg_, src.reg_);
      } else {
        printf("####12\n");
        return -1;
      }
    } else {
      printf("####13\n");
      return -1;
    }
  } else {
    RUDF_ERROR("Unsupported bits:{} to mov", bits);
    return -1;
  }
  return 0;
}
}  // namespace rapidudf