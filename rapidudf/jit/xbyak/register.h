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
#include "xbyak/xbyak.h"
namespace rapidudf {

struct RegisterId {
  Xbyak::Operand::Kind kind;
  uint32_t index = 0;
  RegisterId(const Xbyak::Operand& reg) : kind(reg.getKind()), index(reg.getIdx()) {}
  bool operator==(const RegisterId& other) const { return kind == other.kind && index == other.index; }
};

class Register {
 public:
  Register(Xbyak::CodeGenerator& code_gen, const Xbyak::Reg* reg, bool high = false);
  int Mov(const Register& src, uint32_t bits);
  bool IsRegular() const { return reg_->isREG(); }
  bool IsXmmHigh() const { return reg_->isXMM() && is_xmm_high_; }
  bool IsXmmLow() const { return reg_->isXMM() && is_xmm_low_; }
  const Xbyak::Reg* Raw() const { return reg_; }

 private:
  Xbyak::CodeGenerator& code_gen_;
  const Xbyak::Reg* reg_ = nullptr;
  bool is_xmm_high_ = false;
  bool is_xmm_low_ = false;
};
}  // namespace rapidudf