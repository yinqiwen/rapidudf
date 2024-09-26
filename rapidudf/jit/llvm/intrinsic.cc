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
#include "llvm/Config/llvm-config.h"
#include "rapidudf/jit/llvm/jit.h"
namespace rapidudf {
namespace llvm {
bool JitCompiler::HasIntrinsic(OpToken op) {
  switch (op) {
    case OP_NEGATIVE:
    case OP_NOT:
    case OP_SIN:
    case OP_COS:
    case OP_FLOOR:
    case OP_ABS:
    case OP_SQRT:
    case OP_CEIL:
    case OP_ROUND:
    case OP_EXP:
    case OP_EXP2:
    case OP_LOG:
    case OP_LOG2:
    case OP_LOG10:
    case OP_RINT:
    case OP_TRUNC:
    case OP_FMA:
    case OP_POW:
    case OP_MAX:
    case OP_MIN: {
      return true;
    }
#if LLVM_VERSION_MAJOR >= 19
    case OP_TAN:
    case OP_ASIN:
    case OP_ACOS:
    case OP_ATAN:
    case OP_SINH:
    case OP_COSH:
    case OP_TANH: {
      return true;
    }
#endif
    default: {
      return false;
    }
  }
}
}  // namespace llvm
}  // namespace rapidudf