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
#include <vector>
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace simd {

template <typename T>
struct EvalOperand {
  void* vector_ptr = nullptr;
  T scalar;
  bool is_vector = false;
  bool operator==(const EvalOperand<T>& other) const {
    if (is_vector != other.is_vector) {
      return false;
    }
    if (is_vector) {
      return vector_ptr == other.vector_ptr;
    } else {
      return scalar == other.scalar;
    }
  }
};

template <typename T>
struct EvalSubTree {
  EvalOperand<T> operands[3];
  uint8_t count = 0;
  OpToken op = OP_INVALID;

  bool operator==(const EvalSubTree<T>& other) const {
    if (op != other.op) {
      return false;
    }
    for (uint8_t i = 0; i < count; i++) {
      bool v = (operands[i] == other.operands[i]);
      if (!v) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace simd
}  // namespace rapidudf