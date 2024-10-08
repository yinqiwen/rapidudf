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
#include <cstdint>
// #include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {

struct EvalValue {
  uint64_t dtype;
  union {
    simd::VectorData vector;
    StringView scalar_sv;
    uint64_t scalar_u64;
    int64_t scalar_i64;
    uint32_t scalar_u32;
    int32_t scalar_i32;
    uint16_t scalar_u16;
    int16_t scalar_i16;
    uint8_t scalar_u8;
    int8_t scalar_i8;
    double scalar_f64;
    float scalar_f32;
    bool scalar_bv;
    simd::Column* column;
    uint32_t op;
  };
  EvalValue() : dtype(0) {}
  explicit EvalValue(OpToken v) {
    dtype = 0;
    op = static_cast<uint32_t>(v);
  }
  explicit EvalValue(uint64_t dtype, simd::VectorData v) {
    this->dtype = dtype;
    vector = v;
  }
  template <typename T>
  simd::Vector<T> ToVector() {
    return simd::Vector<T>(vector);
  }
};

}  // namespace rapidudf