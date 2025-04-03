/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "hwy/bit_set.h"
#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector_op.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {
namespace functions {
void simd_vector_bits_not(Vector<Bit> src, Vector<Bit> dst);
void simd_vector_bits_and(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst);
void simd_vector_bits_or(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst);
void simd_vector_bits_xor(Vector<Bit> left, Vector<Bit> right, Vector<Bit> dst);
size_t simd_vector_bits_count_true(Vector<Bit> src);

}  // namespace functions

namespace bits {
class MaskIterator {
 public:
  inline MaskIterator() noexcept = default;

  inline explicit MaskIterator(uint64_t mask) noexcept { bitset_.Set(mask); }

  explicit operator bool() const noexcept { return bitset_.Any(); }

  size_t Advance() noexcept {
    size_t n = bitset_.First();
    bitset_.Clear(n);
    return n;
  }

 private:
  hwy::BitSet64 bitset_;
};
}  // namespace bits
}  // namespace rapidudf