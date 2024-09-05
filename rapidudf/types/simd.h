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
#include <stddef.h>
#include <cstdint>
#include <vector>

#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
namespace simd {

// struct Bit {
//   bool val;
//   Bit(bool v) : val(v) {}
//   Bit operator+(const Bit& other) const { return *this; }
//   Bit operator-(const Bit& other) const { return *this; }
//   Bit operator*(const Bit& other) const { return *this; }
//   Bit operator/(const Bit& other) const { return *this; }
//   Bit operator%(const Bit& other) const { return *this; }
//   Bit operator==(const Bit& other) const { return Bit(val == other.val); }
//   Bit operator!=(const Bit& other) const { return Bit(val != other.val); }
//   Bit operator>=(const Bit& other) const { return Bit(val >= other.val); }
//   Bit operator<=(const Bit& other) const { return Bit(val <= other.val); }
//   Bit operator>(const Bit& other) const { return Bit(val > other.val); }
//   Bit operator<(const Bit& other) const { return Bit(val < other.val); }
//   Bit operator&&(const Bit& other) const { return Bit(val && other.val); }
//   Bit operator||(const Bit& other) const { return Bit(val || other.val); }
//   operator bool() { return val; }
// };
template <typename T>
struct InternalType {
  using internal_type = T;
};

template <>
struct InternalType<Bit> {
  using internal_type = uint8_t;
};
template <typename T>
class Vector;

class VectorData {
 public:
  VectorData(const void* data = nullptr, size_t size = 0) : data_(data), size_(size) {}
  size_t Size() const { return size_; }
  const void* Data() const { return data_; }

 private:
  const void* data_ = nullptr;
  size_t size_;
};

template <typename T>
class Vector {
 public:
  using value_type = T;
  Vector() {}
  Vector(const T* data, size_t size) {
    VectorData vdata(data, size);
    vec_data_ = vdata;
  }
  Vector(VectorData vdata) { vec_data_ = vdata; }
  Vector(const std::vector<T>& vec) {
    VectorData vdata(vec.data(), vec.size());
    vec_data_ = vdata;
  }
  size_t Size() const { return vec_data_.Size(); }
  auto Data() const {
    if constexpr (std::is_same_v<Bit, T>) {
      return reinterpret_cast<const uint8_t*>(vec_data_.Data());
    } else {
      return reinterpret_cast<const T*>(vec_data_.Data());
    }
  }
  T operator[](size_t idx) const {
    if constexpr (std::is_same_v<Bit, T>) {
      size_t byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t bits = *(reinterpret_cast<const uint8_t*>(vec_data_.Data()) + byte_idx);
      uint8_t v = (bits >> bit_cursor) & 1;
      return Bit(v > 0);
    } else {
      return Data()[idx];
    }
  }

 private:
  VectorData vec_data_;
};

}  // namespace simd

}  // namespace rapidudf