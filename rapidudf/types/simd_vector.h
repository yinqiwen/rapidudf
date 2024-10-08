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
  VectorData(const void* data = nullptr, size_t size = 0, size_t bytes_capacity = 0)
      : temporary_(0), size_(size), reserved_(0), bytes_capacity_(bytes_capacity), data_(data) {}
  inline size_t Size() const { return size_; }
  inline size_t BytesCapacity() const { return bytes_capacity_; };
  inline const void* Data() const { return data_; }
  inline void SetTemporary(bool v) { temporary_ = v ? 1 : 0; }
  inline bool IsTemporary() const { return temporary_; };

  template <typename T>
  T* MutableData() {
    return reinterpret_cast<T*>(const_cast<void*>(data_));
  }

 private:
  union {
    struct {
      uint64_t temporary_ : 1;
      uint64_t size_ : 31;  // corresponds to logical address
      uint64_t reserved_ : 1;
      uint64_t bytes_capacity_ : 31;
    };
  };
  const void* data_ = nullptr;
};

template <typename T>
class Vector {
 public:
  using value_type = T;
  Vector() {}
  Vector(const T* data, size_t size, size_t capacity = 0) {
    if (capacity == 0) {
      capacity = size;
    }
    VectorData vdata(data, size, capacity * sizeof(T));
    vec_data_ = vdata;
  }
  Vector(VectorData vdata) { vec_data_ = vdata; }
  Vector(const std::vector<T>& vec) {
    if constexpr (std::is_same_v<T, Bit>) {
      static_assert(sizeof(T) == -1, "unsupported constructor");
    } else {
      VectorData vdata(vec.data(), vec.size(), vec.capacity() * sizeof(T));
      vec_data_ = vdata;
    }
  }
  VectorData RawData() { return vec_data_; }

  size_t Size() const { return vec_data_.Size(); }
  size_t ElementSize() const {
    if constexpr (std::is_same_v<Bit, T>) {
      size_t n = Size();
      size_t byte_n = n / 8;
      if (n % 8 > 0) {
        byte_n++;
      }
      return byte_n;
    } else {
      return Size();
    }
  }
  size_t BytesCapacity() const { return vec_data_.BytesCapacity(); }
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
  auto ToStdVector() const {
    if constexpr (std::is_same_v<Bit, T>) {
      std::vector<uint8_t> bits;
      size_t bytes = Size() / 8;
      if (Size() % 8 > 0) {
        bytes++;
      }
      bits.assign(Data(), Data() + bytes);
      return bits;
    } else if constexpr (std::is_same_v<StringView, T>) {
      std::vector<std::string> strs;
      for (size_t i = 0; i < Size(); i++) {
        strs.emplace_back((Data() + i)->str());
      }
      return strs;
    } else {
      std::vector<T> datas;
      datas.assign(Data(), Data() + Size());
      return datas;
    }
  }
  bool IsTemporary() const { return vec_data_.IsTemporary(); }

 private:
  VectorData vec_data_;
};

}  // namespace simd

}  // namespace rapidudf