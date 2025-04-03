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
#include <stddef.h>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
static constexpr uint32_t kVectorUnitSize = 64;

template <typename T>
class Vector;

namespace functions {
template <typename T, OpToken op>
int simd_vector_find(Vector<T> data, T v);
template <typename T>
T simd_vector_sum(Vector<T> left);
template <typename T>
T simd_vector_avg(Vector<T> left);
template <typename T>
T simd_vector_reduce_max(Vector<T> left);
template <typename T>
T simd_vector_reduce_min(Vector<T> left);

size_t simd_vector_bits_count_true(Vector<Bit> left);

}  // namespace functions

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

class VectorBuf {
 public:
  explicit VectorBuf(const void* data = nullptr, size_t size = 0, size_t bytes_capacity = 0)
      : temporary_(0), size_(size), writable_(0), bytes_capacity_(bytes_capacity), data_(data) {}
  template <typename T>
  explicit VectorBuf(const std::vector<T>& vec) {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      static_assert(sizeof(T) == -1, "unsupported constructor");
    } else {
      temporary_ = 0;
      size_ = vec.size();
      data_ = vec.data();
      bytes_capacity_ = vec.capacity() * sizeof(T);
      writable_ = 0;
    }
  }
  inline size_t Size() const { return size_; }
  inline void SetSize(size_t n) { size_ = n; }
  inline size_t BytesCapacity() const { return bytes_capacity_; };
  inline const void* Data() const { return data_; }
  inline void SetTemporary(bool v) { temporary_ = v ? 1 : 0; }
  inline bool IsTemporary() const { return temporary_ && writable_; };
  inline void SetReadonly(bool v) { writable_ = (v ? 0 : 1); }
  inline bool IsReadonly() const { return !writable_; };

  template <typename T>
  T* MutableData() {
    return reinterpret_cast<T*>(const_cast<void*>(data_));
  }
  template <typename T>
  const T* ReadableData() {
    return reinterpret_cast<const T*>(data_);
  }

 private:
  union {
    struct {
      uint64_t temporary_ : 1;
      uint64_t size_ : 31;  // corresponds to logical address
      uint64_t writable_ : 1;
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
    VectorBuf vdata(data, size, capacity * sizeof(T));
    vec_data_ = vdata;
  }
  Vector(VectorBuf vdata) {
    size_t byte_capacity = vdata.BytesCapacity();
    if (byte_capacity == 0) {
      if constexpr (std::is_same_v<T, Bit>) {
        byte_capacity = (vdata.Size() + 7) / 8;
      } else {
        byte_capacity = vdata.Size() * sizeof(T);
      }
      vec_data_ = VectorBuf(vdata.Data(), vdata.Size(), byte_capacity);
      vec_data_.SetReadonly(vdata.IsReadonly());
    } else {
      vec_data_ = vdata;
    }
  }
  Vector(const std::vector<T>& vec) {
    if constexpr (std::is_same_v<T, Bit>) {
      static_assert(sizeof(T) == -1, "unsupported constructor");
    } else {
      VectorBuf vdata(vec.data(), vec.size(), vec.capacity() * sizeof(T));
      vdata.SetReadonly(true);
      vec_data_ = vdata;
    }
  }
  Vector(std::vector<T>& vec) {
    if constexpr (std::is_same_v<T, Bit>) {
      static_assert(sizeof(T) == -1, "unsupported constructor");
    } else {
      VectorBuf vdata(vec.data(), vec.size(), vec.capacity() * sizeof(T));
      vec_data_ = vdata;
      vec_data_.SetReadonly(false);
    }
  }
  VectorBuf RawData() { return vec_data_; }

  auto begin() const {
    if constexpr (std::is_same_v<Bit, T>) {
      return reinterpret_cast<const uint8_t*>(vec_data_.Data());
    } else {
      return reinterpret_cast<const T*>(vec_data_.Data());
    }
  }
  auto end() const {
    if constexpr (std::is_same_v<Bit, T>) {
      return reinterpret_cast<const uint8_t*>(vec_data_.Data()) + ElementSize();
    } else {
      return reinterpret_cast<const T*>(vec_data_.Data()) + ElementSize();
    }
  }

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
  void Set(size_t idx, T v) {
    if (idx >= Size()) {
      throw std::logic_error(fmt::format("Can NOT set at pos:{}", idx));
    }
    if constexpr (std::is_same_v<Bit, T>) {
      size_t byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t* bits = vec_data_.MutableData<uint8_t>();
      do {
        uint8_t current_byte_v = bits[byte_idx];
        uint8_t new_byte_v = 0;
        if (v) {
          new_byte_v = bit_set(bits[byte_idx], bit_cursor);
        } else {
          new_byte_v = bit_clear(bits[byte_idx], bit_cursor);
        }
        if (__sync_val_compare_and_swap(&bits[byte_idx], current_byte_v, new_byte_v)) {
          break;
        }
      } while (1);
      // uint8_t current_byte_v = bits[byte_idx];
      // uint8_t new_byte_v = 0;
      // if (v) {
      //   new_byte_v = bit_set(bits[byte_idx], bit_cursor);
      // } else {
      //   new_byte_v = bit_clear(bits[byte_idx], bit_cursor);
      // }
      // bits[byte_idx] = new_byte_v;
    } else {
      vec_data_.MutableData<T>()[idx] = v;
    }
  }

  Vector<T> SubVector(uint32_t pos, uint32_t len) {
    if (pos + len > Size()) {
      THROW_OUT_OF_RANGE_ERR((pos + len), Size());
    }
    if constexpr (std::is_same_v<T, Bit>) {
      // static_assert(sizeof(T) == -1, "unsupported subvector for Vector<Bit>");
      auto* data_ptr = Data();
      if (pos % 8 == 0) {
        auto* subvector_data_ptr = data_ptr + pos / 8;
        VectorBuf subvector_data(subvector_data_ptr, len, (len + 7) / 8);
        return Vector<T>(subvector_data);
      } else {
        throw std::logic_error(fmt::format("Can NOT subvector from pos:{}", pos));
      }

    } else {
      auto* data_ptr = Data();
      auto* subvector_data_ptr = data_ptr + pos;
      VectorBuf subvector_data(subvector_data_ptr, len, sizeof(T) * len);
      return Vector<T>(subvector_data);
    }
  }

  int Find(T v) { return functions::simd_vector_find<T, OP_EQUAL>(*this, v); }
  int FindNeq(T v) { return functions::simd_vector_find<T, OP_NOT_EQUAL>(*this, v); }
  int FindGt(T v) { return functions::simd_vector_find<T, OP_GREATER>(*this, v); }
  int FindGe(T v) { return functions::simd_vector_find<T, OP_GREATER_EQUAL>(*this, v); }
  int FindLt(T v) { return functions::simd_vector_find<T, OP_LESS>(*this, v); }
  int FindLe(T v) { return functions::simd_vector_find<T, OP_LESS_EQUAL>(*this, v); }

  T ReduceSum() { return functions::simd_vector_sum(*this); }
  T ReduceAvg() { return functions::simd_vector_avg(*this); }
  T ReduceMax() { return functions::simd_vector_reduce_max(*this); }
  T ReduceMin() { return functions::simd_vector_reduce_min(*this); }

  template <typename U = T, typename std::enable_if<std::is_same<U, Bit>::value, int>::type = 0>
  size_t CountTrue() {
    return functions::simd_vector_bits_count_true(*this);
  }
  template <typename U = T, typename std::enable_if<std::is_same<U, Bit>::value, int>::type = 0>
  size_t CountFalse() {
    return Size() - CountTrue();
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
  Vector<T> Resize(size_t n) const {
    if (vec_data_.Size() == 0) {
      return *this;
    }
    if (n > vec_data_.Size()) {
      THROW_OUT_OF_RANGE_ERR(n, vec_data_.Size());
    }
    auto new_vec_data = VectorBuf(vec_data_.Data(), n, vec_data_.BytesCapacity());
    new_vec_data.SetReadonly(vec_data_.IsReadonly());
    return Vector<T>(new_vec_data);
  }
  VectorBuf& GetVectorBuf() { return vec_data_; }

  void SetReadonly(bool v) { vec_data_.SetReadonly(v); }
  bool IsReadonly() const { return vec_data_.IsReadonly(); }

 private:
  VectorBuf vec_data_;
};
}  // namespace rapidudf