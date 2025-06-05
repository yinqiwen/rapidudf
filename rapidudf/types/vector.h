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
#include "rapidudf/meta/vector_type_traits.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/pointer.h"
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

class VectorBase {
 public:
  static constexpr size_t kVectorSizeBits = 30;
  static constexpr size_t kVectorCapacityBits = 30;
  static constexpr size_t kMaxVectorSize = ((uint64_t)1 << kVectorSizeBits) - 1;
  static constexpr size_t kMaxVectorCapacity = ((uint64_t)1 << kVectorCapacityBits) - 1;
  template <typename T>
  VectorBase(const T* data = nullptr, size_t size = 0, size_t capacity = 0, bool readonly = true)
      : size_(size), arrow_obj_(0), capacity_(capacity), writable_(readonly ? 0 : 1), data_(data) {
    if (capacity == 0) {
      capacity_ = size_;
    }
  }
  template <typename T>
  explicit VectorBase(const std::vector<T>& vec, bool readonly = true)
      : VectorBase(vec.data(), vec.size(), vec.capacity(), readonly) {}
  inline bool IsArrow() const { return arrow_obj_ != 0; }
  inline void SetReadonly() { writable_ = 0; }
  inline void SetWritable() { writable_ = 1; }
  inline bool IsReadonly() const { return !writable_; };

  inline const void* GetMemory() const { return data_; }

  inline int32_t Size() const { return static_cast<int32_t>(size_); }
  inline size_t Capacity() const { return static_cast<size_t>(capacity_); }

  void SetSize(size_t n) { size_ = n; }

  template <typename T>
  auto Data() const {
    if constexpr (std::is_same_v<Bit, T>) {
      return reinterpret_cast<const uint8_t*>(GetMemory());
    } else {
      return reinterpret_cast<const T*>(GetMemory());
    }
  }

  template <typename T>
  auto MutableData() {
    if constexpr (std::is_same_v<Bit, T>) {
      uint8_t* v = nullptr;
      if (!IsReadonly()) {
        v = const_cast<uint8_t*>(Data<uint8_t>());
      }
      return v;
    } else {
      T* v = nullptr;
      if (!IsReadonly()) {
        v = const_cast<T*>(Data<T>());
      }
      return v;
    }
  }

  static int32_t GetSize(VectorBase v) { return v.Size(); }
  static const void* GetData(VectorBase v) { return v.GetMemory(); }

 protected:
  void CopyFrom(const VectorBase& other) { memcpy(this, &other, sizeof(VectorBase)); }
  union {
    struct {
      uint64_t size_ : 30;
      uint64_t arrow_obj_ : 1;
      uint64_t reserved0_ : 1;
      uint64_t capacity_ : 30;
      uint64_t writable_ : 1;
      uint64_t reserved1_ : 1;
    };
  };
  const void* data_ = nullptr;
};

struct VectorBuildOptions {
  bool readonly = true;
  bool clone = false;
};

template <typename T>
class Vector : public VectorBase {
 public:
  using value_type = T;
  using element_type = typename VectorTypeTraits<T>::ElementType;
  Vector() : VectorBase((element_type*)nullptr) {}
  Vector(const element_type* data, size_t size, size_t capacity = 0, bool readonly = true)
      : VectorBase(data, size, capacity * sizeof(element_type), readonly) {}

  Vector(const std::vector<T>& vec, bool readonly = true) : VectorBase(vec, readonly) {}
  template <typename R, typename U = T, typename std::enable_if<std::is_same<U, Pointer>::value, int>::type = 0>
  Vector(const std::vector<const R*>& vec, bool readonly = true)
      : VectorBase(vec.data(), vec.size(), vec.capacity(), readonly) {}

  Vector(const VectorBase& base) : VectorBase(base) {}

  template <typename R>
  Vector(const Vector<R>& vec) : VectorBase((uint8_t*)nullptr, 0, 0, false) {
    if constexpr (sizeof(element_type) != sizeof(typename VectorTypeTraits<R>::ElementType)) {
      static_assert(sizeof(T) == -1, "unsupported constructor from other simd vector");
    } else {
      CopyFrom(vec);
    }
  }

  VectorBase RawData() const {
    const VectorBase& v = *this;
    return v;
  }

  inline auto operator[](size_t idx) const {
    if constexpr (std::is_same_v<Bit, T>) {
      size_t byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t bits = *(Data() + byte_idx);
      uint8_t v = (bits >> bit_cursor) & 1;
      return Bit(v > 0);
    } else {
      return Data()[idx];
    }
  }
  inline void Set(size_t idx, T v) {
    if (idx >= Size()) {
      throw std::logic_error(fmt::format("Can NOT set at pos:{}", idx));
    }
    if constexpr (std::is_same_v<Bit, T>) {
      size_t byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t* bits = MutableData();
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
    } else {
      MutableData()[idx] = v;
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

  Vector<T> Slice(uint32_t pos, uint32_t len) const {
    if (pos + len > Size()) {
      THROW_OUT_OF_RANGE_ERR((pos + len), Size());
    }
    if constexpr (std::is_same_v<T, Bit>) {
      // static_assert(sizeof(T) == -1, "unsupported subvector for Vector<Bit>");
      auto* data_ptr = Data();
      if (pos % 8 == 0) {
        Vector<T> ret = *this;
        auto* subvector_data_ptr = data_ptr + pos / 8;
        ret.data_ = subvector_data_ptr;
        ret.size_ = len;
        ret.capacity_ = ret.capacity_ - pos;
        return ret;
      } else {
        throw std::logic_error(fmt::format("Can NOT subvector from pos:{}", pos));
      }
    } else {
      Vector<T> ret = *this;
      ret.data_ = Data() + pos;
      ret.size_ = len;
      ret.capacity_ = ret.capacity_ - pos;
      return ret;
    }
  }

  Vector<T> Resize(size_t n) const { return Slice(0, n); }

  auto Data() const { return VectorBase::Data<element_type>(); }
  auto MutableData() { return VectorBase::MutableData<element_type>(); }

  size_t Size() const { return static_cast<size_t>(VectorBase::Size()); }
  size_t SizeByBytes() const {
    if constexpr (std::is_same_v<T, Bit>) {
      size_t n = Size();
      size_t bytes_n = n / 8;
      if (n % 8 != 0) {
        bytes_n++;
      }
      return bytes_n;

    } else {
      return Size() * sizeof(element_type);
    }
  }
};

using simd_vector_bool = Vector<Bit>;
using simd_vector_i8 = Vector<int8_t>;
using simd_vector_u8 = Vector<uint8_t>;
using simd_vector_i16 = Vector<int16_t>;
using simd_vector_u16 = Vector<uint16_t>;
using simd_vector_i32 = Vector<int32_t>;
using simd_vector_u32 = Vector<uint32_t>;
using simd_vector_i64 = Vector<int64_t>;
using simd_vector_u64 = Vector<uint64_t>*;
using simd_vector_f32 = Vector<float>;
using simd_vector_f64 = Vector<double>;
using simd_vector_string = Vector<StringView>;
using simd_vector_pointer = Vector<Pointer>;
}  // namespace rapidudf