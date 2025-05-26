/*
 * Copyright (c) 2025 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include <arrow/array/array_binary.h>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arrow/api.h"
#include "arrow/type_traits.h"
#include "rapidudf/common/arrow_helper.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
static constexpr uint32_t kVectorUnitSize = 64;
template <typename T>
class Vector;
namespace functions {
template <typename T, OpToken op>
int simd_vector_find(const Vector<T>* data, T v);
template <typename T>
T simd_vector_sum(const Vector<T>* left);
template <typename T>
T simd_vector_avg(const Vector<T>* left);
template <typename T>
T simd_vector_reduce_max(const Vector<T>* left);
template <typename T>
T simd_vector_reduce_min(const Vector<T>* left);

size_t simd_vector_bits_count_true(const Vector<Bit>* left);

}  // namespace functions

class VectorBase {
 public:
  VectorBase(std::shared_ptr<arrow::Array> p) : data_(p) {}

  const uint8_t* GetMemory() const { return data_->data()->GetValues<uint8_t>(1, 0); }
  uint8_t* GetMutableMemory() { return const_cast<uint8_t*>(GetMemory()); }
  size_t BytesCapacity() const { return data_->data()->buffers[1]->capacity(); }
  int32_t Size() const { return static_cast<int32_t>(data_->length()); }

  static int32_t GetSize(const VectorBase* v) { return v->Size(); }
  static const uint8_t* GetData(const VectorBase* v) { return v->GetMemory(); }

 protected:
  std::shared_ptr<arrow::Array> data_;
};

template <typename T>
class Vector : public VectorBase {
 public:
  using value_type = typename VectorElementTypeTraits<T>::DataType;
  using arrow_value_type = typename VectorElementTypeTraits<T>::ArrowDataType;
  using value_iterator_func = std::function<std::optional<value_type>()>;

  static absl::StatusOr<Vector> Wrap(arrow::MemoryPool* pool, const T* data, size_t size, size_t capacity = 0,
                                     bool copy = false);
  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const value_iterator_func& iter, size_t length);
  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, size_t size);
  static absl::StatusOr<Vector> Wrap(arrow::MemoryPool* pool, const std::vector<T>& data, bool copy = false) {
    return Wrap(pool, data.data(), data.size(), data.capacity(), copy);
  }
  static absl::StatusOr<Vector> Wrap(arrow::MemoryPool* pool, const absl::Span<T>& data, bool copy = false) {
    return Wrap(pool, data.data(), data.size(), data.size(), copy);
  }

  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const T* data, size_t size) {
    return Wrap(pool, data, size, size, true);
  }

  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const std::vector<T>& data) {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool> || std::is_same_v<T, std::string> ||
                  std::is_same_v<T, std::string_view>) {
      auto iter = data.begin();
      return Make(
          pool,
          [&]() -> std::optional<value_type> {
            if (iter == data.end()) {
              return {};
            }
            value_type v = *iter;
            iter++;
            return v;
          },
          data.size());
    } else {
      return Wrap(pool, data.data(), data.size(), data.capacity(), true);
    }
  }

  explicit Vector(const std::vector<T>& data);

  size_t Size() const { return VectorBase::Size(); }
  auto Data(size_t offset = 0) const {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      return data_->data()->GetValues<uint8_t>(1, offset / 8);
    } else if constexpr (std::is_same_v<value_type, StringView>) {
      auto actual_array = static_cast<arrow::FixedSizeBinaryArray*>(data_.get());
      return reinterpret_cast<const StringView*>(actual_array->GetValue(offset));
    } else {
      auto actual_array = static_cast<typename VectorElementTypeTraits<T>::ArrayType*>(data_.get());
      return actual_array->raw_values() + offset;
    }
  }
  auto MutableData(size_t offset = 0) {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      return data_->data()->GetMutableValues<uint8_t>(1, offset / 8);
    } else if constexpr (std::is_same_v<value_type, StringView>) {
      auto actual_array = static_cast<arrow::FixedSizeBinaryArray*>(data_.get());
      const uint8_t* data = actual_array->GetValue(offset);
      return const_cast<StringView*>(reinterpret_cast<const StringView*>(data));
    } else {
      auto actual_array = static_cast<typename VectorElementTypeTraits<T>::ArrayType*>(data_.get());
      return const_cast<arrow_value_type*>(actual_array->raw_values() + offset);
    }
  }

  auto Value(size_t offset) const {
    if constexpr (std::is_same_v<value_type, StringView>) {
      arrow::FixedSizeBinaryArray* actual_array = static_cast<arrow::FixedSizeBinaryArray*>(data_.get());
      StringView s;
      memcpy(&s, actual_array->GetValue(offset), sizeof(StringView));
      return s;
    } else {
      auto actual_array = static_cast<typename VectorElementTypeTraits<T>::ArrayType*>(data_.get());
      auto v = actual_array->Value(offset);
      if constexpr (std::is_same_v<Pointer, T>) {
        uint64_t ptr_val = v;
        return Pointer(ptr_val);
      } else {
        return value_type(v);
      }
    }
  }

  auto operator[](size_t idx) const { return Value(idx); }

  void Set(size_t offset, T v);

  Vector<T> Slice(uint32_t offset, uint32_t length) const;

  void Resize(size_t new_size);

  int Find(T v) { return functions::simd_vector_find<T, OP_EQUAL>(this, v); }
  int FindNeq(T v) { return functions::simd_vector_find<T, OP_NOT_EQUAL>(this, v); }
  int FindGt(T v) { return functions::simd_vector_find<T, OP_GREATER>(this, v); }
  int FindGe(T v) { return functions::simd_vector_find<T, OP_GREATER_EQUAL>(this, v); }
  int FindLt(T v) { return functions::simd_vector_find<T, OP_LESS>(this, v); }
  int FindLe(T v) { return functions::simd_vector_find<T, OP_LESS_EQUAL>(this, v); }

  T ReduceSum() { return functions::simd_vector_sum(this); }
  T ReduceAvg() { return functions::simd_vector_avg(this); }
  T ReduceMax() { return functions::simd_vector_reduce_max(this); }
  T ReduceMin() { return functions::simd_vector_reduce_min(this); }

  template <typename U = T, typename std::enable_if<std::is_same<U, Bit>::value, int>::type = 0>
  size_t CountTrue() {
    return functions::simd_vector_bits_count_true(*this);
  }
  template <typename U = T, typename std::enable_if<std::is_same<U, Bit>::value, int>::type = 0>
  size_t CountFalse() {
    return Size() - CountTrue();
  }

 private:
  Vector(std::shared_ptr<arrow::Array> p) : VectorBase(p) {}
};
template <typename T>
Vector<T>::Vector(const std::vector<T>& data) : VectorBase(nullptr) {
  auto result = Vector<T>::Wrap(nullptr, data, false);
  std::shared_ptr<arrow::Array> p;
  if (result.ok()) {
    p = result->data_;
  }
  data_ = p;
}

template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Wrap(arrow::MemoryPool* pool, const T* data, size_t size, size_t capacity,
                                          bool copy) {
  if (capacity == 0) {
    capacity = size;
  }
  if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
    static_assert(sizeof(T) == -1, "unsupported bool/bit type to wrap");
  }

  std::shared_ptr<arrow::Array> array;
  if (copy) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
    }
    auto actual_builder = static_cast<typename VectorElementTypeTraits<T>::BuilderType*>(builder.get());
    if constexpr (std::is_pointer_v<T> || std::is_same_v<T, Pointer>) {
      const uint64_t* int_data = reinterpret_cast<const uint64_t*>(data);
      status = actual_builder->AppendValues(int_data, size);
    } else {
      status = actual_builder->AppendValues(data, size);
    }

    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder append values failed:{}", status.ToString());
    }
    status = actual_builder->Finish(&array);
    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
    }
  } else {
    typename arrow::CTypeTraits<typename VectorElementTypeTraits<T>::ArrowDataType>::ArrowType arrow_type;
    size_t byte_length = 0;
    if (arrow_type.byte_width() > 0) {
      byte_length = size * arrow_type.byte_width();
    } else {
      byte_length = size / 8;
      if (size % 8 > 0) {
        byte_length += 1;
      }
    }
    std::shared_ptr<arrow::Buffer> buffer = arrow::Buffer::Wrap(data, byte_length);
    std::shared_ptr<arrow::ArrayData> array_data =
        arrow::ArrayData::Make(get_arrow_dtype<T>(), size, {nullptr, buffer});
    array = arrow::MakeArray(array_data);
  }
  return Vector<T>(array);
}
template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Make(arrow::MemoryPool* pool, const value_iterator_func& iter, size_t length) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
  }
  auto actual_builder = static_cast<typename VectorElementTypeTraits<T>::BuilderType*>(builder.get());
  status = actual_builder->Reserve(length);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder reserve:{} failed:{}", length, status.ToString());
  }
  while (1) {
    auto val = iter();
    if (val.has_value()) {
      if constexpr (std::is_same_v<value_type, StringView>) {
        const StringView& s = val.value();
        status = actual_builder->Append(reinterpret_cast<const uint8_t*>(&s));
      } else {
        status = actual_builder->Append(val.value());
      }
      if (!status.ok()) {
        RUDF_LOG_RETURN_FMT_ERROR("Builder append  failed:{}", status.ToString());
      }
    } else {
      break;
    }
  }
  std::shared_ptr<arrow::Array> array_data;
  status = actual_builder->Finish(&array_data);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
  }
  return Vector<T>(array_data);
}
template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Make(arrow::MemoryPool* pool, size_t size) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
  }
  auto actual_builder = static_cast<typename VectorElementTypeTraits<T>::BuilderType*>(builder.get());
  status = actual_builder->AppendEmptyValues(size);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder append empty values failed:{}", status.ToString());
  }
  std::shared_ptr<arrow::Array> array_data;
  status = actual_builder->Finish(&array_data);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
  }
  return Vector<T>(array_data);
}
template <typename T>
Vector<T> Vector<T>::Slice(uint32_t offset, uint32_t length) const {
  auto new_data = data_->Slice(static_cast<int64_t>(offset), static_cast<int64_t>(length));
  return Vector<T>(new_data);
}
template <typename T>
void Vector<T>::Resize(size_t new_size) {
  if (new_size >= static_cast<size_t>(data_->length())) {
    return;
  }
  data_ = data_->Slice(0, new_size);
}
template <typename T>
void Vector<T>::Set(size_t offset, T v) {
  auto* p = MutableData(offset);
  if constexpr (std::is_same_v<Bit, T> || std::is_same_v<bool, T>) {
    size_t bit_cursor = offset % 8;
    do {
      uint8_t current_byte_v = *p;
      uint8_t new_byte_v = 0;
      if (v) {
        new_byte_v = bit_set(*p, bit_cursor);
      } else {
        new_byte_v = bit_clear(*p, bit_cursor);
      }
      if (__sync_val_compare_and_swap(p, current_byte_v, new_byte_v)) {
        break;
      }
    } while (1);
  } else {
    *p = v;
  }
}

using simd_vector_bool = Vector<Bit>*;
using simd_vector_i8 = Vector<int8_t>*;
using simd_vector_u8 = Vector<uint8_t>*;
using simd_vector_i16 = Vector<int16_t>*;
using simd_vector_u16 = Vector<uint16_t>*;
using simd_vector_i32 = Vector<int32_t>*;
using simd_vector_u32 = Vector<uint32_t>*;
using simd_vector_i64 = Vector<int64_t>*;
using simd_vector_u64 = Vector<uint64_t>*;
using simd_vector_f32 = Vector<float>*;
using simd_vector_f64 = Vector<double>*;
using simd_vector_string = Vector<StringView>*;

}  // namespace rapidudf